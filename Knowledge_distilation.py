import os
import sys
import logging
import datetime
import time
import wandb

from tqdm import tqdm
from torchsummary import summary as model_summary

import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report

import theconf
from theconf import Config as C
import random
import numpy as np

import trainer
from trainer.model.TSViT import TSViT
from tensorboardX import SummaryWriter

#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 Knowledge_distilation.py -c configs/dataset/grid_512.yaml configs/loss/ce.yaml configs/model/TSViT_512-B.yaml configs/optimizer/adamw.yaml configs/scheduler/graual_warmup.yaml configs/infer/checkpoint512B.yaml --save_dir weights/train/512/B

class Knowledge_distilation_Loss:
    def __init__(self,):
        self.criterion = self.knowledge_distillation_loss

    def __call__(self, logits, labels, teacher_logits=None):
        return self.criterion(logits, labels, teacher_logits)

    def knowledge_distillation_loss(self, logits, labels, teacher_logits):
        """Logit adjustment loss."""
        alpha = 0.1
        T = 5
        loss = F.cross_entropy(input=logits, target=labels)

        if teacher_logits == None:
            return loss    
        else:
            D_KL = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits/T, dim=1), F.softmax(teacher_logits/T, dim=1)) * (T * T)
            KD_loss =  (1. - alpha)*loss + alpha*D_KL

        return KD_loss

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True


def train_one_epoch(epoch, teacher_model, student_model, dataloader, criterion, optimizer, lr_scheduler, device, flags):
        
    one_epoch_loss = 0
    train_total = 0
    train_hit = 0

    #student model은 학습, teacher model은 학습 x
    teacher_model.eval()
    student_model.train()

    for step, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="[Train] |{:3d}e".format(epoch), disable=not flags.is_master):
        image = image.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)

        #출력은 학생 모델로
        y_pred_student = student_model(image)
        y_pred_teacher = teacher_model(image)
        
        #distilation loss
        loss = criterion(y_pred_student, label, y_pred_teacher)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        _, y_pred = y_pred_student.max(1)
        train_hit += y_pred.detach().eq(label).sum()
        train_total += image.shape[0]

        one_epoch_loss += loss.item()

    if flags.is_master:
        logging.info(f'[Train] Acc: {train_hit / train_total} Losses: {one_epoch_loss / len(dataloader)}')        
    
    return one_epoch_loss / len(dataloader), train_hit / train_total

#evaluation for student model
@torch.no_grad()
def evaluation(epoch, student_model, dataloader, criterion, device, flags):
    one_epoch_loss = 0
    test_total = 0
    test_hit = 0

    y_true_total = []
    y_pred_total = []

    for step, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Test |{:3d}e".format(epoch), disable=not flags.is_master):
        image = image.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)

        y_pred = student_model(image)
        loss = criterion(y_pred, label)
        
        _, y_pred = y_pred.max(1)
        y_true_total.extend(label.cpu().tolist())
        y_pred_total.extend(y_pred.cpu().tolist())
        test_hit += y_pred.detach().eq(label).sum()
        test_total += image.shape[0]

        one_epoch_loss += loss.item()
    if flags.is_master:
        logging.info(f'[Test] Acc: {test_hit / test_total} Loss: {one_epoch_loss / len(dataloader)}')
        print(f'{classification_report(y_true_total, y_pred_total, target_names=["tailing", "normal"], digits=4)}')

    return one_epoch_loss / len(dataloader), test_hit / test_total, classification_report(y_true_total, y_pred_total, target_names=["tailing", "normal"])

@torch.no_grad()
def inference(model, dataloader, device, flags):
    one_epoch_loss = 0
    test_total = 0
    test_hit = 0
    
    tp, tn, fp, fn = 0, 0, 0, 0

    y_true_total = []
    y_pred_total = []

    for step, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Test", disable=not flags.is_master):
        image = image.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)

        y_pred = model(image)             
        _, y_pred = y_pred.max(1)
        y_true_total.extend(label.cpu().tolist())
        y_pred_total.extend(y_pred.cpu().tolist())

        test_hit += y_pred.detach().eq(label).sum()
        test_total += image.shape[0]

        for i in range(y_pred.shape[0]):
            if y_pred[i] == 0:
                if label[i] == 0:
                    tp += 1
                else:
                    fp += 1
            else:
                if label[i] == 0:
                    fn += 1
                else:
                    tn += 1        

    if flags.is_master:
        logging.info(f'[Test] Test Case : {test_total} Acc: {test_hit / test_total}')
        logging.info(f'[Tailing] TP: {tp} FP: {fp} FN: {fn} TN: {tn}')
        print(f'{classification_report(y_true_total, y_pred_total, target_names=["tailing", "normal"], digits=4)}')
  
    return one_epoch_loss / len(dataloader), test_hit / test_total, classification_report(y_true_total, y_pred_total, target_names=["tailing", "normal"])


if __name__ == "__main__":
    parser = theconf.ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--seed', type=lambda x: int(x, 0), default='0xC0FFEE', help='set seed (default:0xC0FFEE)')
    parser.add_argument('--save_dir', default=None, type=str, help='modrl save_dir')

    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m torch.distributed.launch\'.')
    
    #parser.add_argument("--teacher_checkpoint", type=str, default="/workspace/tailing/weights/train/512/B/TSViT_best_val.ckp", help="512-B Checkpoint")
    flags = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', stream=sys.stderr)

    if flags.save_dir is not None:
        flags.save_dir = os.path.join(flags.save_dir)
        log_dir = f"{os.path.join(flags.save_dir, 'logs')}" 
    else:
        log_dir = None

    set_seeds(flags.seed)

    device = torch.device(type= 'cuda', index=max(0, int(os.environ.get('LOCAL_RANK', -1))))
    torch.backends.cudnn.benchmark = True
    if flags.local_rank >= 0:
        dist.init_process_group(backend=flags.dist_backend, init_method= 'env://', world_size=int(os.environ['WORLD_SIZE']))
        torch.cuda.set_device(device)

        flags.is_master = flags.local_rank < 0 or dist.get_rank() == 0
        if flags.is_master:
            logging.info(f"local batch={C.get()['dataset']['train']['batch_size']}, world_size={dist.get_world_size()} ----> total batch={C.get()['dataset']['train']['batch_size'] * dist.get_world_size()}")
            logging.info(f"lr {C.get()['optimizer']['lr']} -> {C.get()['optimizer']['lr'] * dist.get_world_size()}")
        C.get()['optimizer']['lr'] *= dist.get_world_size()
        flags.optimizer_lr = C.get()['optimizer']['lr']

    #load Teacher model 224-B
    Teacher_model = TSViT(image_size = 224, patch_size = 56, dim = 768, depth = 12, heads = 12, mlp_dim = 3072, 
                    dropout=0.1, emb_dropout=0.1, num_frames = 16, num_classes= 2)
    Teacher_model.to(device=device, non_blocking=True)

    #Student Model 224-T
    Student_model = TSViT(image_size = 224, patch_size = 56, dim = 516, depth = 6, heads = 12, mlp_dim = 1024, 
                    dropout=0.1, emb_dropout=0.1, num_frames = 16, num_classes= 2)
    Student_model.to(device=device, non_blocking=True)

    image_size = C.get()['architecture']['params']['image_size']

    #if flags.is_master:
    #    model_summary(Teacher_model, (16, 3, image_size, image_size))
        

    if flags.local_rank >= 0:
        Teacher_model = DDP(Teacher_model, device_ids=[flags.local_rank], output_device=flags.local_rank, find_unused_parameters=True)
        Student_model = DDP(Student_model, device_ids=[flags.local_rank], output_device=flags.local_rank, find_unused_parameters=True)

    #Teacher Checkpoint
    checkpoint = torch.load(C.get()['inference']['checkpoint'], map_location=device)['model']
    Teacher_model.load_state_dict(checkpoint)

    
    
    #data loading
    train_loader, train_sampler = trainer.dataset.create(C.get()['dataset'],
                                              int(os.environ.get('WORLD_SIZE', 1)), 
                                              int(os.environ.get('LOCAL_RANK', -1)),
                                              mode='train')
    logging.info(f'[Dataset] | train_examples: {len(train_loader)}')
    test_loader, _ = trainer.dataset.create(C.get()['dataset'],
                                              mode='test')
    logging.info(f'[Dataset] | test_examples: {len(test_loader)}')

    optimizer = trainer.optimizer.create(C.get()['optimizer'], Student_model.parameters())
    lr_scheduler = trainer.scheduler.create(C.get()['scheduler'], optimizer)
    criterion = Knowledge_distilation_Loss()
   
    eval_criterion = trainer.loss.create(C.get()['loss']).to(device=device, non_blocking=True)
    
    #Teacher model inference
    print("----------------- Teacher Model Infer Result -----------------------")
    inference(Teacher_model, test_loader, device, flags)


    if flags.local_rank >= 0:
        for name, x in Student_model.state_dict().items():
            dist.broadcast(x, 0)
        torch.cuda.synchronize()

        for name, x in Teacher_model.state_dict().items():
            dist.broadcast(x, 0)
        torch.cuda.synchronize()

    best_val_acc, best_val_epoch, best_val_report, = 0, 0, 0
    best_loss, best_loss_acc, best_loss_epoch, best_loss_report = 100, 0, 0, 0

    # student 모델 학습하기
    for epoch in range(C.get()['scheduler']['epoch']):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        Student_model.train()
        start = time.time()
        train_loss, train_acc = train_one_epoch(epoch, Teacher_model, Student_model, train_loader, criterion, optimizer, lr_scheduler, device, flags)
        end = time.time()
        print(f'[takes {(end-start)} seconds]')
        #wandb.log({"train_loss" : train_loss, "train_acc" : train_acc})


        if flags.is_master:
            Student_model.eval()
            test_loss, test_acc, report = evaluation(epoch, Student_model, test_loader, eval_criterion, device, flags)

            if best_val_acc < test_acc and epoch > 0:
                best_val_acc = test_acc
                best_val_epoch = epoch
                best_val_report = report
                if flags.save_dir is not None:
                    torch.save({
                        'epoch': epoch,
                        'model': Student_model.state_dict(),
                        'acc' : best_val_acc,
                    }, os.path.join(flags.save_dir, f'TSViT_best_KD_val.ckp'))
            logging.info(f'[Best] Acc: {best_val_acc * 100}% Epochs: {best_val_epoch}')
            #wandb.log({"best_val_acc" : best_val_acc, "test_acc" : best_val_epoch})
            
            if train_loss < best_loss and epoch > 0:
                best_loss = train_loss
                best_loss_acc = test_acc
                best_loss_epoch = epoch
                best_loss_report = report
                if flags.save_dir is not None:
                    torch.save({
                        'epoch': epoch,
                        'model': Student_model.state_dict(),
                    }, os.path.join(flags.save_dir, f'TSViT_best_KD_loss.ckp'))
            logging.info(f'[Best] Acc: {best_loss_acc * 100}% Loss: {best_loss} epochs: {best_loss_epoch}')                

        torch.cuda.synchronize()
    

    history = {
        'best_acc': best_val_acc,
        'best_val_epoch': best_val_epoch,
        'best_loss': best_loss,
        'best_loss_acc': best_loss_acc,
        'best_loss_epoch': best_loss_epoch,
        'best_val_report': best_val_report,
        'best_loss_report': best_loss_report
    }

    print(history["best_val_report"])
    print(history["best_loss_report"])