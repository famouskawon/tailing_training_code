import os
import sys
import logging
import datetime
import time
import copy

import wandb

from tqdm import tqdm
from torchsummary import summary as model_summary

import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.utils.prune as prune
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report

import theconf
from theconf import Config as C
import random
import numpy as np

import trainer
from tensorboardX import SummaryWriter

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True

def main(flags):
    if flags.save_dir is not None:
        flags.save_dir = os.path.join(flags.save_dir)
        log_dir = f"{os.path.join(flags.save_dir, 'logs')}" 
    else:
        log_dir = None

    sw = SummaryWriter(log_dir=log_dir)

    set_seeds(flags.seed)
    device = torch.device(type= 'cuda', index=max(0, int(os.environ.get('LOCAL_RANK', -1))))
    if flags.local_rank >= 0:
        dist.init_process_group(backend=flags.dist_backend, init_method= 'env://', world_size=int(os.environ['WORLD_SIZE']))
        torch.cuda.set_device(device)

        flags.is_master = flags.local_rank < 0 or dist.get_rank() == 0
        if flags.is_master:
            logging.info(f"local batch={C.get()['dataset']['train']['batch_size']}, world_size={dist.get_world_size()} ----> total batch={C.get()['dataset']['train']['batch_size'] * dist.get_world_size()}")
            logging.info(f"lr {C.get()['optimizer']['lr']} -> {C.get()['optimizer']['lr'] * dist.get_world_size()}")
        C.get()['optimizer']['lr'] *= dist.get_world_size()
        flags.optimizer_lr = C.get()['optimizer']['lr']
        

    torch.backends.cudnn.benchmark = True
    model = trainer.model.create(C.get()['architecture'])
    model.to(device=device, non_blocking=True)
    #checkpoint = torch.load(C.get()['inference']['checkpoint'], map_location=device)['model']
    #if flags.local_rank >= 0:
    #    model = DDP(model, device_ids=[flags.local_rank], output_device=flags.local_rank, find_unused_parameters=True)

    #model.load_state_dict(checkpoint)


    #load model
    
    try:
        checkpoint = torch.load(C.get()['inference']['checkpoint'], map_location=device)['model']
        for key in list(checkpoint.keys()):
            if 'module.' in key:
                checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]
        model.load_state_dict(checkpoint)

    except Exception as e:
        raise (e)
    
    
    
    image_size = C.get()['architecture']['params']['image_size']

    if flags.is_master:
        model_summary(model, (16, 1, image_size, image_size))
        

    #data loader    
    train_loader, train_sampler = trainer.dataset.create(C.get()['dataset'],
                                              int(os.environ.get('WORLD_SIZE', 1)), 
                                              int(os.environ.get('LOCAL_RANK', -1)),
                                              mode='train')
    logging.info(f'[Dataset] | train_examples: {len(train_loader)}')
    test_loader, _ = trainer.dataset.create(C.get()['dataset'],
                                              mode='test')
    logging.info(f'[Dataset] | test_examples: {len(test_loader)}')

    optimizer = trainer.optimizer.create(C.get()['optimizer'], model.parameters())
    lr_scheduler = trainer.scheduler.create(C.get()['scheduler'], optimizer)

    criterion = trainer.loss.create(C.get()['loss']).to(device=device, non_blocking=True)

    #if flags.local_rank >= 0:
    #    for name, x in model.state_dict().items():
    #        dist.broadcast(x, 0)
    #    torch.cuda.synchronize()

    

    #기존 모델 inference
    model.eval()
    inference(model, test_loader, device, flags)
    
    #create prune model
    prune_model = copy.deepcopy(model)

    start = time.time()
    new_model = pruning(epochs = C.get()['scheduler']['epoch'],
            model = prune_model,
            dataloader = train_loader, test_loader= test_loader,
            criterion = criterion, optimizer = optimizer, lr_scheduler = lr_scheduler,
            device = device, flags = flags, iter_per_epoch = 10, train_sampler = train_sampler
            )
    end = time.time()
    print(f'[takes {(end-start)} seconds]')

    print(prune_model == new_model)
    
    


def pruning(epochs, model, dataloader, test_loader, criterion, optimizer, lr_scheduler, device, flags, iter_per_epoch, train_sampler):
    l1_regularization_strength = 0
    l2_regularization_strength = 1e-4
    best_val_acc, best_val_epoch, best_val_report, = 0, 0, 0
    best_loss, best_loss_acc, best_loss_epoch, best_loss_report = 100, 0, 0, 0

    #print(model.space_transformer.layers[0][1].fn.net2[0])
    
    parameters_to_prune = []
    for l in range(12):
        parameters_to_prune.append((model.space_transformer.layers[l][0].fn.to_q, 'weight'))
        parameters_to_prune.append((model.space_transformer.layers[l][0].fn.to_k, 'weight'))
        parameters_to_prune.append((model.space_transformer.layers[l][0].fn.to_v, 'weight'))

        parameters_to_prune.append((model.space_transformer.layers[l][1].fn.net2[0], 'weight'))
        parameters_to_prune.append((model.temporal_transformer.layers[l][1].fn.net2[0], 'weight'))

        parameters_to_prune.append((model.temporal_transformer.layers[l][0].fn.to_q, 'weight'))
        parameters_to_prune.append((model.temporal_transformer.layers[l][0].fn.to_k, 'weight'))
        parameters_to_prune.append((model.temporal_transformer.layers[l][0].fn.to_v, 'weight'))

    prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.2,
        )

    
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    
    for i in range(12):
        print(
            "Sparsity in Layer {}-th key weight: {:.2f}%".format(
                i+1,
                100. * float(torch.sum(model.space_transformer.layers[i][0].fn.to_q.weight == 0))
                / float(model.space_transformer.layers[i][0].fn.to_q.weight.nelement())
            )
            )
    
    return model
    #model_summary(model, (16, 1, 512, 512))

    #if flags.save_dir is not None:
    #    torch.save({
    #        'model': model.state_dict(),
    #    }, os.path.join(flags.save_dir, f'TSViT_best_val_prune.ckp'))

    

    """
    for i in range(epochs // iter_per_epoch):
        print("Pruning and Finetuning {}/{}".format(i + 1, epochs))
        print("Pruning...")

        
        for j in range(iter_per_epoch):
            epoch = i * 10 + j

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            
            model.train()
            train_loss, train_acc = train_one_epoch(epoch, model, dataloader, criterion, optimizer,lr_scheduler, device, flags,l1_regularization_strength,l2_regularization_strength)

            model.eval()
            test_loss, test_acc, report = evaluation(epoch, model, test_loader, criterion, device, flags)

            if best_val_acc < test_acc and epoch > 0:
                best_val_acc = test_acc
                best_val_epoch = epoch
                best_val_report = report
                if flags.save_dir is not None:
                    torch.save({
                        'model': model.state_dict(),
                    }, os.path.join(flags.save_dir, f'TSViT_best_val_prune.ckp'))
            logging.info(f'[Best] Acc: {best_val_acc * 100}% Epochs: {best_val_epoch}')
        
                
            if train_loss < best_loss and epoch > 0:
                best_loss = train_loss
                best_loss_acc = test_acc
                best_loss_epoch = epoch
                best_loss_report = report
                if flags.save_dir is not None:
                    torch.save({
                        'model': model.state_dict(),
                    }, os.path.join(flags.save_dir, f'TSViT_best_loss_prune.ckp'))
            logging.info(f'[Best] Acc: {best_loss_acc * 100}% Loss: {best_loss} epochs: {best_loss_epoch}')

            
            torch.cuda.synchronize()

    return {
        'best_acc': best_val_acc,
        'best_val_epoch': best_val_epoch,
        'best_loss': best_loss,
        'best_loss_acc': best_loss_acc,
        'best_loss_epoch': best_loss_epoch,
        'best_val_report': best_val_report,
        'best_loss_report': best_loss_report
    }
    """

def train_one_epoch(epoch, model, dataloader, criterion, optimizer,lr_scheduler, device, flags, l1_regularization_strength=0,
                l2_regularization_strength=1e-4,):
        
    one_epoch_loss = 0
    train_total = 0
    train_hit = 0

    for step, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="[Train] |{:3d}e".format(epoch), disable=not flags.is_master):
        image = image.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)
        # torch.Size([16, 1, 1536, 1536])
        y_pred = model(image)
        loss = criterion(y_pred, label)
        optimizer.zero_grad(set_to_none=True)
        
        """
        l1_reg = torch.tensor(0.).to(device)
        for module in model.modules():
            mask = None
            weight = None
            for name, buffer in module.named_buffers():
                if name == "weight_mask":
                    mask = buffer
            for name, param in module.named_parameters():
                if name == "weight_orig":
                    weight = param
            # We usually only want to introduce sparsity to weights and prune weights.
            # Do the same for bias if necessary.
            if mask is not None and weight is not None:
                l1_reg += torch.norm(mask * weight, 1)
        """
        #loss += l1_regularization_strength
        
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        _, y_pred = y_pred.max(1)
        train_hit += y_pred.detach().eq(label).sum()
        train_total += image.shape[0]

        one_epoch_loss += loss.item()
    if flags.is_master:
        logging.info(f'[Train] Acc: {train_hit / train_total} Losse: {one_epoch_loss / len(dataloader)}')        
    
    return one_epoch_loss / len(dataloader), train_hit / train_total

@torch.no_grad()
def evaluation(epoch, model, dataloader, criterion, device, flags):
    one_epoch_loss = 0
    test_total = 0
    test_hit = 0

    y_true_total = []
    y_pred_total = []

    for step, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Test |{:3d}e".format(epoch), disable=not flags.is_master):
        image = image.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)

        y_pred = model(image)
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


if __name__ == '__main__':
    parser = theconf.ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--seed', type=lambda x: int(x, 0), default='0xC0FFEE', help='set seed (default:0xC0FFEE)')
    parser.add_argument('--save_dir', default=None, type=str, help='modrl save_dir')

    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m torch.distributed.launch\'.')
    
    flags = parser.parse_args()
    #wandb.init(project = 'Tailing project', entity='chowk', name = "ViViT tailing_512B_F16_0926_prune", reinit=True)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', stream=sys.stderr)

    history = main(flags)
    
    # logging.info(f'[Done] best accuracy:{(history["best_acc"]) * 100:.2f}% epoch: {history["best_epoch"]}')
    
    
    
    print(history["best_val_report"])
    print(history["best_loss_report"])