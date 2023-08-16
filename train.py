import os
import sys
import logging
import datetime
import time
import wandb

from tqdm import tqdm
from torchsummary import summary as model_summary
from VideoTransformer.video_transformer import Mymodel

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
    """
    ##### transfer learning
    try:
        checkpoint = torch.load("/workspace/tailing/weights/train_ubi/224/B/ViViT_UBI_best_val.ckp")['model']
        for key in list(checkpoint.keys()):
            if 'module.' in key:
                checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]
        model.load_state_dict(checkpoint)
        

    except Exception as e:
        raise (e)
    """
    image_size = C.get()['architecture']['params']['image_size']

    #if flags.is_master:
    #    model_summary(model, (12, 3, 224, 224))
        

    if flags.local_rank >= 0:
        model = DDP(model, device_ids=[flags.local_rank], output_device=flags.local_rank, find_unused_parameters=True)

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

    if flags.local_rank >= 0:
        for name, x in model.state_dict().items():
            dist.broadcast(x, 0)
        torch.cuda.synchronize()

    best_val_acc, best_val_epoch, best_val_report, = 0, 0, 0
    best_loss, best_loss_acc, best_loss_epoch, best_loss_report = 100, 0, 0, 0


    for epoch in range(C.get()['scheduler']['epoch']):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        
        model.train()
        start = time.time()
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, criterion, optimizer, lr_scheduler, device, flags)
        wandb.log({"train_loss" : train_loss, "train_acc" : train_acc})
        end = time.time()
        print(f'[takes {(end-start)} seconds]')

        if flags.is_master:
            model.eval()
            test_loss, test_acc, report = evaluation(epoch, model, test_loader, criterion, device, flags)
            wandb.log({"test_loss" : test_loss, "test_acc" : test_acc})

            if best_val_acc < test_acc and epoch > 0:
                best_val_acc = test_acc
                best_val_epoch = epoch
                best_val_report = report
                if flags.save_dir is not None:
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'acc' : best_val_acc,
                    }, os.path.join(flags.save_dir, f'TSViT_best_val.ckp'))
            logging.info(f'[Best] Acc: {best_val_acc * 100}% Epochs: {best_val_epoch}')
            wandb.log({"best_val_acc" : best_val_acc, "test_acc" : best_val_epoch})
            
            if train_loss < best_loss and epoch > 0:
                best_loss = train_loss
                best_loss_acc = test_acc
                best_loss_epoch = epoch
                best_loss_report = report
                if flags.save_dir is not None:
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                    }, os.path.join(flags.save_dir, f'TSViT_best_loss.ckp'))
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



def train_one_epoch(epoch, model, dataloader, criterion, optimizer, lr_scheduler, device, flags):
        
    one_epoch_loss = 0
    train_total = 0
    train_hit = 0

    for step, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="[Train] |{:3d}e".format(epoch), disable=not flags.is_master):
        image = image.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True)

        
        y_pred = model(image)
        loss = criterion(y_pred, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        _, y_pred = y_pred.max(1)
        train_hit += y_pred.detach().eq(label).sum()
        train_total += image.shape[0]

        one_epoch_loss += loss.item()
    if flags.is_master:
        logging.info(f'[Train] Acc: {train_hit / train_total} Losses: {one_epoch_loss / len(dataloader)}')        
    
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
    
if __name__ == '__main__':
    parser = theconf.ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--seed', type=lambda x: int(x, 0), default='0xC0FFEE', help='set seed (default:0xC0FFEE)')
    parser.add_argument('--save_dir', default=None, type=str, help='modrl save_dir')

    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m torch.distributed.launch\'.')
    
    flags = parser.parse_args()
    wandb.init(project = 'Tailing project', entity='chowk', name = "ViViT-B 16 grid 3fps_rs_tiny2", reinit=True)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', stream=sys.stderr)

    history = main(flags)
    
    # logging.info(f'[Done] best accuracy:{(history["best_acc"]) * 100:.2f}% epoch: {history["best_epoch"]}')
    
    
    
    print(history["best_val_report"])
    print(history["best_loss_report"])
