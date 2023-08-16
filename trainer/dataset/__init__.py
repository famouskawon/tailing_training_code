import torch
import torchvision
import torchvision.transforms as transforms
import os 
from .gridDataset import gridDatasetTrain, gridDatasetTest
from .UBIDataset import UBIDatasetTrain, UBIDatasetTest


# dataloader sampler
def create(config, world_size=1, local_rank=-1, mode='train'):
    
    params = config[mode]
    if params.get('preprocess', None) is not None:
        transformers = transforms.Compose([preprocess(t) for t in params['preprocess']] )
    else:
        transformers = None
        
    if mode == 'train':
        dataset = gridDatasetTrain(params['path'], transformers=transformers)
        # dataset = gridDatasetTest(params['path'], transformers=transformers)
    elif mode == 'test':
        dataset = gridDatasetTest(params['path'], transformers=transformers)
    else:
        raise AttributeError(f'not support dataset mode: {mode}')

    if local_rank >= 0 and mode == 'train':
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size, 
            rank=local_rank, 
            shuffle=params.get('shuffle', False)
        )
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        num_workers=0,#os.cpu_count(),
        pin_memory=True,
        drop_last=params.get('drop_last', False),
        sampler=sampler
    )

    return dataloader, sampler

def preprocess(config):
    if config['type'] == 'pad':
        return transforms.Pad(**config['params'])
    elif config['type'] == 'randomcrop':
        return transforms.RandomCrop(**config['params'])
    elif config['type'] == 'horizontal':
        return transforms.RandomHorizontalFlip()
    elif config['type'] == 'tensor':
        return transforms.ToTensor()
    elif config['type'] == 'normalize':
        return transforms.Normalize(**config['params'])


"""
# UBI

def create(config, world_size=1, local_rank=-1, mode='train'):
    
    params = config[mode]
    if mode=="train":
        transformers = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomHorizontalFlip(p=0.3),
            torchvision.transforms.ColorJitter(brightness=0.1),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transformers = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        
    if mode == 'train':
        dataset = UBIDatasetTrain(params['path'], transformers=transformers)
        # dataset = gridDatasetTest(params['path'], transformers=transformers)
    elif mode == 'test':
        dataset = UBIDatasetTest(params['path'], transformers=transformers)
    else:
        raise AttributeError(f'not support dataset mode: {mode}')

    if local_rank >= 0 and mode == 'train':
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size, 
            rank=local_rank, 
            shuffle=params.get('shuffle', False)
        )
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        num_workers=0,#os.cpu_count(),
        pin_memory=True,
        drop_last=params.get('drop_last', False),
        sampler=sampler
    )

    return dataloader, sampler

def preprocess(config):
    if config['type'] == 'pad':
        return transforms.Pad(**config['params'])
    elif config['type'] == 'randomcrop':
        return transforms.RandomCrop(**config['params'])
    elif config['type'] == 'horizontal':
        return transforms.RandomHorizontalFlip()
    elif config['type'] == 'tensor':
        return transforms.ToTensor()
    elif config['type'] == 'normalize':
        return transforms.Normalize(**config['params'])
"""