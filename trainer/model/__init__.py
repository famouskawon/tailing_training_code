import torch
import torchvision.models as models
from .TSViT import TSViT

def create(config):
    if config['type'] == 'TSViT':
        return TSViT(**config['params'])
    else:
        raise AttributeError(f'not support architecture config: {config}')