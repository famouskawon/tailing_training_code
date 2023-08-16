import torch

def create(config):
    if config['type'] == 'ce':
        return torch.nn.CrossEntropyLoss()
    if config['type'] == 'bce':
        return torch.nn.BCELoss(reduction='sum')
    else:
        raise AttributeError(f'not support loss config: {config}')