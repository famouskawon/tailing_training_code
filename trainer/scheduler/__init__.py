import torch

def create(config, optimizer):
    if config['type'] == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['params'])
    elif config['type'] == 'gradual_warmup':
        from warmup_scheduler import GradualWarmupScheduler

        if config['chain_scheduler']['type'] == 'CosineAnnealingLR':
            chain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['chain_scheduler']['params'])
        else:
            raise AttributeError(f'not support chain_scheduler config: {config["chain_scheduler"]}')

        return GradualWarmupScheduler(optimizer, after_scheduler=chain_scheduler, **config['params'])
        
    else:
        raise AttributeError(f'not support scheduler config: {config}')