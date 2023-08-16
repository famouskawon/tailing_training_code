import os
import torch
import torch.nn.utils.prune as prune

import theconf
from theconf import Config as C
import logging
import sys
from sklearn.metrics import classification_report
from tqdm import tqdm
import time

import trainer
from train import evaluation

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

parser = theconf.ConfigArgumentParser(conflict_handler='resolve')
flags = parser.parse_args()
flags.is_master = True

device = torch.device(type= 'cuda')#, index=0)
model = trainer.model.create(C.get()['architecture'])

#model.to(device=device, non_blocking=True)
checkpoint = torch.load(C.get()['inference']['checkpoint'], map_location=device)['model']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
model.load_state_dict(checkpoint)

previous_model = print_size_of_model(model)

parameters_to_prune = ()
for i in range(12):
    parameters_to_prune += (
        (model.temporal_transformer.layers[i][0].fn.to_qkv, 'weight'),
        (model.space_transformer.layers[i][0].fn.to_qkv, 'weight'),
    )

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

pruned_model = print_size_of_model(model)
torch.save(model, os.path.join('/workspace/tailing/weights/prune', f'TSViT_best_prune_val.ckp'))