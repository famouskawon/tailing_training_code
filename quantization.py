import os
import torch
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

quantized_model = quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
#python inference.py -c configs/dataset/grid_512.yaml configs/loss/ce.yaml configs/model/TSViT_512-B.yaml configs/infer/checkpoint512B.yaml

f=print_size_of_model(model,"fp32")
q=print_size_of_model(quantized_model,"int8")

dummy_inputs = torch.ones([1, 16, 1, 512, 512])
traced_model = torch.jit.trace(quantized_model, dummy_inputs)
#traced_model.to(device="cuda")
torch.jit.save(traced_model, os.path.join('/workspace/tailing/weights/quantization', f'TSViT_best_quan_val.ckp'))

loaded_quantized_model = torch.jit.load("/workspace/tailing/weights/quantization/TSViT_best_quan_val.ckp")
q=print_size_of_model(loaded_quantized_model,"int8")
#print(loaded_quantized_model)