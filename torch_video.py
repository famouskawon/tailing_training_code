import torch
import torch.nn as nn

# model_name = "slow_r50"
# model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
# #print(model)
# layers = list(model.blocks.children())
# _layers = layers[:-1]
# feature_extractor = nn.Sequential(*_layers)
# # 2. Classifier:
# fc = layers[-1]
# fc.proj = nn.Linear(in_features=2048, out_features=2, bias=True)

# print(model)

# img = torch.ones([1, 3, 16, 224, 224]).to(device="cuda")
# model.to(device="cuda", non_blocking=True)
# out = model(img)
# print(out)

import torch
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from trainer.model.module import *
#/workspace/tailing/trainer/model/module.py
# helpers

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101
from VideoTransformer.video_transformer import Mymodel

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
       
    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return 

img = torch.ones([1, 16, 3, 224, 224]).to(device="cuda")
#checkpoint = torch.load("/workspace/tailing/weights/pretrained/vivit_model.pth")
#model = CNNLSTM()
model = Mymodel()
model.to(device="cuda", non_blocking=True)
out = model(img)
print(out)