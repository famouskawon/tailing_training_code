import trainer
from thop import profile, clever_format
import torch
import torch.nn as nn
from trainer.model.TSViT import TSViT
from fvcore.nn import FlopCountAnalysis, flop_count_table
from ptflops import get_model_complexity_info


Large_model = TSViT(image_size = 224, patch_size = 56, dim = 1024, depth = 24, heads = 16, mlp_dim = 4096, 
                    dropout=0.1, emb_dropout=0.1, num_frames = 16, num_classes= 2)

#26.420G 177.286M
Base_model = TSViT(image_size = 224, patch_size = 56, dim = 768, depth = 12, heads = 12, mlp_dim = 3072, 
                    dropout=0.1, emb_dropout=0.1, num_frames = 16, num_classes= 2)
    

#Student Model 512-S
#17.029G 110.405M
Small_model = TSViT(image_size = 224, patch_size = 56, dim = 1024, depth = 6, heads = 16, mlp_dim = 2048, 
                dropout=0.1, emb_dropout=0.1, num_frames = 16, num_classes= 2)

tiny_model = TSViT(image_size = 224, patch_size = 56, dim = 516, depth = 6, heads = 12, mlp_dim = 1024, 
                dropout=0.1, emb_dropout=0.1, num_frames = 16, num_classes= 2)

#84.191G 31.639M
model_name = "slow_r50"
res3dmodel = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=False)
#print(model)
layers = list(res3dmodel.blocks.children())
_layers = layers[:-1]
feature_extractor = nn.Sequential(*_layers)
# 2. Classifier:
fc = layers[-1]
fc.proj = nn.Linear(in_features=2048, out_features=2, bias=True)

#5.120G 3.794M
model_name = "x3d_m"
X3dmodel = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=False)
#print(model)
layers = list(res3dmodel.blocks.children())
_layers = layers[:-1]
feature_extractor = nn.Sequential(*_layers)
# 2. Classifier:
fc = layers[-1]
fc.proj = nn.Linear(in_features=2048, out_features=2, bias=True)

input_img = torch.ones([1, 3, 16, 224, 224])
"""
large_flop = FlopCountAnalysis(Large_model, input_img)
base_flop = FlopCountAnalysis(Base_model, input_img)
small_flop = FlopCountAnalysis(Small_model, input_img)
#print(flops.total()) #27884531472.0
#print(stu_flop.total()) #18,918,948,296.0

print(flop_count_table(base_flop))
#print(flop_count_table(stu_flop))
"""

input = torch.randn(1, 16, 3, 224, 224)
macs, params = profile(tiny_model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
#macs, params = get_model_complexity_info(Base_model, (16,3,224,224), as_strings=True, print_per_layer_stat=True, verbose=True)

#print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print(macs, params)
