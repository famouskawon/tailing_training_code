#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c configs/dataset/grid_256.yaml configs/loss/ce.yaml configs/model/TSViT_256-S.yaml configs/optimizer/adamw.yaml configs/scheduler/graual_warmup.yaml --save_dir weights/train/256/S 
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c configs/dataset/grid_256.yaml configs/loss/ce.yaml configs/model/TSViT_256-B.yaml configs/optimizer/adamw.yaml configs/scheduler/graual_warmup.yaml --save_dir weights/train/256/B
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c configs/dataset/grid_256.yaml configs/loss/ce.yaml configs/model/TSViT_256-L.yaml configs/optimizer/adamw.yaml configs/scheduler/graual_warmup.yaml --save_dir weights/train/256/L


