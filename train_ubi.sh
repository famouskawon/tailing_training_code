#!/bin/bash

#ubi
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 ubi_train.py -c configs/dataset/ubi_224.yaml configs/loss/ce.yaml configs/model/ViViT_224-B.yaml configs/optimizer/sgd.yaml configs/scheduler/graual_warmup.yaml --save_dir weights/train_ubi/224/B

#hockey
#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 ubi_train.py -c configs/dataset/hockey_224.yaml configs/loss/ce.yaml configs/model/ViViT_224-B.yaml configs/optimizer/sgd.yaml configs/scheduler/graual_warmup.yaml --save_dir weights/hockey/224/B