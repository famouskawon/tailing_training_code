#!/bin/bash

#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c configs/dataset/grid_224.yaml configs/loss/ce.yaml configs/model/ViViT_224-B.yaml configs/optimizer/sgd.yaml configs/scheduler/graual_warmup.yaml --save_dir weights/train/224/B

#base
#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c configs/dataset/rgb_224.yaml configs/loss/ce.yaml configs/model/ViViT_224-B.yaml configs/optimizer/sgd.yaml configs/scheduler/graual_warmup.yaml --save_dir weights/train/224/B/detect

#tiny
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c configs/dataset/rgb_224.yaml configs/loss/ce.yaml configs/model/ViViT_224-T.yaml configs/optimizer/sgd.yaml configs/scheduler/graual_warmup.yaml --save_dir weights/train/224/T2

#large
#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c configs/dataset/rgb_224.yaml configs/loss/ce.yaml configs/model/ViViT_224-L.yaml configs/optimizer/sgd.yaml configs/scheduler/graual_warmup.yaml --save_dir weights/train/224/L

#from kinects
#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c configs/dataset/rgb_224.yaml configs/loss/ce.yaml configs/model/ViViT_224-B.yaml configs/optimizer/sgd.yaml configs/scheduler/graual_warmup.yaml --save_dir weights/train_init_kinects/224/B

#small
#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c configs/dataset/rgb_224.yaml configs/loss/ce.yaml configs/model/ViViT_224-S.yaml configs/optimizer/sgd.yaml configs/scheduler/graual_warmup.yaml --save_dir weights/train/224/S

#kd
#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 Knowledge_distilation.py -c configs/dataset/rgb_224.yaml configs/loss/ce.yaml configs/model/ViViT_224-B.yaml configs/optimizer/sgd.yaml configs/scheduler/graual_warmup.yaml configs/infer/checkpoint224B.yaml --save_dir weights/train/224/Kd/T

#3d net
#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train_3dnet.py -c configs/dataset/rgb_224.yaml configs/loss/ce.yaml configs/model/ViViT_224-B.yaml configs/optimizer/sgd.yaml configs/scheduler/graual_warmup.yaml  --save_dir weights/train/224/3d/resnet