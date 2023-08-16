## Preprocessing
```
python utils/dataset/preprocessing.py -c utils/dataset/configs/fps2_512.yaml
```
- data 전처리시 ```utils/configs/fps2_512.yaml``` 수정

## Train
```
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c configs/dataset/grid_512.yaml configs/loss/ce.yaml configs/model/TSViT_512.yaml configs/optimizer/adamw.yaml configs/scheduler/graual_warmup.yaml --save_dir result/
```

- 1 Node, 2 Gpus 학습 코드

## Inference
```
python inference.py -c configs/dataset/grid_512.yaml configs/loss/ce.yaml configs/model/TSViT_512.yaml configs/infer/checkpoint.yaml
```

- checkpoint 변경시 ```configs/infer/checkpoint.yaml``` 변경