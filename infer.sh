python inference.py -c configs/dataset/grid_512.yaml configs/loss/ce.yaml configs/model/TSViT_512-S.yaml configs/infer/checkpoint512S.yaml
python inference.py -c configs/dataset/grid_512.yaml configs/loss/ce.yaml configs/model/TSViT_512-B.yaml configs/infer/checkpoint512B.yaml
python inference.py -c configs/dataset/grid_512.yaml configs/loss/ce.yaml configs/model/TSViT_512-L.yaml configs/infer/checkpoint512L.yaml

# python inference.py -c configs/dataset/grid_256.yaml configs/loss/ce.yaml configs/model/TSViT_256-S.yaml configs/infer/checkpoint256S.yaml
# python inference.py -c configs/dataset/grid_256.yaml configs/loss/ce.yaml configs/model/TSViT_256-B.yaml configs/infer/checkpoint256B.yaml
# python inference.py -c configs/dataset/grid_256.yaml configs/loss/ce.yaml configs/model/TSViT_256-L.yaml configs/infer/checkpoint256L.yaml

