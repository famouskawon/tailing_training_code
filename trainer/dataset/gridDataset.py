# from typing_extensions import Concatenate
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import numpy as np
from glob import glob
import os
import random

# import PIL
# import torchvision.transforms
# import cv2
# from torchvision.utils import save_image

class gridDatasetTrain(Dataset):
    def __init__(self, path, transformers):
        super().__init__()
        self.files = sorted(glob(os.path.join(path, '*/*.npy'))*4)
        self.transformers = transformers

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        #print(self.files[idx])
        label = 1 if os.path.basename(os.path.dirname(self.files[idx])) == 'normal' else 0
        img = np.load(self.files[idx])
        #print(img.shape)

        #1차원 차원 추가, 3차원은 차원 수정만
        #img = img[:,np.newaxis,]
        length, c, w, h = img.shape
        
        sampling_idx = sorted(random.sample(range(0, length), 16))
        #sampling_idx = np.linspace(0, length-1, 16).astype(int)


        ## v2
        #img = img[sampling_idx].reshape((1, 3 * w, 3 * h)) ## 9*512*512 -> 1 * 1536 * 1536
        img = img[sampling_idx]
        img = torch.from_numpy(img)
        
        #(85, 224, 224, 3)   
        img = torch.permute(img,(0,3,1,2))
        #img = torchvision.transforms.Grayscale(num_output_channels = 1)(img)
        #3d res
        #mg = torch.permute(img,(3,0,1,2))
        # save_image(img, f'weights/{idx}.png')
        # print(f'[shape] {img.shape}')

        return (img, label)

class gridDatasetTest(Dataset):
    def __init__(self, path, transformers):
        super().__init__()
        self.files = sorted(glob(os.path.join(path, '*/*.npy')))
        self.transformers = transformers

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = 1 if os.path.basename(os.path.dirname(self.files[idx])) == 'normal' else 0
        img = np.load(self.files[idx])
        #img = img[:,np.newaxis,]
        #(46, 512, 512)
        length, c, w, h = img.shape

        ## v1
        # img1 = np.concatenate((img[0], img[1], img[2]), axis=1)
        # img2 = np.concatenate((img[3], img[4], img[5]), axis=1)
        # img3 = np.concatenate((img[6], img[7], img[8]), axis=1)
        # img = np.concatenate((img1, img2, img3), axis=0)
        # img = torch.from_numpy(img)
        # img = img.unsqueeze(dim=0)

        ## v2
        
        #img = img.reshape((length, 1,  3 * w, 3 * h)) 
        img = torch.from_numpy(img)
        img = torch.permute(img,(0,3,1,2))
        #img = torchvision.transforms.Grayscale(num_output_channels = 1)(img)
        

        #3d res
        #img = torch.permute(img,(3,0,1,2))
        # save_image(img, f'weights/{idx}.png')
        # print(f'[shape] {img.shape}')
        
        return (img, label)

if __name__ == '__main__':
    """
    ds = gridDatasetTrain('/workspace/dataset/server_data6_rgb_fps3/train', transformers=None)
    
    
    dataloader = DataLoader(ds, batch_size=1, shuffle=True)
    print("Train")
    for step, (img, target) in enumerate(dataloader):
        print(f'[{step}] img: {img}')
        print(f'[{step}] img_shape: {img.shape}')
        print(f'[{step}] target: {target}')
        print(img.dtype)
        

        if step == 3:
            break
    
    """
    ds = gridDatasetTrain('/workspace/dataset/server_data6_rgb_fps3/test', transformers=None)

    dataloader = DataLoader(ds, batch_size=1, shuffle=False)
    print("Test")
    for step, (img, target) in enumerate(dataloader):
        print(f'[{step}] img: {img}')
        print(f'[{step}] img_shape: {img.shape}')
        print(f'[{step}] target: {target}')
        

        if step == 3:
            break
    
