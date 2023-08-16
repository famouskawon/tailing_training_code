# from typing_extensions import Concatenate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
from glob import glob
import os
import random

# import PIL
# import torchvision.transforms
# import cv2
# from torchvision.utils import save_image

class UBIDatasetTrain(Dataset):
    def __init__(self, path, transformers):
        super().__init__()
        self.files = sorted(glob(os.path.join(path, '*/*.npy')))
        self.transformers = transformers

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        #fight or normal
        label = 1 if os.path.basename(os.path.dirname(self.files[idx])) == 'normal' else 0
        img = np.load(self.files[idx])
        #print(img.shape)
        #img = img[:,np.newaxis,]
        
        length, h, w, c = img.shape
        
        #sampling_idx = sorted(random.sample(range(0, length), 16))
        sampling_idx = np.linspace(0, length-1, 16).astype(int)
        #print(sampling_idx)
        

        ## v2
        img = img[sampling_idx]#.astype('float32')
        #print(img)
        img = torch.from_numpy(img)    
        img = torch.permute(img,(0,3,1,2))#.type(torch.FloatTensor) #torch.Size([1, 16, 360, 640, 3]) -> torch.Size([1, 16, 360, 640, 3])
        # save_image(img, f'weights/{idx}.png')
        # print(f'[shape] {img.shape}')
        
        if self.transformers is not None:
            img = self.transformers(img)

        return (img, label)

class UBIDatasetTest(Dataset):
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
        sampling_idx = np.linspace(0, length-1, 16).astype(int)

        ## v2
        img = img[sampling_idx]#.astype('float32')
        img = torch.from_numpy(img)
        img = torch.permute(img,(0,3,1,2))#.type(torch.FloatTensor)
        if self.transformers is not None:
            img = self.transformers(img)
        # save_image(img, f'weights/{idx}.png')
        # print(f'[shape] {img.shape}')
        
        return (img, label)

if __name__ == '__main__':
    
    
    train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

    
    ds = UBIDatasetTrain('/workspace/raw_data/UBI_FIGHTS/UBI_FIGHTS/hockey_frame/train', transformers=train_transforms)

    dataloader = DataLoader(ds, batch_size=1, shuffle=True)
    print("Train")
    for step, (img, target) in enumerate(dataloader):
        print(f'[{step}] img: {img}')
        print(f'[{step}] img_shape: {img.shape}')
        print(f'[{step}] target: {target}')

        #img = img.numpy()
        #np.save("test.npy", img)
        if step == 3:
            break
    
    """
    ds = UBIDatasetTest('/workspace/raw_data/UBI_FIGHTS/UBI_FIGHTS/UBI_CLIP_DATASET/test', transformers=test_transforms)

    dataloader = DataLoader(ds, batch_size=1, shuffle=False)
    print("Test")
    for step, (img, target) in enumerate(dataloader):
        print(f'[{step}] img: {img}')
        print(f'[{step}] img_shape: {img.shape}')
        print(f'[{step}] target: {target}')
        

        if step == 1:
            break
    """
    
    