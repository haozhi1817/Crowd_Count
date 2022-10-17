'''
Author: HaoZhi
Date: 2022-09-20 15:00:25
LastEditors: HaoZhi
LastEditTime: 2022-09-20 16:39:34
Description: 
'''
import torch
from torchvision.transforms import transforms

def train_aug(img_size=(1024, 768)):
    trans = transforms.Compose(
        [
            transforms.Resize(size = (img_size[0], img_size[1])),
            #transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.ConvertImageDtype(torch.float)
        ]
    )
    return trans

def valid_aug(img_size = (1024, 768)):
    trans = transforms.Compose([transforms.Resize(size = (img_size[0], img_size[1])), transforms.ConvertImageDtype(torch.float)])
    return trans