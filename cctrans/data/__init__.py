'''
Author: HaoZhi
Date: 2022-10-13 13:35:58
LastEditors: HaoZhi
LastEditTime: 2022-10-14 10:18:42
Description: 
'''
import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader

from dataset import CustomDataset, CustomDataset_v2
from augmeantation import Augmentation, Transformation, Transformation_v2

def collect_fn(batch):
    batch_data = list(zip(*batch))
    imgs = torch.stack(batch_data[0], 0)
    kp = batch_data[1]
    dis_map = torch.stack(batch_data[2], 0)
    filenames = batch_data[3]
    return imgs, kp, dis_map, filenames

def build_train_dataloader(root, crop_size, down_ratio, batch_size):
    dataset = CustomDataset(root, crop_size, down_ratio, aug = Augmentation)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collect_fn
    )
    return dataloader

def build_valid_dataloader(root, crop_size, down_ratio, batch_size):
    dataset = CustomDataset(root, crop_size, down_ratio, aug = Transformation)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collect_fn
    )
    return dataloader

def build_valid_dataloader_v2(root, crop_size, down_ratio, batch_size):
    dataset = CustomDataset_v2(root, crop_size, down_ratio, aug = Transformation_v2)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return dataloader



