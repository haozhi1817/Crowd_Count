'''
Author: HaoZhi
Date: 2022-09-20 16:27:02
LastEditors: HaoZhi
LastEditTime: 2022-09-22 15:11:40
Description: 
'''
import os
import sys
import pickle as pk

sys.path.append(os.path.dirname(__file__))

from torch.utils.data import DataLoader, WeightedRandomSampler

from data import TrainDataSet, ValidDataSet
from augmentation import train_aug, valid_aug


def train_loader(data_folder, dat_path, data_size, batch_size, num_worker):
    dataset = TrainDataSet(data_folder, dat_path, data_size, data_aug = train_aug)
    with open(r'D:\workspace\CC\regression\dataset\train_sample_weight.dat', 'rb') as f:
        sample_weight = pk.load(f)
    sampler = WeightedRandomSampler(weights= sample_weight, num_samples= len(sample_weight))
    dataloader = DataLoader(dataset= dataset, batch_size= batch_size, sampler= sampler, num_workers= num_worker)
    return dataloader

def valid_loader(data_folder, csv_path, data_size, batch_size, num_worker):
    dataset = ValidDataSet(data_folder, csv_path, data_size, data_aug = valid_aug)
    dataloader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= False, num_workers= num_worker)
    return dataloader