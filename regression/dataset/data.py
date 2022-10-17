'''
Author: HaoZhi
Date: 2022-09-20 15:00:13
LastEditors: HaoZhi
LastEditTime: 2022-09-20 16:43:55
Description: 
'''
import os
import pickle as pk

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class TrainDataSet(Dataset):
    def __init__(self, data_folder, dat_path, data_size, data_aug) -> None:
        super().__init__()
        with open(dat_path, "rb") as f:
            self.info_list = pk.load(f)

        self.data_folder = data_folder
        self.data_aug = data_aug

    def __getitem__(self, index):
        file_name, count, labels = self.info_list[index]
        img = Image.open(os.path.join(self.data_folder, file_name))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = np.array(img).transpose((2, 0, 1))
        img = torch.from_numpy(img)
        count = torch.from_numpy(np.array(count)).float()
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        labels = torch.from_numpy(labels).float()

        img = self.data_aug(img)
        return img, count, labels, file_name

    def __len__(self):
        return len(self.info_list)


class ValidDataSet(Dataset):
    def __init__(self, data_folder, csv_path, data_size, data_aug) -> None:
        super().__init__()
        self.file_list = pd.read_csv(csv_path)["name"].to_list()

        self.data_folder = data_folder
        self.data_aug = data_aug

    def __getitem__(self, index):
        file_name = self.file_list[index]
        img = Image.open(os.path.join(self.data_folder, file_name))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = np.array(img).transpose((2, 0, 1))
        img = torch.from_numpy(img)

        img = self.data_aug(img)
        return img, file_name

    def __len__(self):
        return len(self.file_list)
