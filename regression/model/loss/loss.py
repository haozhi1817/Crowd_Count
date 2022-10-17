'''
Author: HaoZhi
Date: 2022-09-20 17:08:38
LastEditors: HaoZhi
LastEditTime: 2022-09-22 15:28:48
Description: 
'''
import torch
import numpy as np
import torch.nn as nn

class ClsRegLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kl_loss = nn.KLDivLoss()

    def forward(self, pred, target):
        pred = torch.clip(pred, min= 1e-8)
        target = torch.clip(target, min = 1e-8)
        pred = torch.log(pred)
        loss = self.kl_loss(pred, target)
        return loss

class MseLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.HuberLoss(delta= 10.)

    def forward(self, pred, target):
        pred = torch.clip(pred, min= 1e-8)
        target = torch.clip(target, min = 1e-8)
        loss = self.mse_loss(pred, target)
        return loss

class L1Loss(nn.Module):
    def __init__(self, mode) -> None:
        super().__init__()
        if mode == 'dist':
            bins = torch.arange(50, 2750, 50)
        elif mode == 'deci':
            bins = torch.from_numpy(np.array([10000, 1000, 100, 10]))
        self.register_buffer('bins', bins)
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, pred, target):
        preds = (pred * self.bins).sum(1)
        loss = self.l1_loss(preds, target)
        return  loss

