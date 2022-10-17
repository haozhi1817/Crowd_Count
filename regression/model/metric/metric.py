'''
Author: HaoZhi
Date: 2022-09-23 16:07:06
LastEditors: HaoZhi
LastEditTime: 2022-09-23 16:13:45
Description: 
'''

import torch
import numpy as np
import torch.nn as nn

class MseMetric(nn.Module):
    def __init__(self, mode) -> None:
        super().__init__()
        if mode == 'dist':
            self.bins = torch.arange(50, 2750, 50)
        elif mode == 'deci':
            self.bins = torch.from_numpy(np.array([10000, 1000, 100, 10]))
    
    def forward(self, x, y):
        self.bins = self.bins.to(x.device)
        b = y.shape[0]
        preds = (x * self.bins).sum(1)
        mse = (((preds - y) ** 2).sum() ** 0.5)/ (b + 1e-8)
        return preds, mse