import torch
import numpy as np
import torch.nn as nn

class PredHead(nn.Module):
    def __init__(self, mode) -> None:
        super().__init__()
        if mode == 'dist':
            self.bins = torch.arange(50, 2750, 50)
        elif mode == 'deci':
            self.bins = torch.from_numpy(np.array([10000, 1000, 100, 10]))
    
    def forward(self, x):
        self.bins = self.bins.to(x.device)
        preds = (x * self.bins).sum(1)
        return preds