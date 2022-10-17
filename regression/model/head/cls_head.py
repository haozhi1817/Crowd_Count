import torch
import torch.nn as nn
import torch.nn.functional as F

class ClsHead_Dist(nn.Module):
    def __init__(self, in_chans=2048, hidden_chans=1024, num_class=54) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_chans, hidden_chans, bias=True),
            nn.LayerNorm(hidden_chans),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Linear(hidden_chans, num_class, bias=True)

    def forward(self, x):
        x = self.gap(x).squeeze(2).squeeze(2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x


class ClsHead_Deci(nn.Module):
    def __init__(self, in_chans=2048, hidden_chans=1024, num_class=4) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_chans, hidden_chans, bias=True),
            nn.LayerNorm(hidden_chans),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Linear(hidden_chans, num_class, bias=True)

    def forward(self, x):
        x = self.gap(x).squeeze(2).squeeze(2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
