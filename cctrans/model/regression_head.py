"""
Author: HaoZhi
Date: 2022-10-12 17:48:58
LastEditors: HaoZhi
LastEditTime: 2022-10-13 18:06:03
Description: 
"""
import torch
from torch import nn


class RegressionHead(nn.Module):
    def __init__(self, embed_dims) -> None:
        super().__init__()

        self.v1 = nn.Sequential(
            nn.Conv2d(embed_dims[-3], embed_dims[-3], 3, padding=1, dilation=1),
            nn.BatchNorm2d(embed_dims[-3]),
            nn.ReLU(inplace=True),
        )

        self.v2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(embed_dims[-2], embed_dims[-3], 3, padding=1, dilation=1),
            nn.BatchNorm2d(embed_dims[-3]),
            nn.ReLU(inplace=True),
        )

        self.v3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(embed_dims[-1], embed_dims[-3], 3, padding=1, dilation=1),
            nn.BatchNorm2d(embed_dims[-3]),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(embed_dims[-3], embed_dims[-4], 3, padding=1, dilation=1),
            nn.BatchNorm2d(embed_dims[-4]),
            nn.ReLU(inplace=True),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(embed_dims[-3], embed_dims[-4], 3, padding=2, dilation=2),
            nn.BatchNorm2d(embed_dims[-4]),
            nn.ReLU(inplace=True),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(embed_dims[-3], embed_dims[-4], 3, padding=3, dilation=3),
            nn.BatchNorm2d(embed_dims[-4]),
            nn.ReLU(inplace=True),
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(embed_dims[-3], embed_dims[-4] * 3, 1),
            nn.BatchNorm2d(embed_dims[-4] * 3),
            nn.ReLU(inplace=True),
        )

        self.res = nn.Sequential(
            nn.Conv2d(embed_dims[-4] * 3, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.init_param()

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, x3):
        #print("debug: ", x1.shape, x2.shape, x3.shape)
        x1 = self.v1(x1)
        x2 = self.v2(x2)
        x3 = self.v3(x3)
        x = x1 + x2 + x3
        y1 = self.stage1(x)
        y2 = self.stage2(x)
        y3 = self.stage3(x)
        y4 = self.stage4(x)
        y = torch.cat((y1, y2, y3), dim=1) + y4
        y = self.res(y)
        return y
