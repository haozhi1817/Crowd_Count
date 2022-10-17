'''
Author: HaoZhi
Date: 2022-09-23 15:29:37
LastEditors: HaoZhi
LastEditTime: 2022-09-23 15:42:11
Description: 
'''
import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch.nn as nn

from backbone import resnet, effnet
from head import cls_head

class Model(nn.Module):
    def __init__(self, backbone_name, head_name) -> None:
        super().__init__()
        if backbone_name == 'resnet':
            model_op = getattr(resnet, 'resnet50')
        elif backbone_name == 'effnet':
            model_op = getattr(effnet, 'effnet')
        self.model = model_op()

        if head_name == 'cls_dist':
            head_op = getattr(cls_head, 'ClsHead_Dist')
        elif head_name == 'cls_deci':
            head_op = getattr(cls_head, 'ClsHead_Deci')
        if backbone_name == 'resnet':
            fc1_in_chans = 2048
            fc2_in_chans = 1024
        elif backbone_name == 'effnet':
            fc1_in_chans = 1280
            fc2_in_chans = 512
        self.head = head_op(in_chans = fc1_in_chans, hidden_chans = fc2_in_chans)

    def forward(self, x):
        features = self.model(x)
        logits = self.head(features)
        return logits


