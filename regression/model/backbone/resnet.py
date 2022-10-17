'''
Author: HaoZhi
Date: 2022-09-20 16:53:23
LastEditors: HaoZhi
LastEditTime: 2022-09-21 15:13:33
Description: 
'''
import timm

def resnet50():
    model = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='')
    return model

