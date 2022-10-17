'''
Author: HaoZhi
Date: 2022-09-20 16:53:14
LastEditors: HaoZhi
LastEditTime: 2022-09-20 17:01:59
Description: 
'''
import timm

def effnet():
    model = timm.create_model('efficientnet_b1', pretrained=True, num_classes=0, global_pool='')
    return model