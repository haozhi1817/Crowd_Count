"""
Author: HaoZhi
Date: 2022-10-14 10:13:29
LastEditors: HaoZhi
LastEditTime: 2022-10-14 10:20:31
Description: 
"""
import os

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from data import build_valid_dataloader
from model.twins import alt_gvt_large, alt_gvt_small
from model.vgg import vgg19
#from model.loss import Count_Loss, OT_Loss, TV_Loss
from model.loss import Count_Loss, TV_Loss
from model.loss_official import OT_Loss

valid_root = "/disk2/haozhi/tmp/data/ShanghaiTech/part_A/test_data"
crop_size = 512
down_ratio = 8
batch_size = 1
norm_cood = 1
ot_iter = 1000
ot_reg = 10.0
cout_loss_weight = 1.0
ot_loss_weight = 0.1
tv_loss_weight = 0.01

device = "cuda:1"
reseum_path = '/disk2/haozhi/tmp/cc_trans/ckpt_sha/model_910.pth'
result_folder = '/disk2/haozhi/tmp/cc_trans/result/result_sha_910'

def main():
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # dataset
    valid_loader = build_valid_dataloader(
        root=valid_root,
        crop_size=crop_size,
        down_ratio=down_ratio,
        batch_size=batch_size,
    )

    num_batch = len(valid_loader)

    #model = alt_gvt_large()
    model = vgg19()
    model.to(device)
    # count_loss_op = Count_Loss()
    # ot_loss_op = OT_Loss(crop_size, down_ratio, norm_cood, device, ot_iter, ot_reg)
    # tv_loss_op = TV_Loss()

    checkpoint = torch.load(reseum_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    result_list = []
    with torch.no_grad():
        model.eval()
        for idx, (imgs, keypoints, target_dis_maps, filenames) in enumerate(
            valid_loader
        ):
            imgs = imgs.to(device)
            keypoints = [keypoint.to(device) for keypoint in keypoints]
            target_dis_maps = target_dis_maps.to(device)
            pred_dis_maps, pred_dis_maps_norm = model(imgs)
            # count_loss = (
            #     count_loss_op(pred_dis_maps, target_dis_maps) * cout_loss_weight
            # )
            # ot_loss = (
            #     ot_loss_op(pred_dis_maps_norm, pred_dis_maps, keypoints)[0]
            #     * ot_loss_weight
            # )
            # ot_loss, ot_wd, ot_value =  ot_loss_op(pred_dis_maps_norm, pred_dis_maps, keypoints)
            # ot_loss = ot_loss * ot_loss_weight
            # tv_loss = tv_loss_op(pred_dis_maps_norm, target_dis_maps) * tv_loss_weight
            # loss = count_loss + ot_loss + tv_loss

            post_process(imgs, pred_dis_maps, target_dis_maps, filenames, result_list, result_folder)
        pd.DataFrame(result_list).to_csv(os.path.join(result_folder, 'result.csv'))
        result_array = np.array(result_list)
        print(result_array.shape)
        mae = np.mean(np.abs((result_array[:, 1]).astype('float32') - (result_array[:, 2]).astype('float32')))
        print('mae: ', mae)


def post_process(imgs, pred_dis_maps, target_dis_maps, filenames, result_list, save_folder):
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    pred_dis_maps = pred_dis_maps.permute(0, 2, 3, 1).cpu().numpy()[...,0]
    target_dis_maps = target_dis_maps.permute(0, 2, 3, 1).cpu().numpy()[...,0]
    for (img, pred_dis_map, target_dis_map, filename) in zip(imgs, pred_dis_maps, target_dis_maps, filenames):
        img = np.array(Image.open(os.path.join(valid_root, 'images', filename + '.jpg')))
        pred_count = np.sum(pred_dis_map)
        target_count = np.sum(target_dis_map)
        result_list.append([filename, target_count, pred_count])
        pred_dis_map = cv2.resize(pred_dis_map, (img.shape[1], img.shape[0]), interpolation= cv2.INTER_LINEAR)
        target_dis_map = cv2.resize(target_dis_map, (img.shape[1], img.shape[0]), interpolation= cv2.INTER_LINEAR)
        pred_dis_map = cv2.applyColorMap(np.uint8(255 * pred_dis_map / np.max(pred_dis_map)), cv2.COLORMAP_JET)
        target_dis_map = cv2.applyColorMap(np.uint8(255 * target_dis_map / np.max(target_dis_map)), cv2.COLORMAP_JET)
        pic = np.concatenate([img, pred_dis_map, target_dis_map], axis = 1).astype('uint8')
        save_path = os.path.join(save_folder, filename + '_gt_' + str(target_count) + '_pd_' + str(pred_count) + '.jpg')
        plt.imsave(save_path, pic)



if __name__ == '__main__':
    main()
