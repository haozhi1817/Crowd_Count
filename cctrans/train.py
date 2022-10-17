"""
Author: HaoZhi
Date: 2022-10-14 10:13:29
LastEditors: HaoZhi
LastEditTime: 2022-10-14 10:20:31
Description: 
"""
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from data import build_train_dataloader
from model.twins import alt_gvt_large, alt_gvt_small
from model.vgg import vgg19
#from model.loss import Count_Loss, OT_Loss, TV_Loss
from model.loss import Count_Loss, TV_Loss
from model.loss_official import OT_Loss

train_root = "/disk2/haozhi/tmp/data/ShanghaiTech/part_A/train_data"
crop_size = 512
down_ratio = 8
batch_size = 32
lr_init = 1e-5
wd = 1e-4
norm_cood = 1
ot_iter = 100
ot_reg = 10.0
num_epoch = 1000
cout_loss_weight = 1.0
ot_loss_weight = 0.1
tv_loss_weight = 0.01

device = "cuda:1"
log_path = "/disk2/haozhi/tmp/cc_trans/log_sha"
ckpt_path = "/disk2/haozhi/tmp/cc_trans/ckpt_sha"
reseum_path = '/disk2/haozhi/tmp/cc_trans/ckpt/model_999.pth'


def main():
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # dataset
    train_loader = build_train_dataloader(
        root=train_root,
        crop_size=crop_size,
        down_ratio=down_ratio,
        batch_size=batch_size,
    )

    num_batch = len(train_loader)

    #model = alt_gvt_large()
    model = vgg19()
    model.to(device)
    count_loss_op = Count_Loss()
    ot_loss_op = OT_Loss(crop_size, down_ratio, norm_cood, device, ot_iter, ot_reg)
    tv_loss_op = TV_Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=lr_init * 1e-2
    )

    if reseum_path:
        checkpoint = torch.load(reseum_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        init_epoch = checkpoint["epoch"]
    else:
        init_epoch = 0

    writer = SummaryWriter(log_dir=log_path)

    for epoch in range(0, num_epoch):
        model.train()
        for idx, (imgs, keypoints, target_dis_maps, filenames) in enumerate(
            train_loader
        ):
            imgs = imgs.to(device)
            keypoints = [keypoint.to(device) for keypoint in keypoints]
            target_dis_maps = target_dis_maps.to(device)
            pred_dis_maps, pred_dis_maps_norm = model(imgs)
            count_loss = (
                count_loss_op(pred_dis_maps, target_dis_maps) * cout_loss_weight
            )
            # ot_loss = (
            #     ot_loss_op(pred_dis_maps_norm, pred_dis_maps, keypoints)[0]
            #     * ot_loss_weight
            # )
            ot_loss, ot_wd, ot_value =  ot_loss_op(pred_dis_maps_norm, pred_dis_maps, keypoints)
            ot_loss = ot_loss * ot_loss_weight
            tv_loss = tv_loss_op(pred_dis_maps_norm, target_dis_maps) * tv_loss_weight
            loss = count_loss + ot_loss + tv_loss
            writer.add_scalar(
                "lr", scheduler.get_last_lr()[0], global_step=epoch * num_batch + idx
            )
            writer.add_scalar(
                "count_loss", count_loss.item(), global_step=epoch * num_batch + idx
            )
            writer.add_scalar(
                "ot_loss", ot_loss.item(), global_step=epoch * num_batch + idx
            )
            writer.add_scalar(
                "tv_loss", tv_loss.item(), global_step=epoch * num_batch + idx
            )
            writer.add_scalar(
                "total_loss", loss.item(), global_step=epoch * num_batch + idx
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                "|Epoch = {:>5d} | Steps = {:>5d} | Lr= {:.3e} | Loss = {:.7f} Count_Loss = {:5f} OT_Loss = {:3e} TV_Loss = {:5f} | OT_Wd = {:.5f}".format(
                    epoch,
                    epoch * num_batch + idx,
                    scheduler.get_last_lr()[0],
                    loss.item(),
                    count_loss.item(),
                    ot_loss.item(),
                    tv_loss.item(),
                    ot_wd,
                )
            )
        scheduler.step()
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(
            checkpoint,
            os.path.join(ckpt_path, "model_" + str(epoch) + ".pth"),
        )

if __name__ == '__main__':
    main()