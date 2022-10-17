"""
Author: HaoZhi
Date: 2022-09-23 15:28:09
LastEditors: HaoZhi
LastEditTime: 2022-09-23 16:05:00
Description: 
"""
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import train_loader, train_loader_compose
from model import Model
from model.loss.loss import ClsRegLoss, MseLoss
from model.metric.metric import MseMetric

device = "cuda:1"

train_folder = "/disk2/haozhi/tmp/data/train"
train_dat_path = "/disk2/haozhi/tmp/code/dataset/train_distribute.dat"
data_size = (1024, 1024)
batch_size = 12
num_worker = 4

# compose_dataset
if_data_compose = True
train_folder_compose = "/disk2/haozhi/tmp/data/compose_train"
train_dat_path_compose = "/disk2/haozhi/tmp/code/dataset/train_distribute_compose.dat"

backbone_name = "effnet"
reg_mode = "dist"
lr = 1e-3
wd = 1e-6

reseum_path = False
num_epoch = 500

log_dir = "./log_dist_effnet_compose"
ckpt_dir = "./ckpt_dist_effnet_compose"


def main():

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if reg_mode == "dist":
        head_name = "cls_dist"
        metric_mode = "dist"
        loss_op = ClsRegLoss().to(device)
    elif reg_mode == "deci":
        head_name = "cls_deci"
        metric_mode = "deci"
        loss_op = MseLoss().to(device)

    trainloader = train_loader(
        data_folder=train_folder,
        dat_path=train_dat_path,
        data_size=data_size,
        batch_size=batch_size,
        num_worker=num_worker,
    )

    trainloader_compose = train_loader_compose(
        data_folder=train_folder_compose,
        dat_path=train_dat_path_compose,
        data_size=data_size,
        batch_size=batch_size,
        num_worker=num_worker,
    )

    num_batch = len(trainloader)

    model = Model(backbone_name=backbone_name, head_name=head_name).to(device)
    metric_op = MseMetric(metric_mode).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-7
    )

    if reseum_path:
        checkpoint = torch.load(reseum_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epoch):
        model.train()
        for idx, (imgs, counts, labels, pathes) in enumerate(trainloader):
            imgs = imgs.to(device)
            counts = counts.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = loss_op(logits, labels)
            preds, metric = metric_op(logits, counts)
            writer.add_scalar(
                "lr", scheduler.get_last_lr()[0], global_step=epoch * num_batch + idx
            )
            writer.add_scalar("mse", metric, global_step=epoch * num_batch + idx)
            writer.add_scalar("loss", loss, global_step=epoch * num_batch + idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                "Epoch = {:>5d} Steps = {:>5d} Lr = {:.3e} Mse = {:.5f} loss = {:.5f}".format(
                    epoch, idx, scheduler.get_last_lr()[0], metric.item(), loss.item()
                )
            )
        if if_data_compose:
            for idx, (imgs, counts, labels, pathes) in enumerate(trainloader_compose):
                imgs = imgs.to(device)
                counts = counts.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                loss = loss_op(logits, labels)
                preds, metric = metric_op(logits, counts)
                writer.add_scalar(
                    "lr", scheduler.get_last_lr()[0], global_step=epoch * num_batch + idx
                )
                writer.add_scalar("mse", metric, global_step=epoch * num_batch + idx)
                writer.add_scalar("loss", loss, global_step=epoch * num_batch + idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(
                    "Epoch = {:>5d} Steps = {:>5d} Lr = {:.3e} Mse = {:.5f} loss = {:.5f}".format(
                        epoch, idx, scheduler.get_last_lr()[0], metric.item(), loss.item()
                    )
                )


        scheduler.step()
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
        }
        torch.save(checkpoint, os.path.join(ckpt_dir, "model_" + str(epoch) + ".pth"))


if __name__ == "__main__":
    main()
