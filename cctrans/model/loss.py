"""
Author: HaoZhi
Date: 2022-10-11 16:13:24
LastEditors: HaoZhi
LastEditTime: 2022-10-14 09:58:26
Description: 
"""
import torch
from torch import nn
from model.bregman_pytorch import sinkhorn_knopp


class OT_Loss(nn.Module):
    def __init__(
        self, c_size, stride, norm_cood, device, num_of_iter_in_ot=100, reg=10.0
    ) -> None:
        super().__init__()
        assert c_size % stride == 0

        self.c_size = c_size
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg

        self.cood = (
            torch.arange(0, c_size, step=stride, dtype=torch.float32, device=device)
            + stride / 2
        )
        self.density_size = self.cood.size(0)
        self.cood.unsqueeze_(0)

        if self.norm_cood:
            self.cood = self.cood / c_size * 2 - 1
        self.output_size = self.cood.size(1)

    def forward(self, normed_density, unnormed_density, points):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        assert self.output_size == normed_density.size(2)
        loss = torch.zeros([1]).to(self.device)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0
        for idx, im_points in enumerate(points):
            #print('debug: ', im_points)
            if len(im_points) > 0:
                if self.norm_cood:
                    im_points = im_points / self.c_size * 2 - 1
                # gen cost metric
                x = im_points[:, 0].unsqueeze_(1)
                y = im_points[:, 1].unsqueeze_(1)
                x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
                y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
                x_dis.unsqueeze_(1)
                y_dis.unsqueeze_(2)
                dis = x_dis + y_dis
                dis = dis.view((dis.size(0), -1))

                source_prob = normed_density[idx][0].view([-1]).detach()
                target_prob = (torch.ones([len(im_points)])/len(im_points)).to(
                    self.device
                )
                P, log = sinkhorn_knopp(
                    target_prob,
                    source_prob,
                    dis,
                    self.reg,
                    maxIter=self.num_of_iter_in_ot,
                    log=True,
                )
                beta = log["beta"]
                ot_obj_values += torch.sum(
                    normed_density[idx]
                    * beta.view([1, self.output_size, self.output_size])
                )
                source_density = unnormed_density[idx][0].view([-1]).detach()
                source_count = source_density.sum()
                im_grad_1 = (source_count) / (source_count * source_count + 1e-8) * beta
                im_grad_2 = (source_density * beta).sum() / (
                    source_count * source_count + 1e-8
                )
                im_grad = im_grad_1 - im_grad_2
                im_grad = im_grad.detach().view([1, self.output_size, self.output_size])
                loss += torch.sum(unnormed_density[idx] * im_grad)
                wd += torch.sum(dis * P).item()
        return loss, wd, ot_obj_values


class TV_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss(reduction="none")

    def forward(self, pred_dis_map_norm, target_dis_map):
        #print('tv_loss1: ', pred_dis_map_norm.shape, target_dis_map.shape)
        target_count = torch.sum(target_dis_map, dim=(1, 2, 3)).float()
        target_dis_map_norm = target_dis_map / (target_count.unsqueeze(1).unsqueeze(
            2
        ).unsqueeze(3) + 1e-6)
        #print('tv_loss1: ', pred_dis_map_norm.shape, target_dis_map_norm.shape)
        tv_loss = (
            self.loss(pred_dis_map_norm, target_dis_map_norm).sum(1).sum(1).sum(1)
            * target_count
        )
        tv_loss = tv_loss.mean(0)
        return tv_loss


class Count_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred_dis_map, target_dis_map):
        #print('count_loss1: ', pred_dis_map.shape, target_dis_map.shape)
        target_count = torch.sum(target_dis_map, dim=(1, 2, 3)).float()
        pred_count = pred_dis_map.sum(1).sum(1).sum(1)
        #print('count_loss2: ', pred_count, target_count)
        #print('count_loss: ', pred_count.shape, target_count.shape)
        count_loss = self.loss(pred_count, target_count)
        return count_loss
