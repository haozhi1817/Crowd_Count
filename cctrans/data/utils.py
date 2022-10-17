'''
Author: HaoZhi
Date: 2022-10-13 09:49:30
LastEditors: HaoZhi
LastEditTime: 2022-10-15 16:37:10
Description: 
'''
import math
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from scipy import ndimage, spatial

def gen_discrete_map(img_h, img_w, points):
    discrete_map = np.zeros([img_h, img_w], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h - 1] * num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w - 1] * num_gt).astype(int))
    p_idx = torch.from_numpy(p_h * img_w + p_w).to(torch.int64)
    discrete_map = (
        torch.zeros(img_w * img_h)
        .scatter_add_(0, index=p_idx, src=torch.ones(img_w * img_h))
        .view(img_h, img_w)
        .numpy()
    )
    assert np.sum(discrete_map) == num_gt
    return discrete_map

def gen_discrete_map_slow(img_h, img_w, points):
    discrete_map = np.zeros([img_h, img_w], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map
    points_np = np.array(points).round().astype(int)
    for (x, y) in points_np:
        discrete_map[y, x] = 1
    assert np.sum(discrete_map) == num_gt
    return discrete_map


class Gaussian(nn.Module):
    def __init__(self, in_chans, sigmalist, kernel_size = 64, stride = 1, padding = 0, froze = True) -> None:
        super().__init__()
        out_chans = len(sigmalist) * in_chans
        mu = kernel_size // 2
        gaussFuncTemp = lambda x: (lambda sigma : math.exp(-(x - mu) ** 2 / float(2 * sigma ** 2)))
        gaussFuncs = [gaussFuncTemp(x) for x in range(kernel_size)]
        windows = []
        for sigma in sigmalist:
            gauss = torch.Tensor([gaussFunc(sigma) for gaussFunc in gaussFuncs])
            gauss /= gauss.sum()
            _in_window = gauss.unsqueeze(1)
            _2d_window = _in_window.mm(_in_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = Variable(_2d_window.expand(in_chans, 1, kernel_size, kernel_size).contiguous())
            windows.append(window)
        kernels = torch.stack(windows)
        kernels = kernels.permute(1, 0, 2, 3, 4)
        weight = kernels.reshape(out_chans, in_chans, kernel_size, kernel_size)
        self.gkernel = nn.Conv2d(in_chans, out_chans, kernel_size, stride= stride, padding= padding, groups= in_chans, bias = False)
        self.gkernel.weight = torch.nn.Parameter(weight)
        if froze:
            self.frozePara()
    
    def forward(self, dotmaps):
        gaussianmaps = self.gkernel(dotmaps)
        return gaussianmaps
    
    def frozePara(self):
        for para in self.parameters():
            para.requires_grad = False

class SumPool2d(nn.Module):
    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.avp = nn.AvgPool2d(kernel_size, stride = 1, padding= kernel_size // 2)
        if type(kernel_size) is not int:
            self.area = kernel_size[0] * kernel_size[1]
        else:
            self.area = kernel_size * kernel_size
    
    def forward(self, dotmap):
        return self.avp(dotmap) * self.area

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype = np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    tree = spatial.KDTree(pts.copy(), leafsize = leafsize)
    distance, locations = tree.query(pts, k = 4)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype = np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distance[i][1] + distance[i][2] + distance[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) /2. / 2.
        print('sigma: ', sigma)
        density += ndimage.filters.gaussian_filter(pt2d, sigma, mode = 'constant')
    return density



if __name__ == '__main__':
    from PIL import Image
    from scipy import io as sio
    import matplotlib.pyplot as plt

    img = Image.open(r'D:\workspace\tmp\ShanghaiTech\part_B\train_data\images\IMG_1.jpg').convert('RGB')
    w, h = img.size
    img = np.array(img)
    keypoints = sio.loadmat(r'D:\workspace\tmp\ShanghaiTech\part_B\train_data\ground-truth\GT_IMG_1.mat')["image_info"][0][0][0][0][0]
    dis_map = gen_discrete_map(h, w, keypoints)
    dis_map1 = gen_discrete_map_slow(img.shape[0], img.shape[1], keypoints)

    dis_map_gauss_a = gaussian_filter_density(dis_map)
    gauss = Gaussian(in_chans= 1, sigmalist=[1,], kernel_size= 5, padding=2)
    dis_map_gauss_t = gauss(torch.from_numpy(dis_map[np.newaxis, np.newaxis, ...])).permute(0, 2, 3, 1).numpy().squeeze()

    pic = np.concatenate([dis_map, dis_map_gauss_a, dis_map_gauss_t], axis = 1)
    

