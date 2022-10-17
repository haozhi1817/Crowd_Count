"""
Author: HaoZhi
Date: 2022-10-13 09:49:37
LastEditors: HaoZhi
LastEditTime: 2022-10-13 10:53:53
Description: 
"""
import random

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

from utils import gen_discrete_map


class Augmentation(object):
    def __init__(self, crop_size, down_ratio) -> None:
        self.crop_size = crop_size
        self.down_ratio = down_ratio
        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _random_crop(self, img_h, img_w, crop_h, crop_w):
        res_h = img_h - crop_h
        res_w = img_w - crop_w
        i = random.randint(0, res_h)
        j = random.randint(0, res_w)
        return i, j, crop_h, crop_w

    def __call__(self, img, keypoints):
        w, h = img.size
        st_size = 1.0 * min(w, h)
        if st_size < self.crop_size:
            rr = 1.0 * self.crop_size / st_size
            w = round(w * rr)
            h = round(h * rr)
            st_size = 1.0 * min(w, h)
            img = img.resize((w, h), Image.BICUBIC)
            keypoints = keypoints * rr
        assert len(keypoints) >= 0
        i, j, h, w = self._random_crop(h, w, self.crop_size, self.crop_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (
                (keypoints[:, 0] >= 0)
                * (keypoints[:, 0] < w)
                * (keypoints[:, 1] >= 0)
                * (keypoints[:, 1] < h)
            )
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        discrete_map = gen_discrete_map(h, w, keypoints)
        down_w = w // self.down_ratio
        down_h = h // self.down_ratio
        discrete_map = discrete_map.reshape(
            [down_h, self.down_ratio, down_w, self.down_ratio]
        ).sum(axis=(1, 3))
        assert np.sum(discrete_map) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                discrete_map = np.fliplr(discrete_map)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                discrete_map = np.fliplr(discrete_map)
        
        discrete_map = np.expand_dims(discrete_map, 0)

        return (
            self.trans(img),
            torch.from_numpy(keypoints.copy()).float(),
            torch.from_numpy(discrete_map.copy()).float(),
        )

class Transformation(object):
    def __init__(self, crop_size, down_ratio) -> None:
        self.crop_size = crop_size
        self.down_ratio = down_ratio
        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, img, keypoints):
        w, h = img.size
        st_size = 1.0 * min(w, h)
        if st_size < self.crop_size:
            rr = 1.0 * self.crop_size / st_size
            w = round(w * rr)
            h = round(h * rr)
            st_size = 1.0 * min(w, h)
            img = img.resize((w, h), Image.BICUBIC)
            keypoints = keypoints * rr
        assert len(keypoints) >= 0

        discrete_map = gen_discrete_map(h, w, keypoints)
        down_w = w // self.down_ratio
        down_h = h // self.down_ratio
        discrete_map = discrete_map.reshape(
            [down_h, self.down_ratio, down_w, self.down_ratio]
        ).sum(axis=(1, 3))
        assert np.sum(discrete_map) == len(keypoints)

        discrete_map = np.expand_dims(discrete_map, 0)

        return (
            self.trans(img),
            torch.from_numpy(keypoints.copy()).float(),
            torch.from_numpy(discrete_map.copy()).float(),
        )

class Transformation_v2(object):
    def __init__(self, crop_size, down_ratio) -> None:
        self.crop_size = crop_size
        self.down_ratio = down_ratio
        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, img):
        w, h = img.size
        st_size = 1.0 * min(w, h)
        if st_size < self.crop_size:
            rr = 1.0 * self.crop_size / st_size
            w = round(w * rr)
            h = round(h * rr)
            st_size = 1.0 * min(w, h)
            img = img.resize((w, h), Image.BICUBIC)
        return self.trans(img)

