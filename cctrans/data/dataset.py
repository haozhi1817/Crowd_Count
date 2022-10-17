"""
Author: HaoZhi
Date: 2022-10-13 09:49:24
LastEditors: HaoZhi
LastEditTime: 2022-10-13 10:54:02
Description: 
"""
import os
from PIL import Image
import scipy.io as sio
from torch.utils.data import Dataset

from augmeantation import Augmentation


class CustomDataset(Dataset):
    def __init__(self, root, crop_size, down_ratio, aug = Augmentation) -> None:
        super().__init__()
        self.root = root
        self.filenames = list(
            map(
                lambda x: os.path.splitext(x)[0],
                os.listdir(os.path.join(self.root, "images")),
            )
        )
        print("num of datas: ", len(self.filenames))

        self.aug = aug(crop_size=crop_size, down_ratio=down_ratio)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_path = os.path.join(self.root, "images", filename + ".jpg")
        label_path = os.path.join(self.root, "ground-truth", "GT_" + filename + ".mat")

        img = Image.open(img_path).convert("RGB")
        keypoints = sio.loadmat(label_path)["image_info"][0][0][0][0][0]

        img_aug, keypoints_aug, discrete_map_aug = self.aug(img, keypoints)

        return img_aug, keypoints_aug, discrete_map_aug, filename

    def __len__(self):
        return len(self.filenames)

class CustomDataset_v2(Dataset):
    def __init__(self, root, crop_size, down_ratio, aug = Augmentation) -> None:
        super().__init__()
        self.root = root
        self.filenames = list(
            map(
                lambda x: os.path.splitext(x)[0],
                os.listdir(self.root)),
            )
        print("num of datas: ", len(self.filenames))

        self.aug = aug(crop_size=crop_size, down_ratio=down_ratio)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_path = os.path.join(self.root, filename + ".jpg")

        img = Image.open(img_path).convert("RGB")
        img_aug = self.aug(img)

        return img_aug, filename

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    import cv2
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    root = r"D:\workspace\tmp\ShanghaiTech\part_B\train_data"
    crop_size = 512
    down_ratio = 8

    def collect_fn(batch):
        batch_data = list(zip(*batch))
        imgs = torch.stack(batch_data[0], 0)
        kp = batch_data[1]
        dis_map = torch.stack(batch_data[2], 0)
        filenames = batch_data[3]
        return imgs, kp, dis_map, filenames

    dataset = CustomDataset(root, crop_size, down_ratio)
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, drop_last=False, collate_fn=collect_fn
    )

    for e in range(3):
        for idx, (imgs, kps, dis_maps, files) in enumerate(dataloader):
            for (img, kp, dis_map, file) in zip(imgs, kps, dis_maps, files):
                img = np.uint8(img.numpy().transpose(1, 2, 0) * 255)
                print(np.max(img))
                kp = kp.numpy().round().astype(int)
                kp_img = np.zeros_like(img)
                for (x, y) in kp:
                    kp_img[y, x, :] = [255, 0, 0]
                kernel = np.ones(shape = (3, 3), dtype = np.uint8)
                kp_img = cv2.dilate(kp_img, kernel)
                dis_map = dis_map.numpy()
                dis_map = cv2.resize(
                    dis_map, (512, 512), interpolation=cv2.INTER_LINEAR
                )
                dis_map = cv2.applyColorMap(
                    np.uint8(
                        255
                        * (
                            1
                            - cv2.resize(
                                dis_map, (512, 512), interpolation=cv2.INTER_LINEAR
                            )
                        )
                    ),
                    cv2.COLORMAP_JET,
                )
                pic = np.concatenate([img, kp_img, dis_map], axis = 1)
                plt.imsave(os.path.join(root, file + '.jpg'), pic)
            break
