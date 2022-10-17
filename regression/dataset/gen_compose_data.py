'''
Author: HaoZhi
Date: 2022-09-23 14:36:23
LastEditors: HaoZhi
LastEditTime: 2022-09-23 15:17:52
Description: 
'''
import os
import numpy as np
import pandas as pd
from PIL import Image

def load_csv(csv_path):
    data = pd.read_csv(csv_path).to_numpy()
    return data

def load_img(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert("RGB")
    return img

def gen_data_info(img_folder, data_infos):
    result= []
    for name , count in data_infos:
        img = load_img(os.path.join(img_folder, name))
        result.append([name, img.size[0], img.size[1], count])
    return result

def select_wh(data_infos, thresh):
    result = list(filter(lambda x: (x[1] <= thresh) or (x[2] <= thresh), data_infos))
    return result

def compose_data(data_info, target_size):
    result = []
    for i in data_info:
        for j in data_info:
            if ((i[1] + j[1]) <= target_size) and ((i[2] + j[2]) <= target_size):
                result.append([i,j])
    return result

def gen_new_data(data_info, src_img_folder, tgt_img_forder):
    new_data_info = []
    for i, j in data_info:
        img1 = load_img(os.path.join(src_img_folder, i[0]))
        img2 = load_img(os.path.join(src_img_folder, j[0]))
        size1 = img1.size
        size2 = img2.size
        if np.random.rand() > 0.5:
            joint = Image.new('RGB', (size1[0] + size2[0], max(size1[1], size2[1])))
            loc1, loc2 = (0, 0), (size1[0], 0)
        else:
            joint = Image.new('RGB', (max(size1[0], size2[0]), size1[1] + size2[1]))
            loc1, loc2 = (0, 0), (0, size1[1])
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        new_name = '_'.join([i[0].split('.jpg')[0], j[0]])
        new_count = i[-1] + j[-1]
        new_data_info.append([new_name, new_count])
        joint.save(os.path.join(tgt_img_forder, new_name))
    pd.DataFrame(new_data_info).to_csv(os.path.join(tgt_img_forder, 'train_label.csv'), index= None)

        


def main(csv_path, img_folder, thresh, target_folder):
    data_info = load_csv(csv_path)
    data_info = gen_data_info(img_folder, data_info)
    data_info = select_wh(data_info, thresh)
    compose_list = compose_data(data_info, thresh * 2)
    gen_new_data(compose_list, img_folder, target_folder)

if __name__ == '__main__':
    csv_path = r'd:\workspace\tmp\人员聚集识别挑战赛数据集\train_label.csv'
    img_folder = r'D:\workspace\tmp\人员聚集识别挑战赛数据集\train'
    thresh = 512
    target_folder = r'D:\workspace\tmp\人员聚集识别挑战赛数据集\compose_train'
    main(csv_path, img_folder, thresh, target_folder)




