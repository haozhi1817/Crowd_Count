'''
Author: HaoZhi
Date: 2022-09-20 15:03:38
LastEditors: HaoZhi
LastEditTime: 2022-09-22 15:20:26
Description: 
'''
import os
import pickle as pk

import numpy as np
import pandas as pd

bins = np.arange(50, 2750, 50).tolist()

def gausion_func(x, mu, sigma):
    y = 1 / ((2 * np.pi) ** 0.5) * np.exp(-1 * (x - mu) ** 2 / (2 * sigma**2))
    return y

def gen_distribute_labels(x, sigma = 40):
    labels = []
    for bin in bins:
        label = gausion_func(x, mu = bin, sigma= sigma)
        labels.append(label)
    labels = np.array(labels) / np.sum(labels)
    return labels

def gen_decimal_labels(x, digits = 4):
    labels = []
    for i in range(1, digits + 1):
        ret = x % 10 ** i
        x -= ret
        labels.append(ret/10 ** i)
    labels.reverse()
    return labels

def gen_dat(csv_path, save_path):
    result = []
    data_csv = pd.read_csv(csv_path)
    files = data_csv['name'].to_numpy()
    labels = data_csv['count'].to_numpy()
    for f, l in zip(files, labels):
        new_label = gen_distribute_labels(l)
        result.append([f, l, new_label])
    with open(os.path.join(save_path, 'train_distribute_compose.dat'), 'wb') as f:
        pk.dump(result,f)


if __name__ == '__main__':
    csv_path = '/disk2/haozhi/tmp/data/compose_train/train_label.csv'
    save_path = '/disk2/haozhi/tmp/data/compose_train'
    gen_dat(csv_path, save_path)






