"""
Author: HaoZhi
Date: 2022-09-23 15:28:09
LastEditors: HaoZhi
LastEditTime: 2022-09-23 16:05:00
Description: 
"""
import torch
import pandas as pd

from dataset import valid_loader
from model import Model
from model.head.pred_head import PredHead


device = "cuda:1"

valid_folder = "/disk2/haozhi/tmp/data/test"
valid_csv_path = "/disk2/haozhi/tmp/data/sample_submit.csv"
data_size = (512, 512)
batch_size = 12
num_worker = 4


backbone_name = "effnet"
reg_mode = "dist"


reseum_path = '/disk2/haozhi/tmp/code/ckpt_dist_effnet_compose/model_374.pth'
result_path = "result_dist_effnet374_512.csv"

  

def main():

    if reg_mode == "dist":
        head_name = "cls_dist"
    elif reg_mode == "deci":
        head_name = "cls_deci"

    validloader = valid_loader(
        data_folder=valid_folder,
        csv_path=valid_csv_path,
        data_size=data_size,
        batch_size=batch_size,
        num_worker=num_worker,
    )


    model = Model(backbone_name=backbone_name, head_name=head_name).to(device)
    pred_op = PredHead(mode = reg_mode)

    checkpoint = torch.load(reseum_path)
    model.load_state_dict(checkpoint["model_state_dict"])


    with torch.no_grad():
        model.eval()
        result = []
        for idx, (imgs, pathes) in enumerate(validloader):
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = pred_op(logits)
            for path, pred in zip(pathes, preds):
                result.append([path, pred.detach().cpu().numpy().astype('uint8')])
        pd.DataFrame(result).to_csv(result_path, index= None)



if __name__ == "__main__":
    main()
