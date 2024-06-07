import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import glob

class PaipDataset():
    def __init__(self, data_path):
        super().__init__()

        self.data_path = data_path
        self.csv_path = data_path + f"/train_data.csv"

        self._prepare()
        
    def __len__(self):
        return len(self.data_df)
        
    def __getitem__(self, index: int):
        img_path = os.path.join(self.data_path, self.filename_imgs[index])
        mask_path = os.path.join(self.data_path, self.filename_masks[index])

        img = np.asarray(Image.open(img_path), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path), dtype=np.uint8).clip(0, 1)
        mask = np.expand_dims(mask, axis=2)
        img = (img*mask).astype(np.float64)/255.0
        
        return img, self.num_masked_pixels[index]

    def _prepare(self):
        self.data_df = pd.read_csv(self.csv_path)
        print(f"Reading {len(self.data_df)} files in {self.csv_path}...")
        self.filename_imgs = self.data_df['filename_img'].tolist()
        self.filename_masks = self.data_df['filename_mask'].tolist()
        self.num_masked_pixels = self.data_df['num_masked_pixels'].tolist()

def cal_mean_and_std(dataset):
    sum = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    sum_sq = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    n_pixels = 0

    for images, masked_pixel in tqdm(dataset):
        sum += images.sum(axis=(0, 1))
        sum_sq += (images ** 2).sum(axis=(0, 1))
        n_pixels += masked_pixel

    # mean and std
    total_mean = sum / n_pixels
    total_var = (sum_sq / n_pixels) - (total_mean ** 2)
    total_std = np.sqrt(total_var)

    # output
    print(f"Dataset MEAN is: {total_mean}")
    print(f"Dataset STD is: {total_std}")

if __name__ == '__main__':
    # data_path = "../../data/paip19/train"
    # dataset = PaipDataset(data_path)
    # cal_mean_and_std(dataset)

    # val_path = "/mnt/Hao/projs/data/PAIP19/val1"
    # files = glob.glob("/mnt/Hao/projs/data/PAIP19/train*/*")
    # val_set = [f.split("/")[-1] for f in files]
    # val_set = list(filter(lambda x: x.split(".")[-1]=="svs", val_set))
    # val_set = [f.split(".")[0] for f in val_set]
    # val_set = sorted(random.sample(val_set, 10))
    # print(val_set)
    # for f in files:
    #     file_name = f.split("/")[-1]
    #     if any(map(file_name.__contains__, val_set)):
    #         os.symlink(f, f"{val_path}/{file_name}")

    import pickle as pkl
    import torch

    # input = "/home/dylan/projs/SLF-WSI/logs/unet/slf_hist/tenpercent_resnet18.pth"
    input = "./projs/SLF-WSI/logs/unet/slf_hist/tenpercent_resnet18.ckpt"

    obj = torch.load(input, map_location="cpu")
    obj = obj["state_dict"]

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("model.resnet.fc"):
            old_k = k
            k = k.replace("model.resnet.", "")
            newmodel[k] = v

    torch.save(newmodel, "./logs/unet/slf_hist/tenpercent_resnet18.pth")
    for k in newmodel.keys():
        print(k)
    
    # data_path = "/mnt/Hao/data/BCSS/L0/images"
    # names = os.listdir(data_path)
    # # names = set([i.split("-")[1] for i in names])
    # # print(names)
    # # print(len(names))
    # from collections import Counter
    # import random
    # names = Counter([i.split("-")[1] for i in names])
    # random_name = random.sample(names.keys(), 6)
    # print(names)
    # print(random_name)
    # print(sum([names[i] for i in random_name]))
    # a = ["OL", "LL", "E2", "EW", "GM", "S3"]
    # print(sum([names[i] for i in a]))

    # from dataset.paip import *
    # from utils.logger import setup_logger
    # logger = setup_logger("./logs/temp")
    # train_dataset = PaipPretrainDatasetMS5_2(
    #     "../data/paip19/train",
    #     None,
    #     return_index=False,
    #     fold=1
    # )

    # train_dataset = PaipSegDatasetMS("../data/paip19/train", None, frac=1, fold=1)
    # val_dataset = PaipSegDatasetValMS("../data/paip19/val2", None)