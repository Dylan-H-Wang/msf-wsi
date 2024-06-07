import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


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

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = np.expand_dims(mask>0, axis=2)
        img = (img*mask).astype(np.float64)/255.0
        
        return img, self.num_masked_pixels[index]

    def _prepare(self):
        self.data_df = pd.read_csv(self.csv_path)
        print(f"Reading {len(self.data_df)} files in {self.csv_path}...")
        self.filename_imgs = self.data_df['filename_img'].tolist()
        self.filename_masks = self.data_df['filename_mask'].tolist()
        self.num_masked_pixels = self.data_df['num_masked_pixels'].tolist()


class C16Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self._prepare()
        
    def __len__(self):
        return len(self.filename_imgs)
        
    def __getitem__(self, index: int):
        img_path = os.path.join(self.data_path, self.filename_imgs[index])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        n_pixel = img.shape[0]*img.shape[1]

        # img = img.astype(np.float64)/255.0
        img = torch.tensor(img, dtype=torch.float64)
        n_pixel = torch.tensor(n_pixel, dtype=torch.float64)
        return img, n_pixel

    def _prepare(self):
        self.filename_imgs = [] 
        for image_dir in os.listdir(f"{self.data_path}/imagesTr"):
            self.filename_imgs += [f"imagesTr/{image_dir}/{i}" for i in os.listdir(f"{self.data_path}/imagesTr/{image_dir}")]
        for image_dir in os.listdir(f"{self.data_path}/imagesTs"):
            self.filename_imgs += [f"imagesTs/{image_dir}/{i}" for i in os.listdir(f"{self.data_path}/imagesTs/{image_dir}")]


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


@torch.no_grad()
def cal_mean_and_std_fast(dataset):
    sum = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device="cuda:0")
    sum_sq = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device="cuda:0")
    all_pixels = torch.tensor(0, dtype=torch.float64, device="cuda:0")

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    for images, n_pixels in tqdm(loader):
        images = images.to("cuda:0").view(-1, 3)/255.0
        sum += images.sum(dim=0)
        sum_sq += (images ** 2).sum(dim=0)
        all_pixels += n_pixels.sum().to("cuda:0")

    # mean and std
    total_mean = sum / all_pixels
    total_var = (sum_sq / all_pixels) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print(f"Dataset MEAN is: {total_mean}")
    print(f"Dataset STD is: {total_std}")


if __name__ == '__main__':
    # data_path = "../../data/paip19/train"
    # dataset = PaipDataset(data_path)
    # cal_mean_and_std(dataset)

    data_path = "./data/Dataset001_Camelyon16-1024"
    dataset = C16Dataset(data_path)
    cal_mean_and_std_fast(dataset)
    
