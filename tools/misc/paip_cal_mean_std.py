import os
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    data_path = "../../data/paip19/train"
    dataset = PaipDataset(data_path)
    cal_mean_and_std(dataset)
