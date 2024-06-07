# read 1024x1024 patch but slide with 256 size

import argparse
import numpy as np
import os
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
from PIL import Image

from torch.utils.data import Dataset, DataLoader


class BCSSDataset(Dataset):
    def __init__(
        self, filename, img_path, mask_path, scales=(1, 4), tile_size=256,
    ):
        super().__init__()

        img_path = os.path.join(img_path, filename + ".png")
        mask_path = os.path.join(mask_path, filename + ".png")
        self.scales = np.asarray(scales)
        self.tile_size = tile_size
        self.img = np.array(Image.open(img_path))
        self.mask = np.array(Image.open(mask_path))

        self.classes = {
            "1": [1, 19, 20],
            "2": [2],
            "3": [3, 10, 11, 14],
            "4": [4],
            "5": [5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 18, 21],
        }

        for k, v in self.classes.items():
            self.mask[np.isin(self.mask, v)] = k

        self.h, self.w = self.img.shape[0], self.img.shape[1]
        self.sz = self.tile_size*self.scales
        self.pad_h = (self.h-self.sz[1])%self.sz[0]
        self.pad_w = (self.w-self.sz[1])%self.sz[0]
        self.num_h = (self.h + self.pad_h - self.sz[1]) // self.sz[0] + 1
        self.num_w = (self.w + self.pad_w - self.sz[1]) // self.sz[0] + 1

    def __len__(self):
        return self.num_h * self.num_w

    # Patchs are padded with 0s if reach the edge
    def __getitem__(self, idx):  # idx = i_h * self.num_w + i_w
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h * self.sz[0]
        x = i_w * self.sz[0]
        py0, py1 = max(0, y), min(y + self.sz[1], self.h)
        px0, px1 = max(0, x), min(x + self.sz[1], self.w)

        # TODO: hardcoded for two scales only
        img_patch = np.zeros((self.sz[1], self.sz[1], 3), np.uint8)
        mask_patch = np.zeros((self.sz[1], self.sz[1]), np.uint8)

        img_patch[0 : py1 - py0, 0 : px1 - px0] = self.img[py0:py1, px0:px1]
        mask_patch[0 : py1 - py0, 0 : px1 - px0] = self.mask[py0:py1, px0:px1]

        return {"img": img_patch, "mask": mask_patch}


def generate_data(filename, i, img_patch, mask_patch, output_path):
    mask_clip = np.clip(mask_patch, 0, 1)
    num_masked_pixels = mask_clip.sum()
    ratio_masked_area = mask_clip.sum() / (mask_clip.shape[0] * mask_clip.shape[1])

    if num_masked_pixels == 0:
        return None

    ratio_masked_1_area = (mask_patch == 1).sum() / (
        mask_clip.shape[0] * mask_clip.shape[1]
    )
    ratio_masked_2_area = (mask_patch == 2).sum() / (
        mask_clip.shape[0] * mask_clip.shape[1]
    )
    ratio_masked_3_area = (mask_patch == 3).sum() / (
        mask_clip.shape[0] * mask_clip.shape[1]
    )
    ratio_masked_4_area = (mask_patch == 4).sum() / (
        mask_clip.shape[0] * mask_clip.shape[1]
    )
    ratio_masked_5_area = (mask_patch == 5).sum() / (
        mask_clip.shape[0] * mask_clip.shape[1]
    )

    img_save_path = os.path.join(output_path, filename, f"images/{i}.png")
    img_patch = img_patch.copy()
    img_patch[~mask_clip.astype(bool)] = 0  # Remove outside pixels
    Image.fromarray(img_patch).save(img_save_path)

    mask_save_path = os.path.join(output_path, filename, f"masks/{i}.png")
    Image.fromarray(mask_patch).save(mask_save_path)

    data = [
        f"{filename}/images/{img_save_path.split('/')[-1]}",
        f"{filename}/masks/{mask_save_path.split('/')[-1]}",
        filename,
        num_masked_pixels,
        ratio_masked_area,
        ratio_masked_1_area,
        ratio_masked_2_area,
        ratio_masked_3_area,
        ratio_masked_4_area,
        ratio_masked_5_area
    ]
    return data


def main(data_path, out_path, tile_size):
    base_path = data_path
    img_path = os.path.join(base_path, "images")
    mask_path = os.path.join(base_path, "masks")
    output_path = out_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    val_name_list = ["OL", "LL", "E2", "EW", "GM", "S3"]
    train_list = list(
        filter(lambda x: not x.split("-")[1] in val_name_list, os.listdir(img_path))
    )
    val_list = list(
        filter(lambda x: x.split("-")[1] in val_name_list, os.listdir(img_path))
    )

    # generate training data
    data_list = []
    for idx, filename in enumerate(tqdm(train_list)):
        print("idx = {}, {}".format(idx, filename))
        filename = filename.split(".png")[0]
        os.makedirs(os.path.join(output_path, filename, "images"))
        os.makedirs(os.path.join(output_path, filename, "masks"))

        ds = BCSSDataset(filename, img_path, mask_path, tile_size=tile_size)
        dl = DataLoader(ds, batch_size=256, num_workers=0)
        img_patches = []
        mask_patches = []
        for data in tqdm(dl):
            img_patch = data["img"]
            mask_patch = data["mask"]
            img_patches.append(img_patch)
            mask_patches.append(mask_patch)
        img_patches = np.vstack(img_patches)
        mask_patches = np.vstack(mask_patches)

        data = Parallel(n_jobs=-1)(
            delayed(generate_data)(filename, i, x, y, output_path)
            for i, (x, y) in enumerate(zip(img_patches, mask_patches))
        )
        data = list(filter(lambda x: x is not None, data))
        data_list.append(data)

    # save
    data_df = pd.concat(
        [pd.DataFrame(data_list[i]) for i in range(len(data_list))], axis=0
    ).reset_index(drop=True)
    data_df.columns = [
        "filename_img",
        "filename_mask",
        "filename",
        "num_masked_pixels",
        "ratio_masked_area",
        "ratio_masked_1_area",
        "ratio_masked_2_area",
        "ratio_masked_3_area",
        "ratio_masked_4_area",
        "ratio_masked_5_area"
    ]
    print(data_df.shape)
    data_df.to_csv(os.path.join(output_path, "train_data.csv"), index=False)

    # generate val data
    data_list = []
    for idx, filename in enumerate(tqdm(val_list)):
        print("idx = {}, {}".format(idx, filename))
        filename = filename.split(".png")[0]
        os.makedirs(os.path.join(output_path, filename, "images"))
        os.makedirs(os.path.join(output_path, filename, "masks"))

        ds = BCSSDataset(filename, img_path, mask_path, tile_size=tile_size)
        dl = DataLoader(ds, batch_size=256, num_workers=0)
        img_patches = []
        mask_patches = []
        for data in tqdm(dl):
            img_patch = data["img"]
            mask_patch = data["mask"]
            img_patches.append(img_patch)
            mask_patches.append(mask_patch)
        img_patches = np.vstack(img_patches)
        mask_patches = np.vstack(mask_patches)

        data = Parallel(n_jobs=-1)(
            delayed(generate_data)(filename, i, x, y, output_path)
            for i, (x, y) in enumerate(zip(img_patches, mask_patches))
        )
        data = list(filter(lambda x: x is not None, data))
        data_list.append(data)

    # save
    data_df = pd.concat(
        [pd.DataFrame(data_list[i]) for i in range(len(data_list))], axis=0
    ).reset_index(drop=True)
    data_df.columns = [
        "filename_img",
        "filename_mask",
        "filename",
        "num_masked_pixels",
        "ratio_masked_area",
        "ratio_masked_1_area",
        "ratio_masked_2_area",
        "ratio_masked_3_area",
        "ratio_masked_4_area",
        "ratio_masked_5_area"
    ]
    print(data_df.shape)
    data_df.to_csv(os.path.join(output_path, "val_data.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch extraction for BCSS")
    parser.add_argument(
        "-p", "--data-path", type=str, default="", help="Path to the dataset"
    )
    parser.add_argument(
        "-o",
        "--out-path",
        type=str,
        default="",
        help="Path to the save processed dataset",
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=0,
        choices=range(0, 7),
        help="Region level for WSI, chose from [0, 6]",
    )
    parser.add_argument(
        "-s",
        "--tile-size",
        type=int,
        default=256,
        help="Size of tiles",
    )
    args = parser.parse_args()

    main(args.data_path, args.out_path, args.tile_size)
