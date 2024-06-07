from email.mime import base
import os
import glob
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from PIL import Image
from joblib import Parallel, delayed
from openslide import OpenSlide

import skimage.io as io
from skimage.transform import resize


class PAIPDataset():
    def __init__(self, img_path, tile_size=256):
        super().__init__()

        self.tile_size = tile_size
        self.img = OpenSlide(img_path)

        self.w, self.h = self.img.dimensions
        self.sz = tile_size
        self.pad_h = self.sz - self.h % self.sz  # add to whole slide
        self.pad_w = self.sz - self.w % self.sz  # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.sz
        self.num_w = (self.w + self.pad_w) // self.sz

        print("### generating tissue mask ...")
        whole_img = self.img.read_region((0,0), 0, self.img.dimensions)
        whole_img = np.array(whole_img, dtype=np.uint8)[:, :, :-1]
        r = whole_img[:,:,0] <= 235
        g = whole_img[:,:,1] <= 210
        b = whole_img[:,:,2] <= 235
        del whole_img
        self.tissue_mask = (r & g & b).astype(np.uint8)

    def __len__(self):
        return self.num_h * self.num_w

    # Patchs are padded with 0s if reach the edge
    def __getitem__(self, idx):  # idx = i_h * self.num_w + i_w
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h * self.sz
        x = i_w * self.sz
        py0, py1 = max(0, y), min(y + self.sz, self.h)
        px0, px1 = max(0, x), min(x + self.sz, self.w)

        # TODO: hardcoded
        cpy0 = py0 - self.sz//2*3
        cpx0 = px0 - self.sz//2*3
        cpy00, cpx00 = abs(min(0, cpy0)), abs(min(0, cpx0))
        cpy01, cpx01 = max(0, cpy0), max(0, cpx0)

        cpy1 = cpy0 + self.sz*4
        cpx1 = cpx0 + self.sz*4
        cpy11, cpx11 = min(self.h, cpy1), min(self.w, cpx1)

        # placeholder for input tile (before resize)
        context_img = np.zeros((self.sz*4, self.sz*4, 3), np.uint8) # TODO: hardcoded
        target_img = np.zeros((self.sz, self.sz, 3), np.uint8)
        tissue_mask = np.zeros((self.sz, self.sz), np.uint8)

        tissue_mask[0 : py1 - py0, 0 : px1 - px0] = self.tissue_mask[py0 : py1, px0 : px1]

        if np.count_nonzero(tissue_mask)/tissue_mask.size >= 0.1:
            context_rgba_img = self.img.read_region((cpx01, cpy01), 0, (cpx11 - cpx01, cpy11 - cpy01))
            context_img[cpy00:(cpy11-cpy01+cpy00), cpx00:(cpx11-cpx01+cpx00)] = np.array(context_rgba_img, dtype=np.uint8)[:, :, :-1]

            rgba_img = self.img.read_region((px0, py0), 0, (px1 - px0, py1 - py0))
            target_img[0 : py1 - py0, 0 : px1 - px0] = np.array(rgba_img, dtype=np.uint8)[:, :, :-1]

        context_img = resize(context_img, (self.sz, self.sz, 3), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        # return {"context_img": context_img, "target_img": target_img, "tissue_mask": tissue_mask}
        return context_img, target_img, tissue_mask


def generate_data(filename, i, context_img, target_img, tissue_mask, output_path):
    is_empty = np.count_nonzero(context_img) == 0

    if not is_empty:
        context_img_path = os.path.join(output_path, filename, f"context_imgs/{i}.png")
        Image.fromarray(context_img).save(context_img_path)

        target_img_path = os.path.join(output_path, filename, f"target_imgs/{i}.png")
        Image.fromarray(target_img).save(target_img_path)

        tissue_masks_path = os.path.join(output_path, filename, f"tissue_masks/{i}.png")
        Image.fromarray(tissue_mask).save(tissue_masks_path)

        data = [
            f"{filename}/context_imgs/{context_img_path.split('/')[-1]}",
            f"{filename}/target_imgs/{target_img_path.split('/')[-1]}",
            f"{filename}/tissue_masks/{tissue_masks_path.split('/')[-1]}",
            filename,
        ]
    
    else:
        tissue_masks_path = os.path.join(output_path, filename, f"tissue_masks/{i}.png")
        Image.fromarray(tissue_mask).save(tissue_masks_path)

        data = [
            f"{filename}/context_imgs/{i}_empty",
            f"{filename}/target_imgs/{i}_empty",
            f"{filename}/tissue_masks/{tissue_masks_path.split('/')[-1]}",
            filename,
        ]

    return data


def main(data_path, out_path, tile_size):
    base_path = data_path
    output_path = out_path
    if not os.path.exists(output_path): os.makedirs(output_path)

    data_list = []
    val_path = os.path.join(base_path, "val1")
    imgs = sorted(
        filter(lambda x: x.split(".")[-1] == "svs", os.listdir(val_path))
    )

    print(f"================== Val Statistics ==================")
    print(f"### no. of img: {len(imgs)}")
    print("================== End ==================\n")

    for idx, img in enumerate(imgs):
        print(f"### processing image {idx+1}/{len(imgs)}: {img}")

        filename = img.split(".")[0]
        os.makedirs(os.path.join(output_path, filename, "context_imgs"))
        os.makedirs(os.path.join(output_path, filename, "target_imgs"))
        os.makedirs(os.path.join(output_path, filename, "tissue_masks"))
        os.makedirs(os.path.join(output_path, filename, "gt_masks"))

        img_path = os.path.join(val_path, img)
        # context_imgs = []
        # target_imgs = []
        # tissue_masks = []

        ds = PAIPDataset(img_path, tile_size=tile_size)
        # for i in trange(len(ds)):
        #     data = ds[i]
        #     context_imgs.append(data["context_img"])
        #     target_imgs.append(data["target_img"])
        #     tissue_masks.append(data["tissue_mask"])

        #     context_imgs = np.stack(context_imgs, axis=0)
        #     target_imgs = np.stack(target_imgs, axis=0)
        #     tissue_masks = np.stack(tissue_masks, axis=0)

        #     assert len(context_imgs) == len(target_imgs) == len(tissue_masks) == len(ds)

        #     data = Parallel(n_jobs=-1)(
        #         delayed(generate_data)(filename, i, x, y, z, output_path)
        #         for i, (x, y, z) in enumerate(zip(context_imgs, target_imgs, tissue_masks))
        #     )
        #     data = list(filter(lambda x: x is not None, data))
        #     data_list.append(data)
        print("### tiling data ...")
        data = Parallel(n_jobs=-1)(
            delayed(generate_data)(filename, i, x, y, z, output_path)
            for i, (x, y, z) in enumerate(tqdm(ds))
        )
        data = list(filter(lambda x: x is not None, data))
        data_list.append(data)
          
        wt_mask = io.imread(f"{val_path}/{filename}_whole.tif")
        vt_mask = io.imread(f"{val_path}/{filename}_viable.tif")
        gt_mask = ds.tissue_mask.copy()
        gt_mask += wt_mask
        gt_mask += vt_mask
        Image.fromarray(gt_mask).save(f"{output_path}/{filename}/gt_masks/gt_mask.png")

    # save
    data_df = pd.concat(
        [pd.DataFrame(data_list[i]) for i in range(len(data_list))], axis=0
    ).reset_index(drop=True)
    data_df.columns = [
        "context_img",
        "target_img",
        "tissue_mask",
        "filename",
    ]
    print(data_df.shape)
    data_df.to_csv(os.path.join(output_path, "val_data.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch extraction for PAIP19")
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
        "-s",
        "--tile-size",
        type=int,
        default=256,
        help="Size of tiles",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true"
    )
    args = parser.parse_args()

    main(args.data_path, args.out_path, args.tile_size)
