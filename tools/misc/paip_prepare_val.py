import os
import gc
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


class PAIPDataset:
    def __init__(
        self,
        img_path,
        wt_mask_path,
        vt_mask_path,
        tile_size=256,
        img_sz=3072,
        n_crops=4,
    ):
        super().__init__()

        self.tile_size = tile_size
        self.img_sz = img_sz
        self.n_crops = n_crops
        self.slide = OpenSlide(img_path)
        self.w, self.h = self.slide.dimensions
        print("Image size HxW:", self.h, self.w)

        wt_mask = io.imread(wt_mask_path)
        vt_mask = io.imread(vt_mask_path)

        print("### generating tissue mask ...")
        whole_img = self.slide.read_region((0, 0), 0, self.slide.dimensions)
        whole_img = np.array(whole_img, dtype=np.uint8)[:, :, :-1]
        r = whole_img[:, :, 0] <= 235
        g = whole_img[:, :, 1] <= 210
        b = whole_img[:, :, 2] <= 235
        del whole_img
        self.tissue_mask = (r & g & b).astype(np.uint8)

        print("### generating ground truth mask ...")
        self.gt_mask = self.tissue_mask.copy()
        self.gt_mask += wt_mask
        self.gt_mask += vt_mask

        print("### random croping images ...")
        rng = np.random.default_rng()
        self.coords = []  # (x1,y1,x2,y2)   

        for crop_i in trange(1, 4):
            rows, cols = np.nonzero(self.gt_mask==crop_i)
            while True:
                idx = rng.integers(0, len(rows))
                y = rows[idx]
                x = cols[idx]
                if y + self.img_sz > self.h or x + self.img_sz > self.w:
                    continue
                mask = self.gt_mask[y : y + self.img_sz, x : x + self.img_sz]
                ratio = np.count_nonzero(mask == crop_i) / mask.size
                if ratio > 0.3:
                    self.coords.append((x, y, x + self.img_sz, y + self.img_sz))
                    del rows, cols
                    gc.collect()
                    break

        print("### finding balanced crop ...")
        rows, cols = np.nonzero(self.gt_mask>=3)
        row_start, col_start = np.min(rows), np.min(cols)
        row_end, col_end = np.max(rows), np.max(cols)
        del rows, cols
        gc.collect()
        
        for i in range(row_start-self.img_sz//4, row_end-self.img_sz//4, 64):
            for j in range(col_start-self.img_sz//4, col_end-self.img_sz//4, 64):
                y = max(i, 0)
                x = max(j, 0)
                mask = self.gt_mask[y : y + self.img_sz, x : x + self.img_sz]

                tissue_ratio = np.count_nonzero(mask >= 1) / mask.size
                whole_ratio = np.count_nonzero(mask >= 2) / mask.size
                viable_ratio = np.count_nonzero(mask >= 3) / mask.size

                if tissue_ratio >= 0.7:
                    if whole_ratio >= 0.5:
                        if viable_ratio >= 0.3:
                            if tissue_ratio-whole_ratio >= 0.1 and whole_ratio-viable_ratio >= 0.1:
                                self.coords.append((x, y, x + self.img_sz, y + self.img_sz))
                                break   
            if len(self.coords) >= self.n_crops:
                break

        if len(self.coords) < self.n_crops:
            print("Not enough crops found, add an unbalnced crop!")
            rows, cols = np.nonzero(self.gt_mask==3)
            idx = np.arange(0, len(rows))
            rng.shuffle(idx)
            for i in idx:
                y = rows[i]
                x = cols[i]
                if y + self.img_sz > self.h or x + self.img_sz > self.w:
                    continue
                mask = self.gt_mask[y : y + self.img_sz, x : x + self.img_sz]
                ratio = np.count_nonzero(mask == 3) / mask.size
                if ratio > 0.3:
                    self.coords.append((x, y, x + self.img_sz, y + self.img_sz))
                    break

    def __len__(self):
        return self.n_crops

    # Patchs are padded with 0s if reach the edge
    def __getitem__(self, idx):
        x1, y1, x2, y2 = self.coords[idx]

        context_imgs = []
        target_imgs = []
        tissue_masks = []
        gt_masks = []

        for y0 in range(y1, y2, self.tile_size):
            for x0 in range(x1, x2, self.tile_size):
                if x0 + self.tile_size > x2 or y0 + self.tile_size > y2:
                    continue

                # TODO: hardcoded
                cpy0 = y0 - self.tile_size // 2 * 3
                cpx0 = x0 - self.tile_size // 2 * 3
                cpy00, cpx00 = abs(min(0, cpy0)), abs(min(0, cpx0))
                cpy01, cpx01 = max(0, cpy0), max(0, cpx0)

                cpy1 = cpy0 + self.tile_size * 4
                cpx1 = cpx0 + self.tile_size * 4
                cpy11, cpx11 = min(self.h, cpy1), min(self.w, cpx1)

                context_img = np.zeros(
                    (self.tile_size * 4, self.tile_size * 4, 3), np.uint8
                )
                context_rgba_img = self.slide.read_region(
                    (cpx01, cpy01), 0, (cpx11 - cpx01, cpy11 - cpy01)
                )
                context_img[
                    cpy00 : (cpy11 - cpy01 + cpy00), cpx00 : (cpx11 - cpx01 + cpx00)
                ] = np.asarray(context_rgba_img, dtype=np.uint8)[:, :, :-1]
                context_img = resize(
                    context_img,
                    (self.tile_size, self.tile_size, 3),
                    anti_aliasing=True,
                    preserve_range=True,
                ).astype(np.uint8)

                target_img = self.slide.read_region(
                    (x0, y0), 0, (self.tile_size, self.tile_size)
                )
                target_img = np.asarray(target_img, dtype=np.uint8)[:, :, :-1]
                tissue_mask = self.tissue_mask[
                    y0 : y0 + self.tile_size, x0 : x0 + self.tile_size
                ]
                gt_mask = self.gt_mask[
                    y0 : y0 + self.tile_size, x0 : x0 + self.tile_size
                ]

                context_imgs.append(context_img)
                target_imgs.append(target_img)
                tissue_masks.append(tissue_mask)
                gt_masks.append(gt_mask)

        assert (
            len(context_imgs)
            == len(target_imgs)
            == len(tissue_masks)
            == len(gt_masks)
            == (self.img_sz // self.tile_size) ** 2
        )
        return context_imgs, target_imgs, tissue_masks, gt_masks


def generate_data(
    filename, i, context_imgs, target_imgs, tissue_masks, gt_masks, output_path
):
    data = []
    for idx, (context_img, target_img, tissue_mask, gt_mask) in enumerate(
        zip(context_imgs, target_imgs, tissue_masks, gt_masks)
    ):
        context_img_path = os.path.join(
            output_path, filename, f"context_imgs/{i*len(context_imgs)+idx:2d}.png"
        )
        Image.fromarray(context_img).save(context_img_path)

        target_img_path = os.path.join(
            output_path, filename, f"target_imgs/{i*len(context_imgs)+idx:2d}.png"
        )
        Image.fromarray(target_img).save(target_img_path)

        tissue_masks_path = os.path.join(
            output_path, filename, f"tissue_masks/{i*len(context_imgs)+idx:2d}.png"
        )
        Image.fromarray(tissue_mask).save(tissue_masks_path)

        gt_masks_path = os.path.join(
            output_path, filename, f"gt_masks/{i*len(context_imgs)+idx:2d}.png"
        )
        Image.fromarray(gt_mask).save(gt_masks_path)

        data.append(
            [
                f"{filename}/context_imgs/{context_img_path.split('/')[-1]}",
                f"{filename}/target_imgs/{target_img_path.split('/')[-1]}",
                f"{filename}/tissue_masks/{tissue_masks_path.split('/')[-1]}",
                f"{filename}/gt_masks/{gt_masks_path.split('/')[-1]}",
                filename,
            ]
        )

    return data


def main(data_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_list = []
    imgs = sorted(filter(lambda x: x.split(".")[-1] == "svs", os.listdir(data_path)))

    print(f"================== Val Statistics ==================")
    print(f"### no. of img: {len(imgs)}")
    print("================== End ==================\n")

    for idx, img in enumerate(imgs):
        # img = "01_01_0106.svs"
        print(f"### processing image {idx+1}/{len(imgs)}: {img}")
        filename = img.split(".")[0]
        os.makedirs(os.path.join(output_path, filename, "context_imgs"))
        os.makedirs(os.path.join(output_path, filename, "target_imgs"))
        os.makedirs(os.path.join(output_path, filename, "tissue_masks"))
        os.makedirs(os.path.join(output_path, filename, "gt_masks"))

        img_path = os.path.join(data_path, img)
        wt_mask_path = f"{data_path}/{filename}_whole.tif"
        vt_mask_path = f"{data_path}/{filename}_viable.tif"
        ds = PAIPDataset(img_path, wt_mask_path, vt_mask_path)

        print("### tiling data ...")
        data = Parallel(n_jobs=1)(
            delayed(generate_data)(filename, i, x, y, z, k, output_path)
            for i, (x, y, z, k) in enumerate(tqdm(ds))
        )
        data_list.append([j for i in data for j in i])

    # save
    data_df = pd.concat(
        [pd.DataFrame(data_list[i]) for i in range(len(data_list))], axis=0
    ).reset_index(drop=True)
    data_df.columns = [
        "context_img",
        "target_img",
        "tissue_mask",
        "gt_mask",
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
    parser.add_argument("-d", "--dry-run", action="store_true")
    args = parser.parse_args()

    main(args.data_path, args.out_path)
