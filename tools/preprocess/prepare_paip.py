import os
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
    def __init__(
        self, img_path, wt_path, vt_path, tile_size=256, shift_h=0, shift_w=0
    ):
        super().__init__()

        self.tile_size = tile_size
        self.img = OpenSlide(img_path)
        whole_img = self.img.read_region((0,0), 0, self.img.dimensions)
        whole_img = np.array(whole_img, dtype=np.uint8)[:, :, :-1]
        r = whole_img[:,:,0] <= 235
        g = whole_img[:,:,1] <= 210
        b = whole_img[:,:,2] <= 235
        self.img_mask = (r & g & b).astype(np.uint8)

        self.wt_mask = io.imread(wt_path)
        self.vt_mask = io.imread(vt_path)

        self.w, self.h = self.img.dimensions
        self.sz = tile_size
        self.shift_h = shift_h
        self.shift_w = shift_w
        self.pad_h = self.sz - self.h % self.sz  # add to whole slide
        self.pad_w = self.sz - self.w % self.sz  # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.sz
        self.num_w = (self.w + self.pad_w) // self.sz

        if self.h % self.sz < self.shift_h:
            self.num_h -= 1
        if self.w % self.sz < self.shift_w:
            self.num_w -= 1

    def __len__(self):
        return self.num_h * self.num_w

    # Patchs are padded with 0s if reach the edge
    def __getitem__(self, idx):  # idx = i_h * self.num_w + i_w
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h * self.sz + self.shift_h
        x = i_w * self.sz + self.shift_w
        py0, py1 = max(0, y), min(y + self.sz, self.h)
        px0, px1 = max(0, x), min(x + self.sz, self.w)

        # placeholder for input tile (before resize)
        img_patch = np.zeros((self.sz, self.sz, 3), np.uint8)
        mask_patch = np.zeros((self.sz, self.sz), np.uint8)

        rgba_img = self.img.read_region((px0, py0), 0, (px1 - px0, py1 - py0))
        # img_patch[0 : py1 - py0, 0 : px1 - px0] = rgba2rgb(np.array(rgba_img, dtype=np.uint8))
        img_patch[0 : py1 - py0, 0 : px1 - px0] = np.array(rgba_img, dtype=np.uint8)[:, :, :-1]

        mask_patch[0 : py1 - py0, 0 : px1 - px0] += self.img_mask[py0:py1, px0:px1].astype(np.uint8)
        mask_patch[0 : py1 - py0, 0 : px1 - px0] += self.wt_mask[py0:py1, px0:px1].astype(np.uint8)
        mask_patch[0 : py1 - py0, 0 : px1 - px0] += self.vt_mask[py0:py1, px0:px1].astype(np.uint8)

        return {"img": img_patch, "mask": mask_patch}


def generate_data(filename, i, img_patch, mask_patch, output_path):
    num_masked_pixels = np.count_nonzero(mask_patch)
    ratio_masked_area = num_masked_pixels / mask_patch.size

    if ratio_masked_area <= 0.1:
        return None

    ratio_masked_1_area = (mask_patch == 1).sum() / mask_patch.size
    ratio_masked_2_area = (mask_patch == 2).sum() / mask_patch.size
    ratio_masked_3_area = (mask_patch == 3).sum() / mask_patch.size

    img_save_path = os.path.join(output_path, filename, f"images/{i}.png")
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
    ]
    return data


def main(data_path, out_path, tile_size):
    base_path = data_path
    output_path = out_path
    if not os.path.exists(output_path): os.makedirs(output_path)

    data_list = []
    for i in ["train_1", "train_2"]:
        train_path = os.path.join(base_path, i)
        imgs = sorted(
            filter(lambda x: x.split(".")[-1] == "svs", os.listdir(train_path))
        )
        whole_tumour = sorted(
            filter(
                lambda x: x.split(".")[0].split("_")[-1] == "whole",
                os.listdir(train_path),
            )
        )
        viable_tumour = sorted(
            filter(
                lambda x: x.split(".")[0].split("_")[-1] == "viable",
                os.listdir(train_path),
            )
        )
        assert len(imgs) == len(whole_tumour) == len(viable_tumour)

        print(f"================== {i} Statistics ==================")
        print(f"### no. of img: {len(imgs)}")
        print(f"### no. of whole tumour: {len(whole_tumour)}")
        print(f"### no. of viable tumour: {len(viable_tumour)}")
        print("================== End ==================\n")

        for idx, (img, wt, vt) in enumerate(zip(imgs, whole_tumour, viable_tumour)):
            print(f"### processing image {idx+1}/{len(imgs)}: {img}")

            filename = img.split(".")[0]
            os.makedirs(os.path.join(output_path, filename, "images"))
            os.makedirs(os.path.join(output_path, filename, "masks"))

            img_path = os.path.join(train_path, img)
            wt_path = os.path.join(train_path, wt)
            vt_path = os.path.join(train_path, vt)
            
            ds = PAIPDataset(img_path, wt_path, vt_path, tile_size=tile_size)
            img_patches = []
            mask_patches = []
            for i in trange(len(ds)):
                data = ds[i]
                img_patches.append(data["img"])
                mask_patches.append(data["mask"])
            img_patches = np.stack(img_patches, axis=0)
            mask_patches = np.stack(mask_patches, axis=0)
            assert len(img_patches) == len(mask_patches) == len(ds)

            data = Parallel(n_jobs=-1)(
                delayed(generate_data)(filename, i, x, y, output_path)
                for i, (x, y) in enumerate(zip(img_patches, mask_patches))
            )
            data = list(filter(lambda x: x is not None, data))
            data_list.append(data)

            # print("####")
            # print(data)
            # print("####")

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
    ]
    print(data_df.shape)
    data_df.to_csv(os.path.join(output_path, "train_data.csv"), index=False)


def sanity_check(data_path, out_path):
    def img_loader(fn):
        slide = OpenSlide(fn)
        img = slide.read_region((0, 0), 2, slide.level_dimensions[2])
        img = np.array(img)[:, :, :-1]
        print("### img shape:", img.shape)
        return img


    def mask_loader(fn):
        """
        This is a simplest loader for the given tif mask labels, which are compressed in 'LZW' format for logistic convenience.
        Scikit-image library can automatically decompress and load them on your physical memory.
        """
        assert os.path.isfile(fn)
        mask = io.imread(fn)
        return mask

    OVERLAY_MASK_RATIO=0.3
    def gen_overlay(orig_img, wt_msk, vt_msk):
        """
        We don't give a loader for original svs image because there are well-known open source libraries already.
        (e.g. openslide, pyvips, etc.)
        We assume that original image has [H, W, C(=3)] dimension and mask has [H, W] dimension.
        """
        assert wt_msk.shape == vt_msk.shape
        if orig_img.shape[:-1] != wt_msk.shape:
            print(
                f"### align mask size with img size from {wt_msk.shape} to {orig_img.shape[:-1]}"
            )
            wt_msk = resize(wt_msk, orig_img.shape[:-1], preserve_range=True)
            vt_msk = resize(vt_msk, orig_img.shape[:-1], preserve_range=True)

        img_dark = (orig_img * (1.0 - OVERLAY_MASK_RATIO)).astype(np.uint8)

        r = orig_img[:,:,0] <= 235
        g = orig_img[:,:,1] <= 210
        b = orig_img[:,:,2] <= 235
        img_mask = (r & g & b).astype(np.uint8)

        gmsk = np.zeros(orig_img.shape, dtype=np.uint8)
        gmsk[:, :, 0] += (img_mask * 128 * OVERLAY_MASK_RATIO).astype(
            np.uint8
        )  # assign RED color for whole mask labels
        gmsk[:, :, 1] += (wt_msk * 255 * OVERLAY_MASK_RATIO).astype(
            np.uint8
        )  # assign GREEN color for whole mask labels
        gmsk[:, :, 2] += (vt_msk * 255 * OVERLAY_MASK_RATIO).astype(
            np.uint8
        )  # assign BLUE color for viable mask labels

        img_dark += gmsk
        img_dark[img_dark > 255] = 255
        return img_dark.astype(np.uint8)


    def gen_overlay_img(filename, i, orig_img, msk, output_path):
        img_dark = (orig_img * (1.0 - OVERLAY_MASK_RATIO)).astype(np.uint8)

        gmsk = np.zeros(orig_img.shape, dtype=np.uint8)
        gmsk[:, :, 0] += ((msk>0) * 128 * OVERLAY_MASK_RATIO).astype(
            np.uint8
        )  # assign GREEN color for whole mask labels
        gmsk[:, :, 1] += ((msk>1) * 255 * OVERLAY_MASK_RATIO).astype(
            np.uint8
        )  # assign BLUE color for viable mask labels
        gmsk[:, :, 2] += ((msk>2) * 255 * OVERLAY_MASK_RATIO).astype(
            np.uint8
        )  # assign BLUE color for viable mask labels

        img_dark += gmsk
        img_dark[img_dark > 255] = 255
        overlay = img_dark.astype(np.uint8)

        img_save_path = os.path.join(output_path, filename, f"images/{i}.png")
        Image.fromarray(overlay).save(img_save_path)

    base_path = data_path
    output_path = out_path
    if not os.path.exists(output_path): os.makedirs(output_path)

    for i in ["train_1", "train_2"]:
        train_path = os.path.join(base_path, i)
        imgs = sorted(
            filter(lambda x: x.split(".")[-1] == "svs", os.listdir(train_path))
        )
        whole_tumour = sorted(
            filter(
                lambda x: x.split(".")[0].split("_")[-1] == "whole",
                os.listdir(train_path),
            )
        )
        viable_tumour = sorted(
            filter(
                lambda x: x.split(".")[0].split("_")[-1] == "viable",
                os.listdir(train_path),
            )
        )
        assert len(imgs) == len(whole_tumour) == len(viable_tumour)

        print(f"================== {i} Statistics ==================")
        print(f"### no. of img: {len(imgs)}")
        print(f"### no. of whole tumour: {len(whole_tumour)}")
        print(f"### no. of viable tumour: {len(viable_tumour)}")
        print("================== End ==================\n")

        for idx, (img, wt, vt) in enumerate(zip(imgs, whole_tumour, viable_tumour)):
            print(f"### processing image {idx+1}/{len(imgs)}: {img}")

            filename = img.split(".")[0]
            os.makedirs(os.path.join(output_path, filename, "images"))
            os.makedirs(os.path.join(output_path, filename, "masks"))

            img_path = os.path.join(train_path, img)
            wt_path = os.path.join(train_path, wt)
            vt_path = os.path.join(train_path, vt)

            wt_mask = mask_loader(wt_path)
            vt_mask = mask_loader(vt_path)

            orig_img = img_loader(img_path)
            overlay = gen_overlay(orig_img, wt_mask, vt_mask)

            io.imsave(f"{output_path}/{img.split('.')[0]}.png", overlay)
            print("### original image saved!\n")
            
            ds = PAIPDataset(img_path, wt_path, vt_path, tile_size=1024)
            img_patches = []
            mask_patches = []
            for i in trange(len(ds)):
                data = ds[i]
                img_patches.append(data["img"])
                mask_patches.append(data["mask"])
            img_patches = np.stack(img_patches, axis=0)
            mask_patches = np.stack(mask_patches, axis=0)
            assert len(img_patches) == len(mask_patches) == len(ds)

            data = Parallel(n_jobs=-1)(
                delayed(gen_overlay_img)(filename, i, x, y, output_path)
                for i, (x, y) in enumerate(zip(img_patches, mask_patches))
            )
            break


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

    if args.dry_run:
        print("### dry run")
        print(f"### data path: {args.data_path}")
        print(f"### out path: {args.out_path}")
        sanity_check(args.data_path, args.out_path)
        print("### End")
        exit()
    else:
        main(args.data_path, args.out_path, args.tile_size)
