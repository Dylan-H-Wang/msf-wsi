import os
import logging

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset


VAL_SET = [
    ['01_01_0100', '01_01_0101', '01_01_0103', '01_01_0106', '01_01_0113', '01_01_0115', '01_01_0120', '01_01_0121', '01_01_0133', '01_01_0135'],
    ['01_01_0083', '01_01_0093', '01_01_0096', '01_01_0107', '01_01_0110', '01_01_0113', '01_01_0118', '01_01_0121', '01_01_0123', '01_01_0131'],
    ['01_01_0088', '01_01_0100', '01_01_0104', '01_01_0115', '01_01_0122', '01_01_0128', '01_01_0129', '01_01_0132', '01_01_0133', '01_01_0134'],
    ['01_01_0083', '01_01_0085', '01_01_0094', '01_01_0101', '01_01_0104', '01_01_0108', '01_01_0117', '01_01_0122', '01_01_0124', '01_01_0133'],
    ['01_01_0089', '01_01_0091', '01_01_0094', '01_01_0108', '01_01_0110', '01_01_0122', '01_01_0123', '01_01_0127', '01_01_0134', '01_01_0137']
]


class PaipSegDatasetMS(Dataset):
    """
    Threshold is used to remove images with low tumour area,
    i.e., whole tumour area + viable tumour area >= threshold
    """

    def __init__(self, data_path, transforms, frac=1, threshold=0.7, fold=0) -> None:
        super().__init__()

        self.data_path = data_path
        self.csv_path = data_path + "/train_data.csv"
        self.transforms = transforms
        self.frac = frac
        self.threshold = threshold
        self.fold = fold

        self._prepare()

    def __len__(self) -> int:
        return len(self.filename_imgs)

    def __getitem__(self, index: int) -> Tensor:
        img_path = os.path.join(self.data_path, self.filename_imgs[index])
        img = np.array(Image.open(img_path))

        mask_path = os.path.join(self.data_path, self.filename_masks[index])
        mask = np.array(Image.open(mask_path))

        if self.transforms is not None:
            sample = self.transforms[0](image=img, mask=mask)
            context_img, context_mask = sample["image"], sample["mask"]

            sample = self.transforms[1](image=context_img, mask=context_mask)
            target_img, target_mask = sample["image"], sample["mask"]

            sample = self.transforms[2](image=context_img, mask=context_mask)
            context_img, context_mask = sample["image"], sample["mask"]

            sample = self.transforms[2](image=target_img, mask=target_mask)
            target_img, target_mask = sample["image"], sample["mask"]

        return (context_img, target_img), (context_mask, target_mask)

    def _prepare(self) -> None:
        logger = logging.getLogger("DSF-WSI." + __name__)
        data_df = pd.read_csv(self.csv_path)
        logger.info(f"Reading {len(data_df)} files in {self.csv_path}...")

        data_df = data_df[~data_df["filename"].isin(VAL_SET[self.fold])].reset_index(drop=True)
        logger.info(f"Using fold {self.fold} and keep {len(data_df)} train files only...")

        logger.info(f"Removing images with threshold of {self.threshold}...")
        # tumour_area = data_df["ratio_masked_2_area"] + data_df["ratio_masked_3_area"]
        tumour_area = data_df["ratio_masked_area"]
        data_df = data_df[tumour_area >= self.threshold].reset_index(drop=True)
        logger.info(f"Create train set with {len(data_df)} files...")

        self.data_df = data_df.sample(frac=self.frac, replace=False, random_state=1).reset_index(
            drop=True
        )
        logger.info(f"Use {self.frac} percent of data to train: {len(self.data_df)}!")

        self.filename_imgs = self.data_df["filename_img"].tolist()
        self.filename_masks = self.data_df["filename_mask"].tolist()


class PaipSegDatasetValMS(Dataset):
    """
    Threshold is used to remove images with low tumour area,
    i.e., whole tumour area + viable tumour area >= threshold
    return all patches from the same WSI
    """

    def __init__(self, data_path, transforms, threshold=0.7, fold=0) -> None:
        super().__init__()

        self.data_path = data_path
        self.csv_path = data_path + "/train_data.csv"
        self.transforms = transforms
        self.threshold = threshold
        self.fold = fold

        self._prepare()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tensor:
        filename = self.files[index]
        df = self.data_df[self.data_df["filename"] == filename].reset_index(drop=True)

        context_imgs = []
        context_masks = []
        target_imgs = []
        target_masks = []
        for img_name, mask_name in zip(df["filename_img"], df["filename_mask"]):
            img_path = os.path.join(self.data_path, img_name)
            img = np.array(Image.open(img_path))

            mask_path = os.path.join(self.data_path, mask_name)
            mask = np.array(Image.open(mask_path))

            if self.transforms is not None:
                sample = self.transforms[0](image=img, mask=mask)
                context_img, context_mask = sample["image"], sample["mask"]

                sample = self.transforms[1](image=img, mask=mask)
                target_img, target_mask = sample["image"], sample["mask"]

            context_imgs.append(context_img)
            context_masks.append(context_mask)
            target_imgs.append(target_img)
            target_masks.append(target_mask)

        context_imgs = torch.stack(context_imgs, axis=0)
        context_masks = torch.stack(context_masks, axis=0)
        target_imgs = torch.stack(target_imgs, axis=0)
        target_masks = torch.stack(target_masks, axis=0)
        return (context_imgs, target_imgs), (context_masks, target_masks)

    def _prepare(self) -> None:
        logger = logging.getLogger("DSF-WSI." + __name__)
        data_df = pd.read_csv(self.csv_path)
        logger.info(f"Reading {len(data_df)} files in {self.csv_path}...")

        data_df = data_df[data_df["filename"].isin(VAL_SET[self.fold])].reset_index(drop=True)
        logger.info(f"Using fold {self.fold} and keep {len(data_df)} val files only...")

        logger.info(f"Removing images with threshold of {self.threshold}...")
        # tumour_area = data_df["ratio_masked_2_area"] + data_df["ratio_masked_3_area"]
        tumour_area = data_df["ratio_masked_area"]
        data_df = data_df[tumour_area >= self.threshold].reset_index(drop=True)
        logger.info(f"Create val set with {len(data_df)} files...")

        self.data_df = data_df
        self.files = self.data_df["filename"].unique()


class PaipPretrainDataset(Dataset):
    def __init__(
        self,
        data_path,
        transforms,
        frac=1,
        return_index=False,
        threshold=0.1,
        fold=0,
    ) -> None:
        super().__init__()

        self.data_path = data_path
        self.csv_path = data_path + f"/train_data.csv"
        self.transforms = transforms
        self.frac = frac
        self.return_index = return_index
        self.threshold = threshold
        self.fold = fold

        self._prepare()

    def __len__(self) -> int:
        return len(self.filename_imgs)

    def __getitem__(self, index: int) -> Tensor:
        img_path = os.path.join(self.data_path, self.filename_imgs[index])
        img = np.array(Image.open(img_path))

        context_img = [self.transforms[0](image=img)["image"] for _ in range(2)]
        target_img = [self.transforms[1](image=img)["image"] for _ in range(2)]
        # random drop some target patches
        jigsaw_idx = [torch.randperm(16), torch.randperm(16)]  # TODO: hardcoded
        jigsaw_reverse_idx = [torch.argsort(i) for i in jigsaw_idx]

        for i, j in enumerate(target_img):
            target_grid = blockshaped(j, 256, 256)  # TODO: hardcoded
            assert target_grid.shape == (16, 256, 256, 3)  # TODO: hardcoded
            target_grid = target_grid[jigsaw_idx[i]]
            target_img[i] = torch.stack([self.transforms[2](image=k)["image"] for k in target_grid])

        if self.return_index:
            return index, context_img, target_img, jigsaw_reverse_idx
        return context_img, target_img, jigsaw_reverse_idx

    def _prepare(self) -> None:
        logger = logging.getLogger("DSF-WSI." + __name__)
        data_df = pd.read_csv(self.csv_path)
        logger.info(f"Reading {len(data_df)} files in {self.csv_path}...")

        if self.fold == -1:
            logger.info(f"Using ALL training {len(data_df)} files ...")
        else:
            data_df = data_df[~data_df["filename"].isin(VAL_SET[self.fold])].reset_index(drop=True)
            logger.info(f"Using fold {self.fold} and keep {len(data_df)} train files only...")

        logger.info(f"Removing images with threshold of {self.threshold}...")
        data_df = data_df[data_df["ratio_masked_area"] >= self.threshold].reset_index(drop=True)
        logger.info(f"Create train set with {len(data_df)} files...")

        self.data_df = data_df.sample(frac=self.frac, replace=False, random_state=1).reset_index(
            drop=True
        )
        logger.info(f"Use {self.frac} percent of data to train: {len(self.data_df)}!")

        self.filename_imgs = self.data_df["filename_img"].tolist()


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w, c = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return arr.reshape(h // nrows, nrows, -1, ncols, c).swapaxes(1, 2).reshape(-1, nrows, ncols, c)
