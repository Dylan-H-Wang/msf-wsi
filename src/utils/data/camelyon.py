import os
import json
import random
import logging
import itertools

import numpy as np
from PIL import Image
import torchvision.io as io

import torch
from torch import Tensor
from torch.utils.data import Dataset


class Camelyon16PretrainDataset(Dataset):
    def __init__(self, data_path, transforms, n_sample=500, mode="train", return_index=False) -> None:
        super().__init__()

        self.data_path = data_path
        self.meta_path = data_path + f"/dataset.json"
        self.transforms = transforms
        self.n_sample = n_sample
        self.mode = mode
        self.return_index = return_index

        self._prepare()
        
    def __len__(self) -> int:
        return len(self.filename_imgs)
        
    def __getitem__(self, index: int) -> Tensor:
        img_path = os.path.join(self.data_path, self.filename_imgs[index])
        img = np.asarray(Image.open(img_path))
        
        context_img = [self.transforms[0](image=img)["image"] for _ in range(2)]
        target_img = [self.transforms[1](image=img)["image"] for _ in range(2)]
        # random drop some target patches
        jigsaw_idx = [torch.randperm(16), torch.randperm(16)] # TODO: hardcoded
        jigsaw_reverse_idx = [torch.argsort(i) for i in jigsaw_idx]

        for i, j in enumerate(target_img):
            target_grid = blockshaped(j, 256, 256) # TODO: hardcoded
            assert target_grid.shape == (16, 256, 256, 3) # TODO: hardcoded
            target_grid = target_grid[jigsaw_idx[i]]
            target_img[i] = torch.stack([self.transforms[2](image=k)["image"] for k in target_grid])

        if self.return_index:
            return index, context_img, target_img, jigsaw_reverse_idx
        return context_img, target_img, jigsaw_reverse_idx

    def _prepare(self) -> None:
        logger = logging.getLogger("DSF-WSI."+__name__)
        self.all_data = {}

        with open(self.meta_path, "r") as f:
            data_meta = json.load(f)
            self.train_id = data_meta["train_ids"]
            self.val_id = data_meta["val_ids"]
            self.test_id = data_meta["test_ids"]
            self.file_ending = data_meta["file_ending"]

        if self.mode == "train":
            target_id = self.train_id
            for image_dir in os.listdir(f"{self.data_path}/imagesTr"):
                if image_dir in target_id:
                    self.all_data[image_dir] = [f"imagesTr/{image_dir}/{i}" for i in os.listdir(f"{self.data_path}/imagesTr/{image_dir}")]

        elif self.mode == "all":
            for image_dir in os.listdir(f"{self.data_path}/imagesTr"):
                self.all_data[image_dir] = [f"imagesTr/{image_dir}/{i}" for i in os.listdir(f"{self.data_path}/imagesTr/{image_dir}")]

            for image_dir in os.listdir(f"{self.data_path}/imagesTs"):
                self.all_data[image_dir] = [f"imagesTs/{image_dir}/{i}" for i in os.listdir(f"{self.data_path}/imagesTs/{image_dir}")]

        self.reset_data()
        logger.info(f"Reading {len(self.filename_imgs)}({len(self.all_data)}) files in {self.data_path} with mode {self.mode}...")

    def reset_data(self):
        self.filename_imgs = [random.sample(self.all_data[i], k=len(self.all_data[i]))[:self.n_sample] for i in self.all_data.keys()]
        self.filename_imgs = list(itertools.chain.from_iterable(self.filename_imgs))
        random.shuffle(self.filename_imgs)
        return self.filename_imgs


class Camelyon16PretrainDatasetFast(Dataset):
    def __init__(self, data_path, n_sample=1000, mode="train", return_index=False) -> None:
        super().__init__()

        self.data_path = data_path
        self.meta_path = data_path + f"/dataset.json"
        self.n_sample = n_sample
        self.mode = mode
        self.return_index = return_index

        self._prepare()
        
    def __len__(self) -> int:
        return len(self.filename_imgs)
        
    def __getitem__(self, index: int) -> Tensor:
        img_path = os.path.join(self.data_path, self.filename_imgs[index])
        img = io.read_image(img_path)
        
        if self.return_index:
            return index, img
        return img

    def _prepare(self) -> None:
        logger = logging.getLogger("DSF-WSI."+__name__)
        self.all_data = {}

        with open(self.meta_path, "r") as f:
            data_meta = json.load(f)
            self.train_id = data_meta["train_ids"]
            self.val_id = data_meta["val_ids"]
            self.test_id = data_meta["test_ids"]
            self.file_ending = data_meta["file_ending"]

        if self.mode == "train":
            target_id = self.train_id
            for image_dir in os.listdir(f"{self.data_path}/imagesTr"):
                if image_dir in target_id:
                    self.all_data[image_dir] = [f"imagesTr/{image_dir}/{i}" for i in os.listdir(f"{self.data_path}/imagesTr/{image_dir}")]

        elif self.mode == "all":
            for image_dir in os.listdir(f"{self.data_path}/imagesTr"):
                self.all_data[image_dir] = [f"imagesTr/{image_dir}/{i}" for i in os.listdir(f"{self.data_path}/imagesTr/{image_dir}")]

            for image_dir in os.listdir(f"{self.data_path}/imagesTs"):
                self.all_data[image_dir] = [f"imagesTs/{image_dir}/{i}" for i in os.listdir(f"{self.data_path}/imagesTs/{image_dir}")]

        self.reset_data()
        logger.info(f"Reading {len(self.filename_imgs)}({len(self.all_data)}) files in {self.data_path} with mode {self.mode}...")

    def reset_data(self):
        self.filename_imgs = [random.sample(self.all_data[i], k=len(self.all_data[i]))[:self.n_sample] for i in self.all_data.keys()]
        self.filename_imgs = list(itertools.chain.from_iterable(self.filename_imgs))
        random.shuffle(self.filename_imgs)
        return self.filename_imgs


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
    return (arr.reshape(h//nrows, nrows, -1, ncols, c)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols, c))