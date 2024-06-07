
import random
from typing import Any

from PIL import Image, ImageFilter

import numpy as np

import torch

from torchvision import transforms as T
from torchvision.transforms import functional as F

class FourTilesTransform:
    def __init__(self, base_transform, base_level=True):
        self.base_transform = TwoCropsTransform(base_transform)
        self.base_level = base_level
        self.transform = T.Compose([T.Resize(224), T.ToTensor(), base_transform.transforms[-1]])

    def __call__(self, x):
        q, k = self.base_transform(x)
        p = torch.tensor([0])
        if not self.base_level:
            pic_arr = np.array(x) # (HxWxC)
            h = pic_arr.shape[0]
            w = pic_arr.shape[1]
            p = torch.stack([
                self.transform(Image.fromarray(pic_arr[0:h//2, 0:w//2])),
                self.transform(Image.fromarray(pic_arr[0:h//2, w//2:])),
                self.transform(Image.fromarray(pic_arr[h//2:, 0:w//2])),
                self.transform(Image.fromarray(pic_arr[h//2:, w//2:]))
            ])
        
        return [q, k, p]

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class WholeScale():
    """
        input: torch.Tensor with  […, H, W] shape
    """
    def __init__(self, scale_range) -> None:
        self.scale_levels = list(range(1, scale_range+1))

class PyramidAug():
    """
        from base level
    """
    def __init__(self, level) -> None:
        self.level = level
        self.img_sz = 2**(8+level)
        self.tile_sz = 256
        self.tile_num = self.img_sz//self.tile_sz
        self.idx_range = list(range(2**(level*2)))

    def __call__(self, pic) -> Any:
        # Input: PIL image
        if self.level == 0:
            return pic

        else:
            pic_arr = np.array(pic) # (HxWxC)

            selected_idx = {random.choice(self.idx_range) for _ in self.idx_range}
            mask_idx = np.array(list(set(self.idx_range)-selected_idx))
            # print("masked idx: ", mask_idx)

            i_h = mask_idx // self.tile_num
            i_w = mask_idx % self.tile_num
            y = i_h*self.tile_sz
            x = i_w*self.tile_sz

            for y_,x_ in zip(y,x):
                pic_arr[y_:(y_+self.tile_sz), x_:(x_+self.tile_sz)] = 0
            
            return Image.fromarray(pic_arr)

class PyramidAug2():
    """
        from one previous level
    """
    def __init__(self, level) -> None:
        self.level = level
        self.img_sz = 2**(8+level)
        self.tile_sz = 2**(7+level)
        self.tile_num = self.img_sz//self.tile_sz
        self.idx_range = [0, 1, 2, 3]

    def __call__(self, pic) -> Any:
        # Input: PIL image
        if self.level == 0:
            return pic

        else:
            pic_arr = np.array(pic) # (HxWxC)

            selected_idx = {random.choice(self.idx_range) for _ in self.idx_range}
            mask_idx = np.array(list(set(self.idx_range)-selected_idx))
            # print("masked idx: ", mask_idx)

            i_h = mask_idx // self.tile_num
            i_w = mask_idx % self.tile_num
            y = i_h*self.tile_sz
            x = i_w*self.tile_sz

            for y_,x_ in zip(y,x):
                pic_arr[y_:(y_+self.tile_sz), x_:(x_+self.tile_sz)] = 0

            return Image.fromarray(pic_arr)

class PyramidAug3():
    """
        from all previous levels
    """
    def __init__(self, level) -> None:
        self.level = level
        self.img_sz = 2**(8+level)
        self.prev_levels = list(range(level))
        self.tile_sz = [2**(8+i) for i in self.prev_levels]
        self.tile_num = [self.img_sz//s for s in self.tile_sz]
        self.idx_range =  [list(range(s**2)) for s in self.tile_num]
        assert len(self.prev_levels) == len(self.tile_sz) == len(self.idx_range)

    def __call__(self, pic) -> Any:
        # Input: PIL image
        if self.level == 0:
            return pic

        else:
            pic_arr = np.array(pic) # (HxWxC)
            pic_tmp = np.zeros_like(pic_arr)

            selected_levels = list({random.choice(self.prev_levels) for _ in self.prev_levels})
            
            # for level_ in selected_levels:
            #     selected_idx = np.array({random.choice(self.idx_range[level_]) for _ in self.idx_range[level_]})
            #     i_h = selected_idx // 2
            #     i_w = selected_idx % 2
            #     y = i_h*self.tile_sz[level_]
            #     x = i_w*self.tile_sz[level_]
            #     for y_,x_ in zip(y,x):
            #         pic_tmp[y_:(y_+self.tile_sz[level_]), x_:(x_+self.tile_sz[level_])] = pic_arr[y_:(y_+self.tile_sz[level_]), x_:(x_+self.tile_sz[level_])]
                    
            selected_idx = {self.tile_sz[level_]: np.array(list({random.choice(self.idx_range[level_]) for _ in self.idx_range[level_]})) for level_ in selected_levels}
            start_idx = [((v//(self.img_sz//k))*k, (v%(self.img_sz//k))*k) for (k,v) in selected_idx.items()] # [(y,x)]
            end_idx = [((v//(self.img_sz//k))*k+k, (v%(self.img_sz//k))*k+k) for (k,v) in selected_idx.items()]
            # print(selected_idx)

            for start, end in zip(start_idx, end_idx):
                for y0, y1, x0, x1 in zip(start[0], end[0], start[1], end[1]):
                    # if np.all(pic_tmp[y0, x0] == 0):
                    pic_tmp[y0:y1, x0:x1] = pic_arr[y0:y1, x0:x1]

            return Image.fromarray(pic_tmp)