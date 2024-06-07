import torch
from PIL import ImageFilter
import random

from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

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

class RandomRegion(torch.nn.Module):

    def __init__(self, num=2):
        self.num = num
        self.regions = list(range(num**2))

    def __call__(self, img):
        idx = random.choice(self.regions)

        w, h = img.size
        base = w // self.num
        i_h = idx // self.num
        i_w = idx % self.num
        y = i_h * base
        x = i_w * base
        py0,py1 = max(0,y), min(y+base, h)
        px0,px1 = max(0,x), min(x+base, w)
        img_region = img.crop((px0, py0, px1, py1))

        return img_region

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def RandomShuffledGrid(x, n=3, **kwargs):
    h, w, _ = x.shape
    size = h // n
    imgs = []
    for i in range(n**2):
        hi = i // n
        wi = i % n
        h1 = hi * size
        h2 = (hi+1) * size
        w1 = wi * size
        w2 = (wi+1) * size
        imgs.append(x[h1:h2, w1:w2])
    return np.stack(random.sample(imgs, k=len(imgs)))
    # return np.stack(imgs)

def BatchRandomCrop(x, **kwargs):
    transform = A.RandomCrop(64, 64)
    return np.stack([transform(image=i)["image"] for i in x])

def BatchToTensorV2(x, **kwargs):
    transform = ToTensorV2()
    return torch.stack([transform(image=i)["image"] for i in x])

def BatchResize(x, **kwargs):
    transform = A.Resize(768, 768)
    return np.stack([transform(image=i)["image"] for i in x])