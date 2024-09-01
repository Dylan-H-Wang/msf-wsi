import re
import glob
from pathlib import Path

import torch
import torch.nn as nn


# Copy from YOLOv5
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def cal_mean_and_std(dataset):
    """
    input image shape is (batch, channel, height, width)
    the range of mean and std is (0, 1)

    Examples:
        data_path = "../data/camelyon16_pyramid/train/L0"
        dataset = CamelyonTrainDataset(data_path, transforms.ToTensor())
        cal_mean_and_std(dataset)
    """
    from tqdm import tqdm
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=1,
        shuffle=False
    )

    img,_ = dataset[0]
    height, width = img.shape[1:]
    n_pixels = len(dataset) * height * width
    sum = torch.tensor([0.0, 0.0, 0.0])
    sum_sq = torch.tensor([0.0, 0.0, 0.0])

    for images, _ in tqdm(loader):
        sum += images.sum(axis=[0, 2, 3])
        sum_sq += (images ** 2).sum(axis=[0, 2, 3])

    # mean and std
    total_mean = sum / n_pixels
    total_var = (sum_sq / n_pixels) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print(f"Dataset MEAN is: {total_mean}")
    print(f"Dataset STD is: {total_std}")


class Normalize(nn.Module):
    # input is uint8, mean and std is float
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).float().reshape(1, len(mean), 1, 1).contiguous())
        self.register_buffer("std", torch.tensor(std).float().reshape(1, len(std), 1, 1).reciprocal().contiguous())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input.type_as(self.mean)/255.0 - self.mean) * self.std
        
    def denormalize(self, input: torch.Tensor) -> torch.Tensor:
        return ((input / self.std + self.mean) * 255.0).to(torch.uint8)


def totensor(x):
    return torch.from_numpy(x).permute(2,0,1).contiguous()