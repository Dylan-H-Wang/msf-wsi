import os
from collections import OrderedDict

import torch
from openslide import open_slide

from simsiam.unet import MSUnet


arch = "resnet18"
encoder_weights = "imagenet"
n_classes = 6
pretrain_weights = None

model = MSUnet(encoder_name=arch, encoder_weights=encoder_weights, classes=n_classes)

if pretrain_weights is not None:
    print(
        f"=> loading SLF-WSI pretrained weights {pretrain_weights} into encoder"
    )
    checkpoint = torch.load(pretrain_weights, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    context_state_dict = OrderedDict()
    target_state_dict = OrderedDict()

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith("module.context_encoder") and not k.startswith(
            "module.context_encoder.fc"
        ):
            # remove prefix
            context_state_dict[
                k[len("module.context_encoder.") :]
            ] = state_dict[k]

        elif k.startswith("module.target_encoder") and not k.startswith(
            "module.target_encoder.fc"
        ):
            target_state_dict[
                k[len("module.target_encoder.") :]
            ] = state_dict[k]

        # delete renamed or unused k
        del state_dict[k]
    model.context_branch.encoder.load_state_dict(context_state_dict)
    model.target_branch.encoder.load_state_dict(target_state_dict)

img_path = "/mnt/Hao/projs/data/BCSS/L0/images/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png"
slide = open_slide(img_path)
print(f"dimension: {slide.dimensions}")
slide.read_region((0,0), 0, 1024)
w, h = slide.dimensions

tile_w, tile_h = 256, 256
n_w = w // tile_w
n_h = h // tile_h

for i in range(n_w):
    for j in range(n_h):
        x = i * tile_w
        y = j * tile_h
        print(f"tile: {x}, {y}")
        tile = slide.read_region((x, y), 0, (tile_w, tile_h))

        pred = model(tile)
        