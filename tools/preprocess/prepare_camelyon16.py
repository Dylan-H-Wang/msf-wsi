import os
import json
import argparse

import numpy as np
from tqdm import tqdm

from monai.data import PILWriter
from monai.data.folder_layout import FolderLayout

from .wsi_dataset import MaskedPatchWSIDataset


def make_dirs(dataset_id: int, patch_size: int, out_path: str):
    dataset_name = f"Dataset{dataset_id:03d}_Camelyon16-{patch_size}"
    out_dir = f"{out_path}/{dataset_name}"

    train_dir = FolderLayout(
        output_dir=f"{out_dir}/imagesTr", extension="png", parent=True, makedirs=True
    )
    train_label_dir = FolderLayout(
        output_dir=f"{out_dir}/labelsTr", extension="png", parent=True, makedirs=True
    )
    test_dir = FolderLayout(
        output_dir=f"{out_dir}/imagesTs", extension="png", parent=True, makedirs=True
    )
    test_label_dir = FolderLayout(
        output_dir=f"{out_dir}/labelsTs", extension="png", parent=True, makedirs=True
    )

    return out_dir, train_dir, train_label_dir, test_dir, test_label_dir


def make_json(args, out_dir):
    train_ids = [f"normal_{i:03d}" for i in range(1, 141)]
    train_ids.extend([f"tumor_{i:03d}" for i in range(1, 101)])
    val_ids = [f"normal_{i:03d}" for i in range(141, 161)]
    val_ids.extend([f"tumor_{i:03d}" for i in range(101, 112)])
    test_ids = [f"test_{i:03d}" for i in range(1, 131)]
    labels = {
        "background": 0,
        "normal": 1,
        "tumor": 2,
    }

    dataset_json = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "labels": labels,
        "numTraining": 240,
        "file_ending": ".png", 
        "patch_size": args.patch_size,
        "patch_overlap": args.overlap,
        "wsi_level": args.level,
        "tissue_threshold": args.threshold,
    }

    file_path = f"{out_dir}/dataset.json"
    with open(file_path, 'w') as f:
        json.dump(dataset_json, f, sort_keys=False, indent=4)


def make_patch(data_pair_list, out_data_dir, out_label_dir, desc: str = None):
    img_writer = PILWriter(output_dtype=np.uint8, scale=None)

    for img, mask in tqdm(data_pair_list, desc=desc):
        slide_dataset = MaskedPatchWSIDataset(
            data=[{"image": img, "mask": mask}],
            patch_size=args.patch_size,
            patch_level=args.level,
            tissue_mask_level=args.tissue_mask_level,
            threshold=args.threshold,
            overlap=args.overlap,
            include_label=False,
            center_location=False,
            reader="cuCIM",
            num_workers=args.workers,
        )

        name = os.path.basename(img).split(".")[0]
        for idx, patch in enumerate(tqdm(slide_dataset, desc=f"Patching {name}", miniters=100)):
            patch_image = patch["image"]
            patch_mask = patch["mask"]
            location = patch_image.meta["location"]

            img_save_path = out_data_dir.filename(
                subject=name, idx=f"{idx:04d}", loc=f"{location[0]}-{location[1]}"
            )
            gt_save_path = out_label_dir.filename(
                subject=name, idx=f"{idx:04d}", loc=f"{location[0]}-{location[1]}"
            )

            img_writer.set_data_array(patch_image)
            img_writer.write(img_save_path)
            img_writer.set_data_array(patch_mask, channel_dim=None)
            img_writer.write(gt_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Camelyon16 dataset")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Path to the save processed dataset",
    )
    parser.add_argument(
        "-s", "--patch-size", type=int, default=1024, help="Patch size for WSI"
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=0,
        choices=range(0, 7),
        help="Region level for WSI, chose from [0, 7]",
    )
    parser.add_argument(
        "--tissue-mask-level",
        type=int,
        default=7,
        choices=range(0, 7),
        help="Tissue mask level, chose from [0, 7]",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.8,
        help="Threshold of tissue mask for patch selection",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.0, help="Overlap between patches"
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )
    parser.add_argument("--dataset-id", type=int, default=1, help="ID of the dataset")
    args = parser.parse_args()

    out_dir, out_train_dir, out_train_labels_dir, out_test_dir, out_test_label_dir = make_dirs(
        args.dataset_id, args.patch_size, args.output_path
    )

    img_path = f"{args.input_path}/images"
    mask_path = f"{args.input_path}/masks"

    normal_img_pairs = []
    tumor_img_pairs = []
    test_img_pairs = []

    for name in os.listdir(img_path):
        name = name.split(".")[0]
        if name.startswith("normal"):
            normal_img_pairs.append(
                (f"{img_path}/{name}.tif", f"{mask_path}/{name}_mask.tif")
            )
        elif name.startswith("tumor"):
            tumor_img_pairs.append(
                (f"{img_path}/{name}.tif", f"{mask_path}/{name}_mask.tif")
            )
        elif name.startswith("test"):
            test_img_pairs.append(
                (f"{img_path}/{name}.tif", f"{mask_path}/{name}_mask.tif")
            )

    make_patch(normal_img_pairs, out_train_dir, out_train_labels_dir, "Normal WSIs")
    make_patch(tumor_img_pairs, out_train_dir, out_train_labels_dir, "Tumor WSIs")
    make_patch(test_img_pairs, out_test_dir, out_test_label_dir, "Test WSIs")

    make_json(args, out_dir)