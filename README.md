# [MSF-WSI: A multi-resolution self-supervised learning framework for semantic segmentation in histopathology](https://doi.org/10.1016/j.patcog.2024.110621)

Officail repo for MSF-WSI.

## Note

Current repo provides short documentation and will provide more details when I have time... Feel free to email me for any supports

## Installation
Run the following code to install all dependencies.
```bash
conda env create -f environment.yml
```

## Steps
Check [scripts](https://github.com/Dylan-H-Wang/msf-wsi/tree/main/scripts) in `/script` to see how to run MSF-WSI pre-training for `BCSS`, `PAIP19`, and `Camelyon16` datasets.

Prepare BCSS dataset and generate `csv` file by
```bash
python -u ./tools/bcss_prepare.py -p RAW_DATA_PATH -o ../data/bcss/L0_1024_s512 -s 1024 --overlap 512
```

## Model Weights

| Dataset    | SSL pre-trained                                                                                      | Fine-tuned                                                                                         |
| ---------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| BCSS (fold 1)       | [Link](https://github.com/Dylan-H-Wang/msf-wsi/releases/download/v0.1/bcss_fold0_pretrain_model.pth) | [Link](https://github.com/Dylan-H-Wang/msf-wsi/releases/download/v0.1/bcss_fold0_ft_model.pth.tar) |
| PAIP2019 (fold 1)   | [Link](https://github.com/Dylan-H-Wang/msf-wsi/releases/download/v0.1/paip_fold0_pretrain_model.pth) | [Link](https://github.com/Dylan-H-Wang/msf-wsi/releases/download/v0.1/paip_fold0_ft_model.pth.tar) |
| Camelyon16 | [Link](https://github.com/Dylan-H-Wang/msf-wsi/releases/download/v0.1/c16_pretrain_model.tar)        |             N/A                                                                                       |
