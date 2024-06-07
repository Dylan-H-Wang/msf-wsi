#!/bin/bash

################################ best case
log_path="./logs/paip/tf32_bf16"
CUDA_VISIBLE_DEVICES=0,1 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/ssl_train.py \
  -a resnet18 -j 8 -b 32 --epochs 300 --lr 1e-3 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --data-name paip --data ./data/paip19/train \
  --mean 0.76410981 0.55224932 0.69604445 \
  --std 0.14612035 0.1648203  0.12789637 \
  --log-dir ${log_path} \
  --save-freq 50 \
  --tf32 --amp --bf16 --tensorboard

log_path="./logs/paip/tf32_bf16"
for i in {0299..0149..50}
do
    CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/ssl_ft_paip.py  \
        -j 4 -b 64 --epochs 50 --lr 1e-3 \
        --multiprocessing-distributed --world-size 1 --rank 0 \
        --train-data ../data/paip19/train \
        --val-data ../data/paip19/train \
        --mean 0.76410981 0.55224932 0.69604445 \
        --std 0.14612035 0.1648203  0.12789637 \
        --log-dir ${log_path}/test_${i} \
        --weights ${log_path}/checkpoint_${i}.pth.tar \
        --amp --tensorboard
done


################################ cross validation on PAIP19
log_path="./logs/best/paip"
f=1

CUDA_VISIBLE_DEVICES=0,1 python ms_pretrain7.py \
    -a resnet18 -j 8 -b 32 --lr 1e-3 --seed 3407 --epochs 300 \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --data-name paip --data ../data/paip19/train \
    --mean 0.76410981 0.55224932 0.69604445 \
    --std 0.14612035 0.1648203  0.12789637 \
    --log-dir ${log_path}/fold_${f} \
    --save-freq 50 \
    --fold ${f} \
    --amp --wandb \
    --run-group best_paip_fold_${f} \
    --run-name ssl_paip_fold_${f} \
    --run-tag ssl paip fold_${f} \
    --run-notes "cross validation on paip: ssl, fold ${f}"

for i in {0299..0149..50}
do
    CUDA_VISIBLE_DEVICES=0 python paip_seg_ms.py \
        -j 6 -b 64 --epochs 50 --lr 1e-3 --seed 3407 \
        --multiprocessing-distributed --world-size 1 --rank 0 \
        --train-data ../data/paip19/train \
        --val-data ../data/paip19/train \
        --mean 0.76410981 0.55224932 0.69604445 \
        --std 0.14612035 0.1648203  0.12789637 \
        --log-dir ${log_path}/fold_${f}_r/test_${i} \
        --weights ${log_path}/fold_${f}_r/checkpoint_${i}.pth.tar \
        --fold ${f} \
        --amp --use_ms \
        --wandb --run-group best_paip_fold_${f} \
        --run-name ft_paip_fold_${f} \
        --run-tag fine-tune paip fold_${f} epoch_${i} \
        --run-notes "cross validation on paip: fine-tune, fold ${f}, epoch ${i}"
done

### resume training
# log_path="./logs/best/paip"
# f=4

# CUDA_VISIBLE_DEVICES=0,1 python ms_pretrain7.py \
#     -a resnet18 -j 8 -b 32 --lr 1e-3 --seed 3407 --epochs 300 \
#     --multiprocessing-distributed --world-size 1 --rank 0 \
#     --data-name paip --data ../data/paip19/train \
#     --mean 0.76410981 0.55224932 0.69604445 \
#     --std 0.14612035 0.1648203  0.12789637 \
#     --log-dir ${log_path}/fold_${f}_r300 \
#     --resume /home/dylan/projs/SLF-WSI/logs/best/paip/fold_4_r/checkpoint_0189.pth.tar \
#     --save-freq 10 \
#     --fold ${f} \
#     --amp --wandb \
#     --run-group best_paip_fold_${f} \
#     --run-name ssl_paip_fold_${f}_r300 \
#     --run-tag ssl paip fold_${f}_r300 \
#     --run-notes "cross validation on paip: ssl, fold ${f}"


######### HookNet-dsf-wsi
# FRAC=(1.0 0.5 0.1 0.01)
# FOLDS=(0 1 2 3 4)
# log_path="./logs/best/paip/eval"
# for frac in ${FRAC[@]}
# do
#     for fold in ${FOLDS[@]}
#     do
#         CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate.py  \
#             -j 4 -b 64 --seed 3407 \
#             --multiprocessing-distributed --world-size 1 --rank 0 \
#             --data-name "paip" \
#             --train-data ./data/paip19/train \
#             --mean 0.76410981 0.55224932 0.69604445 \
#             --std 0.14612035 0.1648203 0.12789637 \
#             --frac ${frac} --fold ${fold} \
#             --log-dir ${log_path}/frac_${frac}/fold_${fold} \
#             --weights ./logs/best/paip/frac_${frac}/fold_${fold}/best_ft_model.pth.tar \
#             --amp --tensorboard --dist-url tcp://127.0.0.1:50003
#     done
# done

######### msY-Net-dsf-wsi
FRAC=(1.0 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/best/msynet/paip"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/finetune_msy.py  \
            -j 4 -b 16 --epochs 50 --lr 0.00025 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "paip" \
            --train-data ./data/paip19/train \
            --mean 0.76410981 0.55224932 0.69604445 \
            --std 0.14612035 0.1648203 0.12789637 \
            --frac ${frac} --fold ${fold} \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --weights ./logs/best/bcss/frac_1.0/fold_${fold}/best_pretrain_model.pth \
            --amp --tensorboard --dist-url tcp://127.0.0.1:50002
    done
done