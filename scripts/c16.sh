#!/bin/bash


LOG_PATH="./logs/camelyon/train_set"
CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/ssl_train.py \
    -a resnet18 -j 4 -b 32 --epochs 300 --lr 1e-3 \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --data-name camelyon16 --data ./data/Dataset001_Camelyon16-1024 \
    --mean 0.5783 0.3970 0.6128 \
    --std 0.2424 0.2379 0.1918 \
    --log-dir ${LOG_PATH} \
    --save-freq 50 \
    --amp --bf16 --tf32 --tensorboard


FRAC=(0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/camelyon/paip"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/ssl_finetune.py  \
            -j 4 -b 64 --epochs 50 --lr 1e-3 --seed 3407 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "paip" \
            --train-data ./data/paip19/train \
            --mean 0.76410981 0.55224932 0.69604445 \
            --std 0.14612035 0.1648203 0.12789637 \
            --frac ${frac} --fold ${fold} \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --weights ./logs/camelyon/train_set/checkpoint_0049.pth.tar \
            --amp --bf16 --tensorboard --dist-url tcp://127.0.0.1:50002
    done
done

FRAC=(1.0 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/camelyon/bcss"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/ssl_finetune.py  \
            -j 4 -b 64 --epochs 50 --lr 1e-3 --seed 3407 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "bcss" \
            --train-data ../data/bcss/L0_1024_s512 \
            --mean 0.6998 0.4785 0.6609 \
            --std 0.2203 0.2407 0.1983 \
            --frac ${frac} --fold ${fold} \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --weights ./logs/camelyon/train_set/checkpoint_0049.pth.tar \
            --amp --tensorboard --dist-url tcp://127.0.0.1:50002
    done
done