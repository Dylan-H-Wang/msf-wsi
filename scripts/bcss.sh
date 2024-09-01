#!/bin/bash

##### best BCSS 5-fold cv
log_path="./logs/best/bcss"
folds=(0 1 2 3 4)

for f in "${folds[@]}"
do
    CUDA_VISIBLE_DEVICES=0,1 python tools/ssl_train.py \
        -a resnet18 -j 8 -b 32 --lr 1e-3 --seed 3407 --epochs 500 \
        --multiprocessing-distributed --world-size 1 --rank 0 \
        --data-name bcss --data ../data/bcss/L0_1024_s512 \
        --mean 0.6998 0.4785 0.6609 \
        --std 0.2203 0.2407 0.1983 \
        --log-dir ${log_path}/fold_${f} \
        --save-freq 50 \
        --fold ${f} \
        --amp --wandb \
        --run-group best_bcss_fold_${f} \
        --run-name ssl_bcss_fold_${f} \
        --run-tag ssl bcss fold_${f} \
        --run-notes "cross validation on bcss: ssl, fold ${f}"

    for i in {0499..0249..50}
    do
        CUDA_VISIBLE_DEVICES=0 python tools/ssl_finetune.py \
            -j 4 -b 64 --epochs 50 --lr 1e-3 --seed 3407 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "bcss" \
            --train-data ../data/bcss/L0_1024_s512 \
            --mean 0.6998 0.4785 0.6609 \
            --std 0.2203 0.2407 0.1983 \
            --log-dir ${log_path}/fold_${f}/test_${i} \
            --weights ${log_path}/fold_${f}/checkpoint_${i}.pth.tar \
            --fold ${f} \
            --amp \
            --wandb --run-group best_bcss_fold_${f} \
            --run-name ft_bcss_fold_${f} \
            --run-tag fine-tune bcss fold_${f} \
            --run-notes "cross validation on bcss: fine-tune, fold ${f}"
    done
done


######### HookNet-msf-wsi
FRAC=(1.0 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/best/bcss"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/ssl_finetune.py  \
            -j 4 -b 64 --epochs 50 --lr 1e-3 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "bcss" \
            --train-data ../data/bcss/L0_1024_s512 \
            --mean 0.6998 0.4785 0.6609 \
            --std 0.2203 0.2407 0.1983 \
            --frac ${frac} --fold ${fold} \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --weights ./logs/ablation/bs_lr/bs32_lr1e-3/checkpoint_0249.pth.tar \
            --amp --dist-url tcp://127.0.0.1:50003
    done
done