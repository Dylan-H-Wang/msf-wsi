#!/bin/bash

################################ cross validation on PAIP19
log_path="./logs/best/paip"
folds=(0 1 2 3 4)

for f in "${folds[@]}"
do
    CUDA_VISIBLE_DEVICES=0,1 python tools/ssl_train.py \
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

    for i in {0499..0249..50}
    do
        CUDA_VISIBLE_DEVICES=0 python tools/ssl_finetune.py \
            -j 4 -b 64 --epochs 50 --lr 1e-3 --seed 3407 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "paip" \
            --train-data ../data/paip19/train \
            --mean 0.76410981 0.55224932 0.69604445 \
            --std 0.14612035 0.1648203  0.12789637 \
            --log-dir ${log_path}/fold_${f}_r/test_${i} \
            --weights ${log_path}/fold_${f}_r/checkpoint_${i}.pth.tar \
            --fold ${f} \
            --amp \
            --wandb --run-group best_paip_fold_${f} \
            --run-name ft_paip_fold_${f} \
            --run-tag fine-tune paip fold_${f} epoch_${i} \
            --run-notes "cross validation on paip: fine-tune, fold ${f}, epoch ${i}"
    done
done


######### HookNet-msf-wsi
FRAC=(1.0 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/best/paip/eval"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate.py  \
            -j 4 -b 64 --seed 3407 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "paip" \
            --train-data ./data/paip19/train \
            --mean 0.76410981 0.55224932 0.69604445 \
            --std 0.14612035 0.1648203 0.12789637 \
            --frac ${frac} --fold ${fold} \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --weights ./logs/best/paip/frac_${frac}/fold_${fold}/best_ft_model.pth.tar \
            --amp --tensorboard --dist-url tcp://127.0.0.1:50003
    done
done