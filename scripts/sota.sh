######### UNet
FRAC=(1.0 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/unet/bcss/random/eval"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate_unet.py  \
            -j 4 -b 64 --seed 3407 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "bcss" \
            --train-data ../data/bcss/L0_1024_s512 \
            --mean 0.6998 0.4785 0.6609 \
            --std 0.2203 0.2407 0.1983 \
            --frac ${frac} --fold ${fold} \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --weights ./logs/unet/bcss/random/frac_${frac}/fold_${fold}/best_ft_model.pth.tar \
            --amp --bf16 --dist-url tcp://127.0.0.1:50003
    done
done


FRAC=(1.0 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/unet/paip/random/eval"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate_unet.py  \
            -j 4 -b 64 --seed 3407 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "paip" \
            --train-data ./data/paip19/train \
            --mean 0.76410981 0.55224932 0.69604445 \
            --std 0.14612035 0.1648203 0.12789637 \
            --frac ${frac} --fold ${fold} \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --weights ./logs/unet/paip/random/frac_${frac}/fold_${fold}/best_ft_model.pth.tar \
            --amp --bf16 --dist-url tcp://127.0.0.1:50003
    done
done



######### HookNet-random
FRAC=(1 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/hooknet/bcss/random"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/finetune_sota.py  \
            -j 4 -b 64 --epochs 50 --lr 1e-3 --seed 3407 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "bcss" \
            --train-data ../data/bcss/L0_1024_s512 \
            --mean 0.6998 0.4785 0.6609 \
            --std 0.2203 0.2407 0.1983 \
            --frac ${frac} --fold ${fold} \
            --weights ./none \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --amp --dist-url tcp://127.0.0.1:50003
    done
done

# FRAC=(1.0 0.5 0.1 0.01)
# FOLDS=(0 1 2 3 4)
# log_path="./logs/hooknet/bcss/random/eval"
# for frac in ${FRAC[@]}
# do
#     for fold in ${FOLDS[@]}
#     do
#         CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate.py  \
#             -j 4 -b 64 --seed 3407 \
#             --multiprocessing-distributed --world-size 1 --rank 0 \
#             --data-name "bcss" \
#             --train-data ../data/bcss/L0_1024_s512 \
#             --mean 0.6998 0.4785 0.6609 \
#             --std 0.2203 0.2407 0.1983 \
#             --frac ${frac} --fold ${fold} \
#             --log-dir ${log_path}/frac_${frac}/fold_${fold} \
#             --weights ./logs/hooknet/bcss/random/frac_${frac}/fold_${fold}/best_ft_model.pth.tar \
#             --amp --bf16 --dist-url tcp://127.0.0.1:50003
#     done
# done

FRAC=(1 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/hooknet/paip/random"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/finetune_sota.py  \
            -j 4 -b 64 --epochs 50 --lr 1e-3 --seed 3407 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "paip" \
            --train-data ./data/paip19/train \
            --mean 0.76410981 0.55224932 0.69604445 \
            --std 0.14612035 0.1648203 0.12789637 \
            --frac ${frac} --fold ${fold} \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --weights ./none \
            --amp --bf16 --tensorboard --dist-url tcp://127.0.0.1:50003
    done
done




######### HookNet-SimSiam
# FRAC=(1.0 0.5 0.1 0.01)
# FOLDS=(0 1 2 3 4)
# log_path="./logs/hooknet/bcss/simsiam/eval"
# for frac in ${FRAC[@]}
# do
#     for fold in ${FOLDS[@]}
#     do
#         CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate.py  \
#             -j 4 -b 64 --seed 3407 \
#             --multiprocessing-distributed --world-size 1 --rank 0 \
#             --data-name "bcss" \
#             --train-data ../data/bcss/L0_1024_s512 \
#             --mean 0.6998 0.4785 0.6609 \
#             --std 0.2203 0.2407 0.1983 \
#             --frac ${frac} --fold ${fold} \
#             --log-dir ${log_path}/frac_${frac}/fold_${fold} \
#             --weights ./logs/hooknet/bcss/simsiam/frac_${frac}/fold_${fold}/best_ft_model.pth.tar \
#             --amp --bf16 --dist-url tcp://127.0.0.1:50003
#     done
# done


# FRAC=(1.0 0.5 0.1 0.01)
# FOLDS=(0 1 2 3 4)
# log_path="./logs/hooknet/paip/simsiam/eval"
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
#             --weights ./logs/hooknet/paip/simsiam/frac_${frac}/fold_${fold}/best_ft_model.pth.tar \
#             --amp --bf16 --dist-url tcp://127.0.0.1:50003
#     done
# done




######### HookNet-Slf-Hist
FRAC=(1.0 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/hooknet/bcss/slf_hist"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/finetune_sota.py  \
            -j 4 -b 64 --epochs 50 --lr 1e-3 --seed 3407 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "bcss" \
            --train-data ../data/bcss/L0_1024_s512 \
            --mean 0.6998 0.4785 0.6609 \
            --std 0.2203 0.2407 0.1983 \
            --frac ${frac} --fold ${fold} \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --weight-name slf-hist \
            --weights ./logs/hooknet/bcss/slf_hist/frac_${frac}/fold_${fold}/best_ft_model.pth.tar \
            --amp --dist-url tcp://127.0.0.1:50003
    done
done

# FRAC=(1.0 0.5 0.1 0.01)
# FOLDS=(0 1 2 3 4)
# log_path="./logs/hooknet/bcss/slf_hist/eval"
# for frac in ${FRAC[@]}
# do
#     for fold in ${FOLDS[@]}
#     do
#         CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate.py  \
#             -j 4 -b 64 --seed 3407 \
#             --multiprocessing-distributed --world-size 1 --rank 0 \
#             --data-name "bcss" \
#             --train-data ../data/bcss/L0_1024_s512 \
#             --mean 0.6998 0.4785 0.6609 \
#             --std 0.2203 0.2407 0.1983 \
#             --frac ${frac} --fold ${fold} \
#             --log-dir ${log_path}/frac_${frac}/fold_${fold} \
#             --weights ./logs/hooknet/bcss/slf_hist/frac_${frac}/fold_${fold}/best_ft_model.pth.tar \
#             --amp --bf16 --dist-url tcp://127.0.0.1:50003
#     done
# done


# FRAC=(1.0 0.5 0.1 0.01)
# FOLDS=(0 1 2 3 4)
# log_path="./logs/hooknet/paip/slf_hist/eval"
# for frac in ${FRAC[@]}
# do
#     for fold in ${FOLDS[@]}
#     do
#         CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate.py  \
#             -j 4 -b 64 --seed 3407 \
#             --multiprocessing-distributed --world-size 1 --rank 0 \
#             --data-name "paip" \
#             --train-data ./data/pai1p19/train \
#             --mean 0.76410981 0.55224932 0.69604445 \
#             --std 0.14612035 0.1648203 0.12789637 \
#             --frac ${frac} --fold ${fold} \
#             --log-dir ${log_path}/frac_${frac}/fold_${fold} \
#             --weights ./logs/hooknet/paip/slf_hist/frac_${frac}/fold_${fold}/best_ft_model.pth.tar \
#             --amp --bf16 --dist-url tcp://127.0.0.1:50003
#     done
# done




######### msY-Net-random
# need to re-traind due to loss of weights
FRAC=(1 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/msynet/paip2"
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
            --amp --bf16 --tensorboard --dist-url tcp://127.0.0.1:50003
    done
done

# FRAC=(1.0 0.5 0.1 0.01)
# FOLDS=(0 1 2 3 4)
# log_path="./logs/msynet/bcss/eval"
# for frac in ${FRAC[@]}
# do
#     for fold in ${FOLDS[@]}
#     do
#         CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate_msy.py  \
#             -j 4 -b 64 --seed 3407 \
#             --multiprocessing-distributed --world-size 1 --rank 0 \
#             --data-name "bcss" \
#             --train-data ../data/bcss/L0_1024_s512 \
#             --mean 0.6998 0.4785 0.6609 \
#             --std 0.2203 0.2407 0.1983 \
#             --frac ${frac} --fold ${fold} \
#             --log-dir ${log_path}/frac_${frac}/fold_${fold} \
#             --weights ./logs/msynet/bcss/frac_${frac}/fold_${fold}/best_ft_model.pth.tar \
#             --amp --bf16 --dist-url tcp://127.0.0.1:50003
#     done
# done




######### CTransPath
FRAC=(1.0 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/sotas/ctranspath/bcss"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/finetune_ctranspath.py  \
            -j 4 -b 32 --epochs 50 --lr 1e-4 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "bcss" \
            --train-data ../data/bcss/L0_1024_s512 \
            --mean 0.6998 0.4785 0.6609 \
            --std 0.2203 0.2407 0.1983 \
            --frac ${frac} --fold ${fold} \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --weights ./logs/sotas/ctranspath/ctranspath.pth \
            --amp --bf16 --tensorboard --dist-url tcp://127.0.0.1:50003
    done
done


FOLDS=(0 1 2 3 4)
log_path="./logs/sotas/ctranspath/bcss/eval"
for fold in ${FOLDS[@]}
do
    CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate_ctranspath.py  \
        -j 4 -b 64 --seed 3407 \
        --multiprocessing-distributed --world-size 1 --rank 0 \
        --data-name "bcss" \
        --train-data ../data/bcss/L0_1024_s512 \
        --mean 0.6998 0.4785 0.6609 \
        --std 0.2203 0.2407 0.1983 \
        --fold ${fold} \
        --log-dir ${log_path}/fold_${fold} \
        --weights ./logs/sotas/ctranspath/bcss/fold_${fold}/best_ft_model.pth.tar \
        --amp --bf16 --tensorboard --dist-url tcp://127.0.0.1:50003
done


FOLDS=(0 1 2 3 4)
log_path="./logs/sotas/ctranspath/paip/eval"
for fold in ${FOLDS[@]}
do
    CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate_ctranspath.py  \
        -j 4 -b 64 --seed 3407 \
        --multiprocessing-distributed --world-size 1 --rank 0 \
        --data-name "paip" \
        --train-data ./data/paip19/train \
        --mean 0.76410981 0.55224932 0.69604445 \
        --std 0.14612035 0.1648203 0.12789637 \
        --fold ${fold} \
        --log-dir ${log_path}/fold_${fold} \
        --weights ./logs/sotas/ctranspath/paip/fold_${fold}/best_ft_model.pth.tar \
        --amp --bf16 --tensorboard --dist-url tcp://127.0.0.1:50003
done


######### DSMIL
FOLDS=(0 1 2 3 4)
log_path="./logs/sotas/dsmil/bcss/eval"
for fold in ${FOLDS[@]}
do
    CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate.py  \
        -j 4 -b 64 --seed 3407 \
        --multiprocessing-distributed --world-size 1 --rank 0 \
        --data-name "bcss" \
        --train-data ../data/bcss/L0_1024_s512 \
        --mean 0.6998 0.4785 0.6609 \
        --std 0.2203 0.2407 0.1983 \
        --fold ${fold} \
        --log-dir ${log_path}/fold_${fold} \
        --weights ./logs/sotas/dsmil/bcss/fold_${fold}/best_ft_model.pth.tar \
        --amp --bf16 --tensorboard --dist-url tcp://127.0.0.1:50003
done


FOLDS=(0 1 2 3 4)
log_path="./logs/sotas/dsmil/paip/eval"
for fold in ${FOLDS[@]}
do
    CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate.py  \
        -j 4 -b 64 --seed 3407 \
        --multiprocessing-distributed --world-size 1 --rank 0 \
        --data-name "paip" \
        --train-data ./data/paip19/train \
        --mean 0.76410981 0.55224932 0.69604445 \
        --std 0.14612035 0.1648203 0.12789637 \
        --fold ${fold} \
        --log-dir ${log_path}/fold_${fold} \
        --weights ./logs/sotas/dsmil/paip/fold_${fold}/best_ft_model.pth.tar \
        --amp --bf16 --tensorboard --dist-url tcp://127.0.0.1:50003
done



######### rsp_cr
FOLDS=(0 1 2 3 4)
log_path="./logs/sotas/rsp_cr/bcss/eval"
for fold in ${FOLDS[@]}
do
    CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate.py  \
        -j 4 -b 64 --seed 3407 \
        --multiprocessing-distributed --world-size 1 --rank 0 \
        --data-name "bcss" \
        --train-data ../data/bcss/L0_1024_s512 \
        --mean 0.6998 0.4785 0.6609 \
        --std 0.2203 0.2407 0.1983 \
        --fold ${fold} \
        --log-dir ${log_path}/fold_${fold} \
        --weights ./logs/sotas/rsp_cr/bcss/fold_${fold}/best_ft_model.pth.tar \
        --amp --bf16 --tensorboard --dist-url tcp://127.0.0.1:50003
done


FOLDS=(0 1 2 3 4)
log_path="./logs/sotas/rsp_cr/paip/eval"
for fold in ${FOLDS[@]}
do
    CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/evaluate.py  \
        -j 4 -b 64 --seed 3407 \
        --multiprocessing-distributed --world-size 1 --rank 0 \
        --data-name "paip" \
        --train-data ./data/paip19/train \
        --mean 0.76410981 0.55224932 0.69604445 \
        --std 0.14612035 0.1648203 0.12789637 \
        --fold ${fold} \
        --log-dir ${log_path}/fold_${fold} \
        --weights ./logs/sotas/rsp_cr/paip/fold_${fold}/best_ft_model.pth.tar \
        --amp --bf16 --tensorboard --dist-url tcp://127.0.0.1:50003
done




######### Cerberus
FRAC=(1.0 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/sotas/cerberus/bcss"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/finetune_unet.py  \
            -a resnet34 -j 4 -b 64 --epochs 50 --lr 1e-3 --seed 3407 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "bcss" \
            --train-data ../data/bcss/L0_1024_s512 \
            --mean 0.6998 0.4785 0.6609 \
            --std 0.2203 0.2407 0.1983 \
            --frac ${frac} --fold ${fold} \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --weight-name cerberus \
            --weights ./logs/sotas/cerberus/resnet34_cerberus_torchvision.pth \
            --amp --bf16 --tensorboard --dist-url tcp://127.0.0.1:50003
    done
done


FRAC=(1.0 0.5 0.1 0.01)
FOLDS=(0 1 2 3 4)
log_path="./logs/sotas/cerberus/paip"
for frac in ${FRAC[@]}
do
    for fold in ${FOLDS[@]}
    do
        CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python ./tools/finetune_unet.py  \
            -a resnet34 -j 4 -b 64 --epochs 50 --lr 1e-3 --seed 3407 \
            --multiprocessing-distributed --world-size 1 --rank 0 \
            --data-name "paip" \
            --train-data ./data/paip19/train \
            --mean 0.76410981 0.55224932 0.69604445 \
            --std 0.14612035 0.1648203 0.12789637 \
            --frac ${frac} --fold ${fold} \
            --log-dir ${log_path}/frac_${frac}/fold_${fold} \
            --weight-name cerberus \
            --weights ./logs/sotas/cerberus/resnet34_cerberus_torchvision.pth \
            --amp --bf16 --tensorboard --dist-url tcp://127.0.0.1:50003
    done
done