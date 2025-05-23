#!/bin/bash

# python train_ddp.py \
#     --model_CT resnetMC3_18 \
#     --aggregator TransMIL_seperate \
#     --batch_size 8 \
#     --num_workers 4 \
#     --lr 0.0001 \
#     --gpu 0 \
#     --survival_type OS \
#     --seed 1234 \
#     --save_best

# export CUDA_LAUNCH_BLOCKING=1
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TMPDIR=/tmp/local_temp  # Local directory
dist=$((RANDOM % 99999 + 10000))

python -W ignore train_ddp.py \
    --dist-url tcp://127.0.0.1:$dist \
    --workers 16 \
    --batch_size 16 \
    --distributed \
    --model_CT resnetMC3_18_wMask \
    --aggregator TransMIL_seperate \
    --lr 0.0001 \
    --gpu 0,1,2,3,4,5,6,7 \
    --survival_type OS \
    --save_best

# 使用不同GPU运行的命令示例
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=23458 \
#     train_ddp.py \
#     --multiprocessing_distributed \
#     --world_size 1 \
#     --modality CT \
#     --model_CT resnetMC3_18 \
#     --aggregator TransMIL_seperate \
#     --batch_size 8 \
#     --num_workers 16 \
#     --lr 0.0001 \
#     --gpu 4,5,6,7 \
#     --dist_url tcp://localhost:23458 \
#     --dist-backend nccl \
#     --survival_type OS \
#     --seed 1234 \
#     --save_best

# 使用预训练模型的命令示例
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=23459 \
#     train_ddp.py \
#     --multiprocessing_distributed \
#     --world_size 1 \
#     --modality CT \
#     --model_CT resnetMC3_18 \
#     --aggregator TransMIL_seperate \
#     --batch_size 8 \
#     --num_workers 16 \
#     --lr 0.0001 \
#     --gpu 0,1,2,3 \
#     --dist_url tcp://localhost:23459 \
#     --dist-backend nccl \
#     --survival_type OS \
#     --seed 1234 \
#     --save_best \
#     --resume /path/to/checkpoint.pth.tar