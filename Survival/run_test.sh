#!/bin/bash

export TMPDIR=/tmp/local_temp  # Local directory
dist=$((RANDOM % 99999 + 10000))

# python -W ignore test_ddp.py \
#     --dist-url tcp://127.0.0.1:$dist \
#     --workers 16 \
#     --batch_size 16 \
#     --distributed \
#     --model_CT resnetMC3_18 \
#     --aggregator TransMIL_seperate \
#     --gpu 0,1,2,3 \
#     --test_pth '/home/yyang303/project/Survival/results/SavedModels/modality(1)/resnetMC3_18_wMask(TransMIL_seperate)/[0]2025-02-14-00:31:37/checkpoint_best.pth.tar'

python -W ignore test_ddp.py \
    --dist-url tcp://127.0.0.1:$dist \
    --workers 16 \
    --batch_size 1 \
    --distributed \
    --model_CT resnetMC3_18_wMask \
    --aggregator TransMIL_seperate \
    --lr 0.0001 \
    --gpu 0 \
    --survival_type OS \
    --resume 'results/SavedModels/modality(1)/resnetMC3_18_wMask(TransMIL_seperate)/[0]2025-03-27-00:46:13/checkpoint_last.pth.tar'