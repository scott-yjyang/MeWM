#!/bin/bash
#SBATCH --job-name=segmentation_model

#SBATCH -N 1
#SBATCH -n 8
#SBATCH -G a100:1
##SBATCH --exclusive
#SBATCH --mem=80G
#SBATCH -p general
#SBATCH -t 1-00:00:00
#SBATCH -q public

#SBATCH -o %x_slurm_%j.out     
#SBATCH -e %xslurm_%j.err      
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=zzhou82@asu.edu

# module load mamba/latest # only for Sol

# mamba create -n difftumor python=3.9
# source activate difftumor
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install -r ../requirements.txt

healthy_datapath=/home/yyang303/project/DiffTumor/HealthyCT/HealthyCT/healthy_ct/
datapath=/mnt/realccvl15/yyang303/ucsf/DiffTumor/logs/registered/
cache_rate=1.0
batch_size=8
val_every=20
workers=16
organ=liver
fold=0

# U-Net
# backbone=unet
# logdir="runs/$organ.fold$fold.$backbone"
# datafold_dir=cross_eval/"$organ"_aug_data_fold/
# dist=$((RANDOM % 99999 + 10000))
# python -W ignore main.py --model_name $backbone --cache_rate $cache_rate --dist-url=tcp://127.0.0.1:$dist --workers $workers --max_epochs 2000 --val_every $val_every --batch_size=$batch_size --save_checkpoint --distributed --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir $datafold_dir

# nnU-Net
backbone=nnunet
logdir="runs/hcc4.fold$fold.$backbone"
datafold_dir=cross_eval/hcc_aug_data_fold/
dist=$((RANDOM % 99999 + 10000))
pretrained_dir="/home/yyang303/project/DiffTumor/STEP3.SegmentationModel/nnunet_synt_liver_tumors.pt"
python -W ignore main.py --model_name $backbone --cache_rate $cache_rate --dist-url=tcp://127.0.0.1:$dist --workers $workers --max_epochs 2000 --val_every $val_every --batch_size=$batch_size --save_checkpoint --distributed --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir $datafold_dir --resume_ckpt --pretrained_dir $pretrained_dir




# for fold in 0; do for backbone in unet nnunet swinunetr; do for organ in liver; do sbatch --error=logs/$organ.backbone.$backbone.fold$fold.out --output=logs/$organ.backbone.$backbone.fold$fold.out hg.sh backbone organ fold; done; done; done