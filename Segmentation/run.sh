

python generate_tumor.py \
    --data_root /home/yyang303/project/DiffTumor/STEP2.DiffusionModel/Task03_Liver/ \
    --healthy_data_root /home/yyang303/project/DiffTumor/HealthyCT/HealthyCT/healthy_ct/ \
    --data_list /home/yyang303/project/DiffTumor/STEP3.SegmentationModel/cross_eval/liver_aug_data_fold/real_tumor_train_0.txt \
    --output_dir ./generated_tumors \
    --organ_type liver \
    --tumor_type all \
    --num_samples 1