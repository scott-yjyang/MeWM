import os
import torch
import numpy as np
import nibabel as nib
from TumorGeneration.utils import (
    synthesize_early_tumor, 
    synthesize_medium_tumor, 
    synthesize_large_tumor, 
    synt_model_prepare
)
from monai import transforms, data
from monai.transforms import (
    AsDiscreted,
    Compose,
    Invertd,
)
import argparse

def save_nifti(data, filename, affine=None):
    """保存数据为NIfTI格式"""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if len(data.shape) == 4:
        data = data[0]  # 移除batch维度
    if affine is None:
        affine = np.eye(4)
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, filename)

def get_transforms():
    """获取与验证集相同的数据预处理transform"""
    transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.AddChanneld(keys=["image", "label"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        transforms.ScaleIntensityRanged(
            keys=["image"], 
            a_min=-175, 
            a_max=250, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        transforms.SpatialPadd(
            keys=["image"],
            mode="minimum",
            spatial_size=[96, 96, 96]
        ),
        transforms.ToTensord(keys=["image", "label"])
    ])
    return transform

def get_post_transforms(transform):
    """获取后处理transform"""
    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=3),
        AsDiscreted(keys="label", to_onehot=3),
    ])
    return post_transforms

def denoise_pred(pred: np.ndarray):
    """
    # 0: background, 1: liver, 2: tumor.
    pred.shape: (3, H, W, D)
    """
    denoise_pred = np.zeros_like(pred)
    denoise_pred[1, ...] = pred[1, ...]
    denoise_pred[2, ...] = pred[1, ...] * pred[2,...]
    denoise_pred[0,...] = 1 - np.logical_or(denoise_pred[1,...], denoise_pred[2,...])
    return denoise_pred

class RandCropByPosNegLabeld_select(transforms.RandCropByPosNegLabeld):
    def __init__(self, keys, label_key, spatial_size, 
                 pos=1.0, neg=1.0, num_samples=1, 
                 image_key=None, image_threshold=0.0, allow_missing_keys=True,
                 fg_thresh=0):
        super().__init__(keys=keys, label_key=label_key, spatial_size=spatial_size, 
                 pos=pos, neg=neg, num_samples=num_samples, 
                 image_key=image_key, image_threshold=image_threshold, allow_missing_keys=allow_missing_keys)
        self.fg_thresh = fg_thresh

    def R2voxel(self,R):
        return (4/3*np.pi)*(R)**(3)

    def __call__(self, data):
        d = dict(data)
        data_name = d['name']
        d.pop('name')
        if 'kidney_label' in data_name or 'liver_label' in data_name or 'pancreas_label' in data_name:
            flag=0
            while 1:
                flag+=1
                d_crop = super().__call__(d)
                pixel_num = (d_crop[0]['label']>0).sum()
                if pixel_num > self.R2voxel(self.fg_thresh):
                    break
                if flag>5 and pixel_num > self.R2voxel(max(self.fg_thresh-5, 5)):
                    break
                if flag>10 and pixel_num > self.R2voxel(max(self.fg_thresh-10, 5)):
                    break
                if flag>15 and pixel_num > self.R2voxel(max(self.fg_thresh-15, 5)):
                    break
                if flag>20 and pixel_num > self.R2voxel(max(self.fg_thresh-20, 5)):
                    break
                if flag>25 and pixel_num > self.R2voxel(max(self.fg_thresh-25, 5)):
                    break
                if flag>30:
                    break
        else:
            d_crop = super().__call__(d)
        d_crop[0]['name'] = data_name
        return d_crop

def get_loader(args):
    """准备数据加载器"""
    transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.AddChanneld(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        transforms.ScaleIntensityRanged(
            keys=["image"], 
            a_min=-175, 
            a_max=250, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        transforms.SpatialPadd(
            keys=["image", "label"],
            mode=["minimum", "constant"],
            spatial_size=[96, 96, 96]
        ),
        transforms.CenterSpatialCropd(
            keys=["image", "label"],
            roi_size=[96, 96, 96]
        ),
        transforms.ToTensord(keys=["image", "label"])
    ])

    # 读取数据列表
    img_list = []
    lbl_list = []
    name_list = []
    
    for line in open(args.data_list):
        name = line.strip().split()[1].split('.')[0]
        
        if 'kidney_label' in name or 'liver_label' in name or 'pancreas_label' in name:
            img_list.append(args.healthy_data_root + line.strip().split()[0])
            lbl_list.append(args.healthy_data_root + line.strip().split()[1])
        else:
            img_list.append(args.data_root + line.strip().split()[0])
            lbl_list.append(args.data_root + line.strip().split()[1])
        name_list.append(name)
    
    # 创建数据字典
    data_dicts = [
        {'image': image, 'label': label, 'name': name}
        for image, label, name in zip(img_list, lbl_list, name_list)
    ]
    print('数据集大小: {}'.format(len(data_dicts)))
    
    dataset = data.Dataset(data=data_dicts, transform=transform)
    loader = data.DataLoader(
        dataset,
        batch_size=1,  # 生成时通常使用batch_size=1
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return loader, transform

def main():
    parser = argparse.ArgumentParser(description='批量生成带肿瘤的3D CT图像')
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录')
    parser.add_argument('--healthy_data_root', type=str, required=True, help='健康数据根目录')
    parser.add_argument('--data_list', type=str, required=True, help='数据列表文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--organ_type', type=str, default='liver', choices=['liver', 'kidney', 'pancreas'], help='器官类型')
    parser.add_argument('--tumor_type', type=str, default='all', choices=['early', 'medium', 'large', 'all'], help='肿瘤类型')
    parser.add_argument('--ddim_steps', type=int, default=50, help='DDIM采样步数')
    parser.add_argument('--num_samples', type=int, default=1, help='每个输入生成的样本数量')
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取数据加载器和transform
    loader, transform = get_loader(args)

    # 准备模型
    vqgan, early_sampler, noearly_sampler = synt_model_prepare(
        device=torch.device(args.device), 
        fold=0,
        organ=args.organ_type
    )

    tumor_types = ['medium', 'large'] if args.tumor_type == 'all' else [args.tumor_type]

    # 遍历数据集
    for idx, batch_data in enumerate(loader):
        try:
            print(f"Processing {idx+1}/{len(loader)}")
            
            # 获取数据
            ct_data = batch_data["image"].to(args.device)
            mask_data = batch_data["label"].to(args.device)
            name = batch_data["name"][0]
            original_affine = batch_data["image_meta_dict"]["affine"][0].numpy()

            # 检查mask是否有效
            if torch.sum(mask_data) == 0:
                print(f"Skipping {name}: empty mask")
                continue

            # 保存原始输入
            case_dir = os.path.join(args.output_dir, name)
            os.makedirs(case_dir, exist_ok=True)
            
            save_nifti(ct_data, os.path.join(case_dir, "original_ct.nii.gz"), original_affine)
            save_nifti(mask_data, os.path.join(case_dir, "original_mask.nii.gz"), original_affine)

            for tumor_type in tumor_types:
                for i in range(args.num_samples):
                    try:
                        # 生成带肿瘤的CT
                        if tumor_type == 'early':
                            synt_data, synt_target = synthesize_early_tumor(
                                ct_data, mask_data, args.organ_type, vqgan, early_sampler
                            )
                        elif tumor_type == 'medium':
                            synt_data, synt_target = synthesize_medium_tumor(
                                ct_data, mask_data, args.organ_type, vqgan, noearly_sampler, 
                                ddim_ts=args.ddim_steps
                            )
                        elif tumor_type == 'large':
                            synt_data, synt_target = synthesize_large_tumor(
                                ct_data, mask_data, args.organ_type, vqgan, noearly_sampler, 
                                ddim_ts=args.ddim_steps
                            )

                        # 直接保存生成的结果，不进行post-processing
                        output_prefix = f"{tumor_type}_sample{i+1}"
                        save_nifti(
                            synt_target[0], 
                            os.path.join(case_dir, f"{output_prefix}_segmentation.nii.gz"),
                            original_affine
                        )
                        save_nifti(
                            synt_data[0], 
                            os.path.join(case_dir, f"{output_prefix}_synthetic_ct.nii.gz"),
                            original_affine
                        )

                        print(f"已为 {name} 生成 {output_prefix}")
                            
                    except ValueError as e:
                        print(f"Error generating {tumor_type} tumor for {name}: {str(e)}")
                        continue
                    except Exception as e:
                        print(f"Unexpected error generating {tumor_type} tumor for {name}: {str(e)}")
                        continue

        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
            continue

    print("生成完成！")

if __name__ == "__main__":
    main() 