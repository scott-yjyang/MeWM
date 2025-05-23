import sys, os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

from ddpm import Unet3D, GaussianDiffusion, Tester
from dataset.dataloader import get_loader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Segmentation.TumorGeneration.utils import synt_model_prepare, synthesize_tumor


def save_nifti(img_array, save_path, fixed_spacing=(0.8, 0.8, 3.0)):

   # get original image spacing and size
   img = sitk.GetImageFromArray(img_array)
   original_spacing = [1.0, 1.0, 1.0]
   original_size = img.GetSize()

   # calculate new size after resampling
   new_size = [
       int(round(original_size[0] * (original_spacing[0] / fixed_spacing[0]))),
       int(round(original_size[1] * (original_spacing[1] / fixed_spacing[1]))),
       int(round(original_size[2] * (original_spacing[2] / fixed_spacing[2])))
   ]

   # set resampling parameters
   resampler = sitk.ResampleImageFilter()
   resampler.SetSize(new_size)
   resampler.SetOutputSpacing(fixed_spacing)
#    resampler.SetOutputDirection(image.GetDirection())
#    resampler.SetOutputOrigin(image.GetOrigin())
   resampler.SetTransform(sitk.Transform())
   resampler.SetInterpolator(sitk.sitkLinear)  # linear interpolation

   # execute resampling
   resampled_image = resampler.Execute(img)
   sitk.WriteImage(resampled_image, save_path)


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    # set device
    device = torch.device(f"cuda:{cfg.model.gpus}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(cfg.model.gpus)
    
    # prepare save directory
    cfg.model.results_folder = 'checkpoints/ddpm/'
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, "test_results")
    os.makedirs(cfg.model.results_folder, exist_ok=True)
    
    # load model
    print("Loading model...")
    vqgan, tester = synt_model_prepare(
        device=device,
        cfg=cfg,
        # version=cfg.model.version,
        # fold=cfg.model.fold,
        # organ=cfg.model.organ
    )
    
    # prepare data loader
    print("Loading data...")
    cfg.dataset.phase = 'validation'
    val_dataloader, _, _ = get_loader(cfg.dataset)
    
    # start generating
    print("Start generating CT images...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_dataloader)):
            # get input data
            ct_volume = batch_data['image.pre'].to(device)
            tumor_mask = batch_data['label.pre'].to(device)
            post_ct_volume = batch_data['image.post'].to(device)
            post_tumor_mask = batch_data['label.post'].to(device)
            text_descriptions = batch_data['text']
            # print(text_descriptions)
            # text_descriptions = ['Hydroxycamptothecin']*10
            file_names = batch_data['name']
            tumor_mask[tumor_mask == 1] = 0
            tumor_mask[tumor_mask == 2] = 1
            post_tumor_mask[post_tumor_mask == 1] = 0
            post_tumor_mask[post_tumor_mask == 2] = 1
            # generate CT images
            generated_volumes, blur_mask = synthesize_tumor(
                ct_volume=ct_volume,
                tumor_mask=tumor_mask,
                # organ_type=cfg.model.organ,
                vqgan=vqgan,
                tester=tester,
                text_description=text_descriptions,
            )
            
            # save generated images
            # print(cfg.model.results_folder)
            post_ct_volume = torch.clamp((post_ct_volume + 1.0) / 2.0, min=0.0, max=1.0)
            ct_volume = torch.clamp((ct_volume + 1.0) / 2.0, min=0.0, max=1.0)
            post_ct_volume = post_ct_volume * (600 - (-175)) + (-175)
            ct_volume = ct_volume * (600 - (-175)) + (-175)
            # get unique case name
            unique_cases = {}
            for idx in range(len(file_names)):
                case_name = os.path.basename(file_names[idx])
                if case_name not in unique_cases:
                    unique_cases[case_name] = {
                        'gen_volumes': [],
                        'blur_mask': [],
                        'pre_volume': ct_volume[idx, 0].cpu().numpy().transpose(2, 0, 1),
                        'pre_tumor_mask': tumor_mask[idx, 0].cpu().numpy().transpose(2, 0, 1),
                        'post_ct_volume': post_ct_volume[idx, 0].cpu().numpy().transpose(2, 0, 1),
                        'post_tumor_mask': post_tumor_mask[idx, 0].cpu().numpy().transpose(2, 0, 1),
                        'text': text_descriptions[idx]
                    }
                unique_cases[case_name]['gen_volumes'].append(generated_volumes[idx, 0].cpu().numpy())
                unique_cases[case_name]['blur_mask'].append(blur_mask[idx, 0].cpu().numpy())
            # process each unique case
            for case_name, case_data in unique_cases.items():
                save_dir = os.path.join(cfg.model.results_folder, case_name)
                os.makedirs(save_dir, exist_ok=True)
                
                # save each generated CT image and corresponding blur mask
                for idx, (gen_volume, blur_mask) in enumerate(zip(case_data['gen_volumes'], case_data['blur_mask'])):
                    print(gen_volume.shape)
                    save_nifti(gen_volume, os.path.join(save_dir, f'generated_ct_{idx+1}.nii.gz'))
                    save_nifti(blur_mask, os.path.join(save_dir, f'blur_mask_{idx+1}.nii.gz'))
                
                # save original CT and tumor mask
                save_nifti(case_data['pre_volume'], os.path.join(save_dir, f'pre_ct.nii.gz'))
                save_nifti(case_data['pre_tumor_mask'], os.path.join(save_dir, f'pre_tumor_mask.nii.gz'))
                save_nifti(case_data['post_ct_volume'], os.path.join(save_dir, f'post_ct.nii.gz'))
                save_nifti(case_data['post_tumor_mask'], os.path.join(save_dir, f'post_tumor_mask.nii.gz'))
                
                # save text description
                with open(os.path.join(save_dir, 'description.txt'), 'w') as f:
                    f.write(case_data['text'])
                
                print(f"Save {case_name} generated results")

if __name__ == '__main__':
    run() 