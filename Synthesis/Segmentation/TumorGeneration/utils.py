### Tumor Generateion
import random
import cv2
import elasticdeform
import numpy as np
from scipy.ndimage import gaussian_filter
from .ldm.vq_gan_3d.model.vqgan import VQGAN
import matplotlib.pyplot as plt
import SimpleITK as sitk
from .ldm.ddpm import Unet3D, GaussianDiffusion, Tester
from hydra import initialize, compose
import torch
import yaml
from .ldm.ddpm.ddim import DDIMSampler
# Random select location for tumors.
def random_select(mask_scan, organ_type):
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0].min(), np.where(np.any(mask_scan, axis=(0, 1)))[0].max()

    flag=0
    while 1:
        if flag<=10:
            z = round(random.uniform(0.3, 0.7) * (z_end - z_start)) + z_start
        elif flag>10 and flag<=20:
            z = round(random.uniform(0.2, 0.8) * (z_end - z_start)) + z_start
        elif flag>20 and flag<=30:
            z = round(random.uniform(0.1, 0.9) * (z_end - z_start)) + z_start
        else:
            z = round(random.uniform(0.0, 1.0) * (z_end - z_start)) + z_start
        liver_mask = mask_scan[..., z]

        if organ_type == 'liver':
            kernel = np.ones((5,5), dtype=np.uint8)
            liver_mask = cv2.erode(liver_mask, kernel, iterations=1)
        if (liver_mask == 1).sum() > 0:
            break
        flag+=1

    coordinates = np.argwhere(liver_mask == 1)
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist() # get x,y
    xyz.append(z)
    potential_points = xyz

    return potential_points

def center_select(mask_scan):
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0].min(), np.where(np.any(mask_scan, axis=(0, 1)))[0].max()
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0].min(), np.where(np.any(mask_scan, axis=(1, 2)))[0].max()
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0].min(), np.where(np.any(mask_scan, axis=(0, 2)))[0].max()

    z = round(0.5 * (z_end - z_start)) + z_start
    x = round(0.5 * (x_end - x_start)) + x_start
    y = round(0.5 * (y_end - y_start)) + y_start

    xyz = [x, y, z]
    potential_points = xyz

    return potential_points

# generate the ellipsoid
def get_ellipsoid(x, y, z):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    sh = (4*x, 4*y, 4*z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2*x, 2*y, 2*z])  # center point

    # calculate the ellipsoid 
    bboxl = np.floor(com-radii).clip(0,None).astype(int)
    bboxh = (np.ceil(com+radii)+1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice,bboxl,bboxh))]
    roiaux = aux[tuple(map(slice,bboxl,bboxh))]
    logrid = *map(np.square,np.ogrid[tuple(
            map(slice,(bboxl-com)/radii,(bboxh-com-1)/radii,1j*(bboxh-bboxl)))]),
    dst = (1-sum(logrid)).clip(0,None)
    mask = dst>roiaux
    roi[mask] = 1
    np.copyto(roiaux,dst,where=mask)

    return out

def get_fixed_geo(mask_scan, tumor_type, organ_type):
    if tumor_type == 'large':
        enlarge_x, enlarge_y, enlarge_z = 280, 280, 280
    else:
        enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.int8)
    tiny_radius, small_radius, medium_radius, large_radius = 4, 8, 16, 32

    if tumor_type == 'tiny':
        num_tumor = random.randint(1,3)
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            y = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            z = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            sigma = random.uniform(0.5,1)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan, organ_type)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

    if tumor_type == 'small':
        num_tumor = random.randint(1,3)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            y = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            z = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            sigma = random.randint(1, 2)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan, organ_type)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo


    if tumor_type == 'medium':
        num_tumor = 1
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            y = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            z = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            sigma = random.randint(3, 6)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan, organ_type)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

    if tumor_type == 'large':
        num_tumor = 1
        for _ in range(num_tumor):
            # Large tumor
            
            x = random.randint(int(0.75*large_radius), int(2.0*large_radius))
            y = random.randint(int(0.75*large_radius), int(2.0*large_radius))
            z = random.randint(int(0.75*large_radius), int(2.0*large_radius))
            sigma = random.randint(5, 10)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            if organ_type == 'liver' or organ_type == 'kidney' :
                point = random_select(mask_scan, organ_type)
                new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
                x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
                y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
                z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            else:
                x_start, x_end = np.where(np.any(geo, axis=(1, 2)))[0].min(), np.where(np.any(geo, axis=(1, 2)))[0].max()
                y_start, y_end = np.where(np.any(geo, axis=(0, 2)))[0].min(), np.where(np.any(geo, axis=(0, 2)))[0].max()
                z_start, z_end = np.where(np.any(geo, axis=(0, 1)))[0].min(), np.where(np.any(geo, axis=(0, 1)))[0].max()
                geo = geo[x_start:x_end, y_start:y_end, z_start:z_end]

                point = center_select(mask_scan)

                new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
                x_low = new_point[0] - geo.shape[0]//2
                y_low = new_point[1] - geo.shape[1]//2
                z_low = new_point[2] - geo.shape[2]//2
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_low+geo.shape[0], y_low:y_low+geo.shape[1], z_low:z_low+geo.shape[2]] += geo
    
    geo_mask = geo_mask[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]

    if ((tumor_type == 'medium') or (tumor_type == 'large')) and (organ_type == 'kidney'):
        if random.random() > 0.5:
            geo_mask = (geo_mask>=1)
        else:
            geo_mask = (geo_mask * mask_scan) >=1
    else:
        geo_mask = (geo_mask * mask_scan) >=1

    return geo_mask

from hydra.core.global_hydra import GlobalHydra

def synt_model_prepare(device, cfg):
    vqgan_ckpt = f'Synthesis/Diffusion/pretrained_models/AutoencoderModel.ckpt'
    diffusion_ckpt = f'Synthesis/Diffusion/checkpoints/ddpm/liver_tumor/liver/best_model.pt'
    GlobalHydra.instance().clear()
    with initialize(config_path="diffusion_config/", version_base="1.1"):
        cfg = compose(config_name="ddpm.yaml")
    print('diffusion_ckpt', diffusion_ckpt)
    
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt)
    vqgan = vqgan.to(device)
    vqgan.eval()
    
    early_Unet3D = Unet3D(
        dim=cfg.diffusion_img_size,
        dim_mults=cfg.dim_mults,
        channels=cfg.diffusion_num_channels,
        out_dim=cfg.out_dim,
        clip_model_name="openai/clip-vit-base-patch32" 
    ).to(device)


    early_diffusion = GaussianDiffusion(
        early_Unet3D,
        vqgan_ckpt=vqgan_ckpt,
        image_size=cfg.diffusion_img_size,
        num_frames=cfg.diffusion_depth_size,
        channels=cfg.diffusion_num_channels,
        timesteps=200,
        loss_type=cfg.loss_type,
    ).to(device)
    
    early_tester = Tester(early_diffusion)
    early_tester.load(diffusion_ckpt, map_location=device)

    return vqgan, early_tester

import torch.nn.functional as F
import numpy as np


def synthesize_tumor(ct_volume, tumor_mask, vqgan, tester, text_description=None, ddim_ts=50):
    device = ct_volume.device


    # tumor_types = ['tiny', 'small','medium','large']
    # tumor_probs = np.array([0.25, 0.25,0.25,0.25])
    # total_tumor_mask = []
    # organ_mask_np = organ_mask.cpu().numpy()
    with torch.no_grad():
        # for bs in range(organ_mask_np.shape[0]):
        #     synthetic_tumor_type = np.random.choice(tumor_types, p=tumor_probs.ravel())
        #     tumor_mask = get_fixed_geo(organ_mask_np[bs,0], synthetic_tumor_type, organ_type)
        #     total_tumor_mask.append(torch.from_numpy(tumor_mask)[None,:])
        # total_tumor_mask = torch.stack(total_tumor_mask, dim=0).to(dtype=torch.float32, device=device)
        # volume = ct_volume * 2.0 - 1.0
        volume = ct_volume
        mask = tumor_mask * 2.0 - 1.0


        mask_ = 1 - tumor_mask
        masked_volume = (volume * mask_).detach()
        
        volume = volume.permute(0, 1, -1, -3, -2)
        masked_volume = masked_volume.permute(0, 1, -1, -3, -2)
        mask = mask.permute(0, 1, -1, -3, -2)

        masked_volume_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
        masked_volume_feat = ((masked_volume_feat - vqgan.codebook.embeddings.min()) /
                              (vqgan.codebook.embeddings.max() - vqgan.codebook.embeddings.min())) * 2.0 - 1.0

        cc = torch.nn.functional.interpolate(mask, size=masked_volume_feat.shape[-3:])
        cond = torch.cat((masked_volume_feat, cc), dim=1)

        tester.ema_model.eval()
        sample = tester.sample(batch_size=volume.shape[0], cond=cond, text=text_description)

        mask_01 = torch.clamp((mask + 1.0) / 2.0, min=0.0, max=1.0)
        sigma = 3
        mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy() * 1.0, sigma=[0, 0, sigma, sigma, sigma])

        # volume_ = torch.clamp((volume + 1.0) / 2.0, min=0.0, max=1.0)
        # sample_ = torch.clamp((sample + 1.0) / 2.0, min=0.0, max=1.0)
        
        # volume_ = volume_ * (600 - (-175)) + (-175)
        # sample_ = sample_ * (600 - (-175)) + (-175)
        volume_ = volume
        sample_ = sample

        if sample_.shape != volume_.shape:
            sample_ = torch.nn.functional.interpolate(sample_, size=volume_.shape[2:], mode='trilinear', align_corners=False)

        mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
        final_volume_ = volume_ * (1 - mask_01_blur) + sample_ * mask_01_blur
        # final_volume_ = torch.clamp(final_volume_, min=-100, max=600)
        # final_volume_ = final_volume_ * (600 - (-175)) + (-175)
        # final_volume_ = torch.clamp(final_volume_, max=600)
        # final_volume_ = torch.clamp(final_volume_, -175, 600)

        # final_volume_ = final_volume_.permute(0, 1, -2, -1, -3)
        # organ_tumor_mask = organ_mask + total_tumor_mask

    return final_volume_, mask_01_blur


def get_bbox_from_mask(mask):
    """获取掩码的边界框坐标"""
    indices = torch.nonzero(mask)
    if indices.shape[0] == 0:
        return None
    
    min_d = indices[:, 2].min()
    max_d = indices[:, 2].max()
    min_h = indices[:, 3].min()
    max_h = indices[:, 3].max()
    min_w = indices[:, 4].min()
    max_w = indices[:, 4].max()
    
    return min_d, max_d, min_h, max_h, min_w, max_w

def create_bbox_mask(mask_shape, bbox_coords):
    """创建边界框掩码"""
    device = bbox_coords[0].device  
    mask = torch.zeros(mask_shape, device=device)
    min_d, max_d, min_h, max_h, min_w, max_w = bbox_coords
    mask[:, :, min_d:max_d+1, min_h:max_h+1, min_w:max_w+1] = 1
    return mask
