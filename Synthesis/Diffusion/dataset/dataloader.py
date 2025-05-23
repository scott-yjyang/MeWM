from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py, os

from monai.transforms import Lambda

import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..") 

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey
    
    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img['name'])
            self.data_dic[key].append(img)
        
        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)
    
    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        
        # 创建一个随机状态
        random_state = np.random.RandomState(np.random.randint(2147483647))
        
        # 分别对治疗前后的数据应用相同的随机变换
        pre_data = {'image': data_i['pre']['image'], 'label': data_i['pre']['label']}
        post_data = {'image': data_i['post']['image'], 'label': data_i['post']['label']}
        
        # 对于每个transform,使用相同的随机状态
        if self.transform is not None:
            for t in self.transform.transforms:
                # 如果是随机变换,设置相同的随机种子
                if hasattr(t, 'rand_param'):
                    t.rand_param = random_state
                if hasattr(t, '_do_transform'):
                    t._do_transform = random_state.random() < t.prob
        
            pre_transformed = apply_transform(self.transform, pre_data)
            post_transformed = apply_transform(self.transform, post_data)
        else:
            pre_transformed = pre_data
            post_transformed = post_data
        
        return {
            'pre': pre_transformed,
            'post': post_transformed,
            'name': data_i['name'],
            'text': data_i['text']
        }
    
    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        # data_index = int(index / self.__len__() * self.datasetnum[set_index])
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_statis()
    
    def data_statis(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0
        # breakpoint()
        for img in self.data:
            key = get_key(img['name'])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f'the dataset {key} has no data'
            self.data_num.append(item)
        
        self.datasetlen = len(self.datasetkey)
    
    def index_uniform(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        # print(post_index, self.__len__())
        return self._transform(post_index)

class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        return d

class RandZoomd_select(RandZoomd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']):
            return d
        d = super().__call__(d)
        return d


class RandCropByPosNegLabeld_select(RandCropByPosNegLabeld):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if key in ['10_03', '10_07', '10_08', '04', '05']: # if key in ['10_03', '10_07', '10_08', '04']
            return d
        d = super().__call__(d)
        return d

class CenterTumorRandCropByLabelClassesd(RandCropByLabelClassesd):
    def __init__(self, center_ratio=0.6, **kwargs):
        super().__init__(**kwargs)
        self.center_ratio = center_ratio  # 中心区域占整个裁剪区域的比例
        
    def __call__(self, data):
        d = dict(data)
        label = d[self.label_key]
        
        # 获取肿瘤区域的中心点
        tumor_indices = torch.nonzero((label == 2) | (label == 1))  # 获取肿瘤区域的所有索引
        if len(tumor_indices) == 0:
            return super().__call__(d)  # 如果没有肿瘤，使用原始的随机裁剪
            
        center = torch.mean(tumor_indices.float(), dim=0)  # 计算肿瘤区域的中心点
        
        # 计算中心区域的范围
        spatial_size = self.spatial_size
        center_size = [int(s * self.center_ratio) for s in spatial_size]
        
        # 确保裁剪区域包含肿瘤中心
        min_bound = [c - s//2 for c, s in zip(center, spatial_size)]
        max_bound = [c + s//2 for c, s in zip(center, spatial_size)]
        
        # 调整边界确保在图像范围内
        for i in range(len(min_bound)):
            if min_bound[i] < 0:
                min_bound[i] = 0
                max_bound[i] = spatial_size[i]
            if max_bound[i] > label.shape[i]:
                max_bound[i] = label.shape[i]
                min_bound[i] = max(0, label.shape[i] - spatial_size[i])
        
        # 在调整后的范围内随机选择裁剪起点
        start_coords = [np.random.randint(min_b, max_b - s + 1) 
                       for min_b, max_b, s in zip(min_bound, max_bound, spatial_size)]
        
        # 执行裁剪
        slices = [slice(s, s + sz) for s, sz in zip(start_coords, spatial_size)]
        result = {}
        for key in self.keys:
            result[key] = d[key][slices]
            
        return result

class SeparateRandCropByLabelClassesd(RandCropByLabelClassesd):
    def __init__(self, max_tumor_ratio=0.4, max_liver_ratio=0.8, **kwargs):
        super().__init__(**kwargs)
        self.max_tumor_ratio = max_tumor_ratio  # 最大肿瘤占比
        self.max_liver_ratio = max_liver_ratio  # 最大肝脏占比
        
    def check_ratios(self, label):
        # 计算肿瘤和肝脏的体素数量
        total_voxels = label.size
        tumor_voxels = np.sum((label == 2))
        liver_voxels = np.sum(label > 0)  # 包括肿瘤和肝脏
        
        # 计算比例
        tumor_ratio = tumor_voxels / total_voxels
        liver_ratio = liver_voxels / total_voxels
        
        # 如果任一比例超过阈值，返回False
        if tumor_ratio > self.max_tumor_ratio or liver_ratio > self.max_liver_ratio:
            return False
        return True
        
    def __call__(self, data):
        d = dict(data)
        
        # 继续原来的裁剪逻辑
        random_state = np.random.RandomState(np.random.randint(1000))
        
        pre_results = {'image.pre': d['image.pre'], 'label.pre': d['label.pre']}
        post_results = {'image.post': d['image.post'], 'label.post': d['label.post'], 'image.pre': d['image.pre'], 'label.pre': d['label.pre']}
        # 对 pre 数据进行裁剪
        pre_cropper = RandCropByLabelClassesd(
            keys=["image.pre", "label.pre"],
            label_key="label.pre",
            spatial_size=self.spatial_size,
            ratios=self.ratios,
            num_classes=self.num_classes,
            num_samples=self.num_samples,
            image_key="image.pre",
            image_threshold=self.image_threshold
        )
        pre_cropper._randomization = random_state
        pre_results = pre_cropper(pre_results)  # 返回列表
        
        # 对 post 数据进行裁剪
        try:
            post_cropper = RandCropByLabelClassesd(
                keys=["image.post", "label.post"],
                label_key="label.post",
                spatial_size=self.spatial_size,
                ratios=self.ratios,
                num_classes=self.num_classes,
                num_samples=self.num_samples,
                image_key="image.post",
                image_threshold=self.image_threshold
                )
            post_cropper._randomization = random_state
            post_results = post_cropper(post_results)
        except Exception as e:
            post_cropper = RandCropByLabelClassesd(
                keys=["image.post", "label.post"],
                label_key="label.pre",
                spatial_size=self.spatial_size,
                ratios=self.ratios,
                num_classes=self.num_classes,
                num_samples=self.num_samples,
                image_key="image.pre",
                image_threshold=self.image_threshold
                )
            post_cropper._randomization = random_state
            post_results = post_cropper(post_results)
        
        # 创建结果列表
        results = []
        for i in range(self.num_samples):
            result = dict(d)  # 创建新的字典，保留原始数据的其他字段
            result.update({
                "image.pre": pre_results[i]["image.pre"],
                "label.pre": pre_results[i]["label.pre"],
                "image.post": post_results[i]["image.post"],
                "label.post": post_results[i]["label.post"]
            })
            results.append(result)
        
        return results

class SeparateCenterTumorRandCropByLabelClassesd(SeparateRandCropByLabelClassesd):
    def __init__(self, center_ratio=0.6, min_bg_ratio=0.2, **kwargs):
        super().__init__(**kwargs)
        self.center_ratio = center_ratio
        self.min_bg_ratio = min_bg_ratio  # 最小的非肿瘤区域比例
        
    def get_center_crop(self, image, label, random_state, spatial_size, max_attempts=50):
        # 确保 label 是 PyTorch 张量用于计算
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        
        # 获取肿瘤区域中心点
        tumor_indices = torch.nonzero((label == 2) | (label == 1))
        if len(tumor_indices) == 0:
            return None
            
        center = torch.mean(tumor_indices.float(), dim=0)
        
        # 多次尝试找到合适的裁剪位置
        for attempt in range(max_attempts):
            # 确保裁剪区域包含肿瘤中心
            min_bound = [max(0, int(c - s//2)) for c, s in zip(center, spatial_size)]
            max_bound = [min(int(c + s//2), d) for c, s, d in zip(center, spatial_size, label.shape)]
            
            # 调整边界确保有效的裁剪范围
            for i in range(len(min_bound)):
                # 确保裁剪区域不超过图像边界
                if max_bound[i] - spatial_size[i] < 0:
                    min_bound[i] = 0
                    max_bound[i] = spatial_size[i]
                elif max_bound[i] > label.shape[i]:
                    max_bound[i] = label.shape[i]
                    min_bound[i] = max(0, label.shape[i] - spatial_size[i])
            
                # 确保有效的随机范围
                if min_bound[i] > max_bound[i] - spatial_size[i]:
                    min_bound[i] = max(0, max_bound[i] - spatial_size[i])
            
            # 使用提供的随机状态选择裁剪起点
            np.random.seed(random_state.randint(1234) + attempt)
            start_coords = []
            for min_b, max_b, s in zip(min_bound, max_bound, spatial_size):
                valid_range = max_b - s - min_b + 1
                if valid_range <= 0:
                    start = min_b
                else:
                    start = np.random.randint(min_b, min_b + valid_range)
                start_coords.append(start)
            
            # 执行裁剪
            slices = [slice(s, s + sz) for s, sz in zip(start_coords, spatial_size)]
            cropped_label = label[slices]
            
            # 计算肿瘤区域比例
            tumor_ratio = torch.sum((cropped_label == 1) | (cropped_label == 2)).item() / cropped_label.numel()
            
            # 如果非肿瘤区域比例足够大，接受这个裁剪
            if (1 - tumor_ratio) >= self.min_bg_ratio:
                return {
                    'image': image[slices].copy() if isinstance(image, np.ndarray) else image[slices].numpy(),
                    'label': cropped_label.numpy() if isinstance(cropped_label, torch.Tensor) else cropped_label.copy()
                }
        
        # 如果多次尝试都找不到合适的裁剪，使用最后一次的结果
        return {
            'image': image[slices].copy() if isinstance(image, np.ndarray) else image[slices].numpy(),
            'label': cropped_label.numpy() if isinstance(cropped_label, torch.Tensor) else cropped_label.copy()
        }
    
    def __call__(self, data):
        d = dict(data)
        
        # 设置相同的随机种子
        random_state = np.random.RandomState(np.random.randint(1234))
        
        # 处理 pre 数据
        pre_result = self.get_center_crop(
            d["image.pre"], 
            d["label.pre"], 
            random_state, 
            self.spatial_size
        )
        
        # 处理 post 数据
        post_result = self.get_center_crop(
            d["image.post"], 
            d["label.post"], 
            random_state, 
            self.spatial_size
        )
        
        # 如果任一结果为None，使用原始的随机裁剪
        if pre_result is None or post_result is None:
            return super().__call__(d)
        
        # 合并结果
        d["image.pre"] = pre_result['image']
        d["label.pre"] = pre_result['label']
        d["image.post"] = post_result['image']
        d["label.post"] = post_result['label']
        
        return d

class Compose_Select(Compose):
    def __call__(self, input_):
        name = input_['name']
        key = get_key(name)
        for index, _transform in enumerate(self.transforms):
            # for RandCropByPosNegLabeld and RandCropByLabelClassesd case
            if (key in ['10_03', '10_07', '10_08', '04']) and (index == 8):
                continue
            elif (key not in ['10_03', '10_07', '10_08', '04']) and (index == 9):
                continue
            # for RandZoomd case
            if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']) and (index == 7):
                continue
            input_ = apply_transform(_transform, input_, self.map_items, self.unpack_items, self.log_stats)
        return input_

from itertools import cycle
current_index = 0
# def choose_different_value(x, num_samples):
#     global current_index
#     num_texts = len(x["text"])
#     if current_index >= num_samples:
#         current_index = 0
#     current_choice = x["text"][current_index]
#     current_index = (current_index + 1) % num_texts
#     return {**x, "text": current_choice}

def choose_different_value(x, num_samples):
    global current_index
    num_texts = len(x["text"])
    
    # 处理空列表的情况
    if num_texts == 0:
        return {**x, "text": ""}
        
    # 确保current_index不会超过实际文本数量
    current_index = current_index % num_texts
    
    current_choice = x["text"][current_index]
    current_index = (current_index + 1) % num_texts
    
    return {**x, "text": current_choice}

import random
def get_loader(args):
    # 加载临床数据
    clinical_data = load_clinical_data("/home/yyang303/project/Survival/data/HCC_clinical_data.csv")
    
    def check_shapes(stage):
        def _check_shapes(data):
            image = data['image.pre']
            label = data['label.pre']
            # post_image = data['post']['image']
            # post_label = data['post']['label']
            return data
        return _check_shapes


    train_transforms = Compose(
        [
            LoadImageh5d(keys=["image.pre", "image.post", "label.pre", "label.post"]),
            Lambda(check_shapes("After LoadImageh5d")),
            AddChanneld(keys=["image.pre", "image.post", "label.pre", "label.post"]),
            Lambda(check_shapes("After AddChanneld")),
            Orientationd(keys=["image.pre", "image.post", "label.pre", "label.post"], axcodes="RAS"),
            Lambda(check_shapes("After Orientationd")),
            Spacingd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "bilinear", "nearest", "nearest"),
                align_corners=True,
            ),
            Lambda(check_shapes("After Spacingd")),
            ScaleIntensityRanged(
                keys=["image.pre", "image.post"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            SpatialPadd(
                keys=["image.pre", "image.post", "label.pre", "label.post"], 
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), 
                mode='constant'
            ),
            Lambda(check_shapes("After SpatialPadd")),
            RandCropByLabelClassesd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                label_key="label.pre",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                ratios=[0, 0, 1],
                num_classes=3,
                num_samples=args.num_samples,
                image_key="image.pre",
                image_threshold=-1,
                # max_tumor_ratio=0.4,  # 设置最大肿瘤占比为40%
                # max_liver_ratio=0.8,  # 设置最大肝脏占比为80%
            ),
            Lambda(check_shapes("After SeparateCenterTumorRandCropByLabelClassesd")),
            RandRotate90d(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                prob=0.10,
                max_k=3,
            ),
            Lambda(check_shapes("After RandRotate90d")),
            Lambda(lambda x: choose_different_value(x, args.num_samples)),
            ToTensord(keys=["image.pre", "image.post", "label.pre", "label.post"]),
        ]
    )


    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image.pre", "image.post", "label.pre", "label.post"]),
            AddChanneld(keys=["image.pre", "image.post", "label.pre", "label.post"]),
            Orientationd(keys=["image.pre", "image.post", "label.pre", "label.post"], axcodes="RAS"),
            Spacingd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "bilinear", "nearest", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image.pre", "image.post"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            # 先根据pre的肿瘤区域裁剪
            CropForegroundd(keys=["image.pre", "image.post", "label.pre", "label.post"], source_key="image.pre"),
            RandCropByLabelClassesd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                label_key="label.pre",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                ratios=[0.1, 0.1, 1],
                num_classes=3,
                num_samples=5,
                image_key="image.pre",
                image_threshold=-1,
                # center_ratio=0.6
            ),
            Lambda(lambda x: choose_different_value(x,args.num_samples)),
            ToTensord(keys=["image.pre", "image.post", "label.pre", "label.post"]),
        ]
    )

    # def load_data(file_path, root_path, label_root_path):
    #     img_list = []
    #     lbl_list = []
    #     text_list = []
    #     name_list = []

    #     with open(file_path) as f:
    #         for line in f:
    #             parts = line.strip().split(maxsplit=2)  
    #             name = parts[1].split('.')[0]
    #             img_list.append(root_path + parts[0])
    #             lbl_list.append(label_root_path + parts[1])

    #             text_descriptions = parts[2].split(' ' * 10)  
    #             text_descriptions = [desc.strip() for desc in text_descriptions if desc.strip()] 

    #             text_list.append(text_descriptions)
    #             name_list.append(name)

    #     return img_list, lbl_list, text_list, name_list
    def load_data(file_path, root_path, label_root_path):
        img_list = []
        lbl_list = []
        text_list = []
        name_list = []
        print(file_path)
        with open(file_path) as f:
            for line in f:
                parts = line.strip().split('\t')  
                # if len(parts) == 1:  # 如果没有制表符,则按多个空格分割
                #     parts = line.strip().split('    ')
                print(parts)
                name = parts[0].split('/')[1] + '_' + parts[0].split('/')[2].split('.')[0].split('_')[-1]
                # 只存储路径
                img_list.append([root_path + parts[0], root_path + parts[4]])
                lbl_list.append([label_root_path + parts[1], label_root_path + parts[5]])

                # 保持完整的治疗方案描述
                text_descriptions = parts[-4].strip()
                text_list.append([text_descriptions])  # 用列表包装，保持数据结构一致性
                name_list.append(name)

        return img_list, lbl_list, text_list, name_list

    def load_data_val(file_path, root_path, label_root_path):
        img_list = []
        lbl_list = []
        text_list = []
        name_list = []

        with open(file_path) as f:
            for line in f:
                parts = line.strip().split(maxsplit=8)  
                name = parts[0].split('/')[1]
                # 只存储路径
                img_list.append(root_path + parts[0])
                lbl_list.append(label_root_path + parts[1])

                # 保持完整的治疗方案描述
                text_descriptions = parts[-1].strip()
                text_list.append([text_descriptions])  # 用列表包装，保持数据结构一致性
                name_list.append(name)

        return img_list, lbl_list, text_list, name_list

    if args.phase == 'train':
        train_img, train_lbl, train_text, train_name = [], [], [], []

        for item in args.dataset_list:
            img, lbl, text, name = load_data(os.path.join(args.data_txt_path, item, f'train_paired_all_hcc_under_50_new.txt'),
                                             args.data_root_path, args.label_root_path)
            train_img.extend(img)
            train_lbl.extend(lbl)
            train_text.extend(text)
            train_name.extend(name)
        # print("a",train_img)
        # print("b",train_lbl)
        # print("c",train_text)
        # print("d",train_name)
        data_dicts_train = [
            {
                'image.pre': pre_img,
                'image.post': post_img,
                'label.pre': pre_lbl,
                'label.post': post_lbl,
                'name': name,
                'text': text,
                'clinical_info': torch.tensor(list(clinical_data.get(name, {
                    'sex': 0.0, 'age': 0.0, 'HBV': 0.0, 'AFP': 0.0,
                    'albumin': 0.0, 'bilirubin': 0.0, 'child': 0.0,
                    'BCLC': 0.0, 'numoflesions': 0.0, 'diameter': 0.0,
                    'VI': 0.0
                }).values()), dtype=torch.float32)
            }
            for (pre_img, post_img), (pre_lbl, post_lbl), name, text in zip(train_img, train_lbl, train_name, train_text)
        ]
        
        print('train len {}'.format(len(data_dicts_train)))

        if args.cache_dataset:
            if args.uniform_sample:
                train_dataset = UniformCacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate, datasetkey=args.datasetkey)
            else:
                train_dataset = CacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate)
        else:
            if args.uniform_sample:
                train_dataset = UniformDataset(data=data_dicts_train, transform=train_transforms, datasetkey=args.datasetkey)
            else:
                train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
        
        train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        print(args.batch_size)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                  collate_fn=list_data_collate, sampler=train_sampler)
        return train_loader, train_sampler, len(train_dataset)
    
    if args.phase == 'validation':
        val_img, val_lbl, val_text, val_name = [], [], [], []
        
        for item in args.dataset_list:
            img, lbl, text, name = load_data(os.path.join(args.data_txt_path, item, 'val_paired_all_hcc_under_50_new.txt'),
                                             args.data_root_path, args.label_root_path)
            val_img.extend(img)
            val_lbl.extend(lbl)
            val_text.extend(text)
            val_name.extend(name)
        
        data_dicts_val = [
            {
                'image.pre': pre_img,
                'image.post': post_img,
                'label.pre': pre_lbl,
                'label.post': post_lbl,
                'name': name,
                'text': text,
                'clinical_info': torch.tensor(list(clinical_data.get(name, {
                    'sex': 0.0, 'age': 0.0, 'HBV': 0.0, 'AFP': 0.0,
                    'albumin': 0.0, 'bilirubin': 0.0, 'child': 0.0,
                    'BCLC': 0.0, 'numoflesions': 0.0, 'diameter': 0.0,
                    'VI': 0.0
                }).values()), dtype=torch.float32)
            }
            for (pre_img, post_img), (pre_lbl, post_lbl), name, text in zip(val_img, val_lbl, val_name, val_text)
        ]
        
        print('val len {}'.format(len(data_dicts_val)))

        if args.cache_dataset:
            val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
        
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return val_loader, val_transforms, len(val_dataset)


def get_key(name):
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
    else:
        template_key = name[0:2]
    return template_key

def load_clinical_data(clinical_data_path):
    """
    读取并处理患者的临床数据
    Args:
        clinical_data_path: CSV文件的路径
    Returns:
        clinical_data: 包含患者临床信息的字典，以患者ID为键
    """
    # 读取CSV文件，设置ID列的类型为int
    import pandas as pd
    df = pd.read_csv(clinical_data_path, dtype={'ID': str})
    
    # 定义需要使用的clinical features
    clinical_features = ['sex', 'age', 'HBV', 'AFP', 'albumin', 'bilirubin', 
                        'child', 'BCLC', 'numoflesions', 'diameter', 'VI']
    
    # 处理分类变量
    df['child'] = df['child'].map({'A': 0, 'B': 1})
    df['BCLC'] = df['BCLC'].map({'A': 0, 'B': 1, 'C': 2})
    
    # 处理可能的缺失值
    df = df.fillna({
        'sex': df['sex'].mode()[0],
        'age': df['age'].mean(),
        'HBV': df['HBV'].mode()[0],
        'AFP': df['AFP'].mean(),
        'albumin': df['albumin'].mean(),
        'bilirubin': df['bilirubin'].mean(),
        'child': df['child'].mode()[0],
        'BCLC': df['BCLC'].mode()[0],
        'numoflesions': df['numoflesions'].mean(),
        'diameter': df['diameter'].mean(),
        'VI': df['VI'].mode()[0]
    })
    
    # 将DataFrame转换为字典格式
    clinical_data = {}
    for _, row in df.iterrows():
        patient_id = str(row['ID'])  # 确保ID是字符串格式
        clinical_info = {
            'sex': float(row['sex']),  # 确保所有值都是float类型
            'age': float(row['age']),
            'HBV': float(row['HBV']),
            'AFP': float(row['AFP']),
            'albumin': float(row['albumin']),
            'bilirubin': float(row['bilirubin']),
            'child': float(row['child']),
            'BCLC': float(row['BCLC']),
            'numoflesions': float(row['numoflesions']),
            'diameter': float(row['diameter']),
            'VI': float(row['VI'])
        }
        clinical_data[patient_id] = clinical_info
    
    return clinical_data

if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()