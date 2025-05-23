import os
import torch
from torch.utils.data import Dataset, Sampler, DistributedSampler
import numpy as np
import SimpleITK as sitk
from monai import transforms
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    SpatialPadd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    MapTransform,
    RandCropByLabelClassesd,
)
from monai.config import KeysCollection
from typing import Union, Sequence
from monai.utils import ensure_tuple

from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

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

class ImageDataset(Dataset):
    def __init__(self, data_root, data_list_file, mode='valid'):
        self.data_root = data_root
        self.data_list_file = data_list_file
        self.mode = mode

        with open(self.data_list_file, 'r') as f:
            self.data_lines = f.readlines()
        self.transform = self.get_val_transform()

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        line = self.data_lines[idx].strip()
        items = line.split('\t')
        
        data_dict = {
            'image.pre': os.path.join(self.data_root, items[0]),
            'image.post': os.path.join(self.data_root, items[4]),
            'label.pre': os.path.join(self.data_root, items[1]),
            'label.post': os.path.join(self.data_root, items[5]),
            'survival_time': float(items[-2]),
            'event_indicator': float(items[-1]),
            'name': items[0].split('/')[1] + '_' + items[0].split('/')[2].split('.')[0].split('_')[-1]
        }
        
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return {
            'input_CT_pre': torch.cat([data_dict[i]['image.pre'] for i in range(len(data_dict))],dim=0),
            'input_CT_post': torch.cat([data_dict[i]['image.post'] for i in range(len(data_dict))],dim=0),
            'label_CT_pre': torch.cat([data_dict[i]['label.pre'] for i in range(len(data_dict))],dim=0),
            'label_CT_post': torch.cat([data_dict[i]['label.post'] for i in range(len(data_dict))],dim=0),
            'survival_time': torch.tensor([data_dict[i]['survival_time'] for i in range(len(data_dict))], dtype=torch.float32),
            'event_indicator': torch.tensor([data_dict[i]['event_indicator'] for i in range(len(data_dict))], dtype=torch.float32),
            'name': data_dict[0]['name']
        }

    def get_train_transform(self):
        return transforms.Compose([
            LoadImageh5d(keys=["image.pre", "image.post", "label.pre", "label.post"]),
            AddChanneld(keys=["image.pre", "image.post", "label.pre", "label.post"]),
            Orientationd(keys=["image.pre", "image.post", "label.pre", "label.post"], axcodes="RAS"),
            Spacingd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "nearest", "nearest"),
                align_corners=True,
            ),
            ScaleIntensityRanged(
                keys=["image.pre", "image.post"],
                a_min=-175,
                a_max=600,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(
                keys=["image.pre", "image.post", "label.pre", "label.post"], 
                spatial_size=(96, 96, 96), 
                mode='constant'
            ),
            RandCropByLabelClassesd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                label_key="label.pre",
                spatial_size=(96, 96, 96),
                ratios=[0.1, 0.1, 1],
                num_classes=3,
                num_samples=5,
                image_key="image.pre",
                image_threshold=-1,
            ),
            RandFlipd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                spatial_axis=[0],
                prob=0.2,
            ),
            RandFlipd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                spatial_axis=[1],
                prob=0.2,
            ),
            RandFlipd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                spatial_axis=[2],
                prob=0.2,
            ),
            RandRotate90d(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                prob=0.2,
                max_k=3,
            ),
            RandScaleIntensityd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                factors=0.1,
                prob=0.15
            ),
            RandShiftIntensityd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                offsets=0.1,
                prob=0.15
            ),
            ToTensord(keys=["image.pre", "image.post", "label.pre", "label.post"]),
        ])

    def get_val_transform(self):
        return transforms.Compose([
            LoadImageh5d(keys=["image.pre", "image.post", "label.pre", "label.post"]),
            AddChanneld(keys=["image.pre", "image.post", "label.pre", "label.post"]),
            Orientationd(keys=["image.pre", "image.post", "label.pre", "label.post"], axcodes="RAS"),
            Spacingd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "nearest", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image.pre", "image.post"],
                a_min=-175,
                a_max=600,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image.pre", "image.post", "label.pre", "label.post"], source_key="image.pre"),
            RandCropByLabelClassesd(
                keys=["image.pre", "image.post", "label.pre", "label.post"],
                label_key="label.pre",
                spatial_size=(96, 96, 96),
                ratios=[0.1, 0.1, 1],
                num_classes=3,
                num_samples=5,
                image_key="image.pre",
                image_threshold=-1,
                # center_ratio=0.6
            ),
            # RandCropByPosNegLabeld(
            #     keys=["image.pre", "image.post", "label.pre", "label.post"],
            #     label_key="label.pre",
            #     spatial_size=(96, 96, 96),
            #     pos=1,
            #     neg=1,
            #     num_samples=20,
            #     image_key="image.pre",
            #     image_threshold=0,
            # ),
            ToTensord(keys=["image.pre", "image.post", "label.pre", "label.post"]),
        ])

class CenterSpatialCropByLabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.label_key = label_key
        self.spatial_size = ensure_tuple(spatial_size)

    def __call__(self, data):
        d = dict(data)
        label = d[self.label_key]
        
        nonzero = np.nonzero(label)
        center = np.mean(nonzero, axis=1).round().astype(int)
        
        spatial_size = [s // 2 for s in self.spatial_size]
        start = center - np.array(spatial_size)
        end = start + np.array(self.spatial_size)
        
        for i in range(len(start)):
            if start[i] < 0:
                start[i] = 0
                end[i] = self.spatial_size[i]
            if end[i] > label.shape[i]:
                end[i] = label.shape[i]
                start[i] = end[i] - self.spatial_size[i]
        
        for key in self.keys:
            d[key] = d[key][..., start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        return d

class BalancedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        
        event_indices = []
        censored_indices = []
        
        for idx in range(len(dataset)):
            if dataset.data_lines[idx].strip().split()[-1] == '1':
                event_indices.append(idx)
            else:
                censored_indices.append(idx)
        
        num_samples_per_replica = self.num_samples
        self.event_indices = event_indices[self.rank:len(event_indices):self.num_replicas]
        self.censored_indices = censored_indices[self.rank:len(censored_indices):self.num_replicas]

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.event_indices)
            np.random.shuffle(self.censored_indices)
        
        indices = []
        for e, c in zip(self.event_indices, self.censored_indices):
            indices.extend([e, c])
        
        return iter(indices[:self.num_samples])

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
