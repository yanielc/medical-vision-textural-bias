
import os
import shutil
import tempfile
import json

import matplotlib.pyplot as plt
import numpy as np
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AddChanneld,
    AsChannelFirstd,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
    Randomizable
)
from monai.utils import set_determinism
from monai.data.utils import partition_dataset
from monai.data import CacheDataset

import torch
from torch.utils.data import random_split, ConcatDataset


# Local imports

SOURCE_CODE_PATH = '/homes/yc7620/Documents/medical-vision-textural-bias/source_code/'

import sys
sys.path.append(SOURCE_CODE_PATH)

from utils import ReCompose
from filters_and_operators import SelectChanneld, RandFourierDiskMaskd, WholeTumorTCGA

set_determinism(seed=0)

root_dir = '/vol/bitbucket/yc7620/90_data/53_TCGA_data/' 
print('root_dir', root_dir)


#######################
# preprocessing

class RandConcatd(MapTransform, Randomizable):

    """Transform to concatenate one slice into two channels """

    def __init__(self, keys, seed: int = None, allow_missing_keys: bool = False):

        Randomizable.set_random_state(self, seed=seed)
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data):
        """
        Args:
            data (Mapping): dictionary to transform. Arrays must
                be torch tensors.   
        """

        d = dict(data)
        c = self.R.randint(25,35)

        for key in self.key_iterator(d):
            s = d[key][0,:,:,c][None,:]
            d[key] = torch.cat([s,s])
        return d 

train_transform = ReCompose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys="image"),
        WholeTumorTCGA(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        RandSpatialCropd(
            keys=["image", "label"], roi_size=[128, 128, 64], random_size=False
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd("image", factors=0.1, prob=0.5),
        RandShiftIntensityd("image", offsets=0.1, prob=0.5),
        ToTensord(keys=["image", "label"]),
        RandConcatd("image")
    ]
)

val_transform = ReCompose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys="image"),
        WholeTumorTCGA(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandConcatd("image")
    ]
)

###########################################################################

# Dataloading

# training
with open(os.path.join(root_dir, 'train_sequence_by_modality.json'), 'r') as f:
    data_seqs_4mods = json.load(f)

# split off training and validation     
train_seq_flair, val_seq_flair_inDist = partition_dataset(data_seqs_4mods["FLAIR"], [0.9, 0.1], shuffle=True, seed=0)
train_seq_t1, val_seq_t1_inDist = partition_dataset(data_seqs_4mods["T1"], [0.9, 0.1], shuffle=True, seed=0)
train_seq_t1gd, val_seq_t1gd_inDist = partition_dataset(data_seqs_4mods["T1Gd"], [0.9, 0.1], shuffle=True, seed=0)
train_seq_t2, val_seq_t2_inDist = partition_dataset(data_seqs_4mods["T2"], [0.9, 0.1], shuffle=True, seed=0)

# create training datasets + dataloader
CACHE_NUM = 4

train_ds_flair = CacheDataset(train_seq_flair, train_transform, cache_num=CACHE_NUM)
train_ds_t1 = CacheDataset(train_seq_t1, train_transform, cache_num=CACHE_NUM)
train_ds_t1gd = CacheDataset(train_seq_t1gd, train_transform, cache_num=CACHE_NUM)
train_ds_t2 = CacheDataset(train_seq_t2, train_transform, cache_num=CACHE_NUM)


train_ds = ConcatDataset([train_ds_flair, train_ds_t1, train_ds_t1gd, train_ds_t2])
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

# create in-dist validation datasets + dataloader

val_ds_flair_inDist = CacheDataset(val_seq_flair_inDist, val_transform, cache_num=50)
val_ds_t1_inDist = CacheDataset(val_seq_t1_inDist, val_transform, cache_num=50)
val_ds_t1gd_inDist = CacheDataset(val_seq_t1gd_inDist, val_transform, cache_num=50)
val_ds_t2_inDist = CacheDataset(val_seq_t2_inDist, val_transform, cache_num=50)

# combined dataset and dataloader
val_ds_inDist = ConcatDataset([val_ds_flair_inDist, val_ds_t1_inDist, val_ds_t1gd_inDist, val_ds_t2_inDist])
val_loader_inDist = DataLoader(val_ds_inDist, batch_size=2, shuffle=False, num_workers=4)

# validation out of dist
with open(os.path.join(root_dir, 'test_sequence_by_modality.json'), 'r') as f:
    val_data_seqs_4mods = json.load(f)

# validation modalities     
val_seq_flair  = val_data_seqs_4mods["FLAIR"]
val_seq_t1  = val_data_seqs_4mods["T1"]
val_seq_t1gd = val_data_seqs_4mods["T1Gd"]
val_seq_t2 = val_data_seqs_4mods["T2"]

val_ds_flair = CacheDataset(val_seq_flair, val_transform, cache_num=50)
val_ds_t1 = CacheDataset(val_seq_t1, val_transform, cache_num=50)
val_ds_t1gd = CacheDataset(val_seq_t1gd, val_transform, cache_num=50)
val_ds_t2 = CacheDataset(val_seq_t2, val_transform, cache_num=50)

# combined dataset and dataloader
val_ds = ConcatDataset([val_ds_flair, val_ds_t1, val_ds_t1gd, val_ds_t2])
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4)

print('Data loaders created.\n')