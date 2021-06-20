import os
import shutil
import tempfile

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

import torch
from torch.utils.data import random_split 
#################################################################

# Local imports

SOURCE_CODE_PATH = '/homes/yc7620/Documents/medical-vision-textural-bias/source_code/'

import sys
sys.path.append(SOURCE_CODE_PATH)

from filters_and_operators import (ConvertToMultiChannelBasedOnBratsClassesd,
        SelectChanneld,
        RandFourierDiskMaskd)

from utils import show_slice_and_fourier


# set determinism for reproducibility
from monai.utils import set_determinism
set_determinism(seed=0)

print_config()

root_dir = '/vol/bitbucket/yc7620/90_data/52_MONAI_DATA_DIRECTORY/'
print('root_dir', root_dir)

#########################################
class Sliced(MapTransform, Randomizable):

    """Transform to extract three consecutive slices containing
    a nontrivial segmentation. """

    def __init__(self, keys, seed: int = None, allow_missing_keys: bool = False):

        Randomizable.set_random_state(self, seed=seed)
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data):
        """
        Args:
            data (Mapping): dictionary to transform"""

        d = dict(data)
        c = self.R.randint(20,40)

        for key in self.key_iterator(d):
            d[key] = d[key][:,:,:,c]
        return d

#########################################

train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys="image"),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        SelectChanneld("image", (0, 1)),
        Spacingd(
            keys="image",
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear"),
        ),
        Orientationd(keys="image", axcodes="RAS"),
        RandSpatialCropd(
            keys="image", roi_size=[128, 128, 64], random_size=False
        ),
        RandFlipd(keys="image", prob=0.5, spatial_axis=0),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        Sliced("image"),
        ToTensord(keys="image"),
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys="image"),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        SelectChanneld("image", (0, 1)),
        Spacingd(
            keys="image",
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear"),
        ),
        Orientationd(keys="image", axcodes="RAS"),
        CenterSpatialCropd(keys="image", roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Sliced("image"),
        ToTensord(keys="image"),
        ]
)

#######

# Dataloading
train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    download=False,
    num_workers=4,
    cache_num= 4 #100
)

val_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform,
    section="validation",
    download=False,
    num_workers=4,
    cache_num= 4 #2
)

val_ds, _ = random_split(val_ds, [48, 48], torch.Generator().manual_seed(0))

