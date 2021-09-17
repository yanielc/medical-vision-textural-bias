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
        ToTensord(keys="image"),
        RandConcatd("image")
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
        ToTensord(keys="image"),
        RandConcatd("image"),
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
    cache_num=100
)

val_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform,
    section="validation",
    download=False,
    num_workers=4,
    cache_num= 1 # 4
)

val_ds, _ = random_split(val_ds, [48, 48], torch.Generator().manual_seed(0))


# to plot

# train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)

# import torchvision.utils as vutils
# import matplotlib.pyplot as plt 
# # Plot some training images
# real_batch = next(iter(train_loader))
# real_batch = real_batch["image"]
# plt.figure(figsize=(20,20))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[:64], padding=2, n_row=3, normalize=True).cpu(),(1,2,0))[:,:,0])
# plt.savefig('examples.png')

