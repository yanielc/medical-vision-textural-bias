
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
)
from monai.utils import set_determinism

import torch
from torch.utils.data import random_split
from tqdm import tqdm

#######################
# Local imports

SOURCE_CODE_PATH = '/homes/yc7620/Documents/medical-vision-textural-bias/90_source_code/'
import sys
sys.path.append(SOURCE_CODE_PATH)

from filters_and_operators import ConvertToMultiChannelBasedOnBratsClassesd, RandPlaneWaves_ellipsoid
from utils import show_slice_and_fourier, model_evaluation
######################

# set determinism for reproducibility
set_determinism(seed=0)


root_dir = '/vol/bitbucket/yc7620/90_data/52_MONAI_DATA_DIRECTORY/'
print('root_dir', root_dir)
print_config()

device = torch.device("cuda:0")


#############################################################################

# SET UP TRANSFORMS FOR EACH DATASET

# baseline preprocessing sequence
val_transform_baseline = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"])
    ]
)

# plane waves preprocessing sequences 

val_transform_planes12 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandPlaneWaves_ellipsoid('image',55.,55.,30., intensity_value=12, prob=1.)
    ]
)

val_transform_planes14 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandPlaneWaves_ellipsoid('image',55.,55.,30., intensity_value=14, prob=1.)
    ]
)

val_transform_planes15 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandPlaneWaves_ellipsoid('image',55.,55.,30., intensity_value=15, prob=1.)
    ]
)

val_transform_planes16 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandPlaneWaves_ellipsoid('image',55.,55.,30., intensity_value=16, prob=1.)
    ]
)

val_transform_planes16 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandPlaneWaves_ellipsoid('image',55.,55.,30., intensity_value=16, prob=1.)
    ]
)

val_transform_planes16p5 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandPlaneWaves_ellipsoid('image',55.,55.,30., intensity_value=16.5, prob=1.)
    ]
)

val_transform_planes17 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandPlaneWaves_ellipsoid('image',55.,55.,30., intensity_value=17, prob=1.)
    ]
)

# Gibbs preprocessing sequence


val_transform_gibbs25 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandFourierDiskMaskd(keys='image', r=25 , inside_off=False, prob=1.)
    ]
)

val_transform_gibbs20 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandFourierDiskMaskd(keys='image', r=20 , inside_off=False, prob=1.)
    ]
)

val_transform_gibbs15 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandFourierDiskMaskd(keys='image', r=15 , inside_off=False, prob=1.)
    ]
)

val_transform_gibbs12p5 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandFourierDiskMaskd(keys='image', r=12.5 , inside_off=False, prob=1.)
    ]
)

val_transform_gibbs10 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandFourierDiskMaskd(keys='image', r=10 , inside_off=False, prob=1.)
    ]
)

val_transform_gibbs9 = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AsChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
        RandFourierDiskMaskd(keys='image', r=9 , inside_off=False, prob=1.)
    ]
)

###################################################################################

# Load data using DecathlonDataset

# baseline
val_ds_baseline = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_baseline,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)

_ , test_ds_baseline = random_split(val_ds_baseline, [48, 48],
                                   torch.Generator().manual_seed(0))
test_loader_baseline = DataLoader(test_ds_baseline, batch_size=2, shuffle=False, num_workers=4)

# planes 12
val_ds_planes12 = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_planes12,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)

_ , test_ds_planes12 = random_split(val_ds_planes12, [48, 48],
                                   torch.Generator().manual_seed(0))
test_loader_planes12 = DataLoader(test_ds_planes12, batch_size=2, shuffle=False, num_workers=4)

# planes 14
val_ds_planes14 = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_planes14,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)

_ , test_ds_planes14 = random_split(val_ds_planes14, [48, 48],
                                   torch.Generator().manual_seed(0))
test_loader_planes14 = DataLoader(test_ds_planes14, batch_size=2, shuffle=False, num_workers=4)

# planes 15
val_ds_planes15 = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_planes15,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)

_ , test_ds_planes15 = random_split(val_ds_planes15, [48, 48],
                                   torch.Generator().manual_seed(0))
test_loader_planes15 = DataLoader(test_ds_planes15, batch_size=2, shuffle=False, num_workers=4)

# planes 16
val_ds_planes16 = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_planes16,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)

_ , test_ds_planes16 = random_split(val_ds_planes16, [48, 48],
                                   torch.Generator().manual_seed(0))
test_loader_planes16 = DataLoader(test_ds_planes16, batch_size=2, shuffle=False, num_workers=4)

# planes 16.5
val_ds_planes16p5 = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_planes16p5,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)

_ , test_ds_planes16p5 = random_split(val_ds_planes16p5, [48, 48],
                                   torch.Generator().manual_seed(0))
test_loader_planes16p5 = DataLoader(test_ds_planes16p5, batch_size=2, shuffle=False, num_workers=4)

# planes 17
val_ds_planes17 = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_planes17,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)

_ , test_ds_planes17 = random_split(val_ds_planes17, [48, 48],
                                   torch.Generator().manual_seed(0))
test_loader_planes17 = DataLoader(test_ds_planes17, batch_size=2, shuffle=False, num_workers=4)

#### Gibbs datasets

#Gibbs25
val_ds_gibbs25 = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_gibbs25,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)
_ , test_ds_gibbs25 = random_split(val_ds_gibbs25, [48, 48],
                                   torch.Generator().manual_seed(0))

test_loader_gibbs25 = DataLoader(test_ds_gibbs25, batch_size=2, shuffle=False, num_workers=4)

#Gibbs20
val_ds_gibbs20 = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_gibbs20,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)
_ , test_ds_gibbs20 = random_split(val_ds_gibbs20, [48, 48],
                                   torch.Generator().manual_seed(0))

test_loader_gibbs20 = DataLoader(test_ds_gibbs20, batch_size=2, shuffle=False, num_workers=4)

#Gibbs15
val_ds_gibbs15 = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_gibbs15,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)
_ , test_ds_gibbs15 = random_split(val_ds_gibbs15, [48, 48],
                                   torch.Generator().manual_seed(0))

test_loader_gibbs15 = DataLoader(test_ds_gibbs15, batch_size=2, shuffle=False, num_workers=4)

#Gibbs12.5
val_ds_gibbs12p5 = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_gibbs12p5,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)
_ , test_ds_gibbs12p5 = random_split(val_ds_gibbs12p5, [48, 48],
                                   torch.Generator().manual_seed(0))

test_loader_gibbs12p5 = DataLoader(test_ds_gibbs12p5, batch_size=2, shuffle=False, num_workers=4)

# Gibb10
val_ds_gibbs10 = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_gibbs10,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)
_ , test_ds_gibbs10 = random_split(val_ds_gibbs10, [48, 48],
                                   torch.Generator().manual_seed(0))

test_loader_gibbs10 = DataLoader(test_ds_gibbs10, batch_size=2, shuffle=False, num_workers=4)

# Gibb9
val_ds_gibbs9 = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform_gibbs9,
    section="validation",
    download=False,
    num_workers=4,
    cache_num=50
)
_ , test_ds_gibbs9 = random_split(val_ds_gibbs9, [48, 48],
                                   torch.Generator().manual_seed(0))

test_loader_gibbs9 = DataLoader(test_ds_gibbs9, batch_size=2, shuffle=False, num_workers=4)

###########################################################################################

# Models' evaluations

dataset_dict = dict(
        [('baseline_data', test_loader_baseline),
            ('planes12_data', test_loader_planes12),
            ('planes14_data', test_loader_planes14),
            ('planes15_data', test_loader_planes15),
            ('planes16_data', test_loader_planes16),
            ('planes16.5_data', test_loader_planes16p5),
            ('planes17_data', test_loader_planes17),
            ('gibbs25_data', test_loader_gibbs25),
            ('gibbs20_data', test_loader_gibbs20),
            ('gibbs15_data', test_loader_gibbs15),
            ('gibbs12.5_data', test_loader_gibbs12p5),
            ('gibbs10_data', test_loader_gibbs10),
            ('gibbs9_data', test_loader_gibbs9)]
        )

# instantiate model_evaluations
planes12 = model_evaluation('best_metric_model_planes_a55.0b55.0c30.0_int12.0.pth', 'Planes12 model')
planes14 = model_evaluation('best_metric_model_planes_a55.0b55.0c30.0_int14.0.pth', 'Planes14 model')
planes15 = model_evaluation('best_metric_model_planes_a55.0b55.0c30.0_int15.0.pth', 'Planes15 model')
planes16 = model_evaluation('best_metric_model_planes_a55.0b55.0c30.0_int16.0.pth', 'Planes16 model')
planes16p5 = model_evaluation('best_metric_model_planes_a55.0b55.0c30.0_int16.5.pth', 'Planes16.5 model')
planes17 = model_evaluation('best_metric_model_planes_a55.0b55.0c30.0_int17.0.pth', 'Planes17 model')



gibbs25 = model_evaluation('best_metric_model_Gibbs25.pth', 'Gibbs25 model')
gibbs20 = model_evaluation('best_metric_model_Gibbs20.pth', 'Gibbs20 model')





