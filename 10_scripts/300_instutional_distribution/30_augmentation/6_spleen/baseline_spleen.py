################################################################
# bit to help with torch/monai bug reported at                 #                        
# https://github.com/Project-MONAI/MONAI/issues/701            #
import resource                                                #
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)            #
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))  #
################################################################

import os
import shutil
import json

import matplotlib.pyplot as plt
import numpy as np
from monai.config import print_config
from monai.data import DataLoader, CacheDataset, partition_dataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AddChanneld,
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
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandAffined,
    ToTensord,
)
from monai.utils import set_determinism, NumpyPadMode
from monai.apps.datasets import DecathlonDataset

import torch
from torch.utils.data import ConcatDataset


import matplotlib.pyplot as plt

from math import floor

from typing import Union, List, Tuple

#################################################################

# Local imports

SOURCE_CODE_PATH = '/homes/yc7620/Documents/medical-vision-textural-bias/source_code/'

import sys
sys.path.append(SOURCE_CODE_PATH)

from filters_and_operators import WholeTumorTCGA, RandGibbsNoised 
from utils import ReCompose

# set determinism for reproducibility
set_determinism(seed=0)

print_config()

root_dir = '/vol/bitbucket/yc7620/90_data/53_TCGA_data/' 
print('root_dir', root_dir)
#################################################################
# blurb

print('baseline model on four modalities. excluding one institution\n')

#################################################################
# SCRIPT PARAMETERS 


#print(f'''Using parameters  (IMAGE_CHAN, LABEL_CHAN) =
#        {(IMAGE_CHAN, LABEL_CHAN)}\n\n''')

ALPHA=0.5
JOB_NAME = f"spleen_baseline"
print(f"JOB_NAME = {JOB_NAME}\n")

# create dir

working_dir = os.path.join(root_dir,JOB_NAME)
try:
    os.mkdir(working_dir)
except:
    print('creating version _2 of working dir') 
    JOB_NAME = JOB_NAME + '_2'
    working_dir = os.path.join(root_dir,JOB_NAME)
    os.mkdir(working_dir)
#############################################################################


# Preprocessing transforms. Note we use wrapping artifacts. 

train_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
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
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ]
)



print('\n')
print('training transforms: ', train_transform.transforms,'\n')
print('validation transforms: ', val_transform.transforms, '\n')
###########################################################################

# Dataloading


train_ds = DecathlonDataset(root_dir='/vol/bitbucket/yc7620/90_data/55_spleen/',
                task='Task09_Spleen',
                section='training',
                download=False,
                transform=train_transform,
                cache_num=5)

val_ds = DecathlonDataset(root_dir='/vol/bitbucket/yc7620/90_data/55_spleen/',
                task='Task09_Spleen',
                section='validation',
                download=False,
                transform=val_transform,
                cache_num=5)





train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=2, num_workers=4)

print('Data loaders created.\n')
############################################################################

# Create model, loss, optimizer

device = torch.device("cuda:0")

model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)

optimizer = torch.optim.Adam(
      model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)

print('Model instatitated with number of parameters = ',
      sum([p.numel() for p in model.parameters() if p.requires_grad]))

############################################################################

# Training loop


max_epochs = 100 
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

print('\n Training started... \n')

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        print(inputs.size())
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # print(
        #     f"{step}/{len(train_ds) // train_loader.batch_size}"
        #     f", train_loss: {loss.item():.4f}"
        #     )
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            dice_metric = DiceMetric(include_background=True, reduction="mean")
            post_trans = Compose(
                [Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
            )
            metric_sum = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                print(val_inputs.size())
                val_outputs = model(val_inputs)
                val_outputs = post_trans(val_outputs)
                # compute overall mean dice
                value, not_nans = dice_metric(y_pred=val_outputs, y=val_labels)
                not_nans = not_nans.item()
                metric_count += not_nans
                metric_sum += value.item() * not_nans

            metric = metric_sum / metric_count
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(working_dir, JOB_NAME + '.pth'),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

# print best metric and epoch
print(
    f"train completed, best_metric: {best_metric:.4f}"
    f" at epoch: {best_metric_epoch}"
)

############################################################################

# Save learning curves

print('Plotting learning curves')

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y, color="red")
plt.subplot(1, 2, 2)
plt.title("Val mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y, color="green")
plt.savefig(os.path.join(root_dir, f'trainLoss_and_meanValScore_{JOB_NAME}.png'))
plt.show()



###########################################################################

# save training information

print('Saving epoch_loss_values and metrics')

np.savetxt(os.path.join(working_dir, f'epoch_loss_values_{JOB_NAME}.txt'), np.array(epoch_loss_values))
np.savetxt(os.path.join(working_dir, f'metric_values_{JOB_NAME}.txt'), np.array(metric_values))

############################################################################

print('script ran fully')
