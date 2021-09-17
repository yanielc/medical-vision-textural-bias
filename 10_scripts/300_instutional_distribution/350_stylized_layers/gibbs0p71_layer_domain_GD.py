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
    ToTensord,
)
from monai.utils import set_determinism

import torch
from torch.utils.data import ConcatDataset
import torch.nn as nn


import matplotlib.pyplot as plt

from math import floor

from typing import Union, List, Tuple

#################################################################

# Local imports

SOURCE_CODE_PATH = '/homes/yc7620/Documents/medical-vision-textural-bias/source_code/'

import sys
sys.path.append(SOURCE_CODE_PATH)

from filters_and_operators import WholeTumorTCGA 
from utils import ReCompose
from stylization_layers import GibbsNoiseLayer
# set determinism for reproducibility
set_determinism(seed=0)

print_config()

root_dir = '/vol/bitbucket/yc7620/90_data/53_TCGA_data/' 
print('root_dir', root_dir)
#################################################################
# blurb

print('stylized network on four modalities. excluding one institution\n')

#################################################################
# SCRIPT PARAMETERS 


# gibbs layer starting point
alpha = 0.71

JOB_NAME = f"gibbs{alpha}_layer_GibbsGD_model_sourceDist_4mods_WT"
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
    ]
)


print('\n')
print('training transforms: ', train_transform.transforms,'\n')
print('validation transforms: ', val_transform.transforms, '\n')
###########################################################################

# Dataloading

# training
with open(os.path.join(root_dir, 'train_sequence_by_modality.json'), 'r') as f:
    data_seqs_4mods = json.load(f)

# split off training and validation     
train_seq_flair, _ = partition_dataset(data_seqs_4mods["FLAIR"], [0.9, 0.1], shuffle=True, seed=0)
train_seq_t1, _ = partition_dataset(data_seqs_4mods["T1"], [0.9, 0.1], shuffle=True, seed=0)
train_seq_t1gd, _ = partition_dataset(data_seqs_4mods["T1Gd"], [0.9, 0.1], shuffle=True, seed=0)
train_seq_t2, _ = partition_dataset(data_seqs_4mods["T2"], [0.9, 0.1], shuffle=True, seed=0)

# create training datasets
CACHE_NUM = 100

train_ds_flair = CacheDataset(train_seq_flair, train_transform, cache_num=CACHE_NUM)
train_ds_t1 = CacheDataset(train_seq_t1, train_transform, cache_num=CACHE_NUM)
train_ds_t1gd = CacheDataset(train_seq_t1gd, train_transform, cache_num=CACHE_NUM)
train_ds_t2 = CacheDataset(train_seq_t2, train_transform, cache_num=CACHE_NUM)

# combined dataset and dataloader
train_ds = ConcatDataset([train_ds_flair, train_ds_t1, train_ds_t1gd, train_ds_t2])
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

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
############################################################################

# Create model, loss, optimizer

class Gibbs_UNet(nn.Module):
    """ResUnet with Gibbs layer"""
    
    def __init__(self, alpha=None):
        super().__init__()
        
        self.gibbs = GibbsNoiseLayer(alpha)
        
        self.ResUnet = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
        
    def forward(self,img):
        img = self.gibbs(img) 
        img = self.ResUnet(img)
        return img
    
device = torch.device("cuda:0")

model = Gibbs_UNet(alpha).to(device)

# load trained baseline ResUnet
# baseline_path = '/vol/bitbucket/yc7620/90_data/52_MONAI_DATA_DIRECTORY/10_training_results/imperial_project_data/baseline_model_sourceDist_4mods_WT/baseline_model_sourceDist_4mods_WT.pth'

# model.ResUnet.load_state_dict(torch.load(baseline_path))

loss_function = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)

optimizer = torch.optim.Adam(
      model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)


###########################################################################

# freeze Unet
# for param in model.ResUnet.parameters():
#     param.requires_grad = False
    
print('Model instatitated with number of parameters = ',
      sum([p.numel() for p in model.parameters() if p.requires_grad]))

############################################################################

# Training loop

max_epochs =  110
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
gibbs_values = [] # store the Gibbs trajectory

print('\n Training started... \n')

@torch.no_grad()
def Gibbs_GD(inputs, labels, model, h = 0.01, learning_rate = 0.02):
    """Function to update Gibbs layer via finite different SG"""
#     with torch.no_grad():
    old_alpha = model.gibbs.alpha.clone()
    # loss at alpha
    outputs_0 = model(inputs)
    loss_0 = loss_function(outputs_0, labels)
    # loss at perturbed alpha
    model.gibbs.alpha = old_alpha + h
    outputs_h = model(inputs)
    loss_h = loss_function(outputs_h, labels)
    # approximate gradient
    delta = (loss_h - loss_0) / h
    # update alpha and model
    model.gibbs.alpha = old_alpha - learning_rate * delta
    
    return loss_0.detach().item(), model.gibbs.alpha.item()
    

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    gibbs_loss_epoch = 0
    step = 0
    for batch_data in train_loader:
        #save gibbs trajectory
        gibbs_values.append(model.gibbs.alpha.detach().item())
        
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
         # update the Unet
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # update Gibbs
        gibbs_loss, gibbs_alpha = Gibbs_GD(inputs, labels, model)
        gibbs_values.append(gibbs_alpha)
        #   gibbs_loss_epoch += gibbs_loss
        

    epoch_loss /= step
    # gibbs_loss_epoch /= step
    epoch_loss_values.append(epoch_loss)
   # epoch_loss_values.append(gibbs_loss_epoch) # TODO: use this line with frozen Unet only
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    # test on validation
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
np.savetxt(os.path.join(working_dir, f'gibbs_trajectory_{JOB_NAME}.txt'), np.array(gibbs_values))
############################################################################

print('script ran fully')
