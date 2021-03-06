import numpy as np

import torch
import torch.nn as nn
from torch.fft import (fftn, fft2, 
                       fftshift, ifftshift, 
                       ifft2, ifftn)
from torch.utils.data import IterableDataset
from torch.utils.data import random_split

from monai.apps import DecathlonDataset
from monai.data import DataLoader
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
from monai.metrics import DiceMetric

from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Union, List, Tuple, Optional, Callable, Sequence
import pickle

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
root_dir = '/vol/bitbucket/yc7620/90_data/52_MONAI_DATA_DIRECTORY/'


# Local imports

from filters_and_operators import ConvertToMultiChannelBasedOnBratsClassesd
from stylization_layers import GibbsNoiseLayer, Gibbs_UNet, Spikes_UNet
############################################################

# Display functions

def show(img: torch.Tensor, k_space: bool = True) -> None:
    """
    Displays RBG k-space images nicely. Also accepts 2D images.

    Set k_space = False for regular images.
    """
    img = img.clone()
    if k_space:
        img = img.abs().log()
    min, max = img.min(), img.max()
    img.add_(-min).div_(max-min+1e-5)
    if len(img.shape) == 3:
        plt.imshow(img.permute(1, 2, 0))
    elif len(img.shape) == 2:
        plt.imshow(img)
        

def WL_to_LH(window:float, level:float) -> Tuple[float]:
    '''to compute low and high values for amplitude display'''
    low = level - window/2
    high = level + window/2
    return low, high 


def show_slice_and_fourier(img_2d: torch.tensor, level: Union[int,float]=None, 
                           window: Union[int,float]=None, 
                           level_f: Union[int,float]=None,
                           window_f: Union[int,float]=None,
                           title:str=None):
    
    '''Function to display image slice and corresponding fourier slice
    
    Args:
        img_2d = 2d tensor to image.
        level, window = mean amplitude, amplitude window around the mean/
        level_f, window_f = as above but for the fourier display.
        title = title for the image
        '''
    assert img_2d.dim() == 2, 'Input tensor must have .dim() = 2'
    
    if window is None:
        window = img_2d.max() - img_2d.min()
    if level is None:
        level = window / 2 + img_2d.min()
    low, high = WL_to_LH(window, level)

    if title is None:
        title = 'Image slice and k-space slice'

    plt.figure(figsize=(15,6))
    plt.suptitle(title)
    plt.subplot(1,2,1)
    plt.imshow(img_2d, cmap='gray', vmin=low, vmax=high, 
            interpolation='bilinear', origin='lower')
    plt.colorbar()

    # Fourier and shift
    img_f = fft2(img_2d)
    img_fs = fftshift(img_f,dim=(-2,-1))
    img_fs = img_fs.abs().log()

    if img_fs.min() == -float('Inf'):
        print('supressing -Inf')
        img_fs[img_fs == -float('Inf')] = 0
    if window_f is None:
        window_f = img_fs.max() - img_fs.min()
    if level_f is None:
        level_f = window_f / 2 + img_fs.min()
    low_f, high_f = WL_to_LH(window_f, level_f)

    # plot FT
    plt.subplot(1,2,2)
    plt.imshow(img_fs, cmap='gray', vmin=low_f, vmax=high_f, 
            interpolation='bilinear', origin='lower')
    plt.colorbar()
    plt.show()


##############################################################

class ReCompose(Compose):
    """ Version of Compose allowing use to append a transform to the pipeline. """
    
    def __init__(self, transforms: Optional[Union[Sequence[Callable], Callable]] = None):
        
        super().__init__(transforms)
        
    def append(self, transform: Callable = None):
        """
        Method to append a transform the Compose sequence.
        """
        if transform is not None:
            pipe = list(self.transforms)
            pipe.append(transform)
            self.transforms = tuple(pipe)
        else:
            pass
        
    def __add__(self, transforms: Union[Callable, List[Callable]]):
        """Returns new ReCompose with appended transform"""
        
        old = list(self.transforms)
        transforms = [transforms] if not isinstance(transforms, list) else transforms
        new = old + transforms
        return ReCompose(new)


class BratsValIterDataset(IterableDataset):
    """Super iterable dataset where each sample is a BraTS dataset/dataloader with
       different preprocessing pipeline.
       
       """
    
    def __init__(self, 
                 root_dir: str, 
                 cache_num: int = 0, 
                 transforms: dict=None, 
                 return_loader: bool=False):
        """
        Args:
            root_dir (str): path to Brats dataset to be used by the Decathlon class.
            cache_num (int): number of samples to store in cache.
            transforms (dict): Example, 
                {
                'sap10' : saltAndPepper(0.10),
                'sap20' : saltAndPepper(0.20)
                }
            return_loader (bool): set to True to yield dataloders instead of datasets.
        """

        self.root_dir = root_dir
        self.cache_num = cache_num
        self.transforms = transforms
        self.return_loader =return_loader
        self.pipe = Compose(
            [
            LoadImaged(keys=["image", "label"]),
            AsChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(keys=["image", "label"],
                     pixdim=(1.5, 1.5, 2.0),
                     mode=("bilinear", "nearest"),),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"])
        ])
        
    def __iter__(self):
        """Iterate over created datasets/dataloaders"""
        # apply pipeline with appended transform for each iteration step
        for t in self.transforms:

            pipe = ReCompose(self.pipe.transforms)
            pipe.append(self.transforms[t])

            ds = DecathlonDataset(root_dir=self.root_dir, task="Task01_BrainTumour",
                                  transform=pipe, section="validation", download=False,
                                  num_workers=4,cache_num=self.cache_num)

            _ , test_ds = random_split(ds, [48, 48], torch.Generator().manual_seed(0))
            
            if self.return_loader:
                test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=4)
                yield (t, test_loader)
            else:
                yield (t, test_ds)

    def __getitem__(self, key:str):
        """Returns item based on key of transforms dictionary"""
        
        pipe = ReCompose(self.pipe.transforms)
        pipe.append(self.transforms[key])
        
        ds = DecathlonDataset(root_dir=self.root_dir, task="Task01_BrainTumour",
                                  transform=pipe, section="validation", download=False,
                                  num_workers=4,cache_num=self.cache_num)
        _ , test_ds = random_split(ds, [48, 48], torch.Generator().manual_seed(0))
            
        if self.return_loader:
            test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=4)
            return test_loader
        else:
            return test_ds


##############################################################
from collections import defaultdict

class model_evaluation:
    '''Provides with regularly used tools to compare model
    performance across various datasets.

    Methods:

    * _load_UNet: loads instance of UNet model.

    * dataset_eval: computes metrics on given dataset dataloader.

    * add_eval: keeps record of metrics for given dataset.

    * save: save a pickled version of the class instance.
    
    * load_dict: load pickled version of a class instance.
    '''

    def __init__(self, model_path:str=None, instance_name: str = None, 
            in_channels: int = 4, out_channels: int = 3, gibbs_unet=False, spikes_unet=False):

        '''
        Args:
            model_name = name of saved model.pth. If given it will be
                        loaded on instantiation.
            instance_name = label for the class instantation

        '''
        self.gibbs_unet = gibbs_unet
        self.spikes_unet = spikes_unet
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model_path = model_path
        if model_path:
            if gibbs_unet:
                self.load_gibbs_unet()
            elif spikes_unet:
                self.load_spikes_unet()
            else:
                self.load_UNet()
        else:
            self.model = None
        self.instance_name = instance_name
        self.eval_dict = defaultdict(list)
        
    def load_gibbs_unet(self) -> None:
        """Loads a Gibbs_UNet model"""
        
        self.model = Gibbs_UNet().to(device)
        self.model.load_state_dict(torch.load(self.model_path))
        
    def load_spikes_unet(self) -> None:
        """Loads a Spikes_UNet model"""
        
        self.model = Spikes_UNet().to(device)
        self.model.load_state_dict(torch.load(self.model_path))

    def load_UNet(self) -> None:

        '''Function to load model.
        Args: model_path (string): name of saved model.pth

        Returns: instance of UNet with imported weights'''
        
        self.model = UNet(dimensions=3,
                     in_channels=self.in_channels,
                     out_channels=self.out_channels,
                     channels=(16, 32, 64, 128, 256),
                     strides=(2, 2, 2, 2),
                     num_res_units=2,).to(device)
        self.model.load_state_dict(torch.load(self.model_path))

    def dataset_eval_single(self, test_loader:DataLoader):
        ''' To evaluate model on given data using Dice metric and a single label.
        Args:
            test_data: test data loader
        Returns:
            mean metric
        '''
        if self.model is None:
            raise RuntimeError(f'current model is {self.model}.\
            Use {self.load_UNet.__name__} to load model.')

        self.model.eval()
        with torch.no_grad():
            dice_metric = DiceMetric(include_background=True, reduction="mean")
            post_trans = Compose([Activations(sigmoid=True),
                                  AsDiscrete(threshold_values=True)])

            metric_sum = 0.
            metric_count = 0

            for test_data in tqdm(test_loader):
                val_inputs, val_labels = (
                    test_data["image"].to(device),
                    test_data["label"].to(device),
                )
                val_outputs = self.model(val_inputs)
                val_outputs = post_trans(val_outputs)
                # compute overall mean dice
                value, not_nans = dice_metric(y_pred=val_outputs, y=val_labels)
                not_nans = not_nans.item()
                metric_count += not_nans
                metric_sum += value.item() * not_nans
            
            metric = metric_sum / metric_count
        return metric

    def dataset_eval_multi(self, test_loader:DataLoader):
        ''' To evaluate model on given data using Dice metric. It assumes 3-multilabel.
        Args:
            test_data: test data loader
        Returns:
            mean metrics
        '''
        if self.model is None:
            raise RuntimeError(f'current model is {self.model}.\
            Use {self.load_UNet.__name__} to load model.')

        self.model.eval()
        with torch.no_grad():
            dice_metric = DiceMetric(include_background=True, reduction="mean")
            post_trans = Compose([Activations(sigmoid=True),
                                  AsDiscrete(threshold_values=True)])

            metric_sum = 0.
            metric_sum_tc = 0.
            metric_sum_wt = 0.
            metric_sum_et = 0.

            metric_count = 0
            metric_count_tc  = 0
            metric_count_wt = 0
            metric_count_et = 0

            for test_data in tqdm(test_loader):
                val_inputs, val_labels = (
                    test_data["image"].to(device),
                    test_data["label"].to(device),
                )
                val_outputs = self.model(val_inputs)
                val_outputs = post_trans(val_outputs)
                # compute overall mean dice
                value, not_nans = dice_metric(y_pred=val_outputs, y=val_labels)
                not_nans = not_nans.item()
                metric_count += not_nans
                metric_sum += value.item() * not_nans
                # compute mean dice for TC
                value_tc, not_nans = dice_metric(
                    y_pred=val_outputs[:, 0:1], y=val_labels[:, 0:1]
                )
                not_nans = not_nans.item()
                metric_count_tc += not_nans
                metric_sum_tc += value_tc.item() * not_nans
                # compute mean dice for WT
                value_wt, not_nans = dice_metric(
                    y_pred=val_outputs[:, 1:2], y=val_labels[:, 1:2]
                )
                not_nans = not_nans.item()
                metric_count_wt += not_nans
                metric_sum_wt += value_wt.item() * not_nans
                # compute mean dice for ET
                value_et, not_nans = dice_metric(
                    y_pred=val_outputs[:, 2:3], y=val_labels[:, 2:3]
                )
                not_nans = not_nans.item()
                metric_count_et += not_nans
                metric_sum_et += value_et.item() * not_nans

            metric = metric_sum / metric_count
            metric_tc = metric_sum_tc / metric_count_tc
            metric_wt = metric_sum_wt / metric_count_wt
            metric_et = metric_sum_et / metric_count_et

        return metric, metric_et, metric_tc, metric_wt

    def add_eval(self, name:str, test_loader:DataLoader, data_dict:dict=None) -> None:
        '''Method to add evaluation to the attribute
        eval_dict.

        Args:
            name = string to label evaluation
            test_loader = dataloader of data to test
            data_dict = dictionary of type {name:test_loader}. If this argument is
                        passed, the other arguments are ignored.
        '''
        if self.gibbs_unet or self.spikes_unet:
            if data_dict == None:
                self.eval_dict[name] = self.dataset_eval_single(test_loader)
            else:
                for name in data_dict:
                    self.eval_dict[name] = self.dataset_eval_single(data_dict[name])
        else:
        # work with unets   
            if self.out_channels > 1:
                if data_dict == None:
                    self.eval_dict[name] = self.dataset_eval_multi(test_loader)
                else:
                    for name in data_dict:
                        self.eval_dict[name] = self.dataset_eval_multi(data_dict[name])
            else:
                if data_dict == None:
                    self.eval_dict[name] = self.dataset_eval_single(test_loader)
                else:
                    for name in data_dict:
                        self.eval_dict[name] = self.dataset_eval_single(data_dict[name])

    def save(self):
        '''save a the state the dictionary in a pickle'''
        with open(self.instance_name+'.pickle', 'wb') as f:
            d = self.__dict__.copy()
            d['model'] = None
            pickle.dump(d, f)

    def load_dict(self, filename:str):
        '''load instance of class'''

        # use whether a model name was given at instantiation
        # to decide whether the model should be loaded too.
        original_model_path = self.model_path

        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))
        if original_model_path:
            self.load_UNet(self.model_path)

            
########################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)