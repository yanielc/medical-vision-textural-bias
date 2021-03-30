import numpy as np

import torch
from torch.fft import (fftn, fft2, 
                       fftshift, ifftshift, 
                       ifft2, ifftn)

import matplotlib.pyplot as plt

from typing import Union, List, Tuple


############################################################

# Display functions



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