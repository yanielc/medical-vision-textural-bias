# Code to design circular masks in 2d and 3d, allowing for 
# channel/batch dimensions.

import numpy as np

import torch
from torch.fft import (fftn, fft2, 
                       fftshift, ifftshift, 
                       ifft2, ifftn)

from monai.transforms import Transform, MapTransform, RandomizableTransform, Randomizable
from monai.config import KeysCollection

from math import floor

from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import warnings


############################################################################


class SelectChanneld(MapTransform):
    """
    Transform to keep one channel. It assumes the leading
    dimension is the channel index: [C,H,W,D].

    Args:
        keys: 'image', 'label', or ['image', 'label'] depending on which data
                you need to transform.
        chan_num (int): channel to keep. Provide an int to select the same
            channel for all provided keys, or a tuple to select different
            channels per given key. A 1-tuple has the same behavior as an int.
    """

    def __init__(self, keys, chan_num: Union[int,Sequence[int]], allow_missing_keys=False):

        self.chan_num = chan_num
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):

        d = dict(data)
        if isinstance(self.chan_num, Sequence):
            if len(self.chan_num) > 1:
                for i, key in zip(self.chan_num, self.key_iterator(d)):
                    if d[key].shape[0] - 1 < i:
                        raise AssertionError(f'Provided channel index {i} larger than max channel index for key = {key}')
                    d[key] = d[key][i][None,:] # using None to keep the channel axis
            else:
                for key in self.key_iterator(d):
                    d[key] = d[key][self.chan_num[0]][None,:]
        else:
            for key in self.key_iterator(d):
                d[key] = data[key][self.chan_num][None,:]
        return d


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d

############################################################################
class disk_mask():
    '''
    Class to generate and apply mask with a circular boundary.

    '''

    def __init__(self, k_tensor:torch.tensor, r: float = 2, 
                 dim:int=2, inside_off=True):

        '''
        Args:
        k_tensor = image in k space.
        r = radius for mask.
        dim = dimension of image (2 or 3).
        inside_off  = set True to mask radius < r,
                           set False to mask radius > r. 
        '''

        self.r = r
        self.dim = dim
        self.inside_off = inside_off
        self.last_dims = k_tensor.size(-1)
        
        if self.dim == 2:
            self.binary_mask = self.binary_mask_2d(k_tensor)
        elif self.dim == 3:
            self.binary_mask = self.binary_mask_3d(k_tensor)
        else:
            print('Only 2- and 3-dimensional images.')


    def binary_mask_2d(self, k_tensor) -> torch.tensor:
        '''To build a 2d mask using the last two dimensions'''

        # instatiate mask holder array and reshape to (bunch,H,W)
        mask = torch.zeros(k_tensor.size())
        mask = mask.reshape(-1,k_tensor.size(-2), k_tensor.size(-1))

        # create boolean disk centered at image's center, and
        # with given radius
        center = (floor(k_tensor.size(-2) / 2),
                  floor(k_tensor.size(-1) / 2))
        
        axes = (torch.arange(0, k_tensor.size(-2)),
                torch.arange(0, k_tensor.size(-1)))
        
        select = ( (axes[0][:,None] - center[0])**2 + 
                   (axes[1][None,:] - center[1])**2 ) < self.r**2
                   
        # add batch dimensions
        select = select.unsqueeze(0).repeat_interleave(mask.size(0),0)

        # create binary mask
        mask[select] = 1
        if self.inside_off:
            mask = 1 - mask
        mask = mask.reshape(k_tensor.size())
        return mask

    
    def binary_mask_3d(self, k_tensor) -> torch.tensor:
        '''To build a 3d mask using the last three dimensions '''

        # instatiate mask holder array and reshape to (bunch,H,W)
        mask = torch.zeros(k_tensor.size())
        mask = mask.reshape(-1, k_tensor.size(-3), 
                                k_tensor.size(-2), 
                                k_tensor.size(-1))
        
        # create boolean disk centered at image's center, and
        # with given radius
        center = (floor(k_tensor.size(-3) / 2),
                  floor(k_tensor.size(-2) / 2),
                  floor(k_tensor.size(-1) / 2))
        
        axes = (torch.arange(0, k_tensor.size(-3)),
                torch.arange(0, k_tensor.size(-2)),
                torch.arange(0, k_tensor.size(-1)))   
                
        select = ( (axes[0][:,None,None] - center[0])**2 + 
                   (axes[1][None,:,None] - center[1])**2 + 
                   (axes[2][None,None,:] - center[2])**2 
                  ) < self.r**2

        # add batch dimensions
        select = select.unsqueeze(0).repeat_interleave(mask.size(0),0)

        # create binary mask
        mask[select] = 1
        if self.inside_off:
            mask = 1 - mask
        mask = mask.reshape(k_tensor.size())
        return mask


    def apply(self, k_tensor:torch.tensor) -> torch.tensor:
        '''Apply obtained mask on given tensor'''

        assert k_tensor.size(-1) == self.last_dims, f'Last dimension of input \
                                must be = {self.last_dims}'

        return k_tensor * self.binary_mask



class RandFourierDiskMaskd(RandomizableTransform,MapTransform):

    '''Monai-dictionary-style class to apply disk masks on
    FT of given data.'''

    def __init__(self, keys: Union[str, List['str']], r: Union[float, List[float]] = float('Inf'), 
                 inside_off:bool=False, prob:float=0.5, 
                 allow_missing_keys:bool = False) -> None:

        '''
        keys = 'image', 'label', or ['image', 'label'] depending on which data
                you need to transform
        r = radius defining disk mask in the fourier space.
        inside_off = True to turn off the disk of radius r; False to turn off 
                the complement.
        prob = probability of applying the transform.
        allow_missing_keys = whether no keys are needed.
        '''

        assert prob <= 1 and prob >=0, 'prob must take values in [0,1]'
        self.r = r 
        self.inside_off = inside_off

        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)

    def __call__(self, data):

        d = dict(data)
        self.randomize()

        if not self._do_transform:
            return d
        else:
            for key in self.key_iterator(d):
                # FT
                k = self.shift_fourier(d[key])
                # mask
                mask = disk_mask(k, r=self.r, dim=3, inside_off=self.inside_off)
                k = mask.apply(k)
                # map back
                d[key] = self.inv_shift_fourier(k).real
            return d 

    def randomize(self) -> None:
        '''
        (1) Get random variable to apply the transform.
        (2) Get radius from uniform distribution.
         '''
        super().randomize(None)
        if type(self.r) == list:
            self.r = self.R.uniform(self.r[0], self.r[1])

    def shift_fourier(self, x:torch.tensor) -> torch.tensor:
        ''' 
        Applies fourier transform and shifts its output.
        Only the the last three dimensions get transformed: (x,y,z)-directions.
        
        Args: x[torch.tensor] = tensor to transform'''
        
        return fftshift(fftn(x, dim=(-3,-2,-1)), 
                        dim=(-3,-2,-1))

    def inv_shift_fourier(self, k:torch.tensor) -> torch.tensor:
        '''
        Applies inverse shift and fourier transform. Only the last
        three dimensions are transformed.
        '''
        return ifftn(ifftshift(k, dim=(-3,-2,-1)), 
                     dim=(-3,-2,-1), norm='backward')


########################################################################

class ellipsoid(Randomizable):

    '''(x-x_0)^2 / a^2 + (y-y_0)^2 / b^2 + (z-z_0)^2 / c^2 = 1'''

    def __init__(self, a:float, b:float, c:float):
    
        self.a = a
        self.b = b
        self.c = c

    def binary_mask_3d(self, k_tensor) -> torch.tensor:

        # instatiate mask holder array and reshape to (Bunch,D,H,W)
        mask = torch.zeros(k_tensor.size())

        mask = mask.reshape(-1, k_tensor.size(-3), 
                                k_tensor.size(-2), 
                                k_tensor.size(-1))  
        
        center = self._get_3d_center(k_tensor)

        axes = self._get_three_axes(k_tensor)

        inner =  ( ( (axes[0][:,None,None] - center[0])**2 )/self.a**2 + 
                  ( (axes[1][None,:,None] - center[1])**2 )/self.b**2 + 
                   ( (axes[2][None,None,:] - center[2])**2 )/self.c**2 
                ) >.95

        outer =  ( ( (axes[0][:,None,None] - center[0])**2 )/self.a**2 + 
                   ( (axes[1][None,:,None] - center[1])**2 )/self.b**2 + 
                   ( (axes[2][None,None,:] - center[2])**2 )/self.c**2
                )< 1.05 
        select = torch.logical_and(inner, outer)


        # add batch dimensions
        select = select.unsqueeze(0).repeat_interleave(mask.size(0),0)

        # create binary mask
        mask[select] = 1
        mask = mask.reshape(k_tensor.size())
        return mask

    def _get_3d_center(self, data):
        '''Helper function to pick the center of the 3d volume'''
        center = (floor(data.size(-3) / 2),
                  floor(data.size(-2) / 2),
                  floor(data.size(-1) / 2))
        return center

    def _get_three_axes(self, data):

        axes = (torch.arange(0, data.size(-3)),
                torch.arange(0, data.size(-2)),
                torch.arange(0, data.size(-1))) 
        return axes


    def sample_ellipsoid(self, k_tensor):
        '''Samples from the locations with value 1 in the
        ellipsoid mask. Returns the indeces of obtained 
        location'''

        ellipsoid_mask = self.binary_mask_3d(k_tensor)
        ones_coords = ellipsoid_mask.nonzero()
        num_ones = len(ones_coords)
        idx = self.R.randint(0, num_ones)
        coord = tuple(ones_coords[idx].numpy())
        return coord


class RandPlaneWaves_ellipsoid(RandomizableTransform, MapTransform):

    def __init__(self, keys: Union[str, List['str']] ='image', 
                 a:float =10, b:float=10, c:float=10, intensity_value:float = 1,
                 prob:float=0.2, allow_missing_keys:bool=False):
        

        MapTransform.__init__(self,keys,allow_missing_keys)
        RandomizableTransform.__init__(self,prob=prob)

        self.ellipsoid = ellipsoid(a,b,c)
        self.intensity_value = intensity_value
        self.idx = None


    def __call__(self, data):

        d = dict(data)
        self.randomize(None)

        # using prob randomly apply
        if not self._do_transform:
            return d
        else:
            for key in self.key_iterator(d):
                # FT
                k = self.shift_fourier(d[key])
                # split phase and amplitude
                k_abs_log = k.abs().log()
                k_angle = k.angle()
                # highlight a point in each channel
                self.idx = self.ellipsoid.sample_ellipsoid(k_abs_log[0])
                k_abs_log[:, self.idx[0], self.idx[1], self.idx[2]] = self.intensity_value
                # put back together
                k_abs_new = k_abs_log.exp()
                k_new = k_abs_new * torch.exp(1j*k_angle)
                img_r = self.inv_shift_fourier(k_new)
                d[key] = img_r
            return d

                

    def shift_fourier(self, x:torch.tensor) -> torch.tensor:
        ''' 
        Applies fourier transform and shifts its output.
        Only the the last three dimensions get transformed: (x,y,z)-directions.
        
        Args: x[torch.tensor] = tensor to transform'''
        
        return fftshift(fftn(x, dim=(-3,-2,-1)), 
                        dim=(-3,-2,-1))
        
    def inv_shift_fourier(self, k:torch.tensor) -> torch.tensor:
        '''
        Applies inverse shift and fourier transform. Only the last
        three dimensions are transformed. At the end we take the 
        real part to remove imaginary leakage.
        '''
        return ifftn(ifftshift(k, dim=(-3,-2,-1)), 
                     dim=(-3,-2,-1), norm='backward').real
        

##############################################################################

class SaltAndPepper(MapTransform, RandomizableTransform):
    """
    Transform to apply salt and pepper noise.

    Each pixel is assigned a value uniformly drawn from [0,1]. A mask is 
    constructed for a given  value p with 0 <= p <= 1. If the pixel probability
    is > p, then the original pixel intensity is kept in place. If the pixel
    probability is <= p/2 then the pixel final amplitude is set to blackish. 
    If the pixel probability lies in [p/2,p] then its amplitude is set to 
    whiteish.

    """

    def __init__(self, p:float=0, keys: Union[str, List['str']] ='image', 
                 prob:float = 1., allow_missing_keys:bool = False):
        
        """
        Args:
            p (float): parameter in [0,1]. Fraction of pixels that will be
                modified. p = 0 corresponds to the identity transformation.
            keys (string, or list of strings): flags to apply transform to
                image and/or label.
            prob (float): probability of the transform being applied to any
                one volume.
        """
        self.p = min(max(0,p),1.)
        if p < 0 or p > 1:
            warnings.warn(f'Setting p to {self.p}.')

        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)


    def __call__(self, data):

        d = dict(data)
        self.randomize(None)

        if not self._do_transform:
            return d
        else:
            for key in self.key_iterator(d):
                d[key] = self.salt_and_pepper(d[key])
            return d


    def salt_and_pepper(self, x:torch.tensor):
        """
        Applies salt and pepper on input volume.

        Returns:
            x (torch.tensor): volume with salt and pepper noise.
        """
        mask = torch.rand(x.size()) # mask[i,j,k] takes value in [0,1]
        x = x.clone()

        # salt and pepper intensities
        MAX, MIN = x.max()/2, x.min()/2

        x[mask <= self.p/2] = MIN
        x[torch.logical_and(mask > self.p/2, mask <= self.p)] = MAX
        x[torch.logical_and(mask > self.p, mask != 1.)] = x[torch.logical_and(mask > self.p, mask != 1.)]

        return x



#############################################################################

class WrapArtifact(Transform):
    """
    Applies wrapping artifacts while keeping the size of the image
    fixed.

    Args:
        alpha (float): regulates amplitude of wraparound artifact 

    When called, it assumes the data has shape: (C,H,W) or (C,H,W,D).
    """

    def __init__(self, alpha: float = 0.5):

        self.alpha = alpha

    def __call__(self, img: torch.tensor):

        n_dims = len(img.shape[1:])
        # FT
        k = self._shift_fourier(img, n_dims)
        # reduce information
        k[:,1:k.size(1):2,:,:] = k[:,1:k.size(1):2,:,:] * self.alpha
        k[:,:,1:k.size(2):2,:] = k[:,:,1:k.size(2):2,:] * self.alpha
        k[:,:,:,1:k.size(3):2] = k[:,:,:,1:k.size(3):2] * self.alpha
        # map back
        img = self._inv_shift_fourier(k, n_dims)
        
        return img

    def _shift_fourier(self, x: torch.Tensor, n_dims: int) -> torch.tensor:
        """
        Applies fourier transform and shifts its output.
        Only the spatial dimensions get transformed.

        Args:
            x (torch.tensor): tensor to fourier transform.
        """
        out: torch.tensor = torch.fft.fftshift(torch.fft.fftn(x, dim=tuple(range(-n_dims, 0))), 
                                               dim=tuple(range(-n_dims, 0)))
        return out

    def _inv_shift_fourier(self, k: torch.Tensor, n_dims: int) -> torch.tensor:
        """
        Applies inverse shift and fourier transform. Only the spatial
        dimensions are transformed.
        """
        out: torch.tensor = torch.fft.ifftn(
            torch.fft.ifftshift(k, dim=tuple(range(-n_dims, 0))), dim=tuple(range(-n_dims, 0))
        ).real
        return out


class WrapArtifactd(MapTransform):
    """
    Dictionary version of WrapArtifact
    
    Args:
        alpha (float): regulates amplitude of wraparound artifact 

    When called, it assumes the data has shape: (C,H,W) or (C,H,W,D).
    """
    def __init__(self, keys: KeysCollection, alpha: float = 0.5, 
                 allow_missing_keys: bool = False):

        MapTransform.__init__(self, keys, allow_missing_keys)
        self.transform = WrapArtifact(alpha)

    def __call__(self, data: Mapping[Hashable, torch.tensor]):

        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d
