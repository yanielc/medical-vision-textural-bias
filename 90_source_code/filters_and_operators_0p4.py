# Code to design circular masks in 2d and 3d, allowing for 
# channel/batch dimensions.

import numpy as np

import torch
from torch.fft import (fftn, fft2, 
                       fftshift, ifftshift, 
                       ifft2, ifftn)

from monai.transforms import Transform, Randomizable

from math import floor
from typing import Union, List, Tuple, Any, Hashable, Dict, Generator
import collections.abc
#############################################################


def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    if isinstance(obj, torch.Tensor):
        return int(obj.dim()) > 0  # a 0-d tensor is not iterable
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str)


def ensure_tuple(vals: Any) -> Tuple[Any, ...]:
    """
    Returns a tuple of `vals`.
    """
    if not issequenceiterable(vals):
        vals = (vals,)

    return tuple(vals)

class RandomizableTransform(Randomizable, Transform):
    """
    An interface for handling random state locally, currently based on a class variable `R`,
    which is an instance of `np.random.RandomState`.
    This is mainly for randomized data augmentation transforms. For example::

        class RandShiftIntensity(RandomizableTransform):
            def randomize():
                self._offset = self.R.uniform(low=0, high=100)
            def __call__(self, img):
                self.randomize()
                return img + self._offset

        transform = RandShiftIntensity()
        transform.set_random_state(seed=0)

    """

    def __init__(self, prob: float = 1.0, do_transform: bool = True):
        self._do_transform = do_transform
        self.prob = min(max(prob, 0.0), 1.0)

    def randomize(self, data: Any) -> None:
        """
        Within this method, :py:attr:`self.R` should be used, instead of `np.random`, to introduce random factors.

        all :py:attr:`self.R` calls happen here so that we have a better chance to
        identify errors of sync the random state.

        This method can generate the random factors based on properties of the input data.
        """
        self._do_transform = self.R.rand() < self.prob



class MapTransform(Transform):
    """
    A subclass of :py:class:`monai.transforms.Transform` with an assumption
    that the ``data`` input of ``self.__call__`` is a MutableMapping such as ``dict``.

    The ``keys`` parameter will be used to get and set the actual data
    item to transform.  That is, the callable of this transform should
    follow the pattern:

        .. code-block:: python

            def __call__(self, data):
                for key in self.keys:
                    if key in data:
                        # update output data with some_transform_function(data[key]).
                    else:
                        # raise exception unless allow_missing_keys==True.
                return data

    Raises:
        ValueError: When ``keys`` is an empty iterable.
        TypeError: When ``keys`` type is not in ``Union[Hashable, Iterable[Hashable]]``.

    """

    def __init__(self, keys: Union[str, List[str]], allow_missing_keys: bool = False) -> None:
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        self.allow_missing_keys = allow_missing_keys
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")

    def __call__(self, data):
        """
        ``data`` often comes from an iteration over an iterable,
        such as :py:class:`torch.utils.data.Dataset`.

        To simplify the input validations, this method assumes:

        - ``data`` is a Python dictionary
        - ``data[key]`` is a Numpy ndarray, PyTorch Tensor or string, where ``key`` is an element
          of ``self.keys``, the data shape can be:

          #. string data without shape, `LoadImaged` transform expects file paths
          #. most of the pre-processing transforms expect: ``(num_channels, spatial_dim_1[, spatial_dim_2, ...])``,
             except that `AddChanneld` expects (spatial_dim_1[, spatial_dim_2, ...]) and
             `AsChannelFirstd` expects (spatial_dim_1[, spatial_dim_2, ...], num_channels)
          #. most of the post-processing transforms expect
             ``(batch_size, num_channels, spatial_dim_1[, spatial_dim_2, ...])``

        - the channel dimension is not omitted even if number of channels is one

        Raises:
            NotImplementedError: When the subclass does not override this method.

        returns:
            An updated dictionary version of ``data`` by applying the transform.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


    def key_iterator(
        self,
        data: Dict[Hashable, Any],
        *extra_iterables,
    ) -> Generator:
        """
        Iterate across keys and optionally extra iterables. If key is missing, exception is raised if
        `allow_missing_keys==False` (default). If `allow_missing_keys==True`, key is skipped.

        Args:
            data: data that the transform will be applied to
            extra_iterables: anything else to be iterated through
        """
        # if no extra iterables given, create a dummy list of Nones
        ex_iters = extra_iterables if extra_iterables else [[None] * len(self.keys)]

        # loop over keys and any extra iterables
        _ex_iters: List[Any]
        for key, *_ex_iters in zip(self.keys, *ex_iters):
            # all normal, yield (what we yield depends on whether extra iterables were given)
            if key in data.keys():
                yield (key,) + tuple(_ex_iters) if extra_iterables else key
            # if missing keys not allowed, raise
            elif not self.allow_missing_keys:
                raise KeyError(f"Key was missing ({key}) and allow_missing_keys==False")

##################################################################################

class disk_mask():
    '''
    Class to generate and apply mask with a circular boundary.

    '''

    def __init__(self, k_tensor:torch.tensor, r:  float = 2, 
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

        # instatiate mask holder array and reshape to (Bunch,D,H,W)
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
                 inside_off:bool=False, prob:float=0.5, allow_missing_keys=False) -> None:

        '''
        keys = 'image', 'label', or ['image', 'label'] depending on which data
                you need to transform
        r = radius defining disk mask in the fourier space. If a list [a,b]
            is given, the radius will be sampled from [a,b).
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
                d[key] = self.inv_shift_fourier(k)
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
        three dimensions are transformed. At the end we take the 
        real part to remove imaginary leakage.
        '''
        return ifftn(ifftshift(k, dim=(-3,-2,-1)), 
                     dim=(-3,-2,-1), norm='backward').real


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
        
