import torch
import torch.nn as nn
from monai.transforms import Transform


class FourierTransform:
    """This class stores fourier mappings we use in various artifact transforms"""
    
    @staticmethod
    def shift_fourier(x: torch.Tensor, n_dims: int) -> torch.tensor:
        """
        Applies fourier transform and shifts its output.
        Only the spatial dimensions get transformed.

        Args:
            x (torch.tensor): tensor to fourier transform.
        """
        out: torch.tensor = torch.fft.fftshift(torch.fft.fftn(x, dim=tuple(range(-n_dims, 0))), 
                                               dim=tuple(range(-n_dims, 0)))
        return out
    
    @staticmethod
    def inv_shift_fourier(k: torch.Tensor, n_dims: int) -> torch.tensor:
        """
        Applies inverse shift and fourier transform. Only the spatial
        dimensions are transformed.
        """
        out: torch.tensor = torch.fft.ifftn(
            torch.fft.ifftshift(k, dim=tuple(range(-n_dims, 0))), dim=tuple(range(-n_dims, 0))
        ).real
        return out
    

class RandZF(Transform, FourierTransform):
    """
    Apply random mask in fourier space.

    Each pixel is assigned a value uniformly drawn from [0,1]. A mask is 
    constructed for a given  value p with 0 <= p <= 1. If the pixel probability
    is > p, then the original pixel intensity is kept in place. Otherwise, it is
    set to zero.
    """

    def __init__(self, p:float=0):
        
        """
        Args:
            p (float): parameter in [0,1]. Fraction of pixels that will be
                modified. p = 0 corresponds to the identity transformation.
        """
        self.p = min(max(0,p),1.)
        if p < 0 or p > 1:
            warnings.warn(f'Setting p to {self.p}.')

    def __call__(self, img):
        
        n_dims = len(img.size()[1:])
        k = self.shift_fourier(img, n_dims)
        k = self.rand_mask(k)
        img = self.inv_shift_fourier(k, n_dims)
        return img


    def rand_mask(self, k:torch.tensor):
        """
        Applies random mask on k-data.

        Returns:
            k (torch.tensor): volume with mask.
        """
        mask = torch.rand(k.size()) # mask[i,j,k] takes value in [0,1]
        k = k.clone()
        k[mask <= self.p] = 0
        return k

    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
