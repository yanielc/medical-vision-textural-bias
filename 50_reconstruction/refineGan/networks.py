import torch
import torch.nn as nn

# 
class ResidualBlock(nn.Module):
    """
    To be used in both in the encoder and decoder blocks. The skip connection
    alliviates the vanishing gradient problem.

    Args:
        nf (int): parametrizes numbers of filters for the block. Should be an even
            number.    
    """
    def __init__(self, nf: int = 2):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.Conv2d(nf, nf // 2, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.Conv2d(nf // 2, nf, kernel_size = 3, stride = 1, padding = 1, bias = True),
        )
    
    def __call__(self, x):
        return self.main(x) + x


class ResidualEncoder(nn.Module):
    """Basic unit of the encoding arm.
    Args:
        in_chans (int): number of input channels.
    """
    def __init__(self, in_chans: int = 2, out_chans):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size = 3, stride = 2, padding = 1, bias = True),
            ResidualBlock(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size = 3, stride = 1, padding = 1, bias = True),
        )

    def __call__(self, x):
        return self.main(x)

class ResidualDecoder(nn.Module):
    """Basic unit of the decoding arm.
    Args:
        in_chans (int): number of input channels
    """
    def __init__(self, in_chans: int, out_chans: int):
        
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=True),
            ResidualBlock(out_chans),
            nn.ConvTranspose2d(out_chans, out_chans, kernel_size=3, stride=2, padding=1, bias=True),
        )

    def __call__(self, x):
        return self.main(x)





if __name__ == "__main__":

    from monai.data import DataLoader
    from brats_data import val_ds
    a = ResidualEncoder(1)
    img = val_ds[0]["image"][None,:]
    # b = a(img)
    # b = nn.Conv2d(1, 2, kernel_size = 3, stride = 2, padding = 1, bias = True)(img)
    b = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)(img)
    print(img.size())
    print(b.size())



