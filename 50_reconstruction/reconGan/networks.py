import torch
import torch.nn as nn



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
            nn.InstanceNorm2d(num_features = nf),
            nn.PReLU(),
            nn.Conv2d(nf, nf // 2, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.InstanceNorm2d(num_features = nf // 2),
            nn.PReLU(),
            nn.Conv2d(nf // 2, nf, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.InstanceNorm2d(num_features = nf),
            nn.PReLU(),
        )
    
    def __call__(self, x):
        return self.main(x) + x


class ResidualEncoder(nn.Module):
    """Basic unit of the encoding arm.
    Args:
        in_chans (int): number of input channels.
    """
    def __init__(self, in_chans: int = 3, out_chans: int = 3):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size = 3, stride = 2, padding = 1, bias = True),
            nn.InstanceNorm2d(num_features = out_chans),
            nn.PReLU(),
            ResidualBlock(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.InstanceNorm2d(num_features = out_chans),
            nn.PReLU(),
        )

    def __call__(self, x):
        return self.main(x)

class ResidualDecoder(nn.Module):
    """Basic unit of the decoding arm.
    Args:
        in_chans (int): number of input channels
    """
    def __init__(self, in_chans: int = 3, out_chans: int = 3):
        
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(num_features = out_chans),
            nn.PReLU(),
            ResidualBlock(out_chans),
            nn.ConvTranspose2d(out_chans, out_chans, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(num_features = out_chans),
            nn.PReLU(),
        )

    def __call__(self, x):
        return self.main(x)

class ResUnetGenerator(nn.Module):
    """
    Generator in the ReconGan network. This is a convolutional autoencoder
    with residual units going down and up. It uses contrast-preserving 
    instance normalization layers.

    Args:
        in_chans: number of input channels
        nf: number of starting hidden features
    """
    def __init__(self, in_chans: int = 3, nf: int = 16):

        super().__init__()
        
        # encoder components
        self.e0 = ResidualEncoder(in_chans, nf*1) 
        self.e1 = ResidualEncoder(nf*1, nf*2)
        self.e2 = ResidualEncoder(nf*2, nf*4)
        self.e3 = ResidualEncoder(nf*4, nf*8)

        # decoder components
        self.d3 = ResidualDecoder(nf*8, nf*4)
        self.d2 = ResidualDecoder(nf*4, nf*2)
        self.d1 = ResidualDecoder(nf*2, nf*1)
        self.d0 = ResidualDecoder(nf*1, nf*1)

        self.final = nn.Conv2d(nf*1, in_chans, kernel_size = 3, stride = 1, padding = 1, bias = True)

    def forward(self, x):
        # assuming input size = B x in_chans x 128 x 128
        en1 = self.e0(x)    # size = B x nf*1 x 64 x 64
        en2 = self.e1(en1)  # size = B x nf*2 x 32 x 32
        en3 = self.e2(en2)  # size = B x nf*4 x 16 x 16
        en4 = self.e3(en3)  # size = B x nf*8 x 8 x 8

        de3 = self.d3(en4)
        de2 = self.d2(de3 + en3)
        de1 = self.d1(de2 + en2)
        de0 = self.d0(de1 + en1)

        out = self.final(de0) + x
        return out


class Discriminator(nn.Module):
    """
    Discriminator component of the ReconGan. Identical to encoding arm of the
    generator, followed by a fully connected convolution.
    """
    def __init__(self, in_chans: int = 3, nf: int = 16):
        
        super().__init__()
        
        self.main = nn.Sequential(
            # assuming input size = B x in_chans x 128 x 128
            ResidualEncoder(in_chans, nf*1), # size = B x nf*1 x 64 x 64
            ResidualEncoder(nf*1, nf*2), # size = B x nf*2 x 32 x 32
            ResidualEncoder(nf*2, nf*4), # size = B x nf*4 x 16 x 16
            ResidualEncoder(nf*4, nf*8), # size = B x nf*8 x 8 x 8
            # to output one scalar value
            nn.Conv2d(nf*8, 1, kernel_size = 8, stride=1, bias=True)
        )

    def forward(self, x):

        return self.main(x)







if __name__ == "__main__":

    from monai.data import DataLoader
    from brats_data import val_ds

    img = val_ds[0]["image"][None,:]
    # e = ResidualEncoder(2,2)
    # d = ResidualDecoder(2,2)

    print(img.size())
    # G = ResUnetGenerator(2, 16)
    # print(sum(p.numel() for p in G.parameters() if p.requires_grad))
    D = Discriminator(2, 16)
    out = D(img)
    print(out.size())
    # h = e(img)
    # print(h.size())
    # out = d(h)
    # print(out.size())
    # b = nn.Conv2d(1, 2, kernel_size = 3, stride = 2, padding = 1, bias = True)(img)
    # b = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)(img)
    # print(img.size())
    # print(b.size())



