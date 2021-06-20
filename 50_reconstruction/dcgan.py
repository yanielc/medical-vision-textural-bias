from monai.data import DataLoader

from brats_data import train_ds, val_ds
from networks import Generator, Discriminator, weights_init

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import matplotlib.pyplot as plt


# size of latent vector
nz = 100

# Batch size during training
batch_size = 4

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# device
device = torch.device("cuda:0")



################

# dataloaders
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)


################


# Instatiate networks 

D =  Discriminator().to(device)
G = Generator().to(device)

# initialize weights with normal dist
G.apply(weights_init)
D.apply(weights_init)

# Loss
criterion = nn.BCELoss() # TODO isn't there a trick for a better loss?

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

for epoch in range(num_epochs):
    # do for each batch
    for i, data in enumerate(train_loader):

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        D.zero_grad()
        # using real data
        real_batch = data["image"].to(device)
        b_size = real_batch.size(0)
        label = torch.full((b_size,1), real_label, dtype=torch.float, device=device)
        output = D(real_batch).squeeze(-1).squeeze(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_data_mean = output.mean().item()

        # using synthetic data
        # probe latent space and generate image
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = G(noise)
        label.fill_(fake_label)
        # classify fake image. Use detach to avoid updating G at this point.
        output = D(fake.detach()).squeeze(-1).squeeze(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        # compute error as sum of over real and synthetic images
        errD = errD_real + errD_fake
        D_fake_data_mean = output.mean().item()
        # update D
        optimizerD.step()

        ###
        # (2) Update G network: maximize log(D(G(z)))
        G.zero_grad()
        label.fill_(real_label)
        output = D(fake_label).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_fake_data_mean_2 = output.mean().item()
        # update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 50 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1