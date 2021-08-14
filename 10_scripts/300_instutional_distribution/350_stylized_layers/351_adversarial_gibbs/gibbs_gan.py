# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

from tcga_data import train_loader, val_loader_inDist
from networks import ResUnetGenerator, Discriminator

SOURCE_CODE_PATH = '/homes/yc7620/Documents/medical-vision-textural-bias/source_code/'
import sys
sys.path.append(SOURCE_CODE_PATH)
from filters_and_operators import RandGibbsNoise
from utils import weights_init

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import numpy as np 
import matplotlib.pyplot as plt
import os

# Batch size during training
batch_size = 4

# Number of training epochs
num_epochs = 400  

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# cyclic loss parameters
alpha = 15
gamma = 0.1

# device
device = torch.device("cuda:0")

####################################

# dataloaders
# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)


################

# Gibbs noise
gibbs = RandGibbsNoise(1.0)

# Instatiate networks 
D = Discriminator(in_chans=2, nf=16).to(device)
G = ResUnetGenerator(in_chans=2, nf=16).to(device)

# initialize weights with normal dist
G.apply(weights_init)
D.apply(weights_init)

# Loss
criterion = nn.BCEWithLogitsLoss()
l2_loss = nn.MSELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
mean_G_losses =[]
mean_D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(train_loader, 0):
        data_batch = data["image"].to(device)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        D.zero_grad()
        # Format batch
        b_size = real_batch.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        real_batch = gibbs(data_batch)
        # Forward pass real batch through D
        output = D(real_batch).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of images with undersampled k-space
        downsampled_batch = gibbs(real_batch)
        # Generate fake image batch with G
        fake = G(downsampled_batch)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = D(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        G.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = D(fake).view(-1)
        
        # Adversarial loss: Calculate G's loss based on this output
        adv_loss = criterion(output, label)
        # Cyclic loss:
        img_consistency = l2_loss(real_batch, fake)
        
        # separate real and imaginary parts of freq component
        real_k = torch.fft.fftn(real_batch, dim=(-2,-1))
        real_k_r, real_k_i = real_k.real, real_k.imag
        
        fake_k = torch.fft.fftn(fake, dim=(-2,-1))
        fake_k_r, fake_k_i = fake_k.real, fake_k.imag
        
        freq_conistency = l2_loss(real_k_r, fake_k_r) + l2_loss(real_k_i, fake_k_i)
        
        cyclic_loss = alpha*img_consistency + gamma*freq_conistency
        # total G loss
        errG = adv_loss + cyclic_loss

        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

    # Check how the generator is doing by saving G's output on fixed_noise
    if (epoch % 5 == 0):
        with torch.no_grad():
            fake = G(downsampled_batch).detach().cpu()
        
            plt.figure(figsize=(20,20))
            plt.axis("off")
            plt.title(f"Images at epoch {epoch}")
            plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, n_row=3, normalize=True).cpu(),(1,2,0))[:,:,0])
            plt.savefig(f'images/generated_epoch{epoch}.png')

        iters += 1

    mean_D_losses.append(np.mean(D_losses))
    mean_G_losses.append(np.mean(G_losses))

# save generator
path = '/vol/bitbucket/yc7620/90_data/90_recon/'
torch.save(G.state_dict(), os.path.join(path, f"reconGan_G_epochs{num_epochs}.pth"))

# Grab a batch of real images from the dataloader
# real_batch = next(iter(train_loader))
# real_batch = real_batch["image"]
# # Plot the real images
# plt.figure(figsize=(15,15))
# plt.subplot(1,2,1)
# plt.axis("off")
# plt.title("Real Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0))[:,:,0])

# Plot the fake images from the last epoch
# plt.subplot(1,2,2)
# plt.axis("off")
# plt.title("Fake Images")
# plt.imshow(np.transpose(img_list[-1],(1,2,0))[:,:,0])
# plt.savefig(f'images_epochs{num_epochs}.png')
# plt.show()

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(mean_G_losses,label="G")
plt.plot(mean_D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"losses_{num_epochs}.png")
plt.show()
