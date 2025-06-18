# py libs
import os
import sys
import argparse
import numpy as np
from PIL import Image

# pytorch libs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image

# local libs
from nets.commons import Weights_Normal, VGG19_PercepLoss
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from utils.data_utils import GetTrainingPairs, GetValImage

# get configs and training options
parser = argparse.ArgumentParser()

# Add arguments for configuration values
parser.add_argument("--dataset_name", type=str, default="euvp", help="A")
parser.add_argument("--dataset_path", type=str, default="D:/int/FUnIE-GAN-master/data/test", help="Path to the dataset")
parser.add_argument("--img_width", type=int, default=256, help="Width of the input images")
parser.add_argument("--img_height", type=int, default=256, help="Height of the input images")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--val_interval", type=int, default=1000, help="Interval for validation during training")
parser.add_argument("--ckpt_interval", type=int, default=10, help="Interval for saving model checkpoints")

# Other training parameters
parser.add_argument("--epoch", type=int, default=0, help="Which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=201, help="Number of epochs for training")
parser.add_argument("--batch_size", type=int, default=8, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="Adam optimizer: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="Adam optimizer: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="Adam optimizer: decay of 2nd order momentum")

args = parser.parse_args()

# Extract arguments
dataset_name = args.dataset_name
dataset_path = args.dataset_path
img_width = args.img_width
img_height = args.img_height
channels = args.channels
val_interval = args.val_interval
ckpt_interval = args.ckpt_interval
epoch = args.epoch
num_epochs = args.num_epochs
batch_size = args.batch_size
lr_rate = args.lr
lr_b1 = args.b1
lr_b2 = args.b2

# Create directories for model and validation data
samples_dir = os.path.join("samples/FunieGAN/", dataset_name)
checkpoint_dir = os.path.join("checkpoints/FunieGAN/", dataset_name)
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# FunieGAN specifics: loss functions and patch-size
Adv_cGAN = nn.MSELoss()
L1_G = nn.L1Loss()  # similarity loss (L1)
L_vgg = VGG19_PercepLoss()  # content loss (VGG)
lambda_1, lambda_con = 7, 3  # Loss weightings
patch = (1, img_height // 16, img_width // 16)  # PatchGAN patch size

# Initialize generator and discriminator
generator = GeneratorFunieGAN()
discriminator = DiscriminatorFunieGAN()

# Check if CUDA is available
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    Adv_cGAN = Adv_cGAN.cuda()
    L1_G = L1_G.cuda()
    L_vgg = L_vgg.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Initialize weights or load pretrained models
if args.epoch == 0:
    generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)
else:
    generator.load_state_dict(torch.load(f"checkpoints/FunieGAN/{dataset_name}/generator_{epoch}.pth"))
    discriminator.load_state_dict(torch.load(f"checkpoints/FunieGAN/{dataset_name}/discriminator_{epoch}.pth"))
    print(f"Loaded model from epoch {epoch}")

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))

# Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)

val_dataloader = DataLoader(
    GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir="validation"),
    batch_size=4,
    shuffle=True,
    num_workers=1,
)

# Training loop
for epoch in range(epoch, num_epochs):
    for i, batch in enumerate(dataloader):
        imgs_distorted = Variable(batch["A"].type(Tensor))
        imgs_good_gt = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_distorted.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_distorted.size(0), *patch))), requires_grad=False)

        # Train Discriminator
        optimizer_D.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_real = discriminator(imgs_good_gt, imgs_distorted)
        loss_real = Adv_cGAN(pred_real, valid)
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        loss_fake = Adv_cGAN(pred_fake, fake)
        loss_D = 0.5 * (loss_real + loss_fake) * 10.0  # Scale by 10 for stability
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        loss_GAN = Adv_cGAN(pred_fake, valid)  # GAN loss
        loss_1 = L1_G(imgs_fake, imgs_good_gt)  # Similarity loss
        loss_con = L_vgg(imgs_fake, imgs_good_gt)  # Content loss
        loss_G = loss_GAN + lambda_1 * loss_1 + lambda_con * loss_con
        loss_G.backward()
        optimizer_G.step()

        # Log progress
        if not i % 50:
            sys.stdout.write(f"\r[Epoch {epoch}/{num_epochs}: batch {i}/{len(dataloader)}] [D Loss: {loss_D.item():.3f}, G Loss: {loss_G.item():.3f}, Adv Loss: {loss_GAN.item():.3f}]")

        # Save images at validation intervals
        batches_done = epoch * len(dataloader) + i
        if batches_done % val_interval == 0:
            imgs = next(iter(val_dataloader))
            imgs_val = Variable(imgs["val"].type(Tensor))
            imgs_gen = generator(imgs_val)
            img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
            save_image(img_sample, f"samples/FunieGAN/{dataset_name}/{batches_done}.png", nrow=5, normalize=True)

    # Save model checkpoints at intervals
    if epoch % ckpt_interval == 0:
        torch.save(generator.state_dict(), f"checkpoints/FunieGAN/{dataset_name}/generator_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"checkpoints/FunieGAN/{dataset_name}/discriminator_{epoch}.pth")
