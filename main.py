import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from utils.train_utils import save_samples
import os

# Hyperparams
lr = 2e-4
batch_size = 64
image_size = 28*28
noise_dim = 100
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [-1, 1]
])

dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

# Optimizers & Loss
criterion = nn.BCELoss()
opt_gen = optim.Adam(generator.parameters(), lr=lr)
opt_disc = optim.Adam(discriminator.parameters(), lr=lr)

# Create output dir
os.makedirs("outputs", exist_ok=True)
fixed_noise = torch.randn(64, noise_dim).to(device)

# Training loop
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, image_size).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake = generator(noise)
        disc_real = discriminator(real).view(-1)
        disc_fake = discriminator(fake.detach()).view(-1)
        lossD = criterion(disc_real, torch.ones_like(disc_real)) + \
                criterion(disc_fake, torch.zeros_like(disc_fake))
        opt_disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        ### Train Generator
        output = discriminator(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))  # Want fake to be classified as real
        opt_gen.zero_grad()
        lossG.backward()
        opt_gen.step()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {lossD.item():.4f} | Loss G: {lossG.item():.4f}")
    save_samples(generator, epoch+1, fixed_noise)
