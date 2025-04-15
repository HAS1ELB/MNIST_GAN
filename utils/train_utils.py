import torch
import torchvision
from torchvision.utils import save_image
import os

def save_samples(generator, epoch, noise, folder="outputs"):
    generator.eval()
    with torch.no_grad():
        fake = generator(noise).reshape(-1, 1, 28, 28)
        fake = (fake + 1) / 2  # Remettre en [0,1]
        save_image(fake, os.path.join(folder, f"epoch_{epoch}.png"))
    generator.train()
