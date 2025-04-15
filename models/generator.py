import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_dim=28*28):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh()  # [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)
