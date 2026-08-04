"""Shared auxiliary decoder used by steps 3 (train) and 5 (decode)."""
import torch.nn as nn


class Decoder(nn.Module):
    """Mirror the GAGA Autoencoder.decoder sizing: latent -> 64 -> 128 -> out."""
    def __init__(self, latent_dim, out_dim, hidden_dims=(64, 128)):
        super().__init__()
        layers, prev = [], latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
