import torch.nn as nn
import torch.nn.functional as F

from ml.tensor import Tensor


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        # Encoder: increase number of channels, decrease the spatial dimensions
        # Each MNIST input image has shape (B, 1, 28, 28).
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, padding=1),  # (B, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, padding=1),  # (B, 32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),  # (B, 64, 1, 1)
        )

        # Decoder: decrease the number of channels but increase the spatial dimensions
        # Input to decoder has shape (B, 64, 1, 1).
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # (B, 32, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, 3, 2, padding=1, output_padding=1
            ),  # (B, 16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, 3, 2, padding=1, output_padding=1
            ),  # (B, 1, 28, 28)
            nn.Sigmoid(),  # Scale values to be in [0, 1]
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        output = self.decoder(z)
        return output
