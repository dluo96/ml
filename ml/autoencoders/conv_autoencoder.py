import torch.nn as nn
import torch.nn.functional as F

from ml.tensor import Tensor


class ConvAutoencoder(nn.Module):
    """Minimal implementation of an autoencoder whose encoder and decoder are CNNs."""

    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder: increase number of channels, decrease the spatial dimensions
        # Each MNIST input image has shape (B, 1, 28, 28).
        self.enc_conv1 = nn.Conv2d(1, 16, 3, 2, padding=1)  # (B, 16, 14, 14)
        self.enc_conv2 = nn.Conv2d(16, 32, 3, 2, padding=1)  # (B, 32, 7, 7)
        self.enc_conv3 = nn.Conv2d(32, 64, 7)  # (B, 64, 1, 1)

        # Decoder: decrease the number of channels but increase the spatial dimensions
        # Input to decoder has shape (B, 64, 1, 1).
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, 7)  # (B, 32, 7, 7)
        self.dec_conv2 = nn.ConvTranspose2d(
            32, 16, 3, 2, padding=1, output_padding=1
        )  # (B, 16, 14, 14)
        self.dec_conv3 = nn.ConvTranspose2d(
            16, 1, 3, 2, padding=1, output_padding=1
        )  # (B, 1, 28, 28)

    def encoder(self, x: Tensor) -> Tensor:
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        z = F.relu(self.enc_conv3(x))
        return z

    def decoder(self, z: Tensor) -> Tensor:
        z = F.relu(self.dec_conv1(z))
        z = F.relu(self.dec_conv2(z))
        out = F.sigmoid(self.dec_conv3(z))
        return out

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out
