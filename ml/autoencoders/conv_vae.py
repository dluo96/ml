import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.tensor import Tensor


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        # Encoder: (B, 16, 14, 14) -> (B, 32, 7, 7) -> (B, 64, 1, 1)
        self.enc_conv1 = nn.Conv2d(1, 16, 3, 2, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, 3, 2, padding=1)
        self.enc_conv3_mu = nn.Conv2d(32, 64, 7)
        self.enc_conv3_log_var = nn.Conv2d(32, 64, 7)

        # Decoder: # (B, 32, 7, 7) -> (B, 16, 14, 14) -> (B, 1, 28, 28)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, 7)
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, 3, 2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(16, 1, 3, 2, padding=1, output_padding=1)

    def encoder(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        mu = F.relu(self.enc_conv3_mu(x))
        log_var = F.relu(self.enc_conv3_log_var(x))
        return mu, log_var

    def sample_z(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(log_var / 2)

        # Reparametrization trick
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon

        return z

    def decoder(self, z: Tensor) -> Tensor:
        z = F.relu(self.dec_conv1(z))
        z = F.relu(self.dec_conv2(z))
        out = F.sigmoid(self.dec_conv3(z))
        return out

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        output = self.decoder(z)
        return output, z, mu, log_var
