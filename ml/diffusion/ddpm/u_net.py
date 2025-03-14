"""Backward process i.e. the denoising process.
    - We use a simple form of a U-Net to predict the noise in each sampled image.
    - The input is a noisy image, the output of the model is the predicted noise
      in the image.
    - Because the parameters are shared across time, we must tell the network in
      which timestep we are: the timestep `t` is positionally encoded.
    - We output one single value (mean), because the variance is fixed.

References:
    - https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=buW6BaNga-XH
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ml.diffusion.ddpm.dataset import create_datasets
from ml.tensor import Tensor


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_embd_time: int,
        upsampling: bool = False,
    ) -> None:
        """Block used in U-net.

        It is either a downsampling block or an upsampling block.
        - Downsampling reduces the height and width but increases the number of
            channels. This allows the network to capture important features, enabling
            the network to learn high-level patterns.
        - Upsampling gradually grows the image back to its original size, while
            shrinking the number of channels. This allows the network to reconstruct
            a detailed output while preserving learned features from downsampling.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            d_embd_time: the dimensionality of the embedding space for the positional
                encoding of time steps.
            upsampling: whether the block is used for upsampling. If False, the block
                is for downsampling.
        """
        super().__init__()

        # Position embedding for time step
        self.time_mlp = nn.Linear(d_embd_time, out_channels)

        if upsampling:
            # `2 * in_ch` is due to residual connections during upsampling (recall
            # that U-net consists of downsampling followed by upsampling)
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, padding=1)

            # Transpose convolution is used for upsampling because it increases height
            # and width
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

            # Convolution is used for downsampling because it reduces height and width
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """Forward pass of downsampling (or upsampling) block.

        Args:
            x: embedding tensor of shape (B, C, H, W)
            t_emb: time embedding tensor of shape (B, d_embd_time)

        Returns:
            Embedding tensor.
        """
        h = self.bn1(F.relu(self.conv1(x)))  # (B, out_channels, H, W)

        # Position embedding for time step
        t_emb = F.relu(self.time_mlp(t_emb))  # (B, d_embd_time) -> (B, out_channels)
        t_emb = t_emb[:, :, None, None]  # (B, out_channels) -> (B, out_channels, 1, 1)

        # Add position embedding to hidden representation. Broadcasting occurs:
        # (B, out_channels, H, W) + (B, out_channels, 1, 1) -> (B, out_channels, H, W)
        h = h + t_emb

        # By design, the second convolution does not change the shape
        h = self.bn2(F.relu(self.conv2(h)))  # (B, out_channels, H, W)

        # Downsampling: (B, out_channels, H, W) -> (B, out_channels, H/2, W/2)
        # Upsampling: (B, out_channels, H, W) -> (B, out_channels, 2H, 2W)
        h = self.transform(h)  # (B, out_channels, H, W) -> (B, out_channels, 2H, 2W)

        return h


class SinPosEmbed(nn.Module):
    """Positional encoding is needed because the U-Net uses the same network parameters
    regardless of the timestep `t` in question.
    """

    def __init__(self, d_embd: int) -> None:
        super().__init__()
        self.d_embd = d_embd

    def forward(self, t: Tensor) -> Tensor:
        """Compute a positional embedding for the time step `t`. For full details, see
        Section 3.5 in https://arxiv.org/abs/1706.03762.

        Args:
            t: the time step indicating the amount of noise. In {1, 2, ..., T}.

        Returns:
            1D tensor representing the positional encoding of the timestep `t`.
        """
        device = t.device
        half_d_embd = self.d_embd // 2

        # Compute inverse frequencies 1/10000^(2i/d_embd)
        i = torch.arange(half_d_embd, device=device)  # (d_embd // 2,)
        inv_freqs = torch.exp(-2 * i / self.d_embd * math.log(10000))

        # Compute angles `pos * 1/10000^(2i/d_embd)` where `pos` is the time step `t`
        # (B, 1) * (1, d_embd // 2) -> (B, d_embd // 2)
        angles = t[:, None] * inv_freqs[None, :]

        # Concatenating (B, d_embd // 2) and (B, d_embd // 2) along the last dimension
        # results in shape (B, d_embd)
        pos_embeddings = torch.cat((angles.sin(), angles.cos()), dim=-1)

        return pos_embeddings


class Unet(nn.Module):
    """A simplified variant of the U-net architecture."""

    def __init__(self) -> None:
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)  # Downsampling increases channels
        up_channels = (1024, 512, 256, 128, 64)  # Upsampling decreases channels
        out_dim = 3
        d_embd_time = 32

        # Position embedding for time step followed by linear layer and ReLU
        self.time_mlp = nn.Sequential(
            SinPosEmbed(d_embd_time),
            nn.Linear(d_embd_time, d_embd_time),
            nn.ReLU(),
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsampling
        self.downsampling_blocks = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1], d_embd_time)
                for i in range(len(down_channels) - 1)
            ]
        )

        # Upsampling
        self.upsampling_blocks = nn.ModuleList(
            [
                Block(up_channels[i], up_channels[i + 1], d_embd_time, upsampling=True)
                for i in range(len(up_channels) - 1)
            ]
        )

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Do a forward pass of the neural network in the denoising process.

        Args:
            x: during training, this is usually a noisy version of some starting
                image `x_0` sampled at a timestep `t`.
            t: the timestep indicating how much noise has been added to the
                starting image `x_0`. Is in {1, 2, ..., T}.

        Returns:
            A prediction of the noise in the input `x`.
        """
        # Compute the positional encoding of the timestep
        t_emb = self.time_mlp(t)

        # Initial convolution
        x = self.conv0(x)

        # Use a stack to store residuals (needed for upsampling)
        residuals = []

        # U-net: downsampling followed by upsampling
        for down_block in self.downsampling_blocks:
            x = down_block(x, t_emb)
            residuals.append(x)
        for up_block in self.upsampling_blocks:
            # Pop stack to get residual x
            residual_x = residuals.pop()

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up_block(x, t_emb)

        # Output layer
        out = self.output(x)

        return out


if __name__ == "__main__":
    model = Unet()

    # Load example image
    train_data, test_data = create_datasets()
    dataloader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    image_and_label = next(iter(dataloader))
    image = image_and_label[0]
    timestep = torch.Tensor([3])

    # Pass a noised image through the neural network in the backward process
    pred_noise = model(image, timestep)
