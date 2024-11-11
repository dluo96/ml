"""Backward process i.e. the denoising process.
    - We use a simple form of a U-Net to predict the noise in each sampled image.
    - The input is a noisy image, the output of the model is the predicted noise
      in the image.
    - Because the parameters are shared across time, we must tell the network in
      which timestep we are: the timestep `t` is positionally encoded.
    - We output one single value (mean), because the variance is fixed.
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
        self, in_ch: int, out_ch: int, time_emb_dim: int, upsampling: bool = False
    ) -> None:
        """Block used in U-net.

        It is either a downsampling block or an upsampling block.
        - Downsampling reduces the spatial dimensions of the input while capturing
            important features, enabling the network to learn high-level patterns.
        - Upsampling gradually restores the spatial dimensions, allowing the network
            to reconstruct a detailed output while preserving learned features from
            downsampling.

        Args:
            in_ch: number of input channels.
            out_ch: number of output channels.
            time_emb_dim: the dimensionality of the embedding space for the positional
                encoding of time steps.
            upsampling: whether the block is used for upsampling. If False, the block
                is for downsampling.
        """
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if upsampling:
            # `2 * in_ch` is needed because of skip connection
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass of the downsampling/upsampling block.

        Args:
            x: embedding tensor of shape (B, C, H, W)
            t: the timestep of the sample in question.

        Returns:
            Embedding tensor.
        """
        # First convolution, ReLU, and batch normalisation
        h = self.bn1(F.relu(self.conv1(x)))

        # Time embedding
        time_emb = F.relu(self.time_mlp(t))  # (1, time_emb_dim)

        # Extend last 2 dimensions to get shape (1, time_emb_dim, 1, 1)
        time_emb = time_emb[(...,) + (None,) * 2]

        h = h + time_emb  # Add time channel (broadcasting occurs)
        h = self.bn2(F.relu(self.conv2(h)))

        # Upsample/downsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    """Positional encoding is needed because the U-Net uses the same network parameters
    regardless of the timestep `t` in question.
    """

    def __init__(self, dim: int) -> None:
        """Positional encoding.

        Args:
            dim: dimensionality of the embedding space.
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """Compute a positional encoding for the provided timestep `t`.

        Args:
            t: the timestep indicating the amount of noise. In {1, 2, ..., T}.

        Returns:
            1D tensor representing the positional encoding of the timestep `t`.
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class Unet(nn.Module):
    """A simplified variant of the U-net architecture."""

    def __init__(self) -> None:
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsampling
        self.downsampling_blocks = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1], time_emb_dim)
                for i in range(len(down_channels) - 1)
            ]
        )

        # Upsampling
        self.upsampling_blocks = nn.ModuleList(
            [
                Block(up_channels[i], up_channels[i + 1], time_emb_dim, upsampling=True)
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
        t_encoded = self.time_mlp(t)

        # Initial convolution
        x = self.conv0(x)

        # U-net: downsampling followed by upsampling.
        residual_inputs = []  # Use a stack to store residuals
        for down_blocks in self.downsampling_blocks:
            x = down_blocks(x, t_encoded)
            residual_inputs.append(x)
        for up_block in self.upsampling_blocks:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up_block(x, t_encoded)

        return self.output(x)


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
