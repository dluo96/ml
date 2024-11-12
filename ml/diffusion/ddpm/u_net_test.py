import unittest

import torch
import torch.nn as nn

from ml.diffusion.ddpm.dataset import IMG_SIZE
from ml.diffusion.ddpm.u_net import Block


class TestBlock(unittest.TestCase):
    def test_init(self):
        block = Block(3, 6, 4, upsampling=True)
        assert isinstance(block.transform, nn.ConvTranspose2d)

        block = Block(3, 6, 4, upsampling=False)
        assert isinstance(block.transform, nn.Conv2d)

    def test_forward__upsampling(self):
        # Start with 64 channels and height and width of 1
        B, C, H, W = 3, 64, 1, 1
        d_embd_time = 7
        x = torch.randn(B, C, H, W)
        residual_x = torch.randn(B, C, H, W)
        x = torch.cat([x, residual_x], dim=1)  # (B, C, H, W) -> (B, 2C, H, W)
        t = torch.randint(0, 10, size=(B, d_embd_time)).float()

        # Create upsamling block (reduces channels and increases height and width)
        out_channels = 32
        assert out_channels < C, "Output channels must be less than input channels"
        block = Block(
            in_channels=C,
            out_channels=out_channels,
            d_embd_time=d_embd_time,
            upsampling=True,
        )

        # Check output
        out = block(x, t)
        assert out.shape == (B, out_channels, 2 * W, 2 * H)

    def test_forward__downsampling(self):
        block = Block(3, 6, 4, upsampling=False)
        B = 4
