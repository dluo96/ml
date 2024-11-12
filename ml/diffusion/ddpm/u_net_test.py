import unittest

import torch
import torch.nn as nn

from ml.diffusion.ddpm.u_net import Block


class TestBlock(unittest.TestCase):
    def setUp(self):
        self.B = 5
        self.d_embd_time = 7

    def test_init(self):
        block = Block(3, 6, 4, upsampling=True)
        assert isinstance(block.transform, nn.ConvTranspose2d)

        block = Block(3, 6, 4, upsampling=False)
        assert isinstance(block.transform, nn.Conv2d)

    def test_forward__upsampling(self):
        # Start with 64 channels and height and width of 1
        C, H, W = 64, 1, 1
        x = torch.randn(self.B, C, H, W)
        residual_x = torch.randn(self.B, C, H, W)  # Needed for upsampling
        x = torch.cat([x, residual_x], dim=1)  # (B, C, H, W) -> (B, 2C, H, W)
        t = torch.randint(0, 10, size=(self.B, self.d_embd_time)).float()

        # Create upsamling block (reduces channels and increases height and width)
        out_channels = 32
        assert out_channels < C, "Output channels must be less than input channels"
        upsampling_block = Block(
            in_channels=C,
            out_channels=out_channels,
            d_embd_time=self.d_embd_time,
            upsampling=True,
        )

        # Check output
        assert upsampling_block(x, t).shape == (self.B, out_channels, 2 * W, 2 * H)

    def test_forward__downsampling(self):
        # Start with 3 channels and height and width of 64
        C, H, W = 3, 64, 64
        x = torch.randn(self.B, C, H, W)
        t = torch.randint(0, 10, size=(self.B, self.d_embd_time)).float()

        # Create downsampling block (increases channels and reduces height and width)
        out_channels = 32
        assert out_channels > C, "Output channels must be greater than input channels"
        downsampling_block = Block(
            in_channels=C,
            out_channels=out_channels,
            d_embd_time=self.d_embd_time,
            upsampling=False,
        )

        # Check output
        assert downsampling_block(x, t).shape == (self.B, out_channels, W / 2, H / 2)
