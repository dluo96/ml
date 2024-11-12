import unittest

import torch
import torch.nn as nn

from ml.diffusion.ddpm.u_net import Block, SinusoidalPositionEmbeddings, Unet


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


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.d_embd_time = 10
        self.pos_embd = SinusoidalPositionEmbeddings(d_embd=self.d_embd_time)

    def test_forward__single(self):
        t = torch.tensor([2]).float()
        t_emb = self.pos_embd(t)
        assert t_emb.shape == (1, self.d_embd_time)

    def test_forward__batch(self):
        B = 5
        t = torch.randint(0, 10, size=(B,)).float()
        t_emb = self.pos_embd(t)
        assert t_emb.shape == (B, self.d_embd_time)


class TestUnet(unittest.TestCase):
    def setUp(self):
        self.B = 5
        self.d_embd_time = 10

    def test_init(self):
        u_net = Unet()
        assert len(u_net.downsampling_blocks) == 4
        assert len(u_net.upsampling_blocks) == 4

    def test_forward(self):
        # Input
        C, H, W = 3, 64, 64
        x = torch.randn(self.B, C, H, W)
        t = torch.randint(0, 10, size=(self.B, self.d_embd_time)).float()

        # Model
        u_net = Unet()

        # Check output
        assert u_net(x, t).shape == x.shape, "Output shape must match input shape"
