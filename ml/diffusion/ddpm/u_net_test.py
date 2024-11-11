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

    def test_forward(self):
        block = Block(3, 6, 4, upsampling=True)
        B = 4  # Batch size
        x = torch.randn(B, 3, IMG_SIZE, IMG_SIZE)
        residual_x = torch.randn(B, 3, IMG_SIZE, IMG_SIZE)
        x = torch.cat([x, residual_x], dim=1)  # (B, C, H, W) -> (B, 2C, H, W)
        t = torch.randn(B, 4)
        output = block(x, t)
        assert output.shape == (B, 6, 128, 128)
