import unittest

import torch


class TestConvTranspose2d(unittest.TestCase):
    def test_transpose_convolution(self):
        in_channels = 3
        out_channels = 6
        kernel_size = 4
        stride = 2
        padding = 1
        transpose_conv = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )

        # Define an input
        B = 4
        H = 128
        W = 128
        C = 3
        x = torch.randn(B, C, H, W)

        # Expected height and width after transpose convolution
        H_after = stride * (H - 1) + kernel_size - 2 * padding
        W_after = stride * (W - 1) + kernel_size - 2 * padding

        # Check
        assert transpose_conv(x).shape == (B, out_channels, H_after, W_after)
