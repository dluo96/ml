import unittest

import torch

from ml.diffusion.ddpm.diffuser import DDPM


class TestDDPM(unittest.TestCase):
    def setUp(self):
        self.T = 10
        self.device = "cpu"
        self.ddpm = DDPM(T=self.T, device=self.device)

    def test_init(self):
        assert self.ddpm.T == self.T

        expected_betas = torch.tensor(
            [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
        )
        assert torch.allclose(self.ddpm.betas, expected_betas)

        expected_alphas = torch.tensor(
            [0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.993, 0.992, 0.991, 0.99]
        )
        assert torch.allclose(self.ddpm.alphas, expected_alphas)

        expected_alphabars = torch.tensor(
            [
                0.999,
                0.999 * 0.998,
                0.999 * 0.998 * 0.997,
                0.999 * 0.998 * 0.997 * 0.996,
                0.999 * 0.998 * 0.997 * 0.996 * 0.995,
                0.999 * 0.998 * 0.997 * 0.996 * 0.995 * 0.994,
                0.999 * 0.998 * 0.997 * 0.996 * 0.995 * 0.994 * 0.993,
                0.999 * 0.998 * 0.997 * 0.996 * 0.995 * 0.994 * 0.993 * 0.992,
                0.999 * 0.998 * 0.997 * 0.996 * 0.995 * 0.994 * 0.993 * 0.992 * 0.991,
                0.999 * 0.998 * 0.997 * 0.996 * 0.995 * 0.994 * 0.993 * 0.992 * 0.991 * 0.99,  # fmt: skip
            ]
        )
        assert torch.allclose(self.ddpm.alphabars, expected_alphabars)

        assert self.ddpm.betas.shape == (self.T,)
        assert self.ddpm.alphas.shape == (self.T,)
        assert self.ddpm.alphabars.shape == (self.T,)
        assert self.ddpm.sqrt_recip_alphas.shape == (self.T,)
        assert self.ddpm.sqrt_alphabars.shape == (self.T,)
        assert self.ddpm.sqrt_one_minus_alphabars.shape == (self.T,)
        assert self.ddpm.alphabars_prev.shape == (self.T,)
        assert self.ddpm.posterior_variance.shape == (self.T,)
