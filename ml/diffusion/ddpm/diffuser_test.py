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

        expected_sqrt_recip_alphas = torch.sqrt(
            torch.tensor(
                [
                    1.0 / 0.999,
                    1.0 / 0.998,
                    1.0 / 0.997,
                    1.0 / 0.996,
                    1.0 / 0.995,
                    1.0 / 0.994,
                    1.0 / 0.993,
                    1.0 / 0.992,
                    1.0 / 0.991,
                    1.0 / 0.99,
                ]
            )
        )
        assert torch.allclose(self.ddpm.sqrt_recip_alphas, expected_sqrt_recip_alphas)

        expected_sqrt_alphabars = torch.sqrt(expected_alphabars)
        assert torch.allclose(self.ddpm.sqrt_alphabars, expected_sqrt_alphabars)

        expected_sqrt_one_minus_alphabars = torch.sqrt(1.0 - expected_alphabars)
        assert torch.allclose(
            self.ddpm.sqrt_one_minus_alphabars, expected_sqrt_one_minus_alphabars
        )

        expected_alphabars_prev = torch.tensor(
            [
                1.0,
                0.999,
                0.999 * 0.998,
                0.999 * 0.998 * 0.997,
                0.999 * 0.998 * 0.997 * 0.996,
                0.999 * 0.998 * 0.997 * 0.996 * 0.995,
                0.999 * 0.998 * 0.997 * 0.996 * 0.995 * 0.994,
                0.999 * 0.998 * 0.997 * 0.996 * 0.995 * 0.994 * 0.993,
                0.999 * 0.998 * 0.997 * 0.996 * 0.995 * 0.994 * 0.993 * 0.992,
                0.999 * 0.998 * 0.997 * 0.996 * 0.995 * 0.994 * 0.993 * 0.992 * 0.991,
            ]
        )
        assert torch.allclose(self.ddpm.alphabars_prev, expected_alphabars_prev)

        expected_posterior_variance = (
            expected_betas
            * (1.0 - expected_alphabars_prev)
            / (1.0 - expected_alphabars)
        )
        assert torch.allclose(
            self.ddpm.posterior_variance, expected_posterior_variance, atol=1e-7
        )
