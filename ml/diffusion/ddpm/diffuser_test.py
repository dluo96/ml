import unittest

import torch

from ml.diffusion.ddpm.diffuser import DDPM


class TestDDPM(unittest.TestCase):
    def setUp(self):
        self.T = 4
        self.device = "cpu"
        self.ddpm = DDPM(T=self.T, device=self.device)

    def test_init(self):
        assert self.ddpm.T == self.T

        expected_betas = torch.tensor([0.001, 0.004, 0.007, 0.01])
        assert torch.allclose(self.ddpm.betas, expected_betas)

        expected_alphas = torch.tensor([0.999, 0.996, 0.993, 0.99])
        assert torch.allclose(self.ddpm.alphas, expected_alphas)

        expected_alphabars = torch.tensor(
            [
                0.999,
                0.999 * 0.996,
                0.999 * 0.996 * 0.993,
                0.999 * 0.996 * 0.993 * 0.99,
            ]
        )
        assert torch.allclose(self.ddpm.alphabars, expected_alphabars)

        expected_sqrt_recip_alphas = torch.sqrt(
            torch.tensor([1.0 / 0.999, 1.0 / 0.996, 1.0 / 0.993, 1.0 / 0.99])
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
                0.999 * 0.996,
                0.999 * 0.996 * 0.993,
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
