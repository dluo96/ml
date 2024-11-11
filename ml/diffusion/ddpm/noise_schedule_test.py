import unittest

import torch

from ml.diffusion.ddpm.noise_schedule import linear_beta_schedule


class TestLinearBetaSchedule(unittest.TestCase):
    def test_linear_beta_schedule(self):
        T = 10
        beta_start = 0.1
        beta_final = 0.2
        beta_values = linear_beta_schedule(T, beta_start, beta_final)

        # Check shape
        assert beta_values.shape == (T,)

        # Check values
        expected_step = torch.tensor((beta_final - beta_start) / (T - 1))
        for i in range(1, T):
            assert torch.allclose(beta_values[i] - beta_values[i - 1], expected_step)

        # Check start and end values
        assert torch.equal(beta_values[0], torch.tensor(beta_start))
        assert torch.equal(beta_values[-1], torch.tensor(beta_final))
