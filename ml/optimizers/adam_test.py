import unittest

import torch

from ml.optimizers.adam import AdamOptimizer
from ml.tensor import Tensor


class TestAdamOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up the optimizer and example tensors for testing."""
        self.optim = AdamOptimizer(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.t = 1  # Current iteration

        # Example weights, biases, and their gradients
        self.w = torch.tensor([0.5, -0.3])
        self.dw = torch.tensor([0.01, -0.01])

    def test_init(self):
        assert self.optim.m_dw == 0
        assert self.optim.v_dw == 0

    def test_update__moment_estimates(self):
        self.optim.update(self.t, self.w, self.dw)

        # First moment estimate
        expected_m_dw = self.optim.beta_1 * 0 + (1 - self.optim.beta_1) * self.dw
        assert torch.equal(self.optim.m_dw, expected_m_dw)

        # Second moment estimate:
        expected_v_dw = self.optim.beta_2 * 0 + (1 - self.optim.beta_2) * (self.dw**2)
        assert torch.equal(self.optim.v_dw, expected_v_dw)

    def test_update__bias_correction(self):
        self.optim.update(self.t, self.w, self.dw)

        # Manually compute bias-corrected moments
        expected_m_dw_hat = self.optim.m_dw / (1 - self.optim.beta_1**self.t)
        expected_v_dw_hat = self.optim.v_dw / (1 - self.optim.beta_2**self.t)

        # Test if the bias correction is working as expected
        assert torch.equal(
            expected_m_dw_hat, self.optim.m_dw / (1 - self.optim.beta_1**self.t)
        )
        assert torch.equal(
            expected_v_dw_hat, self.optim.v_dw / (1 - self.optim.beta_2**self.t)
        )

    def test_update__convergence(self):
        """Test if the optimizer converges to the minimum for a simple quadratic function."""

        def grad_fn(w_: Tensor) -> Tensor:
            return 2 * (w_ - 1)  # Derivative of f(w) = (w - 1)^2

        w = torch.tensor(0.0)  # Start far from the minimum at w=1
        for t in range(1, 10_000):
            dw = grad_fn(w)
            w = self.optim.update(t, w, dw)

        # Check that the optimizer converged to the minimum at w=1
        assert torch.allclose(w, torch.tensor(1.0), atol=0.01)


if __name__ == "__main__":
    unittest.main()
