import unittest

import torch

from lm.optimizers.adam import AdamOptimizer


class TestAdamOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up the optimizer and example tensors for testing."""
        self.optim = AdamOptimizer(eta=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.t = 1  # Current iteration

        # Example weights, biases, and their gradients
        self.w = torch.tensor([0.5, -0.3])
        self.b = torch.tensor([0.1])
        self.dw = torch.tensor([0.01, -0.01])
        self.db = torch.tensor([0.005])

    def test_init(self):
        # Check that moment estimates are initialized to zero
        assert self.optim.m_dw == 0
        assert self.optim.v_dw == 0
        assert self.optim.m_db == 0
        assert self.optim.v_db == 0

    def test_update(self):
        new_w, new_b = self.optim.update(self.t, self.w, self.dw, self.b, self.db)

        # Assert that weights and biases are updated
        assert not torch.equal(new_w, self.w), "Weights did not change after update."
        assert not torch.equal(new_b, self.b), "Biases did not change after update."

    def test_update__moment_estimates(self):
        """Test if the moment estimates are updated correctly."""
        # Perform one update step
        self.optim.update(self.t, self.w, self.dw, self.b, self.db)

        # First moment estimates (m_dw, m_db)
        expected_m_dw = self.optim.beta_1 * 0 + (1 - self.optim.beta_1) * self.dw
        expected_m_db = self.optim.beta_1 * 0 + (1 - self.optim.beta_1) * self.db
        assert torch.equal(self.optim.m_dw, expected_m_dw)
        assert torch.equal(self.optim.m_db, expected_m_db)

        # Second moment estimates (v_dw, v_db)
        expected_v_dw = self.optim.beta_2 * 0 + (1 - self.optim.beta_2) * (self.dw**2)
        expected_v_db = self.optim.beta_2 * 0 + (1 - self.optim.beta_2) * (self.db**2)
        assert torch.equal(self.optim.v_dw, expected_v_dw)
        assert torch.equal(self.optim.v_db, expected_v_db)

    def test_update__bias_correction(self):
        """Test the bias correction for moments."""
        # Perform one update step
        self.optim.update(self.t, self.w, self.dw, self.b, self.db)

        # Manually compute bias-corrected moments
        expected_m_dw_hat = self.optim.m_dw / (1 - self.optim.beta_1**self.t)
        expected_m_db_hat = self.optim.m_db / (1 - self.optim.beta_1**self.t)
        expected_v_dw_hat = self.optim.v_dw / (1 - self.optim.beta_2**self.t)
        expected_v_db_hat = self.optim.v_db / (1 - self.optim.beta_2**self.t)

        # Test if the bias correction is working as expected
        assert torch.equal(
            expected_m_dw_hat, self.optim.m_dw / (1 - self.optim.beta_1**self.t)
        )
        assert torch.equal(
            expected_m_db_hat, self.optim.m_db / (1 - self.optim.beta_1**self.t)
        )
        assert torch.equal(
            expected_v_dw_hat, self.optim.v_dw / (1 - self.optim.beta_2**self.t)
        )
        torch.equal(
            expected_v_db_hat, self.optim.v_db / (1 - self.optim.beta_2**self.t)
        )


if __name__ == "__main__":
    unittest.main()
