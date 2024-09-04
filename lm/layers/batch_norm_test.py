import unittest

import torch
from torch.testing import assert_allclose

from lm.layers.batch_norm import BatchNorm1D


class TestBatchNorm1D(unittest.TestCase):
    def setUp(self):
        self.dim = 5
        self.batch_size = 10
        self.eps = 1e-5
        self.momentum = 0.1
        self.bn = BatchNorm1D(dim=self.dim, eps=self.eps, momentum=self.momentum)

    def test_init(self):
        # Check initialization of parameters and buffers
        self.assertEqual(self.bn.gamma.shape, (self.dim,))
        self.assertEqual(self.bn.beta.shape, (self.dim,))
        self.assertEqual(self.bn.running_mean.shape, (self.dim,))
        self.assertEqual(self.bn.running_var.shape, (self.dim,))
        assert_allclose(self.bn.gamma, torch.ones(self.dim))
        assert_allclose(self.bn.beta, torch.zeros(self.dim))
        assert_allclose(self.bn.running_mean, torch.zeros(self.dim))
        assert_allclose(self.bn.running_var, torch.ones(self.dim))

    def test_parameters(self):
        # Ensure that the parameters method returns gamma and beta
        params = self.bn.parameters()
        self.assertEqual(
            len(params), 2, "There should be two parameters (gamma and beta)!"
        )
        self.assertIs(params[0], self.bn.gamma, "First parameter should be gamma!")
        self.assertIs(params[1], self.bn.beta, "Second parameter should be beta!")

    def test_forward_training(self):
        # Set the layer to training mode
        self.bn.training = True
        x = torch.randn(self.batch_size, self.dim)

        # Forward pass
        out = self.bn(x)

        # Check the output shape
        self.assertEqual(
            out.shape,
            (self.batch_size, self.dim),
            msg="Output shape is incorrect during training!",
        )

        # Check that running mean and variance are updated
        self.assertFalse(
            torch.equal(self.bn.running_mean, torch.zeros(self.dim)),
            msg="Running mean should be updated during training!",
        )
        self.assertFalse(
            torch.equal(self.bn.running_var, torch.ones(self.dim)),
            msg="Running variance should be updated during training!",
        )

    def test_forward_evaluation(self):
        # Set the layer to evaluation mode
        self.bn.training = False
        x = torch.randn(self.batch_size, self.dim)

        # Store original running mean and variance
        running_mean_orig = self.bn.running_mean.clone()
        running_var_orig = self.bn.running_var.clone()

        # Forward pass
        out = self.bn(x)

        # Check the output shape
        self.assertEqual(
            out.shape,
            (self.batch_size, self.dim),
            "Output shape is incorrect during evaluation!",
        )

        # Ensure running mean and variance are not updated
        assert_allclose(
            self.bn.running_mean,
            running_mean_orig,
            msg="Running mean should not be updated during evaluation!",
        )
        assert_allclose(
            self.bn.running_var,
            running_var_orig,
            msg="Running variance should not be updated during evaluation!",
        )


if __name__ == "__main__":
    unittest.main()
