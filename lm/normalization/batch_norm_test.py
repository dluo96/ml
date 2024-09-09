import unittest

import torch

from lm.normalization.batch_norm import BatchNorm1D


class TestBatchNorm1D(unittest.TestCase):
    def setUp(self):
        self.n_embd = 5
        self.batch_size = 32
        self.eps = 1e-5
        self.momentum = 0.1
        self.bn = BatchNorm1D(n_embd=self.n_embd, eps=self.eps, momentum=self.momentum)

    def test_init(self):
        # Check initialization of parameters and buffers
        self.assertEqual(self.bn.gamma.shape, (self.n_embd,))
        self.assertEqual(self.bn.beta.shape, (self.n_embd,))
        self.assertEqual(self.bn.running_mean.shape, (self.n_embd,))
        self.assertEqual(self.bn.running_var.shape, (self.n_embd,))
        self.assertTrue(torch.equal(self.bn.gamma, torch.ones(self.n_embd)))
        self.assertTrue(torch.equal(self.bn.beta, torch.zeros(self.n_embd)))
        self.assertTrue(torch.equal(self.bn.running_mean, torch.zeros(self.n_embd)))
        self.assertTrue(torch.equal(self.bn.running_var, torch.ones(self.n_embd)))

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
        x = torch.randn(self.batch_size, self.n_embd)

        # Forward pass
        out = self.bn(x)

        # Check the output shape
        self.assertEqual(
            out.shape,
            (self.batch_size, self.n_embd),
            msg="Output shape is incorrect during training!",
        )

        # Check that the mean and variance are (close to) 0 and 1, respectively
        expected_mean = torch.zeros(self.n_embd, dtype=torch.float32)
        self.assertTrue(
            torch.allclose(out.mean(dim=0), expected_mean, atol=1e-3),
            msg="Mean of the BatchNorm output should be (close to) 0",
        )
        expected_var = torch.ones(self.n_embd, dtype=torch.float32)
        self.assertTrue(
            torch.allclose(out.var(dim=0), expected_var, atol=1e-3),
            msg="Variance of the BatchNorm output should be (close to) 1",
        )

        # Check that running mean and variance are updated
        self.assertFalse(
            torch.equal(self.bn.running_mean, torch.zeros(self.n_embd)),
            msg="Running mean should be updated during training!",
        )
        self.assertFalse(
            torch.equal(self.bn.running_var, torch.ones(self.n_embd)),
            msg="Running variance should be updated during training!",
        )

    def test_forward_evaluation(self):
        # Set the layer to evaluation mode
        self.bn.training = False
        x = torch.randn(self.batch_size, self.n_embd)

        # Store original running mean and variance
        running_mean_orig = self.bn.running_mean.clone()
        running_var_orig = self.bn.running_var.clone()

        # Forward pass
        out = self.bn(x)

        # Check the output shape
        self.assertEqual(
            out.shape,
            (self.batch_size, self.n_embd),
            "Output shape is incorrect during evaluation!",
        )

        # Ensure running mean and variance are not updated
        self.assertTrue(
            torch.equal(self.bn.running_mean, running_mean_orig),
            msg="Running mean should NOT be updated during evaluation!",
        )
        self.assertTrue(
            torch.equal(self.bn.running_var, running_var_orig),
            msg="Running variance should NOT be updated during evaluation!",
        )

    def test_impact_on_tanh(self):
        torch.manual_seed(24)

        x = torch.randn(10_000, 10)
        w = torch.randn(10, 200)
        y = x @ w

        # Check that input distribution is standard normal
        self.assertAlmostEquals(x.mean().item(), 0, delta=0.05)
        self.assertAlmostEquals(x.std().item(), 1, delta=0.05)

        # Check that output distribution has larger standard deviation
        self.assertAlmostEquals(y.mean().item(), 0, delta=0.05)
        self.assertGreater(y.std().item(), 3)

        # Apply batch normalization and check that the output distribution is now
        # standard normal
        bn = BatchNorm1D(n_embd=200)
        y_bn = bn(y)
        self.assertAlmostEquals(y_bn.mean().item(), 0, delta=0.05)
        self.assertAlmostEquals(y_bn.std().item(), 1, delta=0.05)

    def test_tanh_activations_without_batch_norm(self):
        torch.manual_seed(0)
        batch_size = 1000
        n_embd = 10
        x = torch.randn(batch_size, n_embd)
        w1 = torch.randn(n_embd, n_embd)
        w2 = torch.randn(n_embd, n_embd)
        w3 = torch.randn(n_embd, n_embd)

        # Apply three linear layers
        pre_activations = ((x @ w1) @ w2) @ w3

        # Check that tanh is saturated (all neurons are close/equal to -1 or +1)
        activations = torch.tanh(pre_activations)
        self.assertGreaterEqual(
            (activations.abs() > 0.99).sum().item(),
            activations.numel() * 0.9,
            msg="tanh should be saturated for 90% or more of neurons because we did "
            "not apply batch normalization!",
        )


if __name__ == "__main__":
    unittest.main()
