import unittest

import torch

from lm.normalization.layer_norm import LayerNorm


class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        self.n_embd = 5
        self.batch_size = 32
        self.eps = 1e-5
        self.bn = LayerNorm(n_embd=self.n_embd, eps=self.eps)

    def test_init(self):
        self.assertEqual(self.bn.gamma.shape, (self.n_embd,))
        self.assertEqual(self.bn.beta.shape, (self.n_embd,))
        self.assertTrue(torch.equal(self.bn.gamma, torch.ones(self.n_embd)))
        self.assertTrue(torch.equal(self.bn.beta, torch.zeros(self.n_embd)))

    def test_parameters(self):
        # Ensure that the parameters method returns gamma and beta
        params = self.bn.parameters()
        self.assertEqual(
            len(params), 2, "There should be two parameters (gamma and beta)!"
        )
        self.assertIs(params[0], self.bn.gamma, "First parameter should be gamma!")
        self.assertIs(params[1], self.bn.beta, "Second parameter should be beta!")

    def test_forward(self):
        x = torch.randn(self.batch_size, self.n_embd)
        out = self.bn(x)

        # Check the output shape
        self.assertEqual(
            out.shape,
            (self.batch_size, self.n_embd),
            msg="Output shape must be (batch_size, n_embd)",
        )

        # Check that the mean and variance, calculated across the embedding dimension,
        # are (close to) 0 and 1, respectively
        expected_mean = torch.zeros(self.batch_size, dtype=torch.float32)
        self.assertTrue(
            torch.allclose(out.mean(dim=1), expected_mean, atol=1e-3),
            msg="Mean of the LayerNorm output should be (close to) 0",
        )
        expected_var = torch.ones(self.batch_size, dtype=torch.float32)
        self.assertTrue(
            torch.allclose(out.var(dim=1), expected_var, atol=1e-3),
            msg="Variance of the LayerNorm output should be (close to) 1",
        )


if __name__ == "__main__":
    unittest.main()
