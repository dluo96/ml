import unittest

import torch

from lm.models.bg_nn import Bigram


class TestBigramModel(unittest.TestCase):
    class Config:
        vocab_size = 10  # Example vocabulary size for testing purposes

    def setUp(self):
        self.config = self.Config()
        self.model = Bigram(self.config)

    def test_init(self):
        self.assertEqual(
            self.model.logits.shape, (self.config.vocab_size, self.config.vocab_size)
        )
        self.assertTrue(
            torch.equal(
                self.model.logits,
                torch.zeros(self.config.vocab_size, self.config.vocab_size),
            )
        )
        self.assertIn(
            self.model.logits,
            self.model.parameters(),
            msg="nn.Parameter is automatically assigned to the module parameters!",
        )
        self.assertTrue(
            self.model.logits.requires_grad,
            msg="nn.Parameter has requires_grad=True by default!",
        )

    def test_get_block_size(self):
        expected_block_size = 1  # Bigram model uses previous character to predict next
        self.assertEqual(self.model.get_block_size(), expected_block_size)

    def test_forward_no_targets(self):
        idx = torch.randint(0, self.config.vocab_size, (5,))  # 5 random indices
        logits, loss = self.model.forward(idx)

        # Check the shape of the logits
        self.assertEqual(logits.shape, (5, self.config.vocab_size))
        self.assertIsNone(loss, "Loss should be None when targets are not provided!")

    def test_forward_with_targets(self):
        # Test forward pass with targets
        idx = torch.randint(0, self.config.vocab_size, (5,))  # Random indices
        targets = torch.randint(0, self.config.vocab_size, (5,))  # Random targets
        logits, loss = self.model.forward(idx, targets)

        # Check the shape of the logits
        self.assertEqual(logits.shape, (5, self.config.vocab_size))
        self.assertIsNotNone(loss)  # Loss should not be None when targets are provided
        self.assertIsInstance(loss, torch.Tensor, "Loss should be a tensor!")
        self.assertGreaterEqual(loss.item(), 0, "Loss should be non-negative!")


if __name__ == "__main__":
    unittest.main()
