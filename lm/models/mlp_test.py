import unittest

import torch
import torch.nn as nn

from lm.models.mlp import MLP


class MockConfig:
    def __init__(self, vocab_size, block_size, n_embd, n_embd2):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_embd2 = n_embd2


class TestMLP(unittest.TestCase):
    def setUp(self):
        # Mock configuration for testing
        self.config = MockConfig(vocab_size=27, block_size=3, n_embd=4, n_embd2=8)
        self.model = MLP(self.config)

    def test_init(self):
        self.assertEqual(self.model.vocab_size, self.config.vocab_size)
        self.assertEqual(self.model.block_size, self.config.block_size)

        # Run a few checks on the lookup table
        self.assertIsInstance(self.model.lookup_table, nn.Embedding)
        self.assertIsInstance(self.model.lookup_table.weight, nn.Parameter)
        self.assertEqual(
            self.model.lookup_table.weight.shape,
            (self.config.vocab_size, self.config.n_embd),
            msg="Lookup table must have shape (vocab size, embedding dimensionality)!",
        )
        self.assertIn(
            self.model.lookup_table.weight,
            self.model.parameters(),
            msg="nn.Embedding should be automatically registered as a module parameter!",
        )
        self.assertTrue(
            self.model.lookup_table.weight.requires_grad,
            msg="Gradients of lookup table elements should be tracked!",
        )

    def test_forward_without_targets(self):
        # Generate a batch of 5 sequences each of length `block_size`
        idx = torch.randint(0, self.config.vocab_size, (5, self.config.block_size))

        # Forward pass
        logits, loss = self.model(idx)
        self.assertEqual(
            logits.shape,
            (5, self.config.vocab_size),
            msg="Logits must have shape (batch_size, vocab_size)!",
        )
        self.assertIsNone(loss, msg="Loss should be None if targets are not provided!")

    def test_forward_with_targets(self):
        # Generate a batch of 5 input sequences each of length `block_size` as well as
        # a batch of 5 targets (labels)
        idx = torch.randint(0, self.config.vocab_size, (5, self.config.block_size))
        targets = torch.randint(0, self.config.vocab_size, (5,))

        # Forward pass
        logits, loss = self.model(idx, targets)
        self.assertEqual(logits.shape, (5, self.config.vocab_size))
        self.assertEqual(loss.shape, ())
        self.assertIsNotNone(
            loss, msg="Loss should not be None if targets are provided!"
        )
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(
            loss.item(), float, msg="Loss tensor should only contain a single float!"
        )

    def test_lookup_table(self):
        B = 5  # Batch size
        idx = torch.randint(0, self.config.vocab_size, (B, self.config.block_size))
        emb = self.model.lookup_table(idx)
        self.assertEqual(emb.shape, (B, self.config.block_size, self.config.n_embd))


if __name__ == "__main__":
    unittest.main()
