import unittest

import torch
import torch.nn as nn

from lm.generative.model_config import ModelConfig
from lm.generative.models.mlp import MLP


class TestMLP(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig(vocab_size=27, block_size=3, n_embd=4, n_embd2=8)
        self.model = MLP(self.cfg)

    def test_init(self):
        self.assertEqual(self.model.vocab_size, self.cfg.vocab_size)
        self.assertEqual(self.model.block_size, self.cfg.block_size)

        # Run a few checks on the lookup table
        self.assertIsInstance(self.model.lookup_table, nn.Embedding)
        self.assertIsInstance(self.model.lookup_table.weight, nn.Parameter)
        self.assertEqual(
            self.model.lookup_table.weight.shape,
            (self.cfg.vocab_size, self.cfg.n_embd),
            msg="Lookup table must have shape (vocab size, embedding dimensionality)!",
        )
        self.assertTrue(
            "lookup_table.weight" in dict(self.model.named_parameters()),
            msg="nn.Embedding should be automatically registered as a module parameter!",
        )
        self.assertTrue(
            self.model.lookup_table.weight.requires_grad,
            msg="Gradients of lookup table elements should be tracked!",
        )

    def test_lookup_table(self):
        B = 5  # Batch size
        idx = torch.randint(0, self.cfg.vocab_size, (B, self.cfg.block_size))
        emb = self.model.lookup_table(idx)
        self.assertEqual(
            emb.shape,
            (B, self.cfg.block_size, self.cfg.n_embd),
            msg="Embeddings must have shape (batch_size, block_size, embedding dim.)!",
        )

    def test_forward_without_targets(self):
        # Generate a batch of 5 sequences each of length `block_size`
        batch_size = 5
        idx = torch.randint(0, self.cfg.vocab_size, (batch_size, self.cfg.block_size))

        # Forward pass
        logits, loss = self.model(idx)
        self.assertEqual(
            logits.shape,
            (batch_size, self.cfg.vocab_size),
            msg="Logits must have shape (batch_size, vocab_size)!",
        )
        self.assertIsNone(loss, msg="Loss should be None if targets are not provided!")

    def test_forward_with_targets(self):
        # Generate a batch of 5 input sequences each of length `block_size` as well as
        # a batch of 5 targets (labels) each of length 1 (we are only predicting the
        # next character)
        idx = torch.randint(0, self.cfg.vocab_size, (5, self.cfg.block_size))
        targets = torch.randint(0, self.cfg.vocab_size, (5,))

        # Forward pass
        logits, loss = self.model(idx, targets)
        self.assertEqual(logits.shape, (5, self.cfg.vocab_size))
        self.assertEqual(loss.shape, ())
        self.assertIsNotNone(
            loss, msg="Loss should not be None if targets are provided!"
        )
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(
            loss.item(), float, msg="Loss tensor should only contain a single float!"
        )

    def test_forward_equivalence_loss(self):
        idx = torch.randint(0, self.cfg.vocab_size, (5, self.cfg.block_size))
        targets = torch.randint(0, self.cfg.vocab_size, (5,))

        # Forward pass
        _, loss = self.model(idx, targets)

        # Manual forward pass with equivalent loss calculation
        embs = self.model.lookup_table(idx)
        embs = embs.view(embs.shape[0], -1)
        logits = self.model.mlp(embs)
        # Equivalent calculation of cross entropy loss
        counts = logits.exp()
        p = counts / counts.sum(dim=1, keepdim=True)
        n_examples = embs.shape[0]
        p_labels = p[torch.arange(n_examples), targets]
        loss_manual = -p_labels.log().mean()

        self.assertTrue(
            torch.allclose(loss, loss_manual),
            msg="F.cross_entropy calculation must be equivalent to the manual "
            "calculation of average negative log likelihood!",
        )


if __name__ == "__main__":
    unittest.main()
