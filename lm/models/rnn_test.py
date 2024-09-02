import unittest

import torch
import torch.nn as nn

from lm.models.rnn import RNN, GRUCell, RNNCell
from lm.types import ModelConfig


class TestRNN(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(vocab_size=27, block_size=3, n_embd=4, n_embd2=8)

        # Overridden in subclasses
        self.cell_type = "rnn"
        self.model = RNN(self.config, cell_type=self.cell_type)

    def test_init(self):
        # Check that the model attributes are set correctly
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
        self.assertTrue(
            "lookup_table.weight" in dict(self.model.named_parameters()),
            msg="nn.Embedding should be automatically registered as a module parameter!",
        )
        self.assertTrue(
            self.model.lookup_table.weight.requires_grad,
            msg="Gradients of lookup table elements should be tracked!",
        )

        # Check that the start parameter is initialized correctly
        self.assertIsInstance(self.model.start, nn.Parameter)
        self.assertEqual(
            self.model.start.shape,
            (1, self.config.n_embd2),
            msg="Start parameter must have shape (1, hidden state embedding size)!",
        )

        # Check the RNN cell is properly initialized
        if self.cell_type == "rnn":
            self.assertIsInstance(self.model.cell, RNNCell)
            self.assertIsInstance(
                self.model.cell.xh_to_h,
                nn.Linear,
                msg="RNNCell should contain a linear layer to map the input x_{t} and "
                "the previous hidden state h_{t-1} to the next hidden state h_{t}!",
            )
        elif self.cell_type == "gru":
            self.assertIsInstance(self.model.cell, GRUCell)
            self.assertIsInstance(self.model.cell.xh_to_z, nn.Linear)
            self.assertIsInstance(self.model.cell.xh_to_r, nn.Linear)
            self.assertIsInstance(self.model.cell.xh_to_hbar, nn.Linear)
        else:
            raise ValueError(f"Cell type {self.cell_type} is not recognised!")

    def test_lookup_table(self):
        batch_size = 5
        idx = torch.randint(
            0, self.config.vocab_size, (batch_size, self.config.block_size)
        )
        emb = self.model.lookup_table(idx)
        self.assertEqual(
            emb.shape,
            (batch_size, self.config.block_size, self.config.n_embd),
            msg="Embeddings must have shape (batch_size, block_size, embedding dim.)!",
        )

    def test_forward_without_targets(self):
        # Generate a batch of input sequences
        batch_size = 5
        idx = torch.randint(
            0, self.config.vocab_size, (batch_size, self.config.block_size)
        )

        # Forward pass without targets
        logits, loss = self.model(idx)
        self.assertEqual(
            logits.shape,
            (batch_size, self.config.block_size, self.config.vocab_size),
            msg="Output logits must have shape (batch_size, block_size, vocab_size) "
            "because the RNN generates predictions at each step of the input sequence, "
            "effectively providing a prediction for every input position!",
        )
        self.assertIsNone(loss, msg="Loss should be None if targets are not provided!")

    def test_forward_with_targets(self):
        # Craft an example output of `SequenceDataset.__getitem__()`
        x = torch.tensor([0, 5, 13, 13, 1,  0,  0,  0,  0,  0,  0,  0])  # fmt: skip
        y = torch.tensor([5, 13, 13, 1, 0, -1, -1, -1, -1, -1, -1, -1])

        # Since the RNN predicts the next character at each step of the input sequence,
        # the target should have the same shape as the input sequence `idx` to provide
        # a target for each prediction at each time step.
        # Reshape to be consistent with `block_size`
        block_size = 3
        x = x.view(-1, block_size)
        y = y.view(-1, block_size)

        # Infer batch size
        batch_size = x.shape[0]

        # Forward pass with targets
        logits, loss = self.model(idx=x, targets=y)
        self.assertEqual(
            logits.shape,
            (batch_size, self.config.block_size, self.config.vocab_size),
            msg="Logits must have shape (batch_size, block_size, vocab_size)!",
        )
        self.assertIsNotNone(
            loss, msg="Loss should not be None if targets are provided!"
        )
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(
            loss.item(), float, msg="Loss tensor should only contain a single float!"
        )


class TestGRU(TestRNN):
    def setUp(self):
        super().setUp()
        self.cell_type = "gru"
        self.model = RNN(self.config, cell_type=self.cell_type)


if __name__ == "__main__":
    unittest.main()
