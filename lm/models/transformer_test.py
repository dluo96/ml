import math
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from lm.model_config import ModelConfig
from lm.models.transformer import Block, CausalSelfAttention, NewGELU, Transformer


class TestCausalSelfAttention(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(n_embd=64, n_head=4, block_size=8)
        self.model = CausalSelfAttention(self.config)

    def test_init(self):
        # Ensure embedding dimension is divisible by number of heads
        self.assertEqual(self.config.n_embd % self.config.n_head, 0)

        # Ensure the attention and projection layers are initialized correctly
        self.assertIsInstance(self.model.c_attn, nn.Linear)
        self.assertIsInstance(self.model.c_proj, nn.Linear)

        # Check that the causal mask is registered as a buffer
        self.assertTrue(hasattr(self.model, "bias"))
        self.assertEqual(
            self.model.bias.shape,
            (1, 1, self.config.block_size, self.config.block_size),
        )
        expected_bias = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        self.assertTrue(torch.equal(self.model.bias[0, 0, :, :], expected_bias))

        # Check that the causal mask is NOT registered as a parameter
        self.assertTrue("bias" not in dict(self.model.named_parameters()))

    def test_forward(self):
        # Generate a random input tensor of shape (batch_size, sequence_length, embedding_dim)
        batch_size = 2
        sequence_length = 5  # Should match block_size
        x = torch.randn(batch_size, sequence_length, self.config.n_embd)

        # Forward pass
        y = self.model(x)

        # Check the shape of the output
        self.assertEqual(y.shape, (batch_size, sequence_length, self.config.n_embd))

    def test_attention_is_lower_triangular(self):
        # Generate a random input tensor
        B = 2  # Batch size
        T = 8  # Sequence length
        nh = self.config.n_head  # Number of heads
        C = self.config.n_embd  # Embedding dimensionality

        x = torch.randn(B, T, C)

        # Ensure that softmax attention is applied
        q, k, v = self.model.c_attn(x).split(C, dim=2)
        k = k.view(B, T, nh, C // nh).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, nh, C // nh).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, nh, C // nh).transpose(1, 2)  # (B, nh, T, hs)

        # Manually compute attention
        att_manual = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att_manual = att_manual.masked_fill(
            self.model.bias[:, :, :T, :T] == 0, float("-inf")
        )
        att_manual = F.softmax(att_manual, dim=-1)

        # Check that the attention matrix is lower triangular
        for batch_idx in range(B):
            for head_idx in range(nh):
                # Extract attention scores for the current head and batch
                att = att_manual[batch_idx, head_idx, :, :]

                # Check if the matrix is lower triangular
                self.assertTrue(
                    torch.equal(att, torch.tril(att)),
                    msg="Attention matrix must be lower triangular!",
                )


class TestBlock(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(n_embd=64, n_head=4, block_size=10)
        self.model = Block(self.config)

    def test_init(self):
        # Check the LayerNorm layers
        self.assertIsInstance(self.model.ln_1, nn.LayerNorm)
        self.assertIsInstance(self.model.ln_2, nn.LayerNorm)
        self.assertEqual(self.model.ln_1.normalized_shape, (self.config.n_embd,))
        self.assertEqual(self.model.ln_2.normalized_shape, (self.config.n_embd,))

        # Check the CausalSelfAttention layer
        self.assertIsInstance(self.model.attn, CausalSelfAttention)

        # Check the MLP structure
        self.assertIsInstance(self.model.mlp, nn.ModuleDict)
        self.assertIn("c_fc", self.model.mlp)
        self.assertIn("c_proj", self.model.mlp)
        self.assertIn("act", self.model.mlp)
        self.assertIsInstance(self.model.mlp["c_fc"], nn.Linear)
        self.assertIsInstance(self.model.mlp["c_proj"], nn.Linear)
        self.assertIsInstance(self.model.mlp["act"], NewGELU)
        self.assertEqual(
            self.model.mlp["c_fc"].weight.shape,
            (4 * self.config.n_embd, self.config.n_embd),
        )
        self.assertEqual(
            self.model.mlp["c_proj"].weight.shape,
            (self.config.n_embd, 4 * self.config.n_embd),
        )

    def test_forward(self):
        # Create random input tensor
        batch_size = 2
        seq_length = self.config.block_size
        x = torch.randn(batch_size, seq_length, self.config.n_embd)

        # Forward pass
        y = self.model(x)

        # Check output shape
        self.assertEqual(
            y.shape,
            (batch_size, seq_length, self.config.n_embd),
            msg="Output must have shape (batch_size, seq_length, embedding dim.)",
        )

    def test_residual_connections(self):
        # Create random input tensor
        batch_size = 2
        seq_length = self.config.block_size
        x = torch.randn(batch_size, seq_length, self.config.n_embd)

        # Perform forward pass
        y = self.model(x)

        # Check that residual connections are preserved
        with torch.no_grad():
            attn_output = self.model.attn(self.model.ln_1(x))
            mlp_output = self.model.mlpf(self.model.ln_2(x + attn_output))
            expected_output = x + attn_output + mlp_output

        self.assertTrue(
            torch.equal(y, expected_output),
            msg="Residual connections are not preserved properly!",
        )


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            vocab_size=27,
            block_size=6,
            n_embd=64,
            n_layer=7,
            n_head=4,
        )
        self.model = Transformer(self.config)

    def test_init(self):
        # Check if the model's block size matches the config
        self.assertEqual(self.model.block_size, self.config.block_size)

        # Ensure all layers are properly initialized
        self.assertIsInstance(self.model.transformer, nn.ModuleDict)
        self.assertIsInstance(self.model.transformer["lookup_tok_emb"], nn.Embedding)
        self.assertIsInstance(self.model.transformer["lookup_pos_emb"], nn.Embedding)
        self.assertIsInstance(self.model.transformer["ln_f"], nn.LayerNorm)
        self.assertIsInstance(self.model.transformer["h"], nn.ModuleList)
        self.assertEqual(
            len(self.model.transformer["h"]),
            self.config.n_layer,
            msg=f"Transformer should have {self.config.n_layer} transformer blocks!",
        )

        # Check that lm_head is correctly initialized
        self.assertIsInstance(self.model.lm_head, nn.Linear)
        self.assertEqual(
            self.model.lm_head.weight.shape,
            (self.config.vocab_size, self.config.n_embd),
            msg="lm_head must have shape (vocab_size, n_embd)",
        )

    def test_lookup_embedding_layers(self):
        batch_size = 3
        seq_length = 5
        idx = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        pos = torch.arange(0, seq_length).unsqueeze(0)

        tok_emb = self.model.transformer["lookup_tok_emb"](idx)
        pos_emb = self.model.transformer["lookup_pos_emb"](pos)

        self.assertEqual(
            tok_emb.shape,
            (batch_size, seq_length, self.config.n_embd),
            msg="Token embedding must have shape (B, T, n_embd)",
        )
        self.assertEqual(
            pos_emb.shape,
            (1, seq_length, self.config.n_embd),
            msg="Position embedding must have shape (1, T, n_embd)",
        )

        # Verify that broadcasting is done correctly
        x = tok_emb + pos_emb
        self.assertEqual(
            x.shape,
            (batch_size, seq_length, self.config.n_embd),
            msg="Sum of token embeddings and positional embeddings must have shape "
            "(B, T, n_embd) because the positional embeddings are broadcasted across "
            "the batches!",
        )

    def test_forward_without_targets(self):
        # Generate a batch of input sequences
        batch_size, seq_length = 3, 5
        idx = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))

        # Forward pass without targets
        logits, loss = self.model(idx)
        self.assertEqual(
            logits.shape,
            (batch_size, seq_length, self.config.vocab_size),
            msg="Output logits must have shape (B, T, vocab_size)",
        )
        self.assertIsNone(loss, msg="Loss should be None if targets are not provided!")

    def test_forward_with_targets(self):
        """Since the Transformer predicts the next character at each position of the
        input sequence, the target should have the same shape as the input sequence
        `idx` to provide a target for the prediction at each position.
        """
        torch.manual_seed(42)

        # Craft an example output of `SequenceDataset.__getitem__()`
        x = torch.tensor([0,  5, 13, 13, 1,  0,  0,  0,  0,  0,  0,  0])  # fmt: skip
        y = torch.tensor([5, 13, 13,  1, 0, -1, -1, -1, -1, -1, -1, -1])  # fmt: skip

        # Reshape to (B, T)
        x = x.view(-1, self.config.block_size)
        y = y.view(-1, self.config.block_size)

        # Infer batch size and sequence length
        B, T = x.shape

        # Forward pass with targets
        logits, loss = self.model(idx=x, targets=y)

        self.assertEqual(
            logits.shape,
            (B, T, self.config.vocab_size),
            msg="Logits must have shape (B, T, vocab_size)",
        )
        self.assertIsNotNone(
            loss, msg="Loss should not be None if targets are provided!"
        )
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(
            loss.item(), float, msg="Loss tensor should contain a single float!"
        )

        # Check that F.cross_entropy() with `ignore_index=-1` correctly ignores index
        # values of -1 by comparing the loss
        x_trunc = torch.tensor([0,  5, 13, 13, 1,  0])  # fmt: skip
        y_trunc = torch.tensor([5, 13, 13,  1, 0, -1])  # fmt: skip
        x_trunc = x_trunc.view(-1, self.config.block_size)
        y_trunc = y_trunc.view(-1, self.config.block_size)
        _, loss_trunc = self.model(idx=x_trunc, targets=y_trunc)
        self.assertTrue(torch.equal(loss, loss_trunc))

    def test_forward_sequence_too_long(self):
        # Check that an assertion is raised for sequence length > block size
        batch_size = 1
        idx = torch.randint(
            0, self.config.vocab_size, (batch_size, self.config.block_size + 10)
        )
        with self.assertRaises(AssertionError):
            self.model(idx)

    def test_generate(self):
        # Set a random seed for reproducibility
        torch.manual_seed(0)

        # Input tensor: start token (".") with shape (B=1, T=1)
        idx = torch.zeros((1, 1), dtype=torch.long)
        B, T0 = idx.size()
        max_new_chars = 5  # Number of new tokens to generate

        # Generate new tokens
        generated_seq = self.model.generate(idx, max_new_chars)

        # Run a few checks on the generated sequence
        self.assertEqual(
            generated_seq.shape,
            (B, T0 + max_new_chars),
            msg=f"Generated sequence must have shape (B=1, T={T0 + max_new_chars}).",
        )
        self.assertTrue(
            torch.equal(generated_seq[:, :T0], idx),
            msg="The first part of the generated sequence must be the same as the input!",
        )
        self.assertTrue(
            torch.all(generated_seq >= 0)
            and torch.all(generated_seq < self.config.vocab_size),
            msg="Generated sequence must only contain valid character indices (ranging"
            "from 0 to vocab_size-1)",
        )

        # Try another input tensor that is longer: ".em" with shape (B=1, T=3)
        idx = torch.tensor([[0, 5, 13]], dtype=torch.long)
        B, T0 = idx.size()
        max_new_chars = 3
        generated_seq = self.model.generate(idx, max_new_chars)
        self.assertEqual(
            generated_seq.shape,
            (B, T0 + max_new_chars),
            msg=f"Generated sequence must have shape (B={B}, T={T0 + max_new_chars}).",
        )
        self.assertTrue(
            torch.equal(generated_seq[:, :T0], idx),
            msg="The first part of the generated sequence must be the same as the input!",
        )
        self.assertTrue(
            torch.all(generated_seq >= 0)
            and torch.all(generated_seq < self.config.vocab_size),
            msg="Generated sequence must only contain valid character indices!",
        )


if __name__ == "__main__":
    unittest.main()
