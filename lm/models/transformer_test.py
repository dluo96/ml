import math
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from lm.models.transformer import Block, CausalSelfAttention, NewGELU
from lm.types import ModelConfig


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
            msg="Output must have shape (batch_size, seq_length, embedding dim.)!",
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


if __name__ == "__main__":
    unittest.main()
