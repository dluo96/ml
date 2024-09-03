import math
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from lm.models.transformer import CausalSelfAttention
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


if __name__ == "__main__":
    unittest.main()
