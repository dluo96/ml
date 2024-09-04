import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lm.model_config import ModelConfig, Tensor


class NewGELU(nn.Module):
    """Implementation of the GELU activation function currently in the Google BERT repo
    (identical to OpenAI GPT). Reference: Gaussian Error Linear Units (GELU) paper:
    https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip


class CausalSelfAttention(nn.Module):
    """A vanilla multi-head masked self-attention layer with a projection at the end.
    Alternatively, it is possible to use `torch.nn.MultiheadAttention`.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Embedding dimensionality must be divisible by number of heads
        assert config.n_embd % config.n_head == 0

        # Key, query, and value projections for all heads, but in a batch.
        # The 3 is because of key, query, and value
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Causal mask to ensure that attention is only applied to the left in the input
        # sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()  # Batch size, sequence length, embedding dim. (n_embd)

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # Each has shape (B, T, C)

        # Multi-head attention computes multiple attention heads in parallel, allowing
        # each head to focus on different aspects of the input data relationships
        # fmt: off
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # fmt: on

        # Causal self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask to prevent attending to future positions by setting those
        # positions to -inf, ensuring they become zero after applying softmax
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Combine the output from all attention heads by transposing and reshaping
        # back to the original embedding dimension:
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Output projection
        y = self.c_proj(y)  # (B, T, C)

        return y


class Block(nn.Module):
    """A transformer block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=NewGELU(),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))  # MLP forward

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Transformer language model, exactly as seen in GPT-2.

    Represents a decoder-only architecture used for autoregressive tasks, such as text
    generation (like GPT-2). There is no separate encoder, as this model is not
    designed for tasks that require encoding a source sequence and decoding to a
    target sequence (like machine translation). Instead, it is focused on generating
    text based on previous context.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.block_size = config.block_size

        # Define model layers. Importantly, the token embeddings and positional
        # embeddings are learnable parameters!
        self.transformer = nn.ModuleDict(
            dict(
                lookup_tok_emb=nn.Embedding(config.vocab_size, config.n_embd),
                lookup_pos_emb=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Report number of parameters, excluding the decoder parameters in `lm_head`
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"Number of parameters: {(n_params / 1e6):.2f}M")

    def get_block_size(self) -> int:
        return self.block_size

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        B, T = idx.size()

        err_msg = f"Cannot forward sequence of length {T} because {self.block_size=}."
        assert T <= self.block_size, err_msg

        pos = torch.arange(0, T, dtype=torch.long).unsqueeze(0)  # shape (1, T)

        # Forward the GPT model itself
        tok_emb = self.transformer.lookup_tok_emb(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.lookup_pos_emb(pos)  # (1, T, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Project the hidden states of the transformer blocks back to the size of the
        # vocabulary to predict the next token. The output of this layer is used to
        # generate probabilities over the vocabulary for each token position in the
        # sequence.
        logits = self.lm_head(x)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B, T, V) -> (B * T, V)
                targets.view(-1),  # (B, T) -> (B * T,)
                ignore_index=-1,  # Specifies a target value that is ignored
            )

        return logits, loss
