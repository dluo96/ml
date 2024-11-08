import math

import torch.nn.functional as F
from torch import nn

from ml.lm.bert.config import BertConfig
from ml.lm.bert.embeddings import BertEmbeddings
from ml.lm.generative.model_config import Tensor
from ml.lm.generative.models import NewGELU


class MultiHeadedSelfAttention(nn.Module):
    """Multi-headed attention with an optional mask that can be used for MLM (Masked
    Language Modelling), which is one of BERT's pre-training tasks.
    """

    def __init__(self, cfg: BertConfig):
        super().__init__()
        # Embedding dimensionality must be divisible by number of heads
        assert cfg.n_embd % cfg.n_heads == 0

        # Key, query, and value projections for all heads
        self.proj_q = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj_k = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj_v = nn.Linear(cfg.n_embd, cfg.n_embd)

        # Regularization
        self.dropout = nn.Dropout(cfg.p_drop_attn)

        self.n_heads = cfg.n_heads

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, T, C = x.size()  # Batch size, sequence length, embedding dim. (n_embd)
        q = self.proj_q(x)  # (B, T, C)
        k = self.proj_k(x)  # (B, T, C)
        v = self.proj_v(x)  # (B, T, C)

        # Multi-head attention computes multiple attention heads in parallel, allowing
        # each head to focus on different aspects of the input data relationships
        # fmt: off
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # fmt: on

        # Compute attention: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        attn = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))

        # Apply mask in accordance with MLM (masked language modelling)
        # NOTE: this is the most important distinction between BERT (encoder-only) and
        # GPT (decoder-only) models. BERT uses a bidirectional mask, while GPT uses a
        # causal mask.
        if mask is not None:
            mask = mask[:, None, None, :].float()  # (B, T) -> (B, 1, 1, T)

            # Apply mask to prevent attending to the masked positions by setting those
            # attention weights to -inf, which results in their softmax being zero
            attn = attn - float("inf") * (1.0 - mask)

        # Apply softmax to the attention weights
        attn = F.softmax(attn, dim=-1)

        # Apply dropout to the attention weights
        attn = self.dropout(attn)

        # Multiply the attention weights by the values
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = attn @ v

        # Combine the output from all attention heads by transposing and reshaping
        # back to the original embedding dimension:
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, nh * hs) = (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return y


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.n_embd)
        self.gelu = NewGELU()

    def forward(self, x: Tensor) -> Tensor:
        # (B, T, D) -> (B, T, dim_ff) -> (B, T, D)
        return self.fc2(self.gelu(self.fc1(x)))


class Block(nn.Module):
    """A BERT-style transformer block."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.pwff = PositionWiseFeedForward(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        h = self.attn(x, mask)
        h = self.ln_1(x + self.dropout(self.proj(h)))
        h = self.ln_2(h + self.dropout(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """Transformer encoder, as seen in BERT (bidirectional encoder representations
    from transformers).

    Represents an encoder-only architecture used for tasks such as natural language
    inference (NLI), named entity recognition (NER), question answering, sentence
    classification, etc.

    GPT vs. BERT:
    - GPT models are decoder-only, while BERT models are encoder-only.
    - GPT uses a causal mask, where every token can only attend to previous tokens in
        the self-attention layers. In other words, GPT models are left-to-right.
        In contrast, BERT learns bidirectional representations by conditioning on both
        left and right context in all layers.
    """

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.lookup_emb = BertEmbeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x: Tensor, segment: Tensor, mask: Tensor) -> Tensor:
        """Forward pass of BERT.

        For this BERT implementation, let:
        - B denote the batch size,
        - T denote the sequence length.

        Args:
            x: input sequence of shape (B, T). Each element is an integer representing
                the index of a token in the vocabulary.
            segment: segment tensor of shape (B, T). This is used to distinguish
                between the two sentences in the input sequence when the input is a
                sentence pair.
            mask: the mask tensor of shape (B, T). This is used for Masked LM (MLM),
                which is one of BERT's pre-training tasks: MLM masks a percentage of
                the input tokens at random, and then predict those masked tokens.
        """
        x = self.lookup_emb(x, segment)
        for block in self.blocks:
            x = block(x, mask)
        return x
