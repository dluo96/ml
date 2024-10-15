import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Minimal implementation of rotary position embedding (RoPE).

    Paper: https://arxiv.org/abs/2104.09864 (Jianlin Su et al., 2021).
    Implementation is inspired by:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

    Rather than adding a position embedding to each token embedding, RoPE applies a
    rotation to the token embedding. The angle of rotation is proportional to the
    position of the token in the sequence.

    Benefits of RoPE include:
        - Preservation of relative positions: two given tokens will maintain their
            relative distance (angle) even in different contexts.
        - The closer two tokens are, the smaller the angle between them and thus the
            higher their cosine similarity.
        - Stability of vectors: adding tokens at the end of a sentence doesn't affect
            the embeddings for tokens at the beginning, facilitating efficient caching.
    """

    def __init__(self, rope_theta: float, head_dim: int):
        super().__init__()
        base = rope_theta
        D = head_dim

        # Pre-compute and save inverse frequencies as a buffer (not a model parameter)
        # This is the frequencies 1/b^(0/D), 1/b^(2/D), ..., 1/b^((D-2)/D)
        # Note: this is similar to the absolute position embeddings in the original
        # paper 'Attention is All You Need'.
        inv_freq = 1.0 / (base ** (torch.arange(0, D, 2).float() / D))  # (D/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create the vectors of cosines and sines for the different positions:
        [cos(θ), cos(θ), cos(2θ), cos(2θ), ..., cos((D/2)θ), cos((D/2)θ)]
        [sin(θ), sin(θ), sin(2θ), sin(2θ), ..., sin((D/2)θ), sin((D/2)θ)]
        """
        B = pos_ids.shape[0]
        D = self.inv_freq.shape[0] * 2

        # (D/2,) -> (1, D/2, 1) -> (B, D/2, 1)
        inv_freq = self.inv_freq[None, :, None].expand(B, -1, 1)

        # (B, T) -> (B, 1, T)
        pos_ids = pos_ids[:, None, :].float()

        # (B, D/2, 1) @ (B, 1, T) -> (B, D/2, T) -> (B, T, D/2)
        freqs = (inv_freq @ pos_ids).transpose(1, 2)

        # Interleave by (a) stacking the two copies along a new dimension and then
        # (b) flattening the last two dimensions.
        emb = torch.stack((freqs, freqs), dim=-1)  # (B, T, D/2, 2)
        emb = emb.flatten(-2)  # (B, T, D)

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin

    def swap_negate_pairwise(self, x: torch.Tensor) -> torch.Tensor:
        D = x.shape[-1]
        assert D % 2 == 0, "RoPE operates on pairs of features, so D must be even."

        # Take elements at even positions [x1, x3, x5, ...]
        x_at_even = x[..., 0::2]  # (..., D/2)

        # Take elements at odd positions [x2, x4, x6, ...]
        x_at_odd = x[..., 1::2]  # (..., D/2)

        # Negate elements at odd positions to get [-x2, -x4, ..., -x_D]
        x_at_odd *= -1

        # Interleave to get [-x2, x1, -x4, x3, ..., -x_D, x_{D-1}]
        t = torch.stack((x_at_odd, x_at_even), dim=-1).view(-1, D)  # (..., D)

        return t

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies RoPE (Rotary Position Embedding) to the query and key tensors.

        This is done just before the self-attention is computed.

        Args:
            q: query tensor with shape (B, H, T, D).
            k: key tensor with shape (B, H, T, D).

        Returns:
            Rotated (via RoPE) query and key tensors.
        """
        B, H, T, D = q.shape

        pos_ids = torch.arange(T, device=q.device).expand(B, T)
        cos, sin = self.get_cos_sin(pos_ids=pos_ids)  # (B, T, D) for both

        # Implement equation (34) from the paper
        # Broadcasting: (B, H, T, D) * (B, T, D) -> (B, H, T, D)
        q_rope = (q * cos) + (self.swap_negate_pairwise(q) * sin)
        k_rope = (k * cos) + (self.swap_negate_pairwise(k) * sin)

        return q_rope, k_rope
