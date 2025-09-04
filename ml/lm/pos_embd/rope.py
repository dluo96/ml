import torch
import torch.nn as nn

from ml.tensor import Tensor


class RoPE(nn.Module):
    """Minimal implementation of rotary position embedding (RoPE) from Jianlin Su
    et al., 2021 (https://arxiv.org/abs/2104.09864).

    Rather than adding a position embedding to each token embedding, RoPE introduces
    positional information through computationally efficient pairwise rotations applied
    to each token embedding.

    Specifically, in each token embedding, every pair of features is represented as a
    complex number. So if the query tensor has features [q1, q2, q3, q4], it is
    represented by `q1 + iq2` and `q3 + iq4`. Each complex number is rotated by an
    angle that depends on:
        - The position of the token in the input sequence,
        - The feature pair index (identifying the feature pair within the token
            embedding). In the example, this is 0 for [q1, q2] and 1 for [q3, q4].

    Benefits of RoPE include:
        - Preserves cosine similarity between query and key: it is the same pre-RoPE
            and post-RoPE. This is important because it ensures that RoPE doesn't
            distort the self-attention mechanism.
        - Context-independent relative position: the angle between the RoPE-rotated
            embeddings for the tokens 'cat' and 'mat' is the same in 'The cat sat on
            the mat' (position 1 and 5, resp.) and 'In the evening, the cat sat on
            the mat' (position 4 and 8, resp.). This is important because it allows
            the model to maintain a consistent semantic relationship between tokens
            with the same relative position in different contexts.
        - Decaying inter-token dependency with increasing relative positions. This
            makes sense intuitively: a pair of tokens that are far apart in the
            sequence should have less connection.
        - Stability of vectors: adding tokens at the end of a sentence doesn't affect
            the embeddings for tokens at the beginning, facilitating efficient caching.
    """

    def __init__(self, rope_theta: float, head_dim: int):
        super().__init__()
        base = rope_theta
        D = head_dim
        assert D % 2 == 0, "RoPE operates on feature pairs, so D must be even."

        # Pre-compute and save inverse frequencies as a buffer (not a model parameter)
        # The frequencies are 1/b^(0/D), 1/b^(2/D), ..., 1/b^((D-2)/D). There is one
        # frequency for each feature pair.
        i = torch.arange(1, D // 2 + 1).float()  # (D/2,)
        inv_freq = 1.0 / (base ** (2 * (i - 1) / D))  # (D/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, pos_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Create the vectors of cosines and sines for the different token positions:

            [cos(mθ_1), cos(mθ_1), cos(mθ_2), cos(mθ_2), ..., cos(mθ_{D/2})]
            [sin(mθ_1), sin(mθ_1), sin(mθ_2), sin(mθ_2), ..., sin(mθ_{D/2})]

        where `m` is the token position and

            θ_i = 1/b^(2(i-1)/D) where i = 1, 2, ..., D/2.

        See equation (15) in the paper for reference.

        Args:
            pos_ids: the token positions. Shape: (B, T).

        Returns:
            cos: the cosine of the angles. Shape: (B, T, D).
            sin: the sine of the angles. Shape: (B, T, D).

        """
        B = pos_ids.shape[0]
        D = self.inv_freq.shape[0] * 2

        # (D/2,) -> (1, D/2, 1) -> (B, D/2, 1)
        inv_freq = self.inv_freq[None, :, None].expand(B, -1, 1)

        # (B, T) -> (B, 1, T)
        pos_ids = pos_ids[:, None, :].float()

        # Frequency depends on the position
        # (B, D/2, 1) @ (B, 1, T) -> (B, D/2, T) -> (B, T, D/2)
        freqs = (inv_freq @ pos_ids).transpose(1, 2)

        # Interleave by (a) stacking the two copies along a new dimension and then
        # (b) flattening the last two dimensions.
        emb = torch.stack((freqs, freqs), dim=-1)  # (B, T, D/2, 2)
        emb = emb.flatten(-2)  # (B, T, D)

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin

    def swap_negate_pairwise(self, x: Tensor) -> Tensor:
        """Swap and negate the features of `x` pairwise:

            [x1, x2, ..., x_{D-1}, x_{D}] -> [-x2, x1, ..., -x_{D}, x_{D-1}]

        Args:
            x: the tensor to swap and negate (either the query or the key).
                Shape: (..., D).

        Returns:
            The swapped and negated tensor. Shape: (..., D).
        """
        # Features at even indices [x1, x3, ..., x_{D-1}]
        x_at_even = x[..., 0::2]  # (..., D/2)

        # Features at odd indices [x2, x4, ..., x_{D}]
        x_at_odd = x[..., 1::2]  # (..., D/2)

        # Negate elements at odd indices to get [-x2, -x4, ..., -x_D]
        # NB: cannot do this in-place since this would modify the original `x`
        # because x_at_odd is a view  of `x`.
        x_at_odd = -x_at_odd

        # Rearrange to get [-x2, x1, -x4, x3, ..., -x_{D}, x_{D-1}] by
        # (a) stacking the two tensors along a new dimension and then
        # (b) flattening the last two dimensions.
        t = torch.stack((x_at_odd, x_at_even), dim=-1)  # (..., D/2, 2)
        t = t.flatten(start_dim=-2, end_dim=-1)  # (..., D)

        return t

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Applies RoPE (Rotary Position Embedding) to the query and key.

        This is done just before the self-attention is computed.

        Args:
            q: query tensor with shape (B, H, T, D).
            k: key tensor with shape (B, H, T, D).

        Returns:
            Rotated (via RoPE) query and key tensors, each with shape (B, H, T, D).
        """
        B, H, T, D = q.shape

        pos_ids = torch.arange(T, device=q.device).expand(B, T)  # (B, T)
        cos, sin = self.get_cos_sin(pos_ids=pos_ids)  # (B, T, D) for both

        # Implement equation (34) from the paper: this is a computationally efficient
        # implementation of applying a rotation matrix to each feature pair in the
        # query and key tensors.
        q_sw_neg = self.swap_negate_pairwise(q)
        k_sw_neg = self.swap_negate_pairwise(k)

        q_rope = (q * cos) + (q_sw_neg * sin)
        k_rope = (k * cos) + (k_sw_neg * sin)
        # Note: broadcasting occurred above: (B, H, T, D) * (B, T, D) -> (B, H, T, D)

        return q_rope, k_rope
