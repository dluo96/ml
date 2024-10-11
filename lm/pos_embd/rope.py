import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Minimal implementation of rotary position embedding (RoPE).

    Paper: https://arxiv.org/abs/2104.09864 (Jianlin Su et al., 2021).
    Implementation is inspired by:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    """

    def __init__(self, rope_theta: float, head_dim: int):
        super().__init__()
        base = rope_theta
        D = head_dim

        # Compute and save inverse frequencies as a buffer (not a model parameter)
        inv_freq = 1.0 / (base ** (torch.arange(0, D, 2).float() / D))  # (D/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = pos_ids.shape[0]

        # (D/2,) -> (1, D/2, 1) -> (B, D/2, 1)
        inv_freq = self.inv_freq[None, :, None].expand(B, -1, 1)

        # (B, T) -> (B, 1, T)
        pos_ids = pos_ids[:, None, :].float()

        # (B, D/2, 1) @ (B, 1, T) -> (B, D/2, T) -> (B, T, D/2)
        freqs = (inv_freq @ pos_ids).transpose(1, 2)

        emb = torch.cat((freqs, freqs), dim=-1)  # (B, T, D)
        cos = emb.cos()
        sin = emb.sin()

        return cos, sin

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden features (last dimension/axis) of the input."""
        D = x.shape[-1]
        first_half = x[..., : D // 2]  # (..., D/2)
        second_half = x[..., D // 2 :]  # (..., D/2)
        rotated = torch.cat((-second_half, first_half), dim=-1)  # (..., D)
        return rotated

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies RoPE (Rotary Position Embedding) to the query and key tensors.

        Args:
            q: query tensor with shape (B, H, T, D).
            k: key tensor with shape (B, H, T, D)

        Returns:
            Rotated (via RoPE) query and key tensors.
        """
        B, H, T, D = q.shape

        pos_ids = torch.arange(T, device=q.device).expand(B, T)

        cos, sin = self.get_cos_sin(pos_ids=pos_ids)  # (B, T, D) for both

        q_embd = (q * cos) + (self.rotate_half(q) * sin)
        k_embd = (k * cos) + (self.rotate_half(k) * sin)

        return q_embd, k_embd
