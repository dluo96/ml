from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class ModelConfig:
    vocab_size: int | None = None
    block_size: int | None = None
    n_embd: int | None = None
    n_embd2: int | None = None
    n_head: int | None = None
