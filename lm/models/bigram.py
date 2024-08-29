import torch
import torch.nn as nn
from torch.nn import functional as F


class Bigram(nn.Module):
    """Bigram 'neural network' language model: simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, config):
        super().__init__()
        N = config.vocab_size
        self.logits = nn.Parameter(torch.zeros(N, N))

    def get_block_size(self) -> int:
        return 1  # Bigram model only uses one previous character to predict the next

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.logits[idx]  # Forward pass
        """
        Equivalently, the forward pass can be expressed as a matrix multiplication:

        x = F.one_hot(torch.tensor([ix]), num_classes=self.vocab_size).float()
        logits = x @ self.logits
        """
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss
