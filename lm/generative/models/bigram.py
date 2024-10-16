import torch
import torch.nn as nn
from torch.nn import functional as F

from lm.generative.model_config import ModelConfig, Tensor


class Bigram(nn.Module):
    """Bigram 'neural network' language model: simply a lookup table of logits for the
    next character given a previous character. This 'neural network' has a single layer
    with no activation function. The average negative log likelihood (NLL) is minimised
    using gradient descent.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        N = cfg.vocab_size
        self.logits = nn.Parameter(torch.zeros(N, N))

    def get_block_size(self) -> int:
        return 1  # Bigram model only uses one previous character to predict the next

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        # Forward pass
        logits = self.logits[idx]
        """
        Equivalently, the forward pass can be expressed as a matrix multiplication:

        x = F.one_hot(torch.tensor([ix]), num_classes=self.vocab_size).float()
        logits = x @ self.logits

        The lookup table can be interpreted in two equivalent ways:
        1. Indexing the character index into the lookup table.
        2. The first layer of the neural net - this layer doesn't have any
            non-linearity and the weight matrix is simply the lookup table.
            This interpretation would require each input character (index)
            to be one-hot encoded before being multiplied by the lookup table.
        """
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_chars: int) -> Tensor:
        self.eval()
        for _ in range(max_new_chars):
            # Get the previous character (for each batch) since bigram only looks at
            # the previous character to get the next
            prev_idx = idx[:, -1].unsqueeze(dim=-1)  # (B, 1)

            # Forward pass to obtain the logits
            logits, _ = self.forward(prev_idx)  # (B, 1, V)

            # Focus only on the last "time step"
            logits = logits[:, -1, :]  # (B, V)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, V)

            # Sample from the distribution to get the next character
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)

        return idx
