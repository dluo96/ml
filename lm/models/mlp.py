import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    """MLP which takes the previous `block_size` tokens, embeds them with a lookup
    table, concatenates the embeddings, and predicts the next token with an MLP.
    """

    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size

        # Define model layers
        self.lookup_table = nn.Embedding(self.vocab_size, config.n_embd)  # Removed +1
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size),
        )

    def get_block_size(self) -> int:
        return self.block_size

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Get embeddings for the previous `block_size` tokens
        embs = self.lookup_table(idx)

        # To perform matrix multiplication, we need to concatenate the embeddings.
        # The shape goes from
        # (batch_size, block_size, n_embd) to (batch_size, block_size * n_embd)
        embs = embs.view(embs.shape[0], -1)

        # Pass through the MLP
        logits = self.mlp(embs)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss
