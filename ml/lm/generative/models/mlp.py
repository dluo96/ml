import torch.nn as nn
from torch.nn import functional as F

from ml.lm.generative.model_config import ModelConfig
from ml.tensor import Tensor


class MLP(nn.Module):
    """MLP which takes the previous `block_size` characters, embeds them with a lookup
    table, concatenates the `block_size` embeddings, and predicts the next character
    with an MLP.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.vocab_size = cfg.vocab_size
        self.block_size = cfg.block_size

        # Define model layers
        self.lookup_table = nn.Embedding(self.vocab_size, cfg.n_embd)  # Removed +1
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * cfg.n_embd, cfg.n_embd2),
            nn.Tanh(),
            nn.Linear(cfg.n_embd2, self.vocab_size),
        )

    def get_block_size(self) -> int:
        return self.block_size

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
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
            """The line `loss = F.cross_entropy(logits, targets)` is equivalent to:

            counts = logits.exp()
            p = counts / counts.sum(dim=1, keepdim=True)

            # Extract probabilities for labels (the actual characters)
            n_examples = embs.shape[0]
            p_labels = p[torch.arange(n_examples), targets]

            # Compute negative log likelihood
            loss = -p_labels.log().mean()

            However, `F.cross_entropy` is preferred for a few reasons:
                1. Forward pass is much more efficient: it does not create any additional
                    tensors - PyTorch instead runs it in a fused kernel.
                2. Backward pass is much more efficient: it is easier to backpropagate
                    through.
                3. It is more numerically stable. Internally, for each batch,
                    `F.cross_entropy` computes the max value that occurs in the logits
                    and subtracts it from all logits in that batch to prevent overflow
                    of the exp(). This is known as the log-sum-exp trick. Importantly,
                    any offset subtracted from the logits will produce the exact same
                    output probabilities.
            """

        return logits, loss
