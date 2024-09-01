import torch
import torch.nn as nn
import torch.nn.functional as F

from lm.model_config import ModelConfig

T = torch.Tensor


class RNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        # Initialise the beginning hidden state (1 means it is for a single sequence)
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2))

        # Define model layers
        self.lookup_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.cell = RNNCell(config)
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self) -> int:
        return self.block_size

    def forward(self, idx: T, targets: T | None = None) -> tuple[T, T | None]:
        b, t = idx.size()

        # Find the embedding for each index
        emb = self.lookup_table(idx)  # (b, t, n_embd)

        # Create a batch of starting hidden states
        h_prev = self.start.expand((b, -1))

        # Sequentially iterate over the inputs and update the RNN state each tick
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :]  # (b, n_embd)
            ht = self.cell(xt, h_prev)  # (b, n_embd2)
            h_prev = ht
            hiddens.append(ht)

        # Decode the outputs
        hidden = torch.stack(hiddens, 1)  # (b, t, n_embd2)
        logits = self.lm_head(hidden)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class RNNCell(nn.Module):
    """
    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """

    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, h_prev):
        xh = torch.cat([xt, h_prev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht
