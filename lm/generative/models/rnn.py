import torch
import torch.nn as nn
import torch.nn.functional as F

from lm.generative.model_config import ModelConfig
from lm.tensor import Tensor


class RNN(nn.Module):
    """Recurrent neural network (RNN) which takes the previous `block_size` characters,
    embeds them with a lookup table, and sequentially processes them through an RNN
    cell to predict the next character based on the hidden state at each step.
    """

    def __init__(self, cfg: ModelConfig, cell_type: str):
        super().__init__()
        self.block_size = cfg.block_size
        self.vocab_size = cfg.vocab_size

        # Initialise the beginning hidden state h_{0}
        # The 1 means it is for a single sequence
        self.start = nn.Parameter(torch.zeros(1, cfg.n_embd2))

        # Define model layers
        self.lookup_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        if cell_type == "rnn":
            self.cell = RNNCell(cfg)
        elif cell_type == "gru":
            self.cell = GRUCell(cfg)
        else:
            raise ValueError(f"Cell type {cell_type} is not recognised!")

        self.lm_head = nn.Linear(cfg.n_embd2, self.vocab_size)

    def get_block_size(self) -> int:
        return self.block_size

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        # Batch size (B) and sequence length (T)
        B, T = idx.size()

        # Find the embedding for each index
        emb = self.lookup_table(idx)  # (B, T, n_embd)

        # Create a batch of starting hidden states
        h_prev = self.start.expand((B, -1))

        # Sequentially iterate over the input characters and
        # update the RNN state each tick
        hiddens = []
        for i in range(T):
            xt = emb[:, i, :]  # (B, n_embd)
            ht = self.cell(xt, h_prev)  # (B, n_embd2)
            h_prev = ht
            hiddens.append(ht)

        # Decode the outputs
        hidden = torch.stack(hiddens, 1)  # (B, T, n_embd2)
        logits = self.lm_head(hidden)  # (B, T, V)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B, T, V) -> (B * T, V)
                targets.view(-1),  # (B, T) -> (B * T,)
                ignore_index=-1,  # Specifies a target value that is ignored and does
                # not contribute to the input gradient
            )

        return logits, loss


class RNNCell(nn.Module):
    """The job of a 'Cell' is to: take input at current time step x_{t} and the hidden
    state at the previous time step h_{t-1} and return the resulting hidden state h_{t}
    at the current timestep.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Define layer whose input is the concatenated x_{t} and h_{t-1}, and whose
        # output is h_{t}
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt: Tensor, h_prev: Tensor) -> Tensor:
        xh = torch.cat([xt, h_prev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht


class GRUCell(nn.Module):
    """Compared to the RNN cell, the GRU cell uses a slightly more complicated
    recurrence formula that makes the GRU more expressive and easier to optimize.
    """

    def __init__(self, config):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt: Tensor, hprev: Tensor) -> Tensor:
        # Concatenate x_{t} and h_{t-1}
        xh = torch.cat([xt, hprev], dim=1)

        # Use the reset gate to wipe some channels of the hidden state to zero
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev

        # Calculate the candidate new hidden state, hbar
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))

        # Calculate the switch gate that determines if each channel should be updated at all
        z = F.sigmoid(self.xh_to_z(xh))

        # Blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar

        return ht
