import torch
from torch import nn

from ml.lm.bert.config import BertConfig


class BertEmbeddings(nn.Module):
    """Embedding with optional position embeddings and optional segment embeddings.

    For a given token, its input representation is constructed by summing the
    corresponding token, segment, and position embeddings.

    Importantly, all the embeddings are learnable parameters. For example, segment
    embeddings in BERT are learnable parameters that help the model distinguish
    between different segments (namely sentences in a pair) during training.
    """

    def __init__(
        self,
        config: BertConfig,
        use_pos_emb: bool = True,
        use_seg_emb: bool = True,
    ):
        super().__init__()

        # Extract configuration parameters
        n_embd = config.n_embd
        max_len = config.max_len
        n_segments = config.n_segments
        vocab_size = config.vocab_size

        # Token embeddings, positional embeddings, and segment embeddings
        self.lookup_tok_emb = nn.Embedding(vocab_size, n_embd)
        self.lookup_pos_emb = nn.Embedding(max_len, n_embd) if use_pos_emb else None
        self.lookup_seg_emb = nn.Embedding(n_segments, n_embd) if use_seg_emb else None

        # Layer normalization and dropout layers
        self.ln = nn.LayerNorm(n_embd, eps=1e-12)
        self.dropout = nn.Dropout(config.p_drop_hidden)

    def forward(self, x: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        # Token embeddings
        emb = self.lookup_tok_emb(x)

        # Positional embeddings
        if self.lookup_pos_emb is not None:
            T = x.size(1)
            pos = torch.arange(T, dtype=torch.long, device=x.device)  # (T,)
            pos = pos.unsqueeze(0).expand_as(x)  # (B, T)
            emb += self.lookup_pos_emb(pos)

        # Segment embeddings
        if self.lookup_seg_emb is not None:
            emb += self.lookup_seg_emb(segments)  # (B, T) -> (B, T, n_embd)

        # Layer normalization and dropout
        emb = self.dropout(self.ln(emb))

        return emb
