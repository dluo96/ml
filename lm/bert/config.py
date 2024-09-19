from dataclasses import dataclass


@dataclass
class BertConfig:
    vocab_size: int = None  # Size of Vocabulary
    n_embd: int = 768  # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12  # Number of Hidden Layers
    n_heads: int = 12  # Number of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768 * 4  # Dimension of Intermediate Layers in Position-wise FFN
    p_drop_hidden: float = 0.1  # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1  # Probability of Dropout of Attention Layers
    max_len: int = 512  # Maximum Length for Positional Embeddings
    n_segments: int = 2  # Number of Sentence Segments
