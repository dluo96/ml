import argparse
import pathlib

import torch
from torch.utils.data import DataLoader

from lm.datasets import CharDataset, MultiCharDataset, SequenceDataset
from lm.model_config import ModelConfig
from lm.models import MLP, RNN, Bigram, Transformer
from lm.trainer import Trainer


def main() -> None:
    # fmt: off
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Character-level language models")

    # System and I/O
    parser.add_argument("--input-file", type=str, default="names.txt", help="Input file with one word per line")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu|cuda)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Model
    parser.add_argument("--type", type=str, default="transformer", help="Model to use: bigram|mlp|rnn|gru|transformer")
    parser.add_argument("--n-embd", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--n-embd2", type=int, default=64, help="Second embedding dimension")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of consecutive transformer blocks")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads in each transformer block")

    # Optimization
    parser.add_argument("--batch-size", type=int, default=2**10, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num-epochs", type=int, default=10_000, help="Number of complete passes through the dataset")
    # fmt: on

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Create dataset and dataloader
    words = open(args.input_file, "r").read().splitlines()
    if args.type == "bigram":
        dataset = CharDataset(words)
    elif args.type == "mlp":
        dataset = MultiCharDataset(words, block_size=3)
    elif args.type in ["rnn", "gru", "transformer"]:
        dataset = SequenceDataset(words)
    else:
        raise ValueError(f"Model type {args.type} is not recognized!")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Extract vocabulary size and block size
    vocab_size = dataset.get_vocab_size()
    block_size = dataset.get_output_length()  # Only used for RNN/GRU/Transformer

    # Create model
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=args.n_embd,
        n_embd2=args.n_embd2,
        n_layer=args.n_layer,
        n_head=args.n_head,
    )
    if args.type == "bigram":
        model = Bigram(config)
    elif args.type == "mlp":
        model = MLP(config)
    elif args.type == "rnn":
        model = RNN(config, cell_type="rnn")
    elif args.type == "gru":
        model = RNN(config, cell_type="gru")
    elif args.type == "transformer":
        model = Transformer(config)
    else:
        raise ValueError(f"Model type {args.type} is not recognized!")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    # Consolidate everything in the trainer
    trainer = Trainer(
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
    )

    # Launch training
    trainer.train()


if __name__ == "__main__":
    main()
