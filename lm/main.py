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

    # Load data
    words = open(args.input_file, "r").read().splitlines()

    if args.type == "bigram":
        dataset = CharDataset(words)
        vocab_size = dataset.get_vocab_size()
        train_loader = DataLoader(dataset, batch_size=args.batch_size)
        config = ModelConfig(vocab_size=vocab_size)
        model = Bigram(config)
    elif args.type == "mlp":
        # import random
        # random.seed(0)
        # random.shuffle(words)
        # n1 = int(0.8 * len(words))
        # n2 = int(0.9 * len(words))
        # train_words, val_words, test_words = words[:n1], words[n1:n2], words[n2:]
        dataset = MultiCharDataset(words, block_size=3)
        vocab_size = dataset.get_vocab_size()
        block_size = dataset.get_output_length()
        train_loader = DataLoader(dataset, batch_size=args.batch_size)
        config = ModelConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=64,
            n_embd2=64,
        )
        model = MLP(config)
    elif args.type == "rnn":
        dataset = SequenceDataset(words)
        vocab_size = dataset.get_vocab_size()
        block_size = dataset.get_output_length()
        train_loader = DataLoader(dataset, batch_size=args.batch_size)
        config = ModelConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=64,
            n_embd2=64,
        )
        model = RNN(config, cell_type="rnn")
    elif args.type == "gru":
        dataset = SequenceDataset(words)
        vocab_size = dataset.get_vocab_size()
        block_size = dataset.get_output_length()
        train_loader = DataLoader(dataset, batch_size=args.batch_size)
        config = ModelConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=64,
            n_embd2=64,
        )
        model = RNN(config, cell_type="gru")
    elif args.type == "transformer":
        dataset = SequenceDataset(words)
        vocab_size = dataset.get_vocab_size()
        block_size = dataset.get_output_length()
        train_loader = DataLoader(dataset, batch_size=args.batch_size)
        config = ModelConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=64,
            n_layer=4,
            n_head=4,
        )
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
