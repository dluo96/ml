import argparse
import logging

import torch
from torch.utils.data import DataLoader

from ml.lm.generative.datasets import CharDataset, MultiCharDataset, SequenceDataset
from ml.lm.generative.model_config import ModelConfig
from ml.lm.generative.models import MLP, RNN, Bigram, Transformer
from ml.lm.generative.trainer import Trainer


def main() -> None:
    # fmt: off
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Character-level language models")

    # System and I/O
    parser.add_argument("--input-file", type=str, default="names.txt", help="Input file with one word per line")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu|gpu)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Model
    parser.add_argument("--type", type=str, default="transformer", help="Model to use: bigram|mlp|rnn|gru|transformer")
    parser.add_argument("--n-embd", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--n-embd2", type=int, default=64, help="Second embedding dimension")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of consecutive transformer blocks")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads in each transformer block")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")

    # Optimization
    parser.add_argument("--batch-size", type=int, default=2**12, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num-epochs", type=int, default=10_000, help="Number of complete passes through the dataset")
    # fmt: on

    args = parser.parse_args()
    for arg_name, arg_value in vars(args).items():
        logging.info(f"{arg_name}: {arg_value}")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # Create dataset and dataloader
    words = open(args.input_file, "r").read().splitlines()

    # Split the input data into a training set and validation set
    # The validation set is the smallest of 1000 words or 10% of the dataset
    size_val_set = min(1000, int(len(words) * 0.1))
    shuffled_indices = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in shuffled_indices[:-size_val_set]]
    val_words = [words[i] for i in shuffled_indices[-size_val_set:]]
    logging.info(
        f"Split the dataset: {len(train_words)} words in the training set "
        f"and {len(val_words)} words in the validation set."
    )

    if args.type == "bigram":
        train_dataset = CharDataset(train_words)
        val_dataset = CharDataset(val_words)
    elif args.type == "mlp":
        train_dataset = MultiCharDataset(train_words, block_size=3)
        val_dataset = MultiCharDataset(val_words, block_size=3)
    elif args.type in ["rnn", "gru", "transformer"]:
        train_dataset = SequenceDataset(train_words)
        val_dataset = SequenceDataset(val_words)
    else:
        raise ValueError(f"Model type {args.type} is not recognized!")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Extract vocabulary size and block size
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()

    # Create model
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=args.n_embd,
        n_embd2=args.n_embd2,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
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

    # Calculate and report the number of model parameters
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"The {args.type} model has {(num_params / 1e6):.2f}M parameters.")

    # Determine device
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Device {args.device} is not recognized!")
    logging.info(f"Using device: {device}")

    # Move model to device
    model.to(device)

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
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    # Launch training
    trainer.train()


if __name__ == "__main__":
    main()
