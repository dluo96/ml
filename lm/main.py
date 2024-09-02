import pathlib

import torch
from torch.utils.data import DataLoader

from lm.datasets import CharDataset, MultiCharDataset, SequenceDataset
from lm.models import MLP, RNN, Bigram
from lm.trainer import Trainer
from lm.types import ModelConfig


def main() -> None:
    choice = "rnn"

    # Load data
    data_dir = pathlib.Path(__file__).parent
    words = open(f"{data_dir}/names.txt", "r").read().splitlines()

    if choice == "bigram":
        dataset = CharDataset(words)
        vocab_size = dataset.get_vocab_size()
        train_loader = DataLoader(dataset, batch_size=2**14)
        config = ModelConfig(vocab_size=vocab_size)
        model = Bigram(config)
    elif choice == "mlp":
        # import random
        # random.seed(0)
        # random.shuffle(words)
        # n1 = int(0.8 * len(words))
        # n2 = int(0.9 * len(words))
        # train_words, val_words, test_words = words[:n1], words[n1:n2], words[n2:]
        dataset = MultiCharDataset(words, block_size=3)
        vocab_size = dataset.get_vocab_size()
        block_size = dataset.get_output_length()
        train_loader = DataLoader(dataset, batch_size=2**14)
        config = ModelConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=64,
            n_embd2=64,
        )
        model = MLP(config)
    elif choice == "rnn":
        dataset = SequenceDataset(words)
        vocab_size = dataset.get_vocab_size()
        block_size = dataset.get_output_length()
        train_loader = DataLoader(dataset, batch_size=2**10)
        config = ModelConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=64,
            n_embd2=64,
        )
        model = RNN(config)
    else:
        raise ValueError(f"Model type {choice} is not recognized!")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8
    )

    # Consolidate everything in the trainer
    trainer = Trainer(
        num_epochs=20_000,
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
    )

    # Launch training
    trainer.train()


if __name__ == "__main__":
    main()
