import pathlib
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from lm.datasets.char_dataset import CharDataset
from lm.datasets.multi_char_dataset import MultiCharDataset
from lm.models.bigram import Bigram
from lm.models.mlp import MLP
from lm.trainer import Trainer


@dataclass
class ModelConfig:
    vocab_size: int | None = None
    block_size: int | None = None
    n_embd: int | None = None
    n_embd2: int | None = None


def main() -> None:
    choice = "mlp"

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
        dataset = MultiCharDataset(words, block_size=3)
        vocab_size = dataset.get_vocab_size()
        train_loader = DataLoader(dataset, batch_size=2**10)
        config = ModelConfig(vocab_size=vocab_size, block_size=3, n_embd=64, n_embd2=64)
        model = MLP(config)
    else:
        raise ValueError(f"Model type {choice} is not recognized!")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8
    )

    # Consolidate everything in the trainer
    trainer = Trainer(
        num_epochs=10000,
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
    )

    # Launch training
    trainer.train()


if __name__ == "__main__":
    main()
