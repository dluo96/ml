import pathlib
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from lm.datasets.char_dataset import CharDataset
from lm.models.bigram import Bigram
from lm.trainer import Trainer


@dataclass
class ModelConfig:
    vocab_size: int


def main() -> None:
    # Data
    data_dir = pathlib.Path(__file__).parent
    words = open(f"{data_dir}/names.txt", "r").read().splitlines()
    dataset = CharDataset(words)
    train_loader = DataLoader(dataset, batch_size=2)

    # Model
    vocab_size = dataset.get_vocab_size()
    config = ModelConfig(vocab_size=vocab_size)
    model = Bigram(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8
    )

    # Consolidate everything in the trainer
    trainer = Trainer(
        num_epochs=100,
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
    )

    # Launch training
    trainer.train()


if __name__ == "__main__":
    main()
