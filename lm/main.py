import pathlib

import torch
from torch.utils.data import DataLoader

from lm.datasets.char_dataset import CharDataset
from lm.datasets.multi_char_dataset import MultiCharDataset
from lm.model_config import ModelConfig
from lm.models.bigram import Bigram
from lm.models.mlp import MLP
from lm.trainer import Trainer


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
        # import random
        # random.seed(0)
        # random.shuffle(words)
        # n1 = int(0.8 * len(words))
        # n2 = int(0.9 * len(words))
        # train_words, val_words, test_words = words[:n1], words[n1:n2], words[n2:]
        dataset = MultiCharDataset(words, block_size=3)
        vocab_size = dataset.get_vocab_size()
        train_loader = DataLoader(dataset, batch_size=2**14)
        config = ModelConfig(vocab_size=vocab_size, block_size=3, n_embd=64, n_embd2=64)
        model = MLP(config)
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
