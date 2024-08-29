import torch
from torch.utils.data import Dataset


class MultiCharDataset(Dataset):
    def __init__(self, words: list[str], block_size: int = 3):
        self.words = words
        self.block_size = block_size
        self.unique_chars = ["."] + sorted(list(set("".join(words))))
        self.ctoi = {c: i for i, c in enumerate(self.unique_chars)}
        self.itoc = {i: c for i, c in enumerate(self.unique_chars)}

    def __len__(self) -> int:
        return len(self.words)

    def get_vocab_size(self) -> int:
        return len(self.unique_chars)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        word = self.words[idx]
        x, y = [], []
        context = [0] * self.block_size  # Begin with `block_size` start tokens "."
        for ch in word + ".":
            ix = self.ctoi[ch]
            x.append(context)
            y.append(ix)
            context = context[1:] + [ix]  # Crop and append the current character

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
