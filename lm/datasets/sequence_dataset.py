import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, words: list[str]):
        self.words = words
        self.max_word_length = max(len(w) for w in words)
        self.unique_chars = ["."] + sorted(list(set("".join(words))))
        self.ctoi = {c: i for i, c in enumerate(self.unique_chars)}
        self.itoc = {i: c for i, c in enumerate(self.unique_chars)}

    def __len__(self) -> int:
        return len(self.words)

    def get_vocab_size(self) -> int:
        return len(self.unique_chars)

    def encode(self, word: str) -> torch.Tensor:
        ix = torch.tensor([self.ctoi[ch] for ch in word])
        return ix

    def decode(self, ix: torch.Tensor) -> str:
        word = "".join(self.itoc[i.item()] for i in ix)
        return word

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        word = self.words[idx]
        ix = self.encode(word)

        # Initialise input/target tensors with padding (+1 is for start token)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)

        x[1 : 1 + len(ix)] = ix  # Shift by 1 to leave room for the start token "."
        y[: len(ix)] = ix  # Copy encoded word to target tensor
        y[len(ix) + 1 :] = -1  # Mask loss for inactive positions with -1

        return x, y
