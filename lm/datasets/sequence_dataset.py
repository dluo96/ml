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

    def get_output_length(self) -> int:
        return self.max_word_length

    def encode(self, word: str) -> torch.Tensor:
        ix = torch.tensor([self.ctoi[ch] for ch in word])
        return ix

    def decode(self, ix: torch.Tensor) -> str:
        word = "".join(self.itoc[i.item()] for i in ix)
        return word

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the input and target tensors for a given index.

        For example, for `max_word_length=10`, suppose we have the idx representing the
        word "emma". Then, the input and target are:
            - Input:  [0,  5, 13, 13, 1,  0,  0,  0,  0,  0,  0]
            - Target: [5, 13, 13,  1, 0, -1, -1, -1, -1, -1, -1]
        """
        word = self.words[idx]
        ix = self.encode(word)

        # Initialise input/target tensors with padding (+1 is for start token)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)

        # Populate the input. The right shift of 1 leaves room for the start token "."
        x[1 : 1 + len(ix)] = ix

        # Populate the target. Copy the encoded word to target tensor
        y[: len(ix)] = ix

        # Mask the loss (calculated later using F.cross_entropy) for inactive positions
        # with -1. This is done with the keyword argument `ignore_index=-1`, which
        # specifies a target value that is ignored.
        y[len(ix) + 1 :] = -1

        return x, y
