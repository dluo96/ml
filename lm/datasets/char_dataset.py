import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, words: list[str]):
        self.words = words
        self.unique_chars = ["."] + sorted(list(set("".join(words))))
        self.ctoi = {c: i for i, c in enumerate(self.unique_chars)}
        self.itoc = {i: c for i, c in enumerate(self.unique_chars)}

        # Pre-compute dataset
        self.X, self.Y = self._create_dataset()

    def __len__(self) -> int:
        return self.X.numel()

    def _create_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create input-target pairs from all the words in the dataset.

        For example, the word 'emma' will produce 5 examples corresponding to:
            - "." -> "e"
            - "e" -> "m"
            - "m" -> "m"
            - "m" -> "a"
            - "a" -> "."

        In other words,
            - When the input to the NN is 0 ("."), the desired label is 5 ("e")
            - When the input to the NN is 5 ("e"), the desired label is 13 ("m")
            - When the input to the NN is 13 ("m"), the desired label is 13 ("m")
            - When the input to the NN is 13 ("m"), the desired label is 1 ("a")
            - When the input to the NN is 1 ("a"), the desired label is 0 (".")
        """
        xs, ys = [], []
        for w in self.words:
            chs = ["."] + list(w) + ["."]
            for ch1, ch2 in zip(chs, chs[1:]):  # zip truncates the longer iterable
                ix1 = self.ctoi[ch1]
                ix2 = self.ctoi[ch2]
                xs.append(ix1)
                ys.append(ix2)

        xs = torch.tensor(xs, dtype=torch.long)
        ys = torch.tensor(ys, dtype=torch.long)

        return xs, ys

    def get_vocab_size(self) -> int:
        return len(self.unique_chars)

    def get_output_length(self) -> int:
        return 1

    def encode(self, word: str) -> torch.Tensor:
        ix = torch.tensor([self.ctoi[ch] for ch in word])
        return ix

    def decode(self, ix: torch.Tensor) -> str:
        word = "".join(self.itoc[i.item()] for i in ix)
        return word

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
