import torch
from torch.utils.data import Dataset


class MultiCharDataset(Dataset):
    def __init__(self, words: list[str], block_size: int):
        self.words = words
        self.unique_chars = ["."] + sorted(list(set("".join(words))))
        self.ctoi = {c: i for i, c in enumerate(self.unique_chars)}
        self.itoc = {i: c for i, c in enumerate(self.unique_chars)}

        # Context length denotes how many characters we use to predict the next one
        self.block_size = block_size

        # Pre-compute dataset
        self.X, self.Y = self._create_dataset(self.words)

    def __len__(self) -> int:
        return self.Y.numel()

    def _create_dataset(self, words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Create input-target pairs from all the words in the dataset.

        For example, for `block_size=3`, the word "emma", will produce 5 examples
        corresponding to:
            - "..." -> "e"
            - "..e" -> "m"
            - ".em" -> "m"
            - "emm" -> "a"
            - "mma" -> "."

        In other words,
            - When NN input is 0, 0, 0 (".", ".", "."), the desired label is 5 ("e")
            - When NN input is 0, 0, 5 (".", ".", "e"), the desired label is 13 ("m")
            - When NN input is 0, 5, 13 (".", "e", "m"), the desired label is 13 ("m")
            - When NN input is 5, 13, 13 ("e", "m", "m"), the desired label is 1 ("a")
            - When NN input is 13, 13, 1 ("m", "m", "a"), the desired label is 0 (".")
        """
        xs, ys = [], []
        for w in words:
            context = [0] * self.block_size  # Begin with `block_size` start tokens "."
            for ch in w + ".":
                ix = self.ctoi[ch]
                xs.append(context)
                ys.append(ix)
                context = context[1:] + [ix]  # Crop and append

        xs = torch.tensor(xs)
        ys = torch.tensor(ys)

        return xs, ys

    def get_vocab_size(self) -> int:
        return len(self.unique_chars)

    def get_output_length(self) -> int:
        return self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
