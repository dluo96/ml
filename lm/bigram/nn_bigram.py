import pathlib

import torch
import torch.nn.functional as F


class NNBigram:
    def __init__(self, words: list[str]):
        self.words = words
        unique_chars = ["."] + sorted(list(set("".join(words))))
        self.ctoi = {c: i for i, c in enumerate(unique_chars)}
        self.itoc = {i: c for i, c in enumerate(unique_chars)}
        self.n_unique_chars = len(unique_chars)
        self.W = self._initialize_weights()

    def _initialize_weights(self):
        g = torch.Generator().manual_seed(2147483647)
        return torch.randn(
            (self.n_unique_chars, self.n_unique_chars), generator=g, requires_grad=True
        )

    def _get_bigram_pairs(self, words):
        xs, ys = [], []
        for w in words:
            chs = ["."] + list(w) + ["."]
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.ctoi[ch1]
                ix2 = self.ctoi[ch2]
                xs.append(ix1)
                ys.append(ix2)
        return torch.tensor(xs), torch.tensor(ys)

    def train(self):
        xs, ys = self._get_bigram_pairs(self.words)
        n_ex = xs.numel()
        xenc = F.one_hot(xs, num_classes=self.n_unique_chars).float()

        for _ in range(100):
            # Forward pass
            logits = xenc @ self.W  # Log-counts
            counts = logits.exp()  # Un-normalised bigram tensor

            # Get probabilities for next character
            prob = counts / counts.sum(dim=1, keepdim=True)

            # Compute negative log likelihood (NLL)
            loss = -prob[torch.arange(n_ex), ys].log().mean()

            # Backward pass
            self.W.grad = None  # Zero the gradients
            loss.backward()

            with torch.no_grad():
                self.W += -50 * self.W.grad

    def sample(self) -> str:
        g = torch.Generator().manual_seed(2147483647)
        out = []
        ix = 0  # Begin with start token "."
        while True:
            # Forward pass
            xenc = F.one_hot(
                torch.tensor([ix]), num_classes=self.n_unique_chars
            ).float()
            logits = xenc @ self.W

            # Compute softmax to get probability distribution over next character
            counts = logits.exp()
            p = counts / counts.sum(dim=1, keepdim=True)

            # Sample from probability distribution
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()

            out.append(self.itoc[ix])

            # Terminate if end token "." is reached
            if ix == 0:
                break

        return "".join(out)


if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parent.parent
    names = open(f"{data_dir}/names.txt", "r").read().splitlines()

    model = NNBigram(words=names)
    model.train()
    print(model.sample())
