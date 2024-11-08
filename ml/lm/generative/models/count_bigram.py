import torch

from ml.tensor import Tensor


class Bigram:
    """Bigram character-level language model where 'training' consists of counting
    the frequency of all bigrams.
    """

    def __init__(self, words: list[str]) -> None:
        self.words = words
        unique_chars = ["."] + sorted(list(set("".join(words))))
        self.ctoi = {c: i for i, c in enumerate(unique_chars)}
        self.itoc = {i: c for i, c in enumerate(unique_chars)}
        self.n_unique_chars = len(unique_chars)

    def create_bigram_tensor(self, words: list[str]) -> Tensor:
        # Use a single token "." in favour of "<S>" and "<E>"
        bigram_tensor = torch.zeros(
            (self.n_unique_chars, self.n_unique_chars),
            dtype=torch.int32,
        )
        for w in words:
            chs = ["."] + list(w) + ["."]
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.ctoi[ch1]
                ix2 = self.ctoi[ch2]
                bigram_tensor[ix1, ix2] += 1

        return bigram_tensor

    def make_bigrams(self, words: list[str]) -> list[tuple[str, str]]:
        bigrams = []
        for w in words:
            chs = ["<S>"] + list(w) + ["<E>"]
            for ch1, ch2 in zip(chs, chs[1:]):  # zip truncates the longer iterable
                bigrams.append((ch1, ch2))
        return bigrams

    def count_bigrams(self, words: list[str]) -> dict[tuple[str, str], int]:
        bg_counts = {}
        for w in words:
            chs = ["<S>"] + list(w) + ["<E>"]
            for ch1, ch2 in zip(chs, chs[1:]):
                bigram = (ch1, ch2)
                bg_counts[bigram] = bg_counts.get(bigram, 0) + 1
        return bg_counts

    def sample(self, words: list[str]) -> str:
        bigram_tensor = self.create_bigram_tensor(words)

        # Compute matrix of probabilities
        bigram_probs = bigram_tensor.float()
        bigram_probs /= bigram_probs.sum(dim=1, keepdim=True)

        g = torch.Generator().manual_seed(2147483647)
        out = []
        ix = 0  # Start token "." is first
        while True:
            # Get the probability distribution (over all possible characters)
            # for the next character given the current character
            p = bigram_probs[ix]

            # Sample a character index based on the bigram distribution
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()

            # Append character
            out.append(self.itoc[ix])

            # Stop if we sample the end token "."
            if ix == 0:
                break

        return "".join(out)

    def evaluate(self, words: list[str]) -> float:
        """Evaluate the model by computing the average negative log likelihood (NLL).

        Goal is to maximise the likelihood of the data with respect to the model
        parameters, which, for the case of the bigram model, are the elements of the
        bigram probability matrix.

        This is equivalent to maximising the log likelihood (because log is monotonic).
        This is equivalent to minimising the negative log likelihood.
        This is equivalent to minimising the average negative log likelihood.
        """
        # Create matrix of bigram probabilities
        # Smoothing is applied to avoid -inf in the negative log likelihood which would
        # happen for bigrams that never occurred in the dataset. The larger the smoothing
        # parameter, the closer we get to a uniform distribution
        smoothing_param = 0
        bigram_tensor = self.create_bigram_tensor(words)
        bigram_probs = (bigram_tensor + smoothing_param).float()
        bigram_probs /= bigram_probs.sum(dim=1, keepdim=True)

        log_likelihood = 0.0
        n = 0
        for w in words:
            chs = ["."] + list(w) + ["."]
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.ctoi[ch1]
                ix2 = self.ctoi[ch2]
                prob = bigram_probs[ix1, ix2]
                log_prob = torch.log(prob)
                log_likelihood += log_prob.item()
                n += 1

        nll = -log_likelihood
        avg_nll = nll / n
        return avg_nll
