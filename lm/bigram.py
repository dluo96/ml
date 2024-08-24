import torch


def make_bigrams(words: list[str]) -> list[tuple[str, str]]:
    bigrams = []
    for w in words:
        chs = ["<S>"] + list(w) + ["<E>"]
        for ch1, ch2 in zip(chs, chs[1:]):  # zip truncates the longer iterable
            bigrams.append((ch1, ch2))
    return bigrams


def count_bigrams(words: list[str]) -> dict[tuple[str, str], int]:
    bg_counts = {}
    for w in words:
        chs = ["<S>"] + list(w) + ["<E>"]
        for ch1, ch2 in zip(chs, chs[1:]):
            bigram = (ch1, ch2)
            bg_counts[bigram] = bg_counts.get(bigram, 0) + 1
    return bg_counts


def create_bigram_tensor(words: list[str]) -> torch.Tensor:
    # Use a single token "." in favour of "<S>" and "<E>"
    unique_chars = ["."] + sorted(list(set("".join(words))))
    N = len(unique_chars)
    bigram_tensor = torch.zeros((N, N), dtype=torch.int32)
    ctoi = {c: i for i, c in enumerate(unique_chars)}
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = ctoi[ch1]
            ix2 = ctoi[ch2]
            bigram_tensor[ix1, ix2] += 1

    return bigram_tensor


def sample(words: list[str]) -> str:
    unique_chars = ["."] + sorted(list(set("".join(words))))
    itoc = {i: c for i, c in enumerate(unique_chars)}
    bigram_tensor = create_bigram_tensor(words)

    # Compute matrix of probabilities
    bigram_probs = bigram_tensor.float()
    bigram_probs /= bigram_probs.sum(dim=1, keepdim=True)

    g = torch.Generator().manual_seed(2147483647)
    out = []
    ix = 0  # Start token "." is first
    while True:
        # Bigram probabilities given current character index
        p = bigram_probs[ix]

        # Sample a character index based on the bigram distribution
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

        # Append character
        out.append(itoc[ix])

        # Stop if we sample the end token "<E>"
        if ix == 0:
            break

    return "".join(out)


def evaluate(words: list[str]) -> float:
    """Evaluate the model by computing the average negative log likelihood (NLL).

    Goal is to maximise the likelihood of the data with respect to the model
    parameters, which, for the case of the bigram model, are the elements of the
    bigram probability matrix.

    This is equivalent to maximising the log likelihood (because log is monotonic).
    This is equivalent to minimising the negative log likelihood.
    This is equivalent to minimising the average negative log likelihood.
    """
    unique_chars = ["."] + sorted(list(set("".join(words))))
    ctoi = {c: i for i, c in enumerate(unique_chars)}

    # Create matrix of bigram probabilities
    # Smoothing is applied to avoid -inf in the negative log likelihood which would
    # happen for bigrams that never occurred in the dataset. The larger the smoothing
    # parameter, the closer we get to a uniform distribution
    smoothing_param = 0
    bigram_tensor = create_bigram_tensor(words)
    bigram_probs = (bigram_tensor + smoothing_param).float()
    bigram_probs /= bigram_probs.sum(dim=1, keepdim=True)

    log_likelihood = 0.0
    n = 0
    for w in words[:3]:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = ctoi[ch1]
            ix2 = ctoi[ch2]
            prob = bigram_probs[ix1, ix2]
            log_prob = torch.log(prob)
            log_likelihood += log_prob.item()
            n += 1
            print(f"{ch1}{ch2}: {prob:.4f} {log_prob:.4f}")

    nll = -log_likelihood

    print(f"{log_likelihood=}")
    print(f"{nll=}")
    print(f"{nll/n=}")

    avg_nll = nll / n
    return avg_nll


if __name__ == "__main__":
    names = open("names.txt", "r").read().splitlines()
    evaluate(names)
