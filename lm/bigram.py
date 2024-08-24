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
    P = bigram_tensor.float()
    P /= P.sum(dim=1, keepdim=True)

    g = torch.Generator().manual_seed(2147483647)
    out = []
    ix = 0  # Start token "." is first
    while True:
        # Bigram probabilities given current character index
        p = P[ix]

        # Sample a character index based on the bigram distribution
        ix = torch.multinomial(
            input=p, num_samples=1, replacement=True, generator=g
        ).item()

        # Append character
        out.append(itoc[ix])

        # Stop if we sample the end token "<E>"
        if ix == 0:
            break

    return "".join(out)


def test_make_bigrams():
    assert make_bigrams(["emma"]) == [
        ("<S>", "e"),
        ("e", "m"),
        ("m", "m"),
        ("m", "a"),
        ("a", "<E>"),
    ]


def test_count_bigrams():
    assert count_bigrams(["emma"]) == {
        ("<S>", "e"): 1,
        ("e", "m"): 1,
        ("m", "m"): 1,
        ("m", "a"): 1,
        ("a", "<E>"): 1,
    }


def test_create_bigram_tensor():
    assert torch.equal(
        create_bigram_tensor(["emma"]),
        torch.tensor(
            [
                [0, 0, 1, 0],  # (".", "e")
                [1, 0, 0, 0],  # ("a", ".")
                [0, 0, 0, 1],  # ("e", "m")
                [0, 1, 0, 1],  # ("m", "a") and ("m", "m")
            ]
        ),
    )
    assert torch.equal(
        create_bigram_tensor(["ava"]),
        torch.tensor(
            [
                [0, 1, 0],  # (".", "a")
                [1, 0, 1],  # ("a", "v") and ("a", ".")
                [0, 1, 0],  # ("v", "a")
            ]
        ),
    )


def test_sample():
    words = open("names.txt", "r").read().splitlines()
    sampled_name = sample(words)
    assert sampled_name == "cexze."
