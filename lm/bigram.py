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
    unique_chars = sorted(list(set("".join(words))))
    unique_chars = ["<S>"] + unique_chars + ["<E>"]
    N = len(unique_chars)
    bigram_tensor = torch.zeros((N, N), dtype=torch.int32)

    ctoi = {c: i for i, c in enumerate(unique_chars)}
    itoc = {i: c for c, i in enumerate(unique_chars)}

    for w in words:
        chs = ["<S>"] + list(w) + ["<E>"]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1, ix2 = ctoi[ch1], ctoi[ch2]
            bigram_tensor[ix1, ix2] += 1

    return bigram_tensor


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
                [0, 0, 1, 0, 0],  # ("<S>", "e")
                [0, 0, 0, 0, 1],  # ("a", "<E>")
                [0, 0, 0, 1, 0],  # ("e", "m")
                [0, 1, 0, 1, 0],  # ("m", "a") and ("m", "m")
                [0, 0, 0, 0, 0],
            ]
        ),
    )
