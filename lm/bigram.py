def make_bigrams(word: str) -> list[tuple[str, str]]:
    bigrams = []
    chs = ["<S>"] + list(word) + ["<E>"]
    for ch1, ch2 in zip(chs, chs[1:]):  # zip truncates the longer iterable
        bigrams.append((ch1, ch2))
    return bigrams


def count_bigrams(word: str) -> dict[tuple[str, str], int]:
    bg_counts = {}
    chs = ["<S>"] + list(word) + ["<E>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        bg_counts[bigram] = bg_counts.get(bigram, 0) + 1
    return bg_counts


def test_make_bigrams():
    assert make_bigrams("emma") == [
        ("<S>", "e"),
        ("e", "m"),
        ("m", "m"),
        ("m", "a"),
        ("a", "<E>"),
    ]


def test_count_bigrams():
    assert count_bigrams("emma") == {
        ("<S>", "e"): 1,
        ("e", "m"): 1,
        ("m", "m"): 1,
        ("m", "a"): 1,
        ("a", "<E>"): 1,
    }
