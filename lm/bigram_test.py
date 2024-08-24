import torch

from lm.bigram import (
    count_bigrams,
    create_bigram_tensor,
    evaluate,
    make_bigrams,
    sample,
)


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


def test_evaluate():
    words = open("names.txt", "r").read().splitlines()
    avg_nll = evaluate(words)
    assert round(avg_nll, 4) == 2.4241
