import torch

from lm.bigram.neural_network import (
    create_training_set_of_bigrams,
    get_bigram_pairs,
    neural_network,
)


def test_generate_bigram_pairs() -> None:
    words = open("../names.txt", "r").read().splitlines()
    unique_chars = ["."] + sorted(list(set("".join(words))))
    ctoi = {c: i for i, c in enumerate(unique_chars)}

    xs, ys = get_bigram_pairs(["emma"], ctoi)

    # When the input to the NN is 0 ("."), the desired label is 5 ("e")
    # When the input to the NN is 5 ("e"), the desired label is 13 ("m")
    # When the input to the NN is 13 ("m"), the desired label is 13 ("m")
    # When the input to the NN is 13 ("m"), the desired label is 1 ("a")
    # When the input to the NN is 1 ("a"), the desired label is 0 (".")
    expected_xs = torch.Tensor([0, 5, 13, 13, 1])
    expected_ys = torch.Tensor([5, 13, 13, 1, 0])

    assert torch.equal(xs, expected_xs)
    assert torch.equal(ys, expected_ys)


def test_create_training_set_of_bigrams() -> None:
    words = open("../names.txt", "r").read().splitlines()
    unique_chars = ["."] + sorted(list(set("".join(words))))
    ctoi = {c: i for i, c in enumerate(unique_chars)}

    xenc, yenc = create_training_set_of_bigrams(["emma"], ctoi)

    # fmt: off
    expected_xenc = torch.tensor([
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ])
    expected_yenc = torch.tensor([
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ])
    # fmt: on

    assert torch.equal(xenc, expected_xenc)
    assert torch.equal(yenc, expected_yenc)


def test_neural_network():
    words = open("../names.txt", "r").read().splitlines()
    unique_chars = ["."] + sorted(list(set("".join(words))))
    ctoi = {c: i for i, c in enumerate(unique_chars)}

    neural_network(words, ctoi)
    assert True
