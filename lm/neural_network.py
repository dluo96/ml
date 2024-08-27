import torch
import torch.nn.functional as F


def create_training_set_of_bigrams(
    words: list[str], ctoi: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = get_bigram_pairs(words, ctoi)

    n_unique_chars = len(ctoi.keys())
    xenc = F.one_hot(xs, num_classes=n_unique_chars).float()
    yenc = F.one_hot(ys, num_classes=n_unique_chars).float()

    return xenc, yenc


def get_bigram_pairs(words, ctoi):
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = ctoi[ch1]
            ix2 = ctoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    return xs, ys
