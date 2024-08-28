import pathlib
import unittest

import torch
import torch.nn.functional as F

from lm.models.nn_bigram import NNBigram


class TestNNBigram(unittest.TestCase):
    def setUp(self):
        data_dir = pathlib.Path(__file__).parent.parent
        self.words = open(f"{data_dir}/names.txt", "r").read().splitlines()
        self.model = NNBigram(words=self.words)

    def test_initialization(self):
        N = 27  # 26 unique characters in `names.txt` and start/end token "."
        self.assertEqual(len(self.model.ctoi), N)
        self.assertEqual(len(self.model.itoc), N)
        self.assertIsInstance(self.model.W, torch.Tensor)
        self.assertEqual(self.model.W.shape, (N, N))

    def test_get_bigram_pairs(self):
        xs, ys = self.model._get_bigram_pairs(["emma"])

        # When the input to the NN is 0 ("."), the desired label is 5 ("e")
        # When the input to the NN is 5 ("e"), the desired label is 13 ("m")
        # When the input to the NN is 13 ("m"), the desired label is 13 ("m")
        # When the input to the NN is 13 ("m"), the desired label is 1 ("a")
        # When the input to the NN is 1 ("a"), the desired label is 0 (".")
        expected_xs = torch.Tensor([0, 5, 13, 13, 1])
        expected_ys = torch.Tensor([5, 13, 13, 1, 0])
        assert torch.equal(xs, expected_xs)
        assert torch.equal(ys, expected_ys)
        self.assertEqual(xs.numel(), 5)  # ".", "e", "m", "m", "a"
        self.assertEqual(ys.numel(), 5)  # "e", "m", "m", "a", "."

    def test_sampling_before_and_after_training(self):
        # Before training
        self.assertEqual(self.model.sample(), "zexzmkloglquszipczktxhkmpmzisttt.")

        self.model.train()

        # After training
        self.assertEqual(self.model.sample(), "cexze.")

    def test_reproducibility(self):
        # Train and sample twice to check for reproducibility
        self.model.train()
        sample1 = self.model.sample()

        # Reset the model and sample again
        self.model = NNBigram(self.words)
        self.model.train()
        sample2 = self.model.sample()

        self.assertEqual(sample1, sample2)

    def test_training_loss_decreases(self):
        xs, ys = self.model._get_bigram_pairs(self.words)
        n_ex = xs.numel()
        xenc = F.one_hot(xs, num_classes=self.model.n_unique_chars).float()

        # Forward pass before training
        logits = xenc @ self.model.W
        counts = logits.exp()
        prob = counts / counts.sum(dim=1, keepdim=True)
        initial_loss = -prob[torch.arange(n_ex), ys].log().mean().item()

        # Train and get new loss
        self.model.train()
        logits = xenc @ self.model.W
        counts = logits.exp()
        prob = counts / counts.sum(dim=1, keepdim=True)
        final_loss = -prob[torch.arange(n_ex), ys].log().mean().item()

        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
