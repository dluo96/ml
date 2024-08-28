import unittest

import torch
from torch.utils.data import DataLoader

from lm.datasets.char_dataset import CharDataset


class TestCharDataset(unittest.TestCase):
    def setUp(self):
        # Create a toy dataset
        self.words = ["emma"]
        self.dataset = CharDataset(self.words)

    def test_vocab_size(self):
        expected_vocab_size = len(["."] + sorted(list(set("".join(self.words)))))
        self.assertEqual(self.dataset.get_vocab_size(), expected_vocab_size)

    def test_encode(self):
        word = "emma"
        encoded = self.dataset.encode("emma")
        encoded_expected = torch.tensor([2, 3, 3, 1])
        are_equal = torch.equal(encoded, encoded_expected)
        self.assertTrue(are_equal)

    def test_encode_decode(self):
        word = "emma"
        encoded = self.dataset.encode(word)
        decoded = self.dataset.decode(encoded)
        self.assertEqual(decoded, word)

    def test_getitem(self):
        x, y = self.dataset[0]  # Index 0 should be "emma"
        encoded = self.dataset.encode("emma")

        # Check that x contains the correct encoded word with appropriate padding
        self.assertTrue(torch.equal(x[1 : 1 + len(encoded)], encoded))
        # Check that y contains the correct encoded word with appropriate padding
        self.assertTrue(torch.equal(y[: len(encoded)], encoded))
        # Check the mask value in y
        self.assertTrue(torch.all(y[len(encoded) + 1 :] == -1))

    def test_dataloader(self):
        # Check that DataLoader works properly
        dataloader = DataLoader(self.dataset, batch_size=2)
        for batch_x, batch_y in dataloader:
            self.assertEqual(batch_x.shape, (2, self.dataset.max_word_length + 1))
            self.assertEqual(batch_y.shape, (2, self.dataset.max_word_length + 1))


if __name__ == "__main__":
    unittest.main()
