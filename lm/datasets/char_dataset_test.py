import unittest

import torch
from torch.utils.data import DataLoader

from lm.datasets.char_dataset import CharDataset


class TestCharDataset(unittest.TestCase):
    def setUp(self):
        # Create a toy dataset containing all characters a-z
        self.words = [
            "emma",
            "isabella",
            "camila",
            "sadie",
            "faith",
            "margaret",
            "jasmine",
            "kayla",
            "morgan",
            "parker",
            "jacqueline",
            "veronica",
            "winter",
            "alexia",
            "itzel",
        ]
        self.dataset = CharDataset(self.words)

    def test_vocab_size(self):
        self.assertEqual(self.dataset.get_vocab_size(), 27)

    def test_encode(self):
        word = "emma"
        encoded = self.dataset.encode(word)
        encoded_expected = torch.tensor([5, 13, 13, 1])
        are_equal = torch.equal(encoded, encoded_expected)
        self.assertTrue(are_equal)

    def test_encode_decode(self):
        word = "emma"
        encoded = self.dataset.encode(word)
        decoded = self.dataset.decode(encoded)
        self.assertEqual(decoded, word)

    def test_getitem(self):
        x, y = self.dataset[0]

        # The first word is "emma". Check this.
        self.assertEqual(self.dataset.decode(x[:5]), ".emma")
        self.assertEqual(self.dataset.decode(y[:4]), "emma")
        expected_x = torch.tensor([0, 5, 13, 13, 1,  0,  0,  0,  0,  0,  0])
        expected_y = torch.tensor([5, 13, 13, 1, 0, -1, -1, -1, -1, -1, -1])
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

    def test_dataloader(self):
        # Check that DataLoader works properly
        dataloader = DataLoader(self.dataset, batch_size=2)
        for batch_x, batch_y in dataloader:
            self.assertEqual(batch_x.shape, (2, self.dataset.max_word_length + 1))
            self.assertEqual(batch_y.shape, (2, self.dataset.max_word_length + 1))


if __name__ == "__main__":
    unittest.main()
