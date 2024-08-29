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
        expected_vocab_size = 27  # a-z and start/end token "."
        self.assertEqual(self.dataset.get_vocab_size(), expected_vocab_size)

    def test_max_word_length(self):
        expected_max_word_length = len("jacqueline")
        self.assertEqual(self.dataset.max_word_length, expected_max_word_length)

    def test_encode(self):
        word = "emma"
        encoded = self.dataset.encode(word)
        encoded_expected = torch.tensor([5, 13, 13, 1])
        self.assertTrue(torch.equal(encoded, encoded_expected))

    def test_encode_decode(self):
        word = "emma"
        self.assertEqual(self.dataset.decode(self.dataset.encode(word)), word)

    def test_getitem(self):
        x, y = self.dataset[0]

        # Verify the first word is "emma"
        self.assertEqual(self.dataset.decode(x), ".emma......")

        # Check that the input/target tensors have the correct shape
        max_word_length = len("jacqueline")  # "jacqueline" is the longest word
        self.assertEqual(x.shape, (max_word_length + 1,))
        self.assertEqual(x.shape, (max_word_length + 1,))

        # Check that the input/target tensors are correct for "emma"
        expected_x = torch.tensor([0, 5, 13, 13, 1,  0,  0,  0,  0,  0,  0])  # fmt: skip
        expected_y = torch.tensor([5, 13, 13, 1, 0, -1, -1, -1, -1, -1, -1])
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

    def test_dataloader(self):
        B = 2  # Batch size
        num_iter = 0
        dataloader = DataLoader(self.dataset, batch_size=B, drop_last=True)
        for batch_x, batch_y in dataloader:
            self.assertEqual(batch_x.shape, (B, self.dataset.max_word_length + 1))
            self.assertEqual(batch_y.shape, (B, self.dataset.max_word_length + 1))
            num_iter += 1

        expected_num_iter = len(self.words) // B
        self.assertEqual(num_iter, expected_num_iter)


if __name__ == "__main__":
    unittest.main()
