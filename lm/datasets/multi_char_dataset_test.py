import unittest

import torch

from lm.datasets.multi_char_dataset import MultiCharDataset


class TestMultiCharDataset(unittest.TestCase):
    def setUp(self):
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
        self.block_size = 3
        self.dataset = MultiCharDataset(self.words, self.block_size)

    def test_init(self):
        self.assertEqual(len(self.dataset), len(self.words))
        self.assertEqual(self.dataset.block_size, self.block_size)
        self.assertEqual(self.dataset.get_vocab_size(), len(self.dataset.unique_chars))

    def test_getitem(self):
        x, y = self.dataset[0]  # Test on the first word, "emma"

        # Check shapes
        self.assertEqual(x.shape, (len("emma") + 1, self.block_size))
        self.assertEqual(y.shape, (len("emma") + 1,))

        # Check values
        expected_x = torch.tensor(
            [
                [0, 0, 0],  # "..."
                [0, 0, 5],  # "..e"
                [0, 5, 13],  # ".em"
                [5, 13, 13],  # "emm"
                [13, 13, 1],  # "mma"
            ]
        )
        # fmt: off
        expected_y = torch.tensor([
            5,  # "e"
            13,  # "m"
            13,  # "m"
            1,  # "a"
            0  # "."
        ])
        # fmt:on
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))


if __name__ == "__main__":
    unittest.main()
