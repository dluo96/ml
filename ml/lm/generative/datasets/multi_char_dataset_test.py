import unittest

import torch

from ml.lm.generative.datasets.multi_char_dataset import MultiCharDataset


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
        self.assertEqual(self.dataset.block_size, self.block_size)

        expected_vocab_size = 27  # a-z and start/end token "."
        self.assertEqual(self.dataset.get_vocab_size(), expected_vocab_size)

        expected_num_examples = sum([1 + len(w) for w in self.words])
        self.assertEqual(len(self.dataset), expected_num_examples)

    def test_getitem(self):
        # Check shapes on a random example
        x, y = self.dataset[-1]
        self.assertEqual(x.shape, (self.block_size,))
        self.assertEqual(y.shape, ())

        # Check the examples produced by "emma" (first word in the dataset)
        x, y = self.dataset[0]
        expected_x = torch.tensor([0, 0, 0])  # "..."
        expected_y = torch.tensor(5)  # "e"
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

        x, y = self.dataset[1]
        expected_x = torch.tensor([0, 0, 5])  # "..e"
        expected_y = torch.tensor(13)  # "m"
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

        x, y = self.dataset[2]
        expected_x = torch.tensor([0, 5, 13])  # ".em"
        expected_y = torch.tensor(13)  # "m"
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

        x, y = self.dataset[3]
        expected_x = torch.tensor([5, 13, 13])  # "emm"
        expected_y = torch.tensor(1)  # "a"
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

        x, y = self.dataset[4]
        expected_x = torch.tensor([13, 13, 1])  # "mma"
        expected_y = torch.tensor(0)  # "."
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

        x, y = self.dataset[4]
        expected_x = torch.tensor([13, 13, 1])  # "mma"
        expected_y = torch.tensor(0)  # "."
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

        # Check the examples produced by "isabella" (second word in the dataset)
        x, y = self.dataset[5]
        expected_x = torch.tensor([0, 0, 0])  # "..."
        expected_y = torch.tensor(9)  # "i"
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

        x, y = self.dataset[6]
        expected_x = torch.tensor([0, 0, 9])  # "..i"
        expected_y = torch.tensor(19)  # "s"
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

        x, y = self.dataset[7]
        expected_x = torch.tensor([0, 9, 19])  # ".is"
        expected_y = torch.tensor(1)  # "a"
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

        x, y = self.dataset[8]
        expected_x = torch.tensor([9, 19, 1])  # "isa"
        expected_y = torch.tensor(2)  # "b"
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))


if __name__ == "__main__":
    unittest.main()
