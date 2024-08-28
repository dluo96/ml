import pathlib
import unittest

import torch

from lm.mlp.mlp import MLP


class TestBigram(unittest.TestCase):
    def setUp(self):
        # Setup that runs before each test
        data_dir = pathlib.Path(__file__).parent.parent
        self.words = open(f"{data_dir}/names.txt", "r").read().splitlines()
        self.model = MLP(self.words)

    def test_create_dataset(self):
        """Check that, for the word "emma", we correctly get the following input-label
        pairs for a context length of 3 characters:
           - "..." -> "e"
           - "..e" -> "m"
           - ".em" -> "m"
           - "emm" -> "a"
           - "mma" -> "."
        """
        X, Y = self.model.create_dataset(["emma"])
        self.assertEqual(X.shape, (5, 3))
        self.assertEqual(Y.shape, (5,))

        expected_X = torch.tensor(
            [
                [0, 0, 0],  # "..."
                [0, 0, 5],  # "..e"
                [0, 5, 13],  # ".em"
                [5, 13, 13],  # "emm"
                [13, 13, 1],  # "mma"
            ]
        )
        expected_Y = torch.tensor([5, 13, 13, 1, 0])  # "e"  # "m"  # "m"  # "a"  # "."
        self.assertTrue(torch.equal(X, expected_X))
        self.assertTrue(torch.equal(Y, expected_Y))

    # def test_create_lookup_table(self):
    #     X = torch.tensor([
    #         [0, 0, 0],  # "..."
    #         [0, 0, 5],  # "..e"
    #         [0, 5, 13],  # ".em"
    #         [5, 13, 13],  # "emm"
    #         [13, 13, 1],  # "mma"
    #     ])
    #     emb_X = self.model.create_lookup_table(X)
    #     self.assertEqual(emb_X.shape, (5, 3, 2))

    def test_train(self):
        self.model.train()
