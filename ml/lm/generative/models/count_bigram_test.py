import pathlib
import unittest

import torch

from ml.lm.generative.models.count_bigram import Bigram


class TestBigram(unittest.TestCase):
    def setUp(self):
        # Setup that runs before each test
        data_dir = pathlib.Path(__file__).parent.parent
        self.words = open(f"{data_dir}/names.txt", "r").read().splitlines()
        self.model = Bigram(words=self.words)

    def test_make_bigrams(self):
        expected_bigrams = [
            ("<S>", "e"),
            ("e", "m"),
            ("m", "m"),
            ("m", "a"),
            ("a", "<E>"),
        ]
        self.assertEqual(self.model.make_bigrams(["emma"]), expected_bigrams)

    def test_count_bigrams(self):
        expected_bigram_counts = {
            ("<S>", "e"): 1,
            ("e", "m"): 1,
            ("m", "m"): 1,
            ("m", "a"): 1,
            ("a", "<E>"): 1,
        }
        self.assertEqual(self.model.count_bigrams(["emma"]), expected_bigram_counts)

    def test_create_bigram_tensor__one_word_dataset(self):
        # Overwrite model instance in `setUp` to test with a 1-word dataset
        self.model = Bigram(words=["emma"])
        expected_tensor_emma = torch.tensor(
            [
                [0, 0, 1, 0],  # (".", "e")
                [1, 0, 0, 0],  # ("a", ".")
                [0, 0, 0, 1],  # ("e", "m")
                [0, 1, 0, 1],  # ("m", "a") and ("m", "m")
            ]
        )
        actual_tensor_emma = self.model.create_bigram_tensor(["emma"])
        self.assertTrue(torch.equal(expected_tensor_emma, actual_tensor_emma))

        # Reset model instance to test with a different 1-word dataset
        self.model = Bigram(words=["ava"])
        expected_tensor_ava = torch.tensor(
            [
                [0, 1, 0],  # (".", "a")
                [1, 0, 1],  # ("a", "v") and ("a", ".")
                [0, 1, 0],  # ("v", "a")
            ]
        )
        actual_tensor_ava = self.model.create_bigram_tensor(["ava"])
        self.assertTrue(torch.equal(expected_tensor_ava, actual_tensor_ava))

    def test_sample(self):
        self.assertEqual(self.model.sample(self.words), "cexze.")

    def test_evaluate(self):
        avg_nll = self.model.evaluate(self.words)
        self.assertAlmostEqual(avg_nll, 2.454, places=4)


if __name__ == "__main__":
    unittest.main()
