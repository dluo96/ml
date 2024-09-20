import random
import unittest
from unittest.mock import patch

from lm.bert.mlm_dataset import MLMDataset


class TestMLMDataset(unittest.TestCase):
    def setUp(self):
        # Sample words list for testing
        self.words = ["hello", "world"]
        self.dataset = MLMDataset(self.words)

    def test_init(self):
        # Test that unique characters include special tokens and the correct letters
        expected_chars = ["<CLS>", "<SEP>", "<MASK>", "d", "e", "h", "l", "o", "r", "w"]
        self.assertEqual(self.dataset.unique_chars, expected_chars)

        # Check that special tokens are mapped to the correct indices
        self.assertEqual(self.dataset.ctoi["<CLS>"], 0)
        self.assertEqual(self.dataset.ctoi["<SEP>"], 1)
        self.assertEqual(self.dataset.ctoi["<MASK>"], 2)

    @patch("random.choice")
    @patch("random.shuffle")
    @patch("random.random")
    def test_replace_mlm_tokens(self, mock_random, mock_shuffle, mock_choice):
        # Mock to return specific values for controlled testing
        mock_random.side_effect = [0.1, 0.95, 0.85]  # Mask, keep, replace
        mock_shuffle.side_effect = lambda x: x  # No shuffling
        mock_choice.side_effect = lambda x: "q"  # Random choice is "q"

        # Prepare inputs
        tokens = ["<CLS>", "h", "e", "l", "l", "o", "<SEP>"]
        eligible_positions = [1, 2, 3, 4, 5]  # <CLS> and <SEP> are not eligible
        num_mlm_preds = 3  # Since [0.1, 0.95, 0.85] has 3 elements

        mlm_input_tokens, pred_positions_and_labels = self.dataset._replace_mlm_tokens(
            tokens,
            eligible_positions,
            num_mlm_preds,
        )

        # Check output
        self.assertEqual(
            mlm_input_tokens,
            ["<CLS>", "<MASK>", "e", "q", "l", "o", "<SEP>"],
            msg="The first token (after <CLS>) is masked (because 0.1 < 0.8); the "
            "second token is kept (because 0.95 > 0.9); the third token is replaced "
            "by a random token 'q' (because 0.8 < 0.85 < 0.9).",
        )
        self.assertEqual(pred_positions_and_labels, [(1, "h"), (2, "e"), (3, "l")])

    @patch("random.random")
    def test_get_mlm_data_from_tokens(self, mock_random):
        # Mock to return specific values for controlled testing
        # Keep, keep, keep, mask, keep, keep, keep, keep, keep, keep, mask
        mock_random.side_effect = [0.95, 0.95, 0.95, 0.1, 0.95, 0.95, 0.1]

        tokens = ["<CLS>", "h", "e", "l", "l", "o", "w", "o", "r", "l", "d", "<SEP>"]  # fmt: skip
        input_seq, pred_positions, pred_labels = self.dataset._get_mlm_data_from_tokens(
            tokens
        )

        self.assertEqual(
            input_seq,
            [0, 5, 4, 6, 2, 7, 9, 2, 8, 6, 3, 1],
            msg="The tokens ['<CLS>', 'h', 'e', 'l', '<MASK>', 'o', 'w', '<MASK>', "
            "'r', 'l', 'd', '<SEP>'] correspond to the token indices "
            "[0, 5, 4, 6, 2, 7, 1, 9, 2, 8, 6, 3, 1].",
        )
        self.assertEqual(pred_positions, [4, 7], msg="Masked positions are 4 and 7.")
        self.assertEqual(
            pred_labels, [6, 9], msg="Masked labels are 6 ('l') and 9 ('o')."
        )

    def test_dataset_len(self):
        # Test __len__ method (not yet implemented)
        self.assertEqual(len(self.dataset), len(self.words))

    def test_dataset_getitem(self):
        # Test __getitem__ method (not yet implemented)
        idx = 0
        with self.assertRaises(NotImplementedError):
            self.dataset[idx]


if __name__ == "__main__":
    unittest.main()
