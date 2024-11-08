import unittest
from unittest.mock import patch

from ml.lm.bert.mlm_dataset import MLMDataset


class TestMLMDataset(unittest.TestCase):
    def setUp(self):
        # Sample words list for testing
        self.words = ["hello", "world"]
        self.dataset = MLMDataset(self.words)
        self.names = [
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

    def test_init(self):
        # Test that unique characters include special tokens and the correct letters
        expected_chars = ["<CLS>", "<SEP>", "<MASK>", "d", "e", "h", "l", "o", "r", "w"]
        self.assertEqual(self.dataset.unique_chars, expected_chars)

        # Check that special tokens are mapped to the correct indices
        self.assertEqual(self.dataset.ctoi["<CLS>"], 0)
        self.assertEqual(self.dataset.ctoi["<SEP>"], 1)
        self.assertEqual(self.dataset.ctoi["<MASK>"], 2)

        # Check the size of the dataset
        self.assertEqual(len(self.dataset), 2)

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

    @patch("random.shuffle")
    @patch("random.random")
    def test_get_mlm_data_from_tokens(self, mock_random, mock_shuffle):
        # Mock to return specific values for controlled testing
        mock_random.side_effect = [0.1, 0.1]  # Mask, mask

        def shuffle_fn(x):
            """Mock shuffle function which ensures that the 2 tokens (15% of the tokens
            below) chosen for prediction are 4 and 7.
            """
            # Need to use x[:] to modify the list in-place. Just using x would assign a
            # new reference to the variable, leaving the original list unchanged.
            x[:] = [4, 7, 1, 2, 3, 5, 6, 8, 9, 10]

        mock_shuffle.side_effect = lambda x: shuffle_fn(x)

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
            pred_labels, [6, 7], msg="Masked labels are 6 ('l') and 7 ('o')."
        )

    def test_dataset_getitem(self):
        dataset = MLMDataset(self.names)
        token_ids, pred_positions, pred_labels = dataset[0]

        # Check the examples produced by "emma" (first word in the dataset)
        n_tokens = len("emma") + 2  # +2 because of <CLS> and <SEP> tokens
        self.assertEqual(
            token_ids.shape,
            (n_tokens,),
            msg="Combined with <CLS> and <SEP>, 'emma' has 6 tokens.",
        )
        num_tokens_to_predict = max(1, int(n_tokens * 0.15))
        self.assertEqual(
            pred_positions.shape,
            (num_tokens_to_predict,),
            msg="The number of to-be-predicted tokens is the maximum of 1 or 15% of "
            "the total number of tokens!",
        )
        self.assertEqual(pred_labels.shape, (num_tokens_to_predict,))


if __name__ == "__main__":
    unittest.main()
