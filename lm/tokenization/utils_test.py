import unittest

from lm.tokenization.utils import get_pair_counts, merge_new_token


class TestTokenizerUtils(unittest.TestCase):
    def test_get_pair_counts(self):
        # Test case 1: Normal list of integers
        ids = [1, 2, 2, 3, 1, 2]
        expected_pair_counts = {
            (1, 2): 2,
            (2, 2): 1,
            (2, 3): 1,
            (3, 1): 1,
        }
        pair_counts = get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

        # Test case 2: Empty list
        ids = []
        expected_pair_counts = {}
        pair_counts = get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

        # Test case 3: List with one element
        ids = [1]
        expected_pair_counts = {}
        pair_counts = get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

        # Test case 4: List with repeating consecutive elements
        ids = [1, 1, 1, 1]
        expected_pair_counts = {(1, 1): 3}
        pair_counts = get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

        # Test case 5: Longer list of integers
        ids = [1, 2, 3, 4, 5, 3, 4, 1, 2, 3, 4, 5]
        expected_pair_counts = {
            (1, 2): 2,
            (2, 3): 2,
            (3, 4): 3,
            (4, 5): 2,
            (5, 3): 1,
            (4, 1): 1,
        }
        pair_counts = get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

    def test_merge_new_token(self):
        # Test case 1: Merging a pair in a normal list
        ids = [1, 2, 2, 3, 1, 2]
        pair = (1, 2)
        idx = 99
        expected_output = [99, 2, 3, 99]
        result = merge_new_token(ids, pair, idx)
        self.assertEqual(result, expected_output)

        # Test case 2: Pair not present in the list
        ids = [1, 2, 3, 4]
        pair = (2, 5)
        idx = 99
        expected_output = [1, 2, 3, 4]  # No merge occurs
        result = merge_new_token(ids, pair, idx)
        self.assertEqual(result, expected_output)

        # Test case 3: List with consecutive pairs
        ids = [1, 2, 2, 2, 3, 1]
        pair = (2, 2)
        idx = 99
        expected_output = [1, 99, 2, 3, 1]
        result = merge_new_token(ids, pair, idx)
        self.assertEqual(result, expected_output)

        # Test case 4: Entire list replaced by merge
        ids = [1, 2, 1, 2]
        pair = (1, 2)
        idx = 99
        expected_output = [99, 99]
        result = merge_new_token(ids, pair, idx)
        self.assertEqual(result, expected_output)
