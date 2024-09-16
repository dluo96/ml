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
        # Test case 1: merging a pair in a normal list
        self.assertEqual(
            merge_new_token(ids=[1, 2, 2, 3, 1, 2], pair=(1, 2), idx=99),
            [99, 2, 3, 99],
            msg="Merging a pair in a normal list should work!",
        )

        # Test case 2: pair is not present in the list
        self.assertEqual(
            merge_new_token(ids=[1, 2, 3, 4], pair=(2, 5), idx=99),
            [1, 2, 3, 4],
            msg="List should remain unchanged since the pair is not present!",
        )

        # Test case 3: list with consecutive pairs
        self.assertEqual(
            merge_new_token(ids=[1, 2, 2, 2, 3, 1], pair=(2, 2), idx=99),
            [1, 99, 2, 3, 1],
            msg="Merging consecutive pairs should work!",
        )

        # Test case 4: entire list is replaced by merge
        self.assertEqual(
            merge_new_token(ids=[1, 2, 1, 2, 1, 2], pair=(1, 2), idx=99), [99, 99, 99]
        )
