"""Utility functions for tokenization."""


def get_pair_counts(
    ids: list[int], pair_counts: dict[tuple[int, int], int] | None = None
) -> dict[tuple[int, int], int]:
    """Count the occurrences of each pair of integers in the list.

    Args:
        ids: the list of integers in which to count pairs.
        pair_counts: a dictionary to update with the counts.

    Returns:
        A dictionary that maps each pair of integers to the number of times it occurs.
    """
    pair_counts = {} if pair_counts is None else pair_counts
    for pair in zip(ids, ids[1:]):  # Iterate over consecutive elements
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Replace all occurrences of `pair` in `ids` with `idx`. This increases the
    vocabulary size by 1.

    Args:
        ids: the list of integers (token indices) to update.
        pair: the pair of integers to merge.
        idx: the token index to replace the pair with.

    Returns:
        The updated list of integers (token indices).
    """
    new_ids = []
    i = 0  # Index for iterating through the original list of integers
    while i < len(ids):
        # If we are NOT at the last position AND the pair matches, replace it
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids
