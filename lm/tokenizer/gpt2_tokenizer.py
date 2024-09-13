class Tokenizer:
    """Implementation of GPT-2 tokenizer, which uses the byte pair encoding algorithm.

    Suppose we have a byte sequence (vocabulary size of 256). We will go through it and
    find the byte pairs that occur the most. We will iteratively define new tokens,
    append them to our vocabulary, and substitute them in the original byte sequence.
    This way, we end up with a compressed dataset.

    We end up with an algo for taking an arbitrary sequence and encoding, as well as
    decoding it back to strings.
    """

    def __init__(self, final_vocab_size: int):
        self.num_merges = final_vocab_size - 256

    def encode(self, text: str) -> tuple[list[int], dict[tuple[int, int], int]]:
        tokens = text.encode("utf-8")  # Raw bytes
        tokens = list(map(int, tokens))  # Convert to list of integers in 0, ..., 255

        # The more steps we do, the shorter our sequence, but the larger our vocabulary
        # In practice, there is a sweet spot that works best. We make the number of
        # steps a configurable hyperparameter.
        ids = list(tokens)  # Create copy
        merges = {}  # Start with leaves of tree
        for i in range(self.num_merges):
            # Iterate over the tokens to determine how often each byte pair occurs
            pair_counts = self.get_pair_counts(ids)

            # Get the most frequent pair: `max` ranks by value (.get) and returns the
            # associated key
            top_pair = max(pair_counts, key=pair_counts.get)

            # Index for the new token
            idx = 256 + i

            # Merge the new token and get the updated list of integers
            print(f"Merging pair {top_pair} into a new token with {idx=}")
            ids = self.merge(ids, top_pair, idx)

            merges[top_pair] = idx

        return ids, merges

    def get_pair_counts(self, ids: list[int]) -> dict[tuple[int, int], int]:
        pair_counts = {}
        for pair in zip(ids, ids[1:]):  # Iterate consecutive elements
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    def merge(self, ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        """Create a new token (with index `idx`) for the specified `pair` and replace
        all occurrences of it in `ids`. This increases the vocabulary size by 1.
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
