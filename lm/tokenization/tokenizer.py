class BytePairEncodingTokenizer:
    """Implementation of GPT-2 tokenizer, which uses the byte pair encoding (BPE)
    algorithm.

    Suppose we have a byte sequence (vocabulary size of 256). We will go through it and
    find the byte pairs that occur the most. We will iteratively define new tokens,
    append them to our vocabulary, and substitute them in the original byte sequence.
    This way, we end up with a compressed dataset.

    We end up with an algo for taking an arbitrary sequence and encoding, as well as
    decoding it back to strings.

    Note: the tokenizer is a completely separate, independent module from the LLM. It
    has its own training dataset of text (which could be different from that of the
    LLM), on which you train the vocabulary using the Byte Pair Encoding (BPE) algorithm.
    It then translates back and forth between raw text and sequences of tokens. The LLM
    only ever sees the tokens and never directly deals with any text. Therefore, the
    tokenizer can be thought of as a translation layer.
    """

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}  # Map new pair to token index
        self.vocab: dict[int, bytes] = {}  # Map token index to byte object

    def train(self, text: str, final_vocab_size: int) -> None:
        """Train the tokenizer on the specified text and create:
            - A dictionary that maps the byte pair for each new token to its token
                index. This is needed for encoding.
            - A dictionary that maps from token id to byte object. This is needed for
                decoding.

        Args:
            text: the sequence of unicode code points on which to train the tokenizer.
            final_vocab_size: the desired final size of the vocabulary. The larger it
                is, the more new tokens we create (i.e. merges we do), and in turn,
                the shorter encoded sequences will be. In practice, there is a sweet
                spot that works best, hence why this is a configurable parameter.
        """
        num_merges = final_vocab_size - 256

        # Convert text to raw bytes
        tokens = text.encode("utf-8")

        # Convert raw bytes to list of bytes in integer representation. Each byte is
        # represented by an integer in the range 0-255.
        tokens = list(map(int, tokens))

        merges = {}  # Start with leaves of tree
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            # Iterate over the tokens to determine how often each byte pair occurs
            pair_counts = self.get_pair_counts(tokens)

            # Get the most frequent pair: `max` ranks by value (.get) and returns the
            # associated key
            top_pair = max(pair_counts, key=pair_counts.get)

            # Index for the new token
            idx = 256 + i

            # Merge the new token and get the updated list of integers
            print(f"Merging {top_pair=} into a new token with {idx=}")
            tokens = self.merge(tokens, top_pair, idx)

            # Update dictionaries
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

        self.merges = merges  # Needed for encoding
        self.vocab = vocab  # Needed for decoding

    def decode(self, ids: list[int]) -> str:
        """Get the text corresponding to a sequence of integers each in the range
        0, ..., vocab_size - 1.
        """
        # Get raw bytes
        tokens = b"".join(self.vocab[idx] for idx in ids)

        # Decode with UTF-8. Importantly, not every byte sequence is valid UTF-8.
        # If the language model predicts tokens in a bad manner, then they might not
        # be valid UTF-8, and so we won't be able to decode them. The standard practice
        # is to replace them with the Unicode replacement character.
        text = tokens.decode("utf-8", errors="replace")

        return text

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:  # Need at least two tokens, otherwise `min` will fail
            pair_counts = self.get_pair_counts(tokens)

            # Identify pair to merge: we want the pair with the lowest index in `merges`
            # Ranks by value (token index) and returns the associated key (pair)
            # The inf is a fallback for pairs that are not in `merges` - they are not
            # eligible for merging and the inf guarantees they are not selected
            pair = min(pair_counts, key=lambda p: self.merges.get(p, float("inf")))

            # Edge case: none of the pairs were in `merges` and there is nothing to merge
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)

        return tokens

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
