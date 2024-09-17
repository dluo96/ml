import regex as re

from lm.tokenization.split_patterns import GPT4_SPLIT_PATTERN
from lm.tokenization.utils import get_pair_counts, merge_new_token


class RegexTokenizer:
    """Tokenizer that uses a regex pattern to split text into chunks.

    But why isn't `BasicTokenizer` enough? Why do we need to enforce regex?
    The folks making GPT-2 found that BPE would create separate tokens for multiple
    versions of common words like 'dog' since they occurred in many variations such as
    'dog.', 'dog!', 'dog?'. This led to a sub-optimal allocation of limited vocabulary
    slots and model capacity. To prevent this, they enforce a regex pattern that splits
    the text in such a way that some types of characters are never merged (into a new
    token) by BPE. It effectively adds merging rules on top of the BPE algorithm.
    """

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}  # Map new pair to token index
        self.vocab: dict[int, bytes] = {}  # Map token index to byte object
        self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)

    def train(self, text: str, final_vocab_size: int) -> None:
        num_merges = final_vocab_size - 256

        # Get a list of subtexts matching the pattern
        list_texts = re.findall(self.compiled_pattern, text)

        # Convert to a list of a list of bytes (in integer representation)
        tokens = [list(txt.encode("utf-8")) for txt in list_texts]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            pair_counts = {}
            for chunk_tokens in tokens:
                # Passing in `pair_counts` will update it in place
                get_pair_counts(chunk_tokens, pair_counts)

            # Get the most frequent pair
            top_pair = max(pair_counts, key=pair_counts.get)

            # Index for the new token
            idx = 256 + i

            # Merge: replace all occurrences of `top_pair` in `tokens` with idx
            print(f"Merging {top_pair=} into a new token with {idx=}")
            tokens = [
                merge_new_token(chunk_tokens, top_pair, idx) for chunk_tokens in tokens
            ]

            # Update dictionaries
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

        self.merges = merges
        self.vocab = vocab

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        ids = list(text_bytes)
        while len(ids) >= 2:
            pair_counts = get_pair_counts(ids)
            # Find pair with the lowest token index: this is the next pair to merge
            pair = min(pair_counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge_new_token(ids, pair, idx)
        return ids

    def encode(self, text: str) -> list[int]:
        # Split the text into chunks each of which matches the regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)

        # Encode each chunk separately and join the results
        token_ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # Raw bytes
            chunk_token_ids = self._encode_chunk(chunk_bytes)
            token_ids.extend(chunk_token_ids)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Convert a sequence of integers (token indices) to a string. Importantly, we
        do not need the regex pattern to decode the text.
        """
        # Iterate over the tokens: map each token index to its byte object and store
        # it. Each element of the list is a `bytes` object consisting of one or more
        # bytes, e.g. [b'aaab', b'd', b'aaab', b'a', b'c']
        part_bytes = []
        for token_id in token_ids:
            if token_id in self.vocab:
                part_bytes.append(self.vocab[token_id])
            else:
                raise ValueError(f"Invalid token id: {token_id}")

        # Join the `bytes` elements to get the full `bytes` object, e.g. b'aaabdaaabac'
        text_bytes = b"".join(part_bytes)

        # Decode the `bytes` object to get the text
        text = text_bytes.decode("utf-8", errors="replace")

        return text
