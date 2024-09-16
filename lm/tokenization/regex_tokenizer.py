import regex as re

from lm.tokenization.utils import get_pair_counts, merge_new_token

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""  # fmt: skip


class RegexTokenizer:
    """Tokenizer that uses a regex pattern to split text into chunks."""

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

        # All chunks of the text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # Raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
