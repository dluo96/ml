import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""  # fmt: skip


class RegexTokenizer:
    def __init__(self):
        super().__init__()
        self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)

    def train(self, text: str, final_vocab_size: int) -> None:
        num_merges = final_vocab_size - 256

        subtexts = re.findall(self.compiled_pattern, text)

        # Convert text to raw bytes
        tokens = [list(subtext.encode("utf-8")) for subtext in subtexts]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            pair_counts = {}
            for chunk_tokens in tokens:
                # Passing in `pair_counts` will update it in place, adding up counts
                pair_counts = self._get_pair_counts(chunk_tokens)

            # Get the most frequent pair
            top_pair = max(pair_counts, key=pair_counts.get)

            # Index for the new token
            idx = 256 + i

            # Merge: replace all occurrences of `top_pair` in `tokens` with idx
            print(f"Merging {top_pair=} into a new token with {idx=}")
            tokens = [self._merge(t, top_pair, idx) for t in tokens]

            # Update dictionaries
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

        self.merges = merges
        self.vocab = vocab
