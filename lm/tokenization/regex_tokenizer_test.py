import unittest

from lm.tokenization.regex_tokenizer import RegexTokenizer


class TestRegexTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = RegexTokenizer()

    def test_train(self):
        # Train with 2 merges
        final_vocab_size = 256 + 2
        text = "We'll chill until Will grills dill!"
        self.tokenizer.train(text, final_vocab_size)
        self.assertEqual(
            self.tokenizer.merges,
            {(108, 108): 256, (105, 256): 257},
            msg="The most frequent pair is 'l' and 'l', so the first merge should be "
            "(108, 108) -> 256. Afterwards, the most frequent pair is 'i' and 'll', "
            "so the second merge should be (105, 256) -> 257.",
        )
        expected_vocab = {idx: bytes([idx]) for idx in range(256)}
        expected_vocab[256] = b"ll"
        expected_vocab[257] = b"ill"
        self.assertEqual(
            self.tokenizer.vocab,
            expected_vocab,
            msg="The vocabulary should contain all 256 ASCII characters, plus the "
            "two newly created tokens 256 (representing b'll') and 257 (b'ill').",
        )

    def test_encode_chunk(self):
        # Create toy tokenizer with only two merges
        tokenizer = RegexTokenizer()
        tokenizer.merges = {(101, 102): 256, (256, 103): 257}

        # Test case 1: normal chunk with multiple possible merges
        text_bytes = bytes([101, 102, 103])
        self.assertEqual(
            tokenizer._encode_chunk(text_bytes),
            [257],
            msg="The chunk should be merged into 257: first (101, 102) are merged so "
            "that the chunk becomes [256, 103], then (256, 103) are merged into 257, "
            "implying that the final chunk is [257].",
        )

        # Test case 2: chunk with no possible merges
        text_bytes = bytes([104, 105, 106])
        self.assertEqual(
            tokenizer._encode_chunk(text_bytes),
            [104, 105, 106],
            msg="No merges are possible, so the chunk should be left unchanged!",
        )

        # Test case 3: chunk with only one possible merge
        text_bytes = bytes([101, 102, 104])
        self.assertEqual(
            tokenizer._encode_chunk(text_bytes),
            [256, 104],
            msg="Only (101, 102) can be merged into 256, so the chunk should become "
            "[256, 104].",
        )

    def test_encode(self):
        # Create toy tokenizer with only two merges
        tokenizer = RegexTokenizer()
        tokenizer.merges = {
            (104, 101): 256,  # Example: merging byte pair ('h', 'e') into token 256
            (256, 108): 257,  # Example: merging token 256 and 'l' into token 257
        }

        # Test case 1: normal text with multiple possible merges
        text = "hello"
        self.assertEqual(list(text.encode("utf-8")), [104, 101, 108, 108, 111])
        self.assertEqual(
            tokenizer.encode(text),
            [257, 108, 111],
            msg="The text 'hello' should be encoded as [257, 108, 111]: first, the "
            "pair 104 ('h') and 101 ('e') are merged into 256. Then, 256 ('he') and"
            "108 ('l') are merged into 257.",
        )

        # Test case 2: text with multiple chunks
        text = "hello hello"
        self.assertEqual(tokenizer.encode(text), [257, 108, 111, 32, 257, 108, 111])

    def test_encode_decode_wikipedia_example(self):
        tokenizer = RegexTokenizer()
        text = "aaabdaaabac"  # From https://en.wikipedia.org/wiki/Byte_pair_encoding
        tokenizer.train(text, 256 + 3)  # Train with 3 merges (256 is for byte tokens)

        # Check encoding
        self.assertEqual(
            tokenizer.encode(text),
            [258, 100, 258, 97, 99],
            msg="Running BPE on 'aaabdaaabac' for 3 merges results in 'XdXac' where "
            "X=ZY, Y=ab, and Z=aa. Note that our tokenizer always allocates the 256 "
            "individual bytes as tokens, and then merges bytes as needed from there. "
            "Thus, a=97, b=98, c=99, and d=100 (their ASCII values). When (a,a) is "
            "merged to Z, Z becomes 256. Similarly, Y becomes 257 and X becomes 258. "
            "So we start with the 256 bytes, and do 3 merges to get to the result "
            "above, with the expected output of [258, 100, 258, 97, 99].",
        )

        # Check decoding
        self.assertEqual(
            tokenizer.decode([258, 100, 258, 97, 99]),
            "aaabdaaabac",
            msg="Decoding the encoded text should recover the original text!",
        )


if __name__ == "__main__":
    unittest.main()
