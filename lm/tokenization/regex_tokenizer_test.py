import unittest

import regex as re

from lm.tokenization.regex_tokenizer import RegexTokenizer


def test_regex():
    # Define the regex pattern. Note that | means "or"
    pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")  # fmt: skip

    # 's, 't, 're, 've, 'm, 'll, 'd correspond to common contractions
    assert re.findall(pattern, "Hello how're you") == ["Hello", " how", "'re", " you"]

    # ` ?\p{L}+` represents an optional space followed by one or more letters
    assert re.findall(pattern, "Hello how are you") == ["Hello", " how", " are", " you"]

    # ` ?\p{N}+` represents an optional space followed by one or more numbers
    # NOTE: letters and numbers are separated!
    assert re.findall(pattern, "Hello how123 are you") == [
        "Hello",
        " how",
        "123",
        " are",
        " you",
    ]

    # ` ?[^\s\p{L}\p{N}]+` represents an optional space followed by one or more
    # characters that are not letters or numbers. This capture punctuation.
    assert re.findall(pattern, "Hello how are you!!!?") == [
        "Hello",
        " how",
        " are",
        " you",
        "!!!?",
    ]

    # `\s+(?!\S)` represents one or more spaces that are not followed by a non-space character
    assert re.findall(pattern, "Hello how are         you?") == [
        "Hello",
        " how",
        " are",
        "        ",
        " you",
        "?",
    ]

    # Should have added re.IGNORECASE to `pattern` so BPE merges can happen for
    # capitalized versions of contractions. We can see the wrong behaviour this leads
    # to in this example:
    assert re.findall(pattern, "Hello HOW'RE you") == [
        "Hello",
        " HOW",
        "'",
        "RE",
        " you",
    ]

    # `\s+` will catch any trailing spaces
    assert re.findall(pattern, "Hello how are you    ") == [
        "Hello",
        " how",
        " are",
        " you",
        "    ",
    ]

    """Instead of encoding the text directly, we first split it up.
    So,
        1. First, the text is split into a list of subtexts.
        2. Each subtext is processed independently by the tokenizer.
        3. The results of the separate processes are concatenated.
        4. We only ever find merges between the characters within a single subtext.
            For example, the space in " you" would never be merged with the "e" in
            " are" because they belong to separate subtexts.
        5. Once we have done the merges for each subtext, the results are combined.
        6. This approach prevents certain undesirable merges.

    NOTE: the GPT-2 tokenizer does not merge multiple whitespaces. This is unlike the
    GPT-4 tokenizer.
    """

    """GPT-2 has 50257 tokens in its vocabulary. This consists of
        - 256 raw byte tokens
        - 50,000 additional tokens from using the BPE algorithm (50,000 merges)
        - 1 special token '<|endoftext|>', used to delimit documents in the training
            dataset. This signals to the LLM that it has reached the end of a document.
            We expect the language model to learn the meaning of this token.
    """


class TestRegexTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = RegexTokenizer()

    def test_train(self):
        text = "Hello how're you?"
        self.tokenizer.train(text, final_vocab_size=266)
        tokens = self.tokenizer.encode(text)

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
            msg="The text 'hello' should be encoded as [257, 108, 111]: first the "
            "pair 104 ('h') and 101 ('e') are merged into 256. Then, 256 ('he') and"
            "108 ('l') are merged into 257.",
        )

        # Test case: Text with multiple merges
        text = "hello hello"
        expected_output = [104, 101, 108, 108, 111, 32, 104, 101, 108, 108, 111]
        result = self.tokenizer.encode(text)
        self.assertEqual(result, expected_output)

        # Test case: Encoding text with special characters
        text = "hello, world!"
        expected_output = [104, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]
        result = self.tokenizer.encode(text)
        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
