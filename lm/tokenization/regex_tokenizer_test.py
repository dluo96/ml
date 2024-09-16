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
        final_vocab_size = 257
        self.tokenizer.train(text, final_vocab_size)
        self.assertEqual(len(self.tokenizer.vocab), final_vocab_size)
        self.assertIn(final_vocab_size - 1, self.tokenizer.vocab)


if __name__ == "__main__":
    unittest.main()
