import unittest

import regex as re

from lm.tokenization.split_patterns import GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN


class TestSplitPatterns(unittest.TestCase):
    def test_gpt2_split_pattern(self):
        # Define the regex pattern. Note that | means "or"
        pattern = re.compile(GPT2_SPLIT_PATTERN)
        self.assertEqual(
            GPT2_SPLIT_PATTERN,
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # fmt: skip
        )

        # 's, 't, 're, 've, 'm, 'll, 'd correspond to common contractions
        self.assertEqual(
            re.findall(pattern, "Hello how're you"), ["Hello", " how", "'re", " you"]
        )

        # ` ?\p{L}+` represents an optional space followed by one or more letters
        self.assertEqual(
            re.findall(pattern, "Hello how are you"), ["Hello", " how", " are", " you"]
        )

        # ` ?\p{N}+` represents an optional space followed by one or more numbers
        # NOTE: letters and numbers are separated!
        self.assertEqual(
            re.findall(pattern, "Hello how123 are you"),
            ["Hello", " how", "123", " are", " you"],
        )

        # ` ?[^\s\p{L}\p{N}]+` represents an optional space followed by one or more
        # characters that are not letters or numbers. This capture punctuation.
        self.assertEqual(
            re.findall(pattern, "Hello how are you!!!?"),
            ["Hello", " how", " are", " you", "!!!?"],
        )

        # `\s+(?!\S)` represents one or more spaces that are not followed by a non-space character
        self.assertEqual(
            re.findall(pattern, "Hello how are         you?"),
            ["Hello", " how", " are", "        ", " you", "?"],
        )

        # Should have added re.IGNORECASE to `pattern` so BPE merges can happen for
        # capitalized versions of contractions. We can see the wrong behaviour this leads
        # to in this example:
        self.assertEqual(
            re.findall(pattern, "Hello HOW'RE you"),
            ["Hello", " HOW", "'", "RE", " you"],
        )

        # `\s+` will catch any trailing spaces
        self.assertEqual(
            re.findall(pattern, "Hello how are you    "),
            ["Hello", " how", " are", " you", "    "],
        )

    def test_gpt4_split_pattern(self):
        pattern = re.compile(GPT4_SPLIT_PATTERN)
        self.assertEqual(
            GPT4_SPLIT_PATTERN,
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""  # fmt: skip
        )

        # (?i:[sdmt]|ll|ve|re) matches the strings 's, 'd, 'm, 't, 'll, 've, or 're,
        # case-insensitively due to the (?i:) flag
        self.assertEqual(
            re.findall(pattern, "Hello how're you"), ["Hello", " how", "'re", " you"]
        )
        self.assertEqual(
            re.findall(pattern, "Hello HOW'RE you"), ["Hello", " HOW", "'RE", " you"]
        )

        # [^\r\n\p{L}\p{N}]?+\p{L}+ matches an optional sequence of any character
        # except line breaks (\r, \n), Unicode letters (\p{L}), and numbers (\p{N}),
        # followed by one or more Unicode letters (\p{L}+), with the non-greedy +?
        # preventing over-consumption of characters.
        self.assertEqual(re.findall(pattern, "Hello\nWorld"), ["Hello", "\n", "World"])

        # ` ?\p{N}+` represents an optional space followed by one or more numbers
        # NOTE: letters and numbers are separated!
        self.assertEqual(
            re.findall(pattern, "Hello how123 are you"),
            ["Hello", " how", "123", " are", " you"],
        )

        # ` ?[^\s\p{L}\p{N}]+` represents an optional space followed by one or more
        # characters that are not letters or numbers. This capture punctuation.
        self.assertEqual(
            re.findall(pattern, "Hello how are you!!!?"),
            ["Hello", " how", " are", " you", "!!!?"],
        )

        # `\s+(?!\S)` represents one or more spaces that are not followed by a non-space character
        self.assertEqual(
            re.findall(pattern, "Hello how are         you?"),
            ["Hello", " how", " are", "        ", " you", "?"],
        )

        # Unlike the GPT-2 pattern, the GPT-4 pattern handles this case correctly
        self.assertEqual(
            re.findall(pattern, "Hello HOW'RE you"),
            ["Hello", " HOW", "'RE", " you"],
        )

        # `\s+` will catch any trailing spaces
        self.assertEqual(
            re.findall(pattern, "Hello how are you    "),
            ["Hello", " how", " are", " you", "    "],
        )


if __name__ == "__main__":
    unittest.main()
