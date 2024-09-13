import unittest

import regex as re

from lm.tokenization.tokenizer import BytePairEncodingTokenizer


def test_textual_data_in_python():
    """In Python, strings are immutable sequences of unicode code points. A unicode
    code point is defined by The Unicode Standard. It is an integer that is defined
    for each character. Version 15.1 defines ~150,000 characters.

    In Python, the unicode code point of a character is given by ord().
    Importantly, you cannot pass a string to ord().
    """
    assert ord("h") == 104
    assert ord("👋") == 128075
    assert ord("안") == 50504

    # fmt: off
    assert [ord(x) for x in "안녕하세요 👋 (hello in Korean!)"] == [
        50504, 45397, 54616, 49464, 50836, 32, 128075, 32, 40, 104, 101, 108, 108,
        111, 32, 105, 110, 32, 75, 111, 114, 101, 97, 110, 33, 41
    ]
    # fmt: on

    """The Unicode Consortium defines three types of encodings:
        - UTF-8 (most common),
        - UTF-16,
        - UTF-32.
    These encodings are the way by which we can take unicode text and translate it into
    binary data.

    UTF-8 encoding is the most common. It takes every code point and translates it to a
    byte stream (a sequence of bytes) between 1 and 4 bytes (so it is a variable length
    encoding). Out of the three encodings above, UTF-8 is the only encoding that is
    backwards compatible to the much simpler ASCII encoding of text.
    """
    # Encode sentence above in UTF-8
    utf8_encoding_raw_bytes = "안녕하세요 👋 (hello in Korean!)".encode("utf-8")
    # fmt: off
    assert utf8_encoding_raw_bytes == (b"\xec\x95\x88\xeb\x85\x95\xed\x95\x98\xec\x84"
                                       b"\xb8\xec\x9a\x94 \xf0\x9f\x91\x8b (hello in "
                                       b"Korean!)")
    # fmt: on

    # Wrap with list to convert raw bytes to integers
    utf8_encoding_integers = list(utf8_encoding_raw_bytes)
    # fmt: off
    assert utf8_encoding_integers == [
        236, 149, 136, 235, 133, 149, 237, 149, 152, 236, 132, 184, 236, 154, 148, 32,
        240, 159, 145, 139, 32, 40, 104, 101, 108, 108, 111, 32, 105, 110, 32, 75, 111,
        114, 101, 97, 110, 33, 41
    ]
    # fmt: on

    # Common characters (e.g. English letters) are usually encoded to 1 byte whereas
    # many special characters are converted into 2/3/4 bytes
    assert len("h".encode("utf-8")) == 1, "h should be encoded to 1 byte"
    assert list("h".encode("utf-8")) == [104]
    assert len("👋".encode("utf-8")) == 4, "👋 should be encoded to 4 bytes"
    assert list("👋".encode("utf-8")) == [240, 159, 145, 139]
    assert len("안".encode("utf-8")) == 3, "안 should be encoded to 3 bytes"
    assert list("안".encode("utf-8")) == [236, 149, 136]

    """If we used UTF-8 naively, we would have a vocabulary size of 2^8 = 256 because
    each character (unicode code point) becomes 1-4 bytes, and each byte (8 bits) is in
    the range 0, ..., 255 (= 2^8 - 1). This vocabulary size is too small since it would
    lead to sentences being extremely long once tokenized. Thus, we don't want to use
    the raw bytes of the UTF-8 encoding.

    We want to support a larger vocabulary size that we can tune as a hyperparameter.
    However, we still want touse the UTF-8 encoding. How do we do this? We turn to the
    byte pair encoding algorithm. This will allow us to compress the byte sequences to
    a variable length.
    """


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BytePairEncodingTokenizer()

    def test_init(self):
        self.assertEqual(self.tokenizer.merges, {})

    def test_train(self):
        text = (
            "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe "
            "into the hearts of programmers worldwide. We all know we ought to “support Unicode” "
            "in our software (whatever that means—like using wchar_t for all the strings, right?)."
            " But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus "
            "its dozens of supplementary annexes, reports, and notes can be more than a little "
            "intimidating. I don’t blame programmers for still finding the whole thing mysterious, "
            "even 30 years after Unicode’s inception."
        )
        num_code_points = len(text)
        self.assertEqual(num_code_points, 533)

        # Train w/o any merges
        self.tokenizer = BytePairEncodingTokenizer()
        self.tokenizer.train(text, final_vocab_size=256)
        tokens = self.tokenizer.encode(text)
        num_tokens = len(tokens)
        self.assertEqual(num_tokens, 616)
        self.assertGreaterEqual(
            num_tokens,
            num_code_points,
            msg="In UTF-8 encoding, every character is encoded to at least 1 byte "
            "(special characters are typically encoded to 2/3/4 bytes), so the number "
            "of tokens must be greater than or equal to the number of characters!",
        )

        # Run encoding with a single merge
        final_vocab_size = 257
        self.tokenizer = BytePairEncodingTokenizer()
        self.tokenizer.train(text, final_vocab_size)
        tokens = self.tokenizer.encode(text)
        self.assertEqual(len(tokens), 596)
        self.assertIn(final_vocab_size - 1, tokens, msg="Last token should be 256")

        # Run encoding with 10 merges
        final_vocab_size = 266
        self.tokenizer = BytePairEncodingTokenizer()
        self.tokenizer.train(text, final_vocab_size)
        tokens = self.tokenizer.encode(text)
        self.assertIn(final_vocab_size - 1, tokens, msg="Last token should be 265")
        self.assertIn(
            (257, 133),
            self.tokenizer.merges,
            msg="The pair (257, 133) should be present in merges, which demonstrates "
            "that a newly created token (257 in this case) is also eligible for "
            "merging later on!",
        )

    def test_get_pair_counts(self):
        # Test case 1: Normal list of integers
        ids = [1, 2, 2, 3, 1, 2]
        expected_pair_counts = {
            (1, 2): 2,
            (2, 2): 1,
            (2, 3): 1,
            (3, 1): 1,
        }
        pair_counts = self.tokenizer.get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

        # Test case 2: Empty list
        ids = []
        expected_pair_counts = {}
        pair_counts = self.tokenizer.get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

        # Test case 3: List with one element
        ids = [1]
        expected_pair_counts = {}
        pair_counts = self.tokenizer.get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

        # Test case 4: List with repeating consecutive elements
        ids = [1, 1, 1, 1]
        expected_pair_counts = {(1, 1): 3}
        pair_counts = self.tokenizer.get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

        # Test case 5: Longer list of integers
        ids = [1, 2, 3, 4, 5, 3, 4, 1, 2, 3, 4, 5]
        expected_pair_counts = {
            (1, 2): 2,
            (2, 3): 2,
            (3, 4): 3,
            (4, 5): 2,
            (5, 3): 1,
            (4, 1): 1,
        }
        pair_counts = self.tokenizer.get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

    def test_merge(self):
        self.assertEqual(
            self.tokenizer.merge([5, 6, 6, 7, 9, 1], pair=(6, 7), idx=99),
            [5, 6, 99, 9, 1],
        )
        self.assertEqual(
            self.tokenizer.merge([1, 2, 1, 2, 1, 2], pair=(1, 2), idx=3),
            [3, 3, 3],
        )

    def test_decode(self):
        text = (
            "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe "
            "into the hearts of programmers worldwide. We all know we ought to “support Unicode” "
            "in our software (whatever that means—like using wchar_t for all the strings, right?)."
            " But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus "
            "its dozens of supplementary annexes, reports, and notes can be more than a little "
            "intimidating. I don’t blame programmers for still finding the whole thing mysterious, "
            "even 30 years after Unicode’s inception."
        )
        final_vocab_size = 276
        self.tokenizer = BytePairEncodingTokenizer()

        # Encode the text
        tokens = self.tokenizer.encode(text)

        # Decode and recover the original text
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(text, decoded_text)

        # Verify that there are invalid start bytes: for example, 128 is an invalid
        # start byte in UTF-8 encoding: in binary, 128 is a 1 followed by 0s, which
        # does not conform to the UTF-8 encoding rules.
        self.assertRaises(UnicodeDecodeError, lambda: bytes([128]).decode("utf-8"))

        # To handle invalid start bytes, we set the `errors="replace"` argument in
        # decode, which replaces invalid bytes with the Unicode replacement character �.
        self.assertEqual(self.tokenizer.decode([128]), "�")

    def test_encode(self):
        text = (
            "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe "
            "into the hearts of programmers worldwide. We all know we ought to “support Unicode” "
            "in our software (whatever that means—like using wchar_t for all the strings, right?)."
            " But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus "
            "its dozens of supplementary annexes, reports, and notes can be more than a little "
            "intimidating. I don’t blame programmers for still finding the whole thing mysterious, "
            "even 30 years after Unicode’s inception."
        )
        final_vocab_size = 276
        self.tokenizer = BytePairEncodingTokenizer()
        self.tokenizer.train(text, final_vocab_size)

        # Test encoding
        self.assertEqual(
            self.tokenizer.encode("hello world!"),
            [104, 101, 108, 108, 111, 32, 119, 270, 108, 100, 33],
        )

        # Test that decoding the encoded text recovers the original text
        text = "hello world!"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)
        text = "안녕하세요 👋 (hello in Korean!)"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)


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


if __name__ == "__main__":
    unittest.main()
