import unittest

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
        self.text = (
            "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe "
            "into the hearts of programmers worldwide. We all know we ought to “support Unicode” "
            "in our software (whatever that means—like using wchar_t for all the strings, right?)."
            " But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus "
            "its dozens of supplementary annexes, reports, and notes can be more than a little "
            "intimidating. I don’t blame programmers for still finding the whole thing mysterious, "
            "even 30 years after Unicode’s inception."
        )

    def test_init(self):
        self.assertEqual(self.tokenizer.merges, {})

    def test_train(self):
        # Train tokenizer with 1 merge
        final_vocab_size = 257
        self.tokenizer = BytePairEncodingTokenizer()
        self.tokenizer.train(self.text, final_vocab_size)
        self.assertEqual(len(self.tokenizer.merges), final_vocab_size - 256)
        self.assertIn(
            final_vocab_size - 1,
            self.tokenizer.merges.values(),
            msg="Newly added token index should be present in merges!",
        )
        tokens = self.tokenizer.encode(self.text)
        self.assertEqual(len(tokens), 596)
        self.assertIn(final_vocab_size - 1, tokens, msg="Last token should be 256")

        # Train tokenizer with 10 merges
        final_vocab_size = 266
        self.tokenizer = BytePairEncodingTokenizer()
        self.tokenizer.train(self.text, final_vocab_size)
        self.assertEqual(len(self.tokenizer.merges), final_vocab_size - 256)
        self.assertIn(
            final_vocab_size - 1,
            self.tokenizer.merges.values(),
            msg="Newly added token index should be present in merges!",
        )
        self.assertIn(
            (257, 133),
            self.tokenizer.merges,
            msg="The pair (257, 133) should be present in merges, which demonstrates "
            "that a newly created token (257 in this case) is also eligible for "
            "merging later on!",
        )
        tokens = self.tokenizer.encode(self.text)
        self.assertIn(final_vocab_size - 1, tokens, msg="Last token should be 265")

    def test_encode(self):
        final_vocab_size = 276
        self.tokenizer = BytePairEncodingTokenizer()
        self.tokenizer.train(self.text, final_vocab_size)

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

    def test_encode_without_train(self):
        num_code_points = len(self.text)
        self.assertEqual(num_code_points, 533)

        # Train w/o any merges
        final_vocab_size = 256
        self.tokenizer = BytePairEncodingTokenizer()
        self.tokenizer.train(self.text, final_vocab_size)

        self.assertEqual(self.tokenizer.merges, {}, msg="No merges were done!")

        tokens = self.tokenizer.encode(self.text)
        num_tokens = len(tokens)
        self.assertEqual(num_tokens, 616)
        self.assertGreaterEqual(
            num_tokens,
            num_code_points,
            msg="In UTF-8 encoding, every ASCII character is encoded to exactly 1 byte"
            " whereas every non-ASCII characters is encoded to 2/3/4 bytes. Thus, the "
            "number of bytes must be greater than or equal to the number of characters!",
        )

    def test_decode(self):
        # First create and train the tokenizer
        final_vocab_size = 276
        self.tokenizer = BytePairEncodingTokenizer()
        self.tokenizer.train(self.text, final_vocab_size)

        # Test that decoding the encoded text recovers the original text
        self.assertEqual(
            self.tokenizer.decode(self.tokenizer.encode(self.text)),
            self.text,
            msg="Decoding the encoded text should recover the original text!",
        )
        text = "안녕하세요 👋 (hello in Korean!)"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

        # Verify that there are invalid start bytes: for example, 128 is an invalid
        # start byte in UTF-8 encoding: in binary, 128 is a 1 followed by 0s, which
        # does not conform to the UTF-8 encoding rules.
        self.assertRaises(UnicodeDecodeError, lambda: bytes([128]).decode("utf-8"))

        # To handle invalid start bytes, we set the `errors="replace"` argument in
        # decode, which replaces invalid bytes with the Unicode replacement character �.
        self.assertEqual(self.tokenizer.decode([128]), "�")

    def test_get_pair_counts(self):
        # Test case 1: Normal list of integers
        ids = [1, 2, 2, 3, 1, 2]
        expected_pair_counts = {
            (1, 2): 2,
            (2, 2): 1,
            (2, 3): 1,
            (3, 1): 1,
        }
        pair_counts = self.tokenizer._get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

        # Test case 2: Empty list
        ids = []
        expected_pair_counts = {}
        pair_counts = self.tokenizer._get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

        # Test case 3: List with one element
        ids = [1]
        expected_pair_counts = {}
        pair_counts = self.tokenizer._get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

        # Test case 4: List with repeating consecutive elements
        ids = [1, 1, 1, 1]
        expected_pair_counts = {(1, 1): 3}
        pair_counts = self.tokenizer._get_pair_counts(ids)
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
        pair_counts = self.tokenizer._get_pair_counts(ids)
        self.assertEqual(pair_counts, expected_pair_counts)

    def test_merge(self):
        self.assertEqual(
            self.tokenizer._merge([5, 6, 6, 7, 9, 1], pair=(6, 7), idx=99),
            [5, 6, 99, 9, 1],
        )
        self.assertEqual(
            self.tokenizer._merge([1, 2, 1, 2, 1, 2], pair=(1, 2), idx=3),
            [3, 3, 3],
        )

    def test_bytes(self):
        # Check a few examples
        self.assertEqual(bytes([0]), b"\x00")
        self.assertEqual(bytes([127]), b"\x7f")
        self.assertEqual(bytes([255]), b"\xff")

        # Check addition of examples
        self.assertEqual(bytes([0]) + bytes([127]) + bytes([255]), b"\x00\x7f\xff")

        # Verify that each byte must be in the range 0, ..., 255
        self.assertRaises(ValueError, bytes, [256])
        self.assertRaises(ValueError, bytes, [1000])
        self.assertRaises(ValueError, bytes, [-1])


if __name__ == "__main__":
    unittest.main()
