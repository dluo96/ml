import unittest

from ml.lm.tokenization.basic_tokenizer import BasicTokenizer


class TestBasicTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BasicTokenizer()
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
        self.tokenizer = BasicTokenizer()
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
        self.tokenizer = BasicTokenizer()
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
        self.tokenizer = BasicTokenizer()
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
        self.tokenizer = BasicTokenizer()
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
        self.tokenizer = BasicTokenizer()
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

    def test_encode_decode_wikipedia_example(self):
        tokenizer = BasicTokenizer()
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


if __name__ == "__main__":
    unittest.main()
