import unittest

from lm.tokenizer.gpt2_tokenizer import Tokenizer


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

    UTF-8 is the most common. It takes every code point and translates it to a
    byte stream (a sequence of bytes). This byte stream is between 1 and 4 bytes (so it
    is a variable length encoding). Out of the above, UTF-8 is the only encoding that
    is backwards compatible to the much simpler ASCII encoding of text.
    """
    # Encode sentence above in UTF-8
    utf8_encoding_raw_bytes = "안녕하세요 👋 (hello in Korean!)".encode("utf-8")
    assert (
        utf8_encoding_raw_bytes
        == b"\xec\x95\x88\xeb\x85\x95\xed\x95\x98\xec\x84\xb8\xec\x9a\x94 \xf0\x9f\x91\x8b (hello in Korean!)"
    )

    # Wrap with list to convert raw bytes to integers
    utf8_encoding_integers = list(utf8_encoding_raw_bytes)
    # fmt: off
    assert utf8_encoding_integers == [
        236, 149, 136, 235, 133, 149, 237, 149, 152, 236, 132, 184, 236, 154, 148, 32,
        240, 159, 145, 139, 32, 40, 104, 101, 108, 108, 111, 32, 105, 110, 32, 75, 111,
        114, 101, 97, 110, 33, 41
    ]
    # fmt: on

    # If we used UTF-8 naively, we would have a vocabulary size of 256 (2^8 because we have
    # 1-4 bytes per character), but this is too small since it would lead to sentences
    # to be extremely long once tokenized. Thus, we don't want to use the raw bytes of
    # the UTF-8 encoding.
    # We want to support a larger vocabulary size that we can tune as a hyperparameter.
    # However, we want to stick with the UTF-8 encoding. How do we do this? We turn to
    # the byte pair encoding algorithm. This allows us to compress the byte sequences
    # to a variable length.

    # Most English characters are converted to 1 byte, whereas special characters are
    # converted into 2/3/4 bytes.
    assert len("h".encode("utf-8")) == 1
    assert len("안".encode("utf-8")) == 3, "안 should be encoded to 3 bytes"


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.final_vocab_size = 257
        self.tokenizer = Tokenizer(self.final_vocab_size)

    def test_init(self):
        self.assertEqual(self.tokenizer.num_merges, self.final_vocab_size - 256)

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
        num_code_points = len(text)
        self.assertEqual(num_code_points, 533)

        self.tokenizer = Tokenizer(final_vocab_size=256)
        vocab_size = len(self.tokenizer.encode(text))
        self.assertEqual(vocab_size, 616)
        self.assertGreaterEqual(
            vocab_size,
            num_code_points,
            msg="Every character will be encoded to at least 1 byte. Special "
            "characters are usually encoded to 2/3/4 bytes, hence the vocabulary size "
            "should be greater than or equal to the number of characters!",
        )

        self.tokenizer = Tokenizer(final_vocab_size=257)
        self.assertEqual(len(self.tokenizer.encode(text)), 596)

        # assert top_pair == (101, 32)  # 101 is byte encoding for "e", 32 for " "
        # assert chr(top_pair[0]) == "e"
        # assert chr(top_pair[1]) == " "

    def test_merge(self):
        self.assertEqual(
            self.tokenizer.merge([5, 6, 6, 7, 9, 1], pair=(6, 7), idx=99),
            [5, 6, 99, 9, 1],
        )
