import unittest

from adapter.utils import chunk_text_by_words


class UtilsTests(unittest.TestCase):
    def test_chunk_text_by_words_with_overlap(self) -> None:
        text = "one two three four five six seven"
        chunks = chunk_text_by_words(text, chunk_size=3, overlap=1)
        self.assertEqual(chunks, ["one two three", "three four five", "five six seven"])

    def test_chunk_text_validation(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text_by_words("a b", chunk_size=0, overlap=0)
        with self.assertRaises(ValueError):
            chunk_text_by_words("a b", chunk_size=2, overlap=2)


if __name__ == "__main__":
    unittest.main()
