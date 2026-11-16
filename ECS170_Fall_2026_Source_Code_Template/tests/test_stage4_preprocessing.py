"""Tests for Stage 4 tokenization, training vocabularies, and encoding."""

import unittest

from local_code.stage_4_code.Text_Preprocessing import build_vocabulary, encode, tokenize


class TokenizerTests(unittest.TestCase):
    def test_html_negation_apostrophes_and_punctuation(self):
        text = 'I DON\u2019T hate it!<BR />It isn&#39;t bad &amp; costs 12.'
        self.assertEqual(tokenize(text), [
            'i', "don't", 'hate', 'it', '!', 'it', "isn't", 'bad', '&', 'costs', '12', '.',
        ])

    def test_escaped_break_tags_and_unicode_words(self):
        self.assertEqual(tokenize('Caf\u00e9&lt;br&gt;NOT good?'), ['caf\u00e9', 'not', 'good', '?'])


class VocabularyTests(unittest.TestCase):
    def test_frequency_cap_and_alphabetical_ties_are_repeatable(self):
        texts = ['dog cat dog', 'ant cat']
        expected = {'token_to_id': {'<PAD>': 0, '<UNK>': 1, 'cat': 2, 'dog': 3},
                    'id_to_token': ['<PAD>', '<UNK>', 'cat', 'dog']}
        self.assertEqual(build_vocabulary(texts, max_tokens=2), expected)
        self.assertEqual(build_vocabulary(reversed(texts), max_tokens=2), expected)

    def test_validation_and_test_words_remain_unknown(self):
        vocabulary = build_vocabulary(['good movie'])
        self.assertEqual(encode('unseen movie', vocabulary), [1, 3])
        self.assertEqual(encode('testonly', vocabulary), [1])
        self.assertEqual(vocabulary['id_to_token'], ['<PAD>', '<UNK>', 'good', 'movie'])
        self.assertNotIn('unseen', vocabulary['token_to_id'])
        self.assertNotIn('testonly', vocabulary['token_to_id'])

    def test_joke_vocabulary_keeps_all_tokens_after_reserved_ids(self):
        vocabulary = build_vocabulary(['what did what?'], max_tokens=None, include_eos=True)
        self.assertEqual(vocabulary['id_to_token'], ['<PAD>', '<UNK>', '<EOS>', 'what', '?', 'did'])
        self.assertEqual(vocabulary['token_to_id']['<EOS>'], 2)

    def test_nonpositive_vocabulary_cap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'max_tokens'):
            build_vocabulary(['a review'], max_tokens=0)


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = {
            'token_to_id': {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2,
                            'what': 3, 'did': 4, 'the': 5, 'chicken': 6, '?': 7},
            'id_to_token': ['<PAD>', '<UNK>', '<EOS>', 'what', 'did', 'the', 'chicken', '?'],
        }

    def test_review_truncation_keeps_the_first_tokens(self):
        self.assertEqual(encode('What did the chicken?', self.vocabulary, max_length=3), [3, 4, 5])

    def test_joke_encoding_preserves_the_whole_text_and_appends_eos(self):
        self.assertEqual(encode('What did the chicken?', self.vocabulary, add_eos=True),
                         [3, 4, 5, 6, 7, 2])

    def test_empty_review_has_one_unknown_input_token(self):
        self.assertEqual(encode('  <br />\n', self.vocabulary, max_length=256), [1])

    def test_nonpositive_review_length_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'max_length'):
            encode('What did the chicken?', self.vocabulary, max_length=0)


if __name__ == '__main__':
    unittest.main()
