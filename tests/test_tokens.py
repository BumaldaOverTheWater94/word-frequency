from unittest.mock import Mock
import pytest

from word_frequency.tokens import filter_token


class TestFilterToken:
    def create_mock_token(self, **kwargs):
        """Create a mock spaCy token with default valid properties."""
        defaults = {
            'is_ascii': True,
            'ent_type_': '',
            'is_space': False,
            'is_punct': False,
            'is_digit': False,
            'is_quote': False,
            'is_bracket': False,
            'is_currency': False,
            'like_num': False,
            'like_url': False,
            'like_email': False,
            'is_stop': False,
            'pos_': 'NOUN',
            'lemma_': 'hello'
        }
        defaults.update(kwargs)
        
        token = Mock()
        for attr, value in defaults.items():
            setattr(token, attr, value)
        
        return token

    def test_valid_token_passes_filter(self):
        """Test that a valid token passes all filters."""
        token = self.create_mock_token(lemma_='hello')
        assert filter_token(token) is True

    def test_non_ascii_token_filtered(self):
        """Test that non-ASCII tokens are filtered out."""
        token = self.create_mock_token(is_ascii=False)
        assert filter_token(token) is False

    def test_entity_token_filtered(self):
        """Test that named entity tokens are filtered out."""
        token = self.create_mock_token(ent_type_='PERSON')
        assert filter_token(token) is False

    def test_punctuation_token_filtered(self):
        """Test that punctuation tokens are filtered out."""
        token = self.create_mock_token(is_punct=True)
        assert filter_token(token) is False

    def test_digit_token_filtered(self):
        """Test that digit tokens are filtered out."""
        token = self.create_mock_token(is_digit=True)
        assert filter_token(token) is False

    def test_stop_word_filtered(self):
        """Test that stop words are filtered out."""
        token = self.create_mock_token(is_stop=True)
        assert filter_token(token) is False

    def test_symbol_pos_filtered(self):
        """Test that tokens with symbol POS tags are filtered out."""
        for pos in ['SYM', 'PUNCT', 'INTJ', 'NUM']:
            token = self.create_mock_token(pos_=pos)
            assert filter_token(token) is False

    def test_empty_lemma_filtered(self):
        """Test that tokens with empty lemmas are filtered out."""
        token = self.create_mock_token(lemma_='')
        assert filter_token(token) is False

    def test_whitespace_lemma_filtered(self):
        """Test that tokens with only whitespace lemmas are filtered out."""
        token = self.create_mock_token(lemma_='   ')
        assert filter_token(token) is False

    def test_punctuation_in_lemma_filtered(self):
        """Test that lemmas containing punctuation are filtered out."""
        punctuation_cases = [
            'hello,',
            'word.',
            'test!',
            'quote"',
            "word's",
            'para(graph)',
            'list[item]',
            'dict{key}',
            'title:',
            'end;'
        ]
        
        for lemma in punctuation_cases:
            token = self.create_mock_token(lemma_=lemma)
            assert filter_token(token) is False, f"Failed to filter lemma: {lemma}"

    def test_short_word_with_punctuation_filtered(self):
        """Test that short words with punctuation are filtered out."""
        short_punct_cases = ['a.', 'i,', 'he!']
        
        for lemma in short_punct_cases:
            token = self.create_mock_token(lemma_=lemma)
            assert filter_token(token) is False

    def test_unicode_characters_filtered(self):
        """Test that lemmas with excluded unicode characters are filtered out."""
        unicode_cases = [
            'word\u200b',  # Zero-width space
            'text\u200c',  # Zero-width non-joiner
            'hello\ufeff',  # BOM
            'test\u00a0'   # Non-breaking space
        ]
        
        for lemma in unicode_cases:
            token = self.create_mock_token(lemma_=lemma)
            assert filter_token(token) is False

    def test_repeated_characters_filtered(self):
        """Test that words with excessive repeated characters are filtered out."""
        repeated_cases = ['hellooooo', 'wooooord', 'aaaaah', 'hmmmmm']
        
        for lemma in repeated_cases:
            token = self.create_mock_token(lemma_=lemma)
            assert filter_token(token) is False

    def test_no_vowel_words_filtered(self):
        """Test that words without vowels are filtered out."""
        no_vowel_cases = ['xyz', 'bcdfg', 'shh', 'hmm', 'pfft']
        
        for lemma in no_vowel_cases:
            token = self.create_mock_token(lemma_=lemma)
            assert filter_token(token) is False

    def test_valid_words_with_vowels_pass(self):
        """Test that valid words with vowels pass the filter."""
        valid_cases = ['hello', 'world', 'python', 'testing', 'beautiful', 'amazing']
        
        for lemma in valid_cases:
            token = self.create_mock_token(lemma_=lemma)
            assert filter_token(token) is True

    def test_words_with_y_as_vowel_pass(self):
        """Test that words with 'y' as vowel pass the filter."""
        y_vowel_cases = ['my', 'sky', 'gym', 'rhythm']
        
        for lemma in y_vowel_cases:
            token = self.create_mock_token(lemma_=lemma)
            assert filter_token(token) is True

    def test_url_like_tokens_filtered(self):
        """Test that URL-like tokens are filtered out."""
        token = self.create_mock_token(like_url=True)
        assert filter_token(token) is False

    def test_email_like_tokens_filtered(self):
        """Test that email-like tokens are filtered out."""
        token = self.create_mock_token(like_email=True)
        assert filter_token(token) is False

    def test_number_like_tokens_filtered(self):
        """Test that number-like tokens are filtered out."""
        token = self.create_mock_token(like_num=True)
        assert filter_token(token) is False