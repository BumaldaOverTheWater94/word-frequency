import pytest

from word_frequency.tokens import filter_token


class TestFilterToken:
    def create_mock_token(self, mocker, **kwargs):
        """Create a mock spaCy token with default valid properties."""
        defaults = {
            "is_ascii": True,
            "ent_type_": "",
            "is_space": False,
            "is_punct": False,
            "is_digit": False,
            "is_quote": False,
            "is_bracket": False,
            "is_currency": False,
            "like_num": False,
            "like_url": False,
            "like_email": False,
            "is_stop": False,
            "pos_": "NOUN",
            "lemma_": "hello",
        }
        defaults.update(kwargs)

        token = mocker.Mock()
        for attr, value in defaults.items():
            setattr(token, attr, value)

        return token

    def test_valid_token_passes_filter(self, mocker):
        """Test that a valid token passes all filters."""
        token = self.create_mock_token(mocker, lemma_="hello")
        assert filter_token(token) is True

    def test_non_ascii_token_filtered(self, mocker):
        """Test that non-ASCII tokens are filtered out."""
        token = self.create_mock_token(mocker, is_ascii=False)
        assert filter_token(token) is False

    def test_entity_token_filtered(self, mocker):
        """Test that named entity tokens are filtered out."""
        token = self.create_mock_token(mocker, ent_type_="PERSON")
        assert filter_token(token) is False

    def test_punctuation_token_filtered(self, mocker):
        """Test that punctuation tokens are filtered out."""
        token = self.create_mock_token(mocker, is_punct=True)
        assert filter_token(token) is False

    def test_digit_token_filtered(self, mocker):
        """Test that digit tokens are filtered out."""
        token = self.create_mock_token(mocker, is_digit=True)
        assert filter_token(token) is False

    def test_stop_word_filtered(self, mocker):
        """Test that stop words are filtered out."""
        token = self.create_mock_token(mocker, is_stop=True)
        assert filter_token(token) is False

    def test_symbol_pos_filtered(self, mocker):
        """Test that tokens with symbol POS tags are filtered out."""
        for pos in ["SYM", "PUNCT", "INTJ", "NUM"]:
            token = self.create_mock_token(mocker, pos_=pos)
            assert filter_token(token) is False

    def test_empty_lemma_filtered(self, mocker):
        """Test that tokens with empty lemmas are filtered out."""
        token = self.create_mock_token(mocker, lemma_="")
        assert filter_token(token) is False

    def test_whitespace_lemma_filtered(self, mocker):
        """Test that tokens with only whitespace lemmas are filtered out."""
        token = self.create_mock_token(mocker, lemma_="   ")
        assert filter_token(token) is False

    def test_punctuation_in_lemma_filtered(self, mocker):
        """Test that lemmas containing punctuation are filtered out."""
        punctuation_cases = ["hello,", "word.", "test!", 'quote"', "word's", "para(graph)", "list[item]", "dict{key}", "title:", "end;"]

        for lemma in punctuation_cases:
            token = self.create_mock_token(mocker, lemma_=lemma)
            assert filter_token(token) is False, f"Failed to filter lemma: {lemma}"

    def test_short_word_with_punctuation_filtered(self, mocker):
        """Test that short words with punctuation are filtered out."""
        short_punct_cases = ["a.", "i,", "he!"]

        for lemma in short_punct_cases:
            token = self.create_mock_token(mocker, lemma_=lemma)
            assert filter_token(token) is False

    def test_unicode_characters_filtered(self, mocker):
        """Test that lemmas with excluded unicode characters are filtered out."""
        unicode_cases = [
            "word\u200b",  # Zero-width space
            "text\u200c",  # Zero-width non-joiner
            "hello\ufeff",  # BOM
            "te\u00a0st",  # Non-breaking space in middle (not stripped)
        ]

        for lemma in unicode_cases:
            token = self.create_mock_token(mocker, lemma_=lemma)
            assert filter_token(token) is False

    def test_repeated_characters_filtered(self, mocker):
        """Test that words with excessive repeated characters are filtered out."""
        repeated_cases = ["hellooooo", "wooooord", "aaaaah", "hmmmmm"]

        for lemma in repeated_cases:
            token = self.create_mock_token(mocker, lemma_=lemma)
            assert filter_token(token) is False

    def test_no_vowel_words_filtered(self, mocker):
        """Test that words without vowels are filtered out."""
        no_vowel_cases = ["bcdfg", "shh", "hmm", "pfft", "nth"]

        for lemma in no_vowel_cases:
            token = self.create_mock_token(mocker, lemma_=lemma)
            assert filter_token(token) is False

    def test_valid_words_with_vowels_pass(self, mocker):
        """Test that valid words with vowels pass the filter."""
        valid_cases = ["hello", "world", "python", "testing", "beautiful", "amazing"]

        for lemma in valid_cases:
            token = self.create_mock_token(mocker, lemma_=lemma)
            assert filter_token(token) is True

    def test_words_with_y_as_vowel_pass(self, mocker):
        """Test that words with 'y' as vowel pass the filter."""
        y_vowel_cases = ["my", "sky", "gym", "rhythm", "xyz"]

        for lemma in y_vowel_cases:
            token = self.create_mock_token(mocker, lemma_=lemma)
            assert filter_token(token) is True

    def test_url_like_tokens_filtered(self, mocker):
        """Test that URL-like tokens are filtered out."""
        token = self.create_mock_token(mocker, like_url=True)
        assert filter_token(token) is False

    def test_email_like_tokens_filtered(self, mocker):
        """Test that email-like tokens are filtered out."""
        token = self.create_mock_token(mocker, like_email=True)
        assert filter_token(token) is False

    def test_number_like_tokens_filtered(self, mocker):
        """Test that number-like tokens are filtered out."""
        token = self.create_mock_token(mocker, like_num=True)
        assert filter_token(token) is False
