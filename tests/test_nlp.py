import pytest
import spacy
from unittest.mock import patch, Mock

from word_frequency.nlp import load_model


class TestLoadModel:
    def test_load_model_returns_language_object(self):
        """Test that load_model returns a spaCy Language object."""
        try:
            nlp = load_model(max_length=1000)
            assert isinstance(nlp, spacy.language.Language)
        except OSError:
            pytest.skip("spaCy model 'en_core_web_trf' not available")

    def test_load_model_sets_max_length(self):
        """Test that load_model sets the max_length attribute correctly."""
        try:
            max_len = 5000
            nlp = load_model(max_length=max_len)
            assert nlp.max_length == max_len
        except OSError:
            pytest.skip("spaCy model 'en_core_web_trf' not available")

    @patch('spacy.load')
    def test_load_model_with_mocked_spacy(self, mock_spacy_load):
        """Test load_model behavior with mocked spaCy to avoid model dependency."""
        # Create mock objects
        mock_nlp = Mock()
        mock_nlp.Defaults = Mock()
        mock_nlp.Defaults.prefixes = ['default_prefix']
        mock_nlp.Defaults.suffixes = ['default_suffix']  
        mock_nlp.Defaults.infixes = ['default_infix']
        
        mock_tokenizer = Mock()
        mock_nlp.tokenizer = mock_tokenizer
        
        mock_spacy_load.return_value = mock_nlp
        
        # Mock the regex compilation functions
        with patch('word_frequency.nlp.compile_prefix_regex') as mock_prefix, \
             patch('word_frequency.nlp.compile_suffix_regex') as mock_suffix, \
             patch('word_frequency.nlp.compile_infix_regex') as mock_infix:
            
            mock_prefix_regex = Mock()
            mock_suffix_regex = Mock()
            mock_infix_regex = Mock()
            
            mock_prefix.return_value = mock_prefix_regex
            mock_suffix.return_value = mock_suffix_regex
            mock_infix.return_value = mock_infix_regex
            
            # Call function
            result = load_model(max_length=2000)
            
            # Assertions
            mock_spacy_load.assert_called_once_with("en_core_web_trf", exclude=[])
            assert result.max_length == 2000
            
            # Check that tokenizer attributes were set
            assert mock_tokenizer.prefix_search == mock_prefix_regex.search
            assert mock_tokenizer.suffix_search == mock_suffix_regex.search
            assert mock_tokenizer.infix_finditer == mock_infix_regex.finditer

    def test_custom_tokenization_punctuation(self):
        """Test that custom tokenizer properly handles punctuation."""
        try:
            nlp = load_model(max_length=1000)
            
            # Test cases that should be tokenized differently with custom tokenizer
            test_cases = [
                "word,punctuation",  # Should split comma
                "text.period",       # Should split period
                "word!exclamation",  # Should split exclamation
                "text?question",     # Should split question mark
                'word"quote',        # Should split quote
                "word(parenthesis)", # Should split parenthesis
            ]
            
            for text in test_cases:
                doc = nlp(text)
                # Should produce more than one token due to custom tokenization
                assert len(doc) > 1, f"Failed to properly tokenize: {text}"
                
        except OSError:
            pytest.skip("spaCy model 'en_core_web_trf' not available")

    def test_custom_tokenization_numbers_letters(self):
        """Test that custom tokenizer handles number-letter combinations."""
        try:
            nlp = load_model(max_length=1000)
            
            test_cases = [
                "word123",   # Letters followed by numbers
                "123word",   # Numbers followed by letters
                "test(5)",   # Parenthetical numbers
            ]
            
            for text in test_cases:
                doc = nlp(text)
                # Should produce multiple tokens due to custom infix patterns
                tokens = [token.text for token in doc]
                assert len(tokens) > 1, f"Failed to properly tokenize: {text}, got: {tokens}"
                
        except OSError:
            pytest.skip("spaCy model 'en_core_web_trf' not available")

    def test_model_has_required_components(self):
        """Test that loaded model has required components for lemmatization and NER."""
        try:
            nlp = load_model(max_length=1000)
            
            # Model should have essential components for our use case
            assert nlp.has_pipe('tok2vec') or nlp.has_pipe('transformer')
            assert nlp.has_pipe('tagger')
            assert nlp.has_pipe('parser')
            assert nlp.has_pipe('ner')
            assert nlp.has_pipe('attribute_ruler')
            assert nlp.has_pipe('lemmatizer')
            
        except OSError:
            pytest.skip("spaCy model 'en_core_web_trf' not available")

    def test_model_can_process_text(self):
        """Test that loaded model can actually process text."""
        try:
            nlp = load_model(max_length=1000)
            
            text = "This is a test sentence for processing."
            doc = nlp(text)
            
            # Should produce tokens with proper attributes
            assert len(doc) > 0
            for token in doc:
                assert hasattr(token, 'lemma_')
                assert hasattr(token, 'pos_')
                assert hasattr(token, 'is_ascii')
                assert hasattr(token, 'ent_type_')
                
        except OSError:
            pytest.skip("spaCy model 'en_core_web_trf' not available")

    def test_different_max_lengths(self):
        """Test load_model with different max_length values."""
        max_lengths = [500, 1000, 5000, 10000]
        
        for max_len in max_lengths:
            try:
                nlp = load_model(max_length=max_len)
                assert nlp.max_length == max_len
            except OSError:
                pytest.skip("spaCy model 'en_core_web_trf' not available")

    def test_tokenizer_attributes_exist(self):
        """Test that tokenizer has the required custom attributes after loading."""
        try:
            nlp = load_model(max_length=1000)
            tokenizer = nlp.tokenizer
            
            # Check that custom attributes were set
            assert hasattr(tokenizer, 'prefix_search')
            assert hasattr(tokenizer, 'suffix_search') 
            assert hasattr(tokenizer, 'infix_finditer')
            
            # These should be callable
            assert callable(tokenizer.prefix_search)
            assert callable(tokenizer.suffix_search)
            assert callable(tokenizer.infix_finditer)
            
        except OSError:
            pytest.skip("spaCy model 'en_core_web_trf' not available")