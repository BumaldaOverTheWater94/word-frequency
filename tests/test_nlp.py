import pytest
import spacy

from word_frequency.nlp import load_model


def test_load_model_returns_language_object() -> None:
    """Test that load_model returns a spaCy Language object."""
    try:
        nlp = load_model(max_length=1000)
        assert isinstance(nlp, spacy.language.Language)
    except OSError:
        pytest.skip("spaCy model 'en_core_web_trf' not available")


def test_custom_tokenization_punctuation() -> None:
    """Test that custom tokenizer properly handles punctuation."""
    try:
        nlp = load_model(max_length=1000)

        # Test cases that should be tokenized with custom tokenizer
        test_cases = [
            ("word,punctuation", 3),  # Should split comma: ['word', ',', 'punctuation']
            ("text.Period", 3),  # Should split period before uppercase: ['text', '.', 'Period']
            ("text.period", 3),  # Should split period between lowercase letters: ['text', '.', 'period']
            ("word!exclamation", 3),  # Should split exclamation: ['word', '!', 'exclamation']
            ("text?question", 3),  # Should split question mark between letters: ['text', '?', 'question']
            ('word"quote', 3),  # Should split quote: ['word', '"', 'quote']
            ("word(parenthesis)", 4),  # Should split parentheses: ['word', '(', 'parenthesis', ')']
        ]

        for text, expected_tokens in test_cases:
            doc = nlp(text)
            tokens = [token.text for token in doc]
            assert len(doc) == expected_tokens, f"Expected {expected_tokens} tokens for '{text}', got {len(doc)}: {tokens}"

    except OSError:
        pytest.skip("spaCy model 'en_core_web_trf' not available")


def test_custom_tokenization_numbers_letters() -> None:
    """Test that custom tokenizer handles number-letter combinations."""
    try:
        nlp = load_model(max_length=1000)

        test_cases = [
            "word123",  # Letters followed by numbers
            "123word",  # Numbers followed by letters
            "test(5)",  # Parenthetical numbers
        ]

        for text in test_cases:
            doc = nlp(text)
            # Should produce multiple tokens due to custom infix patterns
            tokens = [token.text for token in doc]
            assert len(tokens) > 1, f"Failed to properly tokenize: {text}, got: {tokens}"

    except OSError:
        pytest.skip("spaCy model 'en_core_web_trf' not available")


def test_model_has_required_components() -> None:
    """Test that loaded model has required components for lemmatization and NER."""
    try:
        nlp = load_model(max_length=1000)

        # Model should have essential components for our use case
        assert nlp.has_pipe("tok2vec") or nlp.has_pipe("transformer")
        assert nlp.has_pipe("tagger")
        assert nlp.has_pipe("parser")
        assert nlp.has_pipe("ner")
        assert nlp.has_pipe("attribute_ruler")
        assert nlp.has_pipe("lemmatizer")

    except OSError:
        pytest.skip("spaCy model 'en_core_web_trf' not available")


def test_model_can_process_text() -> None:
    """Test that loaded model can actually process text."""
    try:
        nlp = load_model(max_length=1000)

        text = "This is a test sentence for processing."
        doc = nlp(text)

        # Should produce tokens with proper attributes
        assert len(doc) > 0
        for token in doc:
            assert hasattr(token, "lemma_")
            assert hasattr(token, "pos_")
            assert hasattr(token, "is_ascii")
            assert hasattr(token, "ent_type_")

    except OSError:
        pytest.skip("spaCy model 'en_core_web_trf' not available")


def test_tokenizer_attributes_exist() -> None:
    """Test that tokenizer has the required custom attributes after loading."""
    try:
        nlp = load_model(max_length=1000)
        tokenizer = nlp.tokenizer

        # Check that custom attributes were set
        assert hasattr(tokenizer, "prefix_search")
        assert hasattr(tokenizer, "suffix_search")
        assert hasattr(tokenizer, "infix_finditer")

        # These should be callable
        assert callable(tokenizer.prefix_search)
        assert callable(tokenizer.suffix_search)
        assert callable(tokenizer.infix_finditer)

    except OSError:
        pytest.skip("spaCy model 'en_core_web_trf' not available")


def test_custom_lemmatization_fixes() -> None:
    """Test that custom fallback lemmatizer fixes broken compound word lemmatization."""
    try:
        nlp = load_model(max_length=1000)

        # Test cases for words that spaCy's default lemmatizer breaks
        test_cases = [
            ("shellshocked", "shellshocked"),
            ("multitasker", "multitasker"),
            ("multitasking", "multitasking"),
            ("shocked", "shock"),
            ("tasker", "tasker"),
            ("walker", "walker"),
            ("worked", "work"),
        ]

        for word, expected_lemma in test_cases:
            doc = nlp(word)
            assert len(doc) == 1, f"Expected 1 token for '{word}', got {len(doc)}"
            token = doc[0]
            assert token.lemma_ == expected_lemma, f"Expected lemma '{expected_lemma}' for '{word}', got '{token.lemma_}'"

    except OSError:
        pytest.skip("spaCy model 'en_core_web_trf' not available")
