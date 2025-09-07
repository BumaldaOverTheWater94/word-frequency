from typing import Any

import pytest
import spacy
from pytest_mock import MockerFixture

from word_frequency.tokens import filter_token


def create_mock_token(mocker: MockerFixture, **kwargs: Any) -> spacy.tokens.Token:
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

    return token  # type: ignore


@pytest.mark.parametrize(
    "token_attr,value",
    [
        ("is_ascii", False),
        ("ent_type_", "PERSON"),
        ("is_punct", True),
        ("is_digit", True),
        ("is_stop", True),
        ("lemma_", ""),
        ("lemma_", "   "),
        ("like_url", True),
        ("like_email", True),
        ("like_num", True),
    ],
)
def test_token_filtered_by_attribute(mocker: MockerFixture, token_attr: str, value: Any) -> None:
    """Test that tokens with specific attributes are filtered out."""
    token = create_mock_token(mocker, **{token_attr: value})
    assert filter_token(token) is False


@pytest.mark.parametrize("pos", ["SYM", "PUNCT", "INTJ", "NUM"])
def test_token_filtered_by_pos(mocker: MockerFixture, pos: str) -> None:
    """Test that tokens with specific POS tags are filtered out."""
    token = create_mock_token(mocker, pos_=pos)
    assert filter_token(token) is False


@pytest.mark.parametrize(
    "lemma",
    [
        # Punctuation in lemma
        "hello,",
        "word.",
        "test!",
        'quote"',
        "word's",
        "para(graph)",
        "list[item]",
        "dict{key}",
        "title:",
        "end;",
        # Short words with punctuation
        "a.",
        "i,",
        "he!",
        # Unicode characters
        "word\u200b",  # Zero-width space
        "text\u200c",  # Zero-width non-joiner
        "hello\ufeff",  # BOM
        "te\u00a0st",  # Non-breaking space in middle (not stripped)
        # Repeated characters
        "hellooooo",
        "wooooord",
        "aaaaah",
        "hmmmmm",
        # No vowel words
        "bcdfg",
        "shh",
        "hmm",
        "pfft",
        "nth",
    ],
)
def test_token_filtered_by_lemma_content(mocker: MockerFixture, lemma: str) -> None:
    """Test that tokens with problematic lemma content are filtered out."""
    token = create_mock_token(mocker, lemma_=lemma)
    assert filter_token(token) is False


@pytest.mark.parametrize(
    "lemma",
    [
        "hello",
        "world",
        "python",
        "testing",
        "beautiful",
        "amazing",  # Valid words with vowels
        "my",
        "sky",
        "gym",
        "rhythm",
        "xyz",  # Words with y as vowel
    ],
)
def test_valid_token_passes_filter(mocker: MockerFixture, lemma: str) -> None:
    """Test that valid tokens pass the filter."""
    token = create_mock_token(mocker, lemma_=lemma)
    assert filter_token(token) is True
