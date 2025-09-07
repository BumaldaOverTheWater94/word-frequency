from collections import Counter
from typing import List, Any, Callable
import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from word_frequency.pipeline import init_database, lemmatize_text, process
from pytest_mock import MockerFixture


def test_init_database_creates_db_with_correct_path(mocker: MockerFixture) -> None:
    """Test that init_database creates a database with the correct path."""
    csv_filepath: str = "/path/to/output.csv"
    expected_db_path: str = "/path/to/output.db"

    mock_db_class = mocker.patch("word_frequency.pipeline.CountsDB")
    mock_db_instance = mocker.Mock()
    mock_db_class.return_value = mock_db_instance

    result = init_database(csv_filepath)

    mock_db_class.assert_called_once_with(expected_db_path)
    assert result == mock_db_instance


@pytest.mark.parametrize(
    "csv_path,expected_db_path",
    [
        ("output.csv", "output.db"),
        ("data.txt", "data.db"),
        ("file.log", "file.db"),
        ("path/to/file.csv", "path/to/file.db"),
    ],
)
def test_init_database_with_different_extensions(mocker: MockerFixture, csv_path: str, expected_db_path: str) -> None:
    """Test init_database with different file extensions."""
    mock_db_class = mocker.patch("word_frequency.pipeline.CountsDB")
    mock_db_instance = mocker.Mock()
    mock_db_class.return_value = mock_db_instance

    result = init_database(csv_path)

    mock_db_class.assert_called_once_with(expected_db_path)
    assert result == mock_db_instance


@pytest.fixture
def sample_tokens() -> List[str]:
    """Provide sample token lemmas for testing."""
    return ["hello", "world", "test"]


@pytest.fixture
def duplicate_tokens() -> List[str]:
    """Provide token lemmas with duplicates for testing."""
    return ["hello", "hello", "world", "hello"]


@pytest.fixture
def mixed_tokens() -> List[str]:
    """Provide mixed tokens where some should be filtered."""
    return ["hello", "bad_token", "world"]


@pytest.fixture(scope="session")
def nlp_model() -> Language:
    """Load spaCy model for testing."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Fallback to blank model if en_core_web_sm is not available
        return spacy.blank("en")


@pytest.fixture
def token_doc(nlp_model: Language) -> Callable[[List[str]], Doc]:
    """Create a real spaCy document factory for testing."""

    def _create_doc(text_or_lemmas: List[str]) -> Doc:
        if not text_or_lemmas:
            return nlp_model("")

        # Join the lemmas as text and process through spaCy
        text = " ".join(text_or_lemmas)
        doc = nlp_model(text)

        # Override lemmas if needed for consistent testing
        for i, lemma in enumerate(text_or_lemmas):
            if i < len(doc):
                # This creates a more predictable test environment
                doc[i].lemma_ = lemma

        return doc

    return _create_doc


def test_lemmatize_text_with_valid_tokens(mocker: MockerFixture, sample_tokens: List[str], token_doc: Callable[[List[str]], Doc]) -> None:
    """Test lemmatizing text with tokens that pass filtering."""
    doc: Doc = token_doc(sample_tokens)

    mocker.patch("word_frequency.pipeline.filter_token", return_value=True)
    result: Counter[str] = lemmatize_text(doc)

    assert isinstance(result, Counter)
    assert result["hello"] == 1
    assert result["world"] == 1
    assert result["test"] == 1


def test_lemmatize_text_with_filtered_tokens(mocker: MockerFixture, mixed_tokens: List[str], token_doc: Callable[[List[str]], Doc]) -> None:
    """Test lemmatizing text with some tokens filtered out."""
    doc: Doc = token_doc(mixed_tokens)

    def filter_side_effect(token: spacy.tokens.Token) -> bool:
        return token.lemma_ != "bad_token"

    mocker.patch("word_frequency.pipeline.filter_token", side_effect=filter_side_effect)
    result: Counter[str] = lemmatize_text(doc)

    assert isinstance(result, Counter)
    assert result["hello"] == 1
    assert result["world"] == 1
    assert "bad_token" not in result


def test_lemmatize_text_with_duplicate_lemmas(
    mocker: MockerFixture, duplicate_tokens: List[str], token_doc: Callable[[List[str]], Doc]
) -> None:
    """Test lemmatizing text with duplicate lemmas."""
    doc: Doc = token_doc(duplicate_tokens)

    mocker.patch("word_frequency.pipeline.filter_token", return_value=True)
    result: Counter[str] = lemmatize_text(doc)

    assert isinstance(result, Counter)
    assert result["hello"] == 3
    assert result["world"] == 1


def test_lemmatize_text_empty_doc(token_doc: Callable[[List[str]], Doc]) -> None:
    """Test lemmatizing an empty document."""
    doc: Doc = token_doc([])

    result: Counter[str] = lemmatize_text(doc)

    assert isinstance(result, Counter)
    assert len(result) == 0


def test_process_basic_functionality(mocker: MockerFixture) -> None:
    """Test the basic functionality of the process function."""
    # Setup mocks
    mock_gc = mocker.patch("word_frequency.pipeline.gc")
    mock_tqdm = mocker.patch("word_frequency.pipeline.tqdm")
    mock_lemmatize = mocker.patch("word_frequency.pipeline.lemmatize_text")
    mock_logger = mocker.patch("word_frequency.pipeline.logger")

    mock_nlp = mocker.Mock()
    mock_db = mocker.Mock()
    mock_text_gen: List[str] = ["text1", "text2", "text3"]

    # Mock nlp.pipe to return mock docs
    mock_docs: List[Any] = [mocker.Mock(), mocker.Mock(), mocker.Mock()]
    mock_nlp.pipe.return_value = mock_docs
    mock_tqdm.return_value = mock_docs  # tqdm returns the iterable

    # Mock lemmatize_text to return counters
    mock_counters: List[Counter[str]] = [Counter({"hello": 2, "world": 1}), Counter({"test": 3, "hello": 1}), Counter({"final": 1})]
    mock_lemmatize.side_effect = mock_counters

    # Call function
    process(nlp=mock_nlp, db=mock_db, text_generator=iter(mock_text_gen), total_chunks=3, batch_size=10, n_process=2)

    # Verify nlp.pipe was called correctly
    mock_nlp.pipe.assert_called_once()
    call_args = mock_nlp.pipe.call_args
    assert call_args[1]["batch_size"] == 10
    assert call_args[1]["n_process"] == 2
    assert call_args[1]["as_tuples"] is False

    # Verify tqdm was called with correct total
    mock_tqdm.assert_called_once()
    tqdm_call_args = mock_tqdm.call_args
    assert tqdm_call_args[1]["total"] == 3

    # Verify lemmatize_text was called for each doc
    assert mock_lemmatize.call_count == 3

    # Verify db.bump_many was called for each counter
    assert mock_db.bump_many.call_count == 3
    expected_calls: List[List[tuple[str, int]]] = [[("hello", 2), ("world", 1)], [("test", 3), ("hello", 1)], [("final", 1)]]
    actual_calls = [call[0][0] for call in mock_db.bump_many.call_args_list]

    # Sort each call for comparison (order doesn't matter)
    for expected, actual in zip(expected_calls, actual_calls):
        assert sorted(expected) == sorted(actual)

    # Verify gc.collect was called after each iteration
    assert mock_gc.collect.call_count == 3

    # Verify logging
    mock_logger.info.assert_any_call("Processing 3 chunks with batch size 10, n_process 2")
    mock_logger.info.assert_any_call("Completed processing of all chunks")


def test_process_with_empty_generator(mocker: MockerFixture) -> None:
    """Test process function with empty text generator."""
    mock_gc = mocker.patch("word_frequency.pipeline.gc")
    mock_tqdm = mocker.patch("word_frequency.pipeline.tqdm")
    mock_lemmatize = mocker.patch("word_frequency.pipeline.lemmatize_text")
    mock_logger = mocker.patch("word_frequency.pipeline.logger")

    mock_nlp = mocker.Mock()
    mock_db = mocker.Mock()

    # Empty generator
    empty_list: List[Any] = []
    mock_nlp.pipe.return_value = empty_list
    mock_tqdm.return_value = empty_list

    process(nlp=mock_nlp, db=mock_db, text_generator=iter([]), total_chunks=0, batch_size=10, n_process=2)

    # Should not call lemmatize_text or db.bump_many
    mock_lemmatize.assert_not_called()
    mock_db.bump_many.assert_not_called()
    mock_gc.collect.assert_not_called()

    # Should still log start and completion
    mock_logger.info.assert_any_call("Processing 0 chunks with batch size 10, n_process 2")
    mock_logger.info.assert_any_call("Completed processing of all chunks")
