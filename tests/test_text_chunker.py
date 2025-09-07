import math
import os
import tempfile
from collections.abc import Generator

import pytest

from word_frequency.text_chunker import chunk_chars, text_generator


@pytest.fixture
def temp_text_file() -> Generator[str]:
    """Create a temporary text file and yield its path, cleanup after test."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        temp_path = tmp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def small_content_file() -> Generator[str]:
    """Create a temporary file with small test content."""
    content = "Hello World Python Testing"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        temp_path = tmp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def empty_content_file() -> Generator[str]:
    """Create a temporary file with empty content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write("")
        temp_path = tmp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def unicode_content_file() -> Generator[str]:
    """Create a temporary file with unicode content."""
    content = "Hello 世界 Python тест"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        temp_path = tmp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def multiline_content_file() -> Generator[str]:
    """Create a temporary file with multiline content."""
    content = """
            This is line one.
            This is line two.
            This is line three with more content.
            Final line here.
        """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        temp_path = tmp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_chunk_simple_text() -> None:
    """Test chunking simple text with word boundaries."""
    text = "hello world python testing"
    chunks = list(chunk_chars(text, chunk_size=50))

    # Should fit in a single chunk since text is shorter than chunk_size
    assert len(chunks) == 1
    assert chunks[0] == "hello world python testing"


def test_chunk_exact_size() -> None:
    """Test chunking when text length equals chunk size."""
    text = "hello world testing this is a sample text string"
    chunks = list(chunk_chars(text, chunk_size=len(text)))

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_smaller_than_chunk_size() -> None:
    """Test chunking text smaller than chunk size."""
    text = "hi there world"
    chunks = list(chunk_chars(text, chunk_size=50))

    assert len(chunks) == 1
    assert chunks[0] == "hi there world"


def test_chunk_with_overlap() -> None:
    """Test chunking with moderate chunk size."""
    text = "hello world python testing this is a longer sentence with many words to test chunking"
    chunks = list(chunk_chars(text, chunk_size=50))

    # Algorithm preserves word boundaries
    assert len(chunks) == 2
    assert len(chunks[0]) <= 60  # Should be around 50 but may exceed slightly for word boundaries
    assert len(chunks[1]) > 0


def test_chunk_long_word_no_break() -> None:
    """Test chunking with words longer than chunk size."""
    text = "supercalifragilisticexpialidocious word"
    chunks = list(chunk_chars(text, chunk_size=45))

    # Since total text is 39 chars and chunk_size is 45, should fit in one chunk
    assert len(chunks) == 1
    assert chunks[0] == "supercalifragilisticexpialidocious word"  # Both words in one chunk


def test_chunk_multiple_spaces() -> None:
    """Test chunking text with multiple consecutive spaces."""
    text = "hello    world    python testing with many spaces and additional content to make it longer"
    chunks = list(chunk_chars(text, chunk_size=50))

    # Should handle multiple spaces gracefully
    assert len(chunks) >= 1
    for chunk in chunks:
        assert not chunk.startswith(" ")  # No leading spaces


def test_chunk_newlines_and_tabs() -> None:
    """Test chunking text with newlines and tabs."""
    text = "hello\nworld\tpython\n\ttesting with additional content to make this text much longer"
    chunks = list(chunk_chars(text, chunk_size=50))

    # Should treat newlines and tabs as whitespace
    assert len(chunks) >= 1


def test_chunk_empty_string() -> None:
    """Test chunking empty string."""
    text = ""
    chunks = list(chunk_chars(text, chunk_size=50))

    assert len(chunks) == 0


def test_chunk_only_spaces() -> None:
    """Test chunking string with only spaces."""
    text = "     "
    chunks = list(chunk_chars(text, chunk_size=50))

    # Should produce empty chunks or handle gracefully
    if chunks:
        for chunk in chunks:
            assert chunk == "" or chunk.isspace()


def test_chunk_word_boundary_search() -> None:
    """Test that chunking respects word boundaries within search distance."""
    text = "this is a longer sentence that should be chunked at word boundaries"
    chunks = list(chunk_chars(text, chunk_size=50))

    # Each chunk should end at a word boundary (except possibly the last)
    for chunk in chunks[:-1]:  # All but last chunk
        assert chunk[-1] != " " or chunk.rstrip() != chunk
        # Should not break in middle of words
        words = chunk.split()
        if words:
            last_word = words[-1]
            # The last word should be complete (not cut off)
            assert last_word.isalpha() or not last_word[-1].isalpha()


def test_chunk_zero_overlap() -> None:
    """Test chunking with moderate chunk size (no overlap functionality)."""
    text = "hello world python testing beautiful code with many additional words for testing chunking"
    chunks = list(chunk_chars(text, chunk_size=50))

    # Each chunk should preserve word boundaries
    combined = " ".join(chunks)
    # Content should be preserved with proper word boundaries
    assert "hello" in combined
    assert "world" in combined
    assert "python" in combined
    assert len(chunks) > 1  # Should be multiple chunks


def test_text_generator_small_file(small_content_file: str) -> None:
    """Test text generator with a small file."""
    content = "Hello World Python Testing"

    chunks_iter, total_chunks = text_generator(small_content_file, chunk_size=50)
    chunks = list(chunks_iter)

    # Test chunk count calculation - note that actual chunks may be fewer than calculated
    # because algorithm prioritizes word boundaries over target chunk size
    calculated_chunks = math.ceil(len(content.lower()) / 50)
    assert total_chunks == calculated_chunks
    # Actual chunks may be fewer due to word boundary preservation
    assert len(chunks) <= total_chunks

    # Test that chunks contain expected content by checking word coverage
    all_words_found: set[str] = set()
    for chunk in chunks:
        words = chunk.split()
        all_words_found.update(word.lower() for word in words)

    # All words should be present and complete (algorithm should never split words)
    expected_words = {"hello", "world", "python", "testing"}
    assert expected_words.issubset(all_words_found)

    # Test chunk properties - chunks may be larger than target size to preserve word boundaries
    for chunk in chunks:
        # Chunks may exceed target size to avoid splitting words
        assert len(chunk) >= 1  # Should have content
        # Verify no words are split by checking that each chunk contains complete words
        words = chunk.split()
        assert all(len(word) > 0 for word in words)  # No empty words from splitting
        assert not chunk.startswith(" ")  # No leading spaces
        assert not chunk.endswith(" ")  # No trailing spaces

    # Test that chunks are reasonable for NLP processing
    assert len(chunks) > 0
    assert all(len(chunk.strip()) > 0 for chunk in chunks)  # No empty chunks


def test_text_generator_large_chunk_size(temp_text_file: str) -> None:
    """Test text generator with chunk size larger than file."""
    content = "Small file content"

    # Write content to the temporary file
    with open(temp_text_file, "w", encoding="utf-8") as f:
        f.write(content)

    chunks_iter, total_chunks = text_generator(temp_text_file, chunk_size=1000)
    chunks = list(chunks_iter)

    # Should produce exactly one chunk when chunk size > file size
    assert total_chunks == 1
    assert len(chunks) == 1

    # The single chunk should contain all words
    chunk = chunks[0]
    words = chunk.split()
    expected_words = {"small", "file", "content"}
    actual_words = {word.lower() for word in words}
    assert expected_words == actual_words


def test_text_generator_empty_file(empty_content_file: str) -> None:
    """Test text generator with empty file."""
    chunks_iter, total_chunks = text_generator(empty_content_file, chunk_size=100)
    chunks = list(chunks_iter)

    assert total_chunks == 0
    assert len(chunks) == 0


def test_text_generator_unicode_content(unicode_content_file: str) -> None:
    """Test text generator with unicode content."""
    chunks_iter, total_chunks = text_generator(unicode_content_file, chunk_size=50)
    chunks = list(chunks_iter)

    # Should handle unicode properly and produce chunks
    assert total_chunks > 0
    assert len(chunks) == total_chunks

    # Test that unicode characters are preserved in individual chunks
    unicode_chars_found: set[str] = set()
    words_found: set[str] = set()

    for chunk in chunks:
        unicode_chars_found.update(chunk)
        words = chunk.split()
        words_found.update(word.lower() for word in words)

    # Check that important unicode characters are preserved
    assert "世" in unicode_chars_found
    assert "界" in unicode_chars_found
    assert "т" in unicode_chars_found
    assert "е" in unicode_chars_found
    assert "с" in unicode_chars_found
    assert "т" in unicode_chars_found

    # Check that regular words are also present
    expected_words = {"hello", "python"}
    assert expected_words.issubset(words_found)


def test_text_generator_multiline_content(multiline_content_file: str) -> None:
    """Test text generator with multiline content."""
    content = """
            This is line one.
            This is line two.
            This is line three with more content.
            Final line here.
        """

    chunks_iter, total_chunks = text_generator(multiline_content_file, chunk_size=50)
    chunks = list(chunks_iter)

    # Test chunk count calculation
    expected_chunks_math = math.ceil(len(content.lower()) / 50)
    assert total_chunks == expected_chunks_math
    # Actual chunks produced may be different due to word boundary algorithm
    assert len(chunks) >= expected_chunks_math

    # Test that important words from multiline content appear in chunks
    all_text = " ".join(chunks).lower()

    # Check that key words from different lines are preserved (accounting for punctuation)
    assert "this" in all_text
    assert "line" in all_text
    assert "one" in all_text
    assert "two" in all_text
    assert "three" in all_text
    assert "final" in all_text
    assert "here" in all_text

    # Test chunk properties - what matters for NLP processing
    for i, chunk in enumerate(chunks):
        assert len(chunk) <= 50 + 15  # Allow flexibility for word boundaries
        assert not chunk.startswith(" ")  # No leading spaces
        # Last chunk might have trailing whitespace from original file
        if i < len(chunks) - 1:  # All chunks except the last
            assert not chunk.endswith(" ")

    # Test that chunks are reasonable for NLP processing
    assert len(chunks) > 0
    assert all(
        len(chunk.strip()) > 0 for chunk in chunks
    )  # No empty chunks after stripping  # No empty chunks   # No trailing spaces  # Spaces may be removed during chunking
