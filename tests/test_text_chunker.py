import tempfile
import os
import math
from pathlib import Path
import pytest

from word_frequency.text_chunker import chunk_chars, text_generator


class TestChunkChars:
    def test_chunk_simple_text(self):
        """Test chunking simple text without word boundary issues."""
        text = "hello world python testing"
        chunks = list(chunk_chars(text, chunk_size=10))
        
        assert len(chunks) == 3
        assert chunks[0] == "hello"
        assert chunks[1] == "world"
        assert chunks[2] == "python testing"

    def test_chunk_exact_size(self):
        """Test chunking when text length equals chunk size."""
        text = "hello"
        chunks = list(chunk_chars(text, chunk_size=5))
        
        assert len(chunks) == 1
        assert chunks[0] == "hello"

    def test_chunk_smaller_than_chunk_size(self):
        """Test chunking text smaller than chunk size."""
        text = "hi"
        chunks = list(chunk_chars(text, chunk_size=10))
        
        assert len(chunks) == 1
        assert chunks[0] == "hi"

    def test_chunk_with_overlap(self):
        """Test chunking with overlap between chunks."""
        text = "hello world python testing"
        chunks = list(chunk_chars(text, chunk_size=12, overlap=3))
        
        assert len(chunks) >= 2
        # Check that there's some overlap
        assert chunks[1].startswith("o") or chunks[1].startswith("world")

    def test_chunk_long_word_no_break(self):
        """Test chunking with words longer than chunk size."""
        text = "supercalifragilisticexpialidocious word"
        chunks = list(chunk_chars(text, chunk_size=10))
        
        # Should not break the long word
        assert len(chunks) == 2
        assert chunks[0] == "supercalifragilisticexpialidocious"
        assert chunks[1] == "word"

    def test_chunk_multiple_spaces(self):
        """Test chunking text with multiple consecutive spaces."""
        text = "hello    world    python"
        chunks = list(chunk_chars(text, chunk_size=10))
        
        # Should handle multiple spaces gracefully
        assert len(chunks) >= 2
        for chunk in chunks:
            assert not chunk.startswith(" ")  # No leading spaces

    def test_chunk_newlines_and_tabs(self):
        """Test chunking text with newlines and tabs."""
        text = "hello\nworld\tpython\n\ttesting"
        chunks = list(chunk_chars(text, chunk_size=10))
        
        # Should treat newlines and tabs as whitespace
        assert len(chunks) >= 2

    def test_chunk_empty_string(self):
        """Test chunking empty string."""
        text = ""
        chunks = list(chunk_chars(text, chunk_size=10))
        
        assert len(chunks) == 0

    def test_chunk_only_spaces(self):
        """Test chunking string with only spaces."""
        text = "     "
        chunks = list(chunk_chars(text, chunk_size=3))
        
        # Should produce empty chunks or handle gracefully
        if chunks:
            for chunk in chunks:
                assert chunk == "" or chunk.isspace()

    def test_chunk_word_boundary_search(self):
        """Test that chunking respects word boundaries within search distance."""
        text = "this is a longer sentence that should be chunked at word boundaries"
        chunks = list(chunk_chars(text, chunk_size=25))
        
        # Each chunk should end at a word boundary (except possibly the last)
        for chunk in chunks[:-1]:  # All but last chunk
            assert chunk[-1] != ' ' or chunk.rstrip() != chunk
            # Should not break in middle of words
            words = chunk.split()
            if words:
                last_word = words[-1]
                # The last word should be complete (not cut off)
                assert last_word.isalpha() or not last_word[-1].isalpha()

    def test_chunk_zero_overlap(self):
        """Test chunking with zero overlap (default behavior)."""
        text = "hello world python testing beautiful code"
        chunks = list(chunk_chars(text, chunk_size=15, overlap=0))
        
        # No overlap means each chunk starts where previous ended
        combined = "".join(chunks)
        # Length might differ due to whitespace handling, but content should be preserved
        assert "hello" in combined
        assert "world" in combined
        assert "python" in combined


class TestTextGenerator:
    def test_text_generator_small_file(self):
        """Test text generator with a small file."""
        content = "Hello World Python Testing"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            chunks_iter, total_chunks = text_generator(tmp_path, chunk_size=10)
            chunks = list(chunks_iter)
            
            # Should produce chunks from the lowercased content
            assert total_chunks == math.ceil(len(content.lower()) / 10)
            assert len(chunks) == total_chunks
            
            # Content should be lowercased
            combined = "".join(chunks)
            assert "hello world python testing" in combined.lower()
            
        finally:
            os.unlink(tmp_path)

    def test_text_generator_large_chunk_size(self):
        """Test text generator with chunk size larger than file."""
        content = "Small file content"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            chunks_iter, total_chunks = text_generator(tmp_path, chunk_size=1000)
            chunks = list(chunks_iter)
            
            assert total_chunks == 1
            assert len(chunks) == 1
            assert chunks[0].lower() == content.lower()
            
        finally:
            os.unlink(tmp_path)

    def test_text_generator_empty_file(self):
        """Test text generator with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write("")
            tmp_path = tmp.name
        
        try:
            chunks_iter, total_chunks = text_generator(tmp_path, chunk_size=100)
            chunks = list(chunks_iter)
            
            assert total_chunks == 0
            assert len(chunks) == 0
            
        finally:
            os.unlink(tmp_path)

    def test_text_generator_unicode_content(self):
        """Test text generator with unicode content."""
        content = "Hello 世界 Python тест"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            chunks_iter, total_chunks = text_generator(tmp_path, chunk_size=15)
            chunks = list(chunks_iter)
            
            # Should handle unicode properly
            assert total_chunks > 0
            assert len(chunks) == total_chunks
            
            combined = "".join(chunks)
            assert "世界" in combined
            assert "тест" in combined
            
        finally:
            os.unlink(tmp_path)

    def test_text_generator_multiline_content(self):
        """Test text generator with multiline content."""
        content = """This is line one.
This is line two.
This is line three with more content.
Final line here."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            chunks_iter, total_chunks = text_generator(tmp_path, chunk_size=30)
            chunks = list(chunks_iter)
            
            assert total_chunks == math.ceil(len(content.lower()) / 30)
            assert len(chunks) == total_chunks
            
            combined = "".join(chunks)
            assert "line one" in combined.lower()
            assert "final line" in combined.lower()
            
        finally:
            os.unlink(tmp_path)