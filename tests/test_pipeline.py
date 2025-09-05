import tempfile
import os
from collections import Counter
from unittest.mock import Mock, patch, MagicMock
import pytest

from word_frequency.pipeline import init_database, lemmatize_text, process
from word_frequency.db import CountsDB


class TestInitDatabase:
    def test_init_database_creates_db_with_correct_path(self):
        """Test that init_database creates a database with the correct path."""
        csv_filepath = "/path/to/output.csv"
        expected_db_path = "/path/to/output.db"
        
        with patch('word_frequency.pipeline.CountsDB') as mock_db_class:
            mock_db_instance = Mock()
            mock_db_class.return_value = mock_db_instance
            
            result = init_database(csv_filepath)
            
            mock_db_class.assert_called_once_with(expected_db_path)
            assert result == mock_db_instance

    def test_init_database_with_different_extensions(self):
        """Test init_database with different file extensions."""
        test_cases = [
            ("output.csv", "output.db"),
            ("data.txt", "data.db"),
            ("file.log", "file.db"),
            ("path/to/file.csv", "path/to/file.db")
        ]
        
        for csv_path, expected_db_path in test_cases:
            with patch('word_frequency.pipeline.CountsDB') as mock_db_class:
                mock_db_instance = Mock()
                mock_db_class.return_value = mock_db_instance
                
                result = init_database(csv_path)
                
                mock_db_class.assert_called_once_with(expected_db_path)
                assert result == mock_db_instance

    @patch('word_frequency.pipeline.logger')
    def test_init_database_logs_message(self, mock_logger):
        """Test that init_database logs an info message."""
        csv_filepath = "test.csv"
        expected_db_path = "test.db"
        
        with patch('word_frequency.pipeline.CountsDB'):
            init_database(csv_filepath)
            
            mock_logger.info.assert_called_once_with(f"Initialized database at {expected_db_path}")


class TestLemmatizeText:
    def create_mock_token(self, lemma="test", should_filter=True):
        """Create a mock spaCy token."""
        token = Mock()
        token.lemma_ = lemma
        return token

    def test_lemmatize_text_with_valid_tokens(self):
        """Test lemmatizing text with tokens that pass filtering."""
        # Create mock doc with tokens
        mock_doc = Mock()
        mock_tokens = [
            self.create_mock_token("hello"),
            self.create_mock_token("world"),
            self.create_mock_token("test")
        ]
        mock_doc.__len__ = Mock(return_value=len(mock_tokens))
        mock_doc.__iter__ = Mock(return_value=iter(mock_tokens))
        
        with patch('word_frequency.pipeline.filter_token', return_value=True):
            result = lemmatize_text(mock_doc)
            
            assert isinstance(result, Counter)
            assert result['hello'] == 1
            assert result['world'] == 1
            assert result['test'] == 1

    def test_lemmatize_text_with_filtered_tokens(self):
        """Test lemmatizing text with some tokens filtered out."""
        mock_doc = Mock()
        mock_tokens = [
            self.create_mock_token("hello"),
            self.create_mock_token("bad_token"),
            self.create_mock_token("world")
        ]
        mock_doc.__len__ = Mock(return_value=len(mock_tokens))
        mock_doc.__iter__ = Mock(return_value=iter(mock_tokens))
        
        def filter_side_effect(token):
            return token.lemma_ != "bad_token"
        
        with patch('word_frequency.pipeline.filter_token', side_effect=filter_side_effect):
            result = lemmatize_text(mock_doc)
            
            assert isinstance(result, Counter)
            assert result['hello'] == 1
            assert result['world'] == 1
            assert 'bad_token' not in result

    def test_lemmatize_text_with_duplicate_lemmas(self):
        """Test lemmatizing text with duplicate lemmas."""
        mock_doc = Mock()
        mock_tokens = [
            self.create_mock_token("hello"),
            self.create_mock_token("hello"),
            self.create_mock_token("world"),
            self.create_mock_token("hello")
        ]
        mock_doc.__len__ = Mock(return_value=len(mock_tokens))
        mock_doc.__iter__ = Mock(return_value=iter(mock_tokens))
        
        with patch('word_frequency.pipeline.filter_token', return_value=True):
            result = lemmatize_text(mock_doc)
            
            assert isinstance(result, Counter)
            assert result['hello'] == 3
            assert result['world'] == 1

    def test_lemmatize_text_empty_doc(self):
        """Test lemmatizing an empty document."""
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=0)
        mock_doc.__iter__ = Mock(return_value=iter([]))
        
        result = lemmatize_text(mock_doc)
        
        assert isinstance(result, Counter)
        assert len(result) == 0

    @patch('word_frequency.pipeline.logger')
    def test_lemmatize_text_logs_token_count(self, mock_logger):
        """Test that lemmatize_text logs the number of tokens."""
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=5)
        mock_doc.__iter__ = Mock(return_value=iter([]))
        
        lemmatize_text(mock_doc)
        
        mock_logger.info.assert_called_once_with("Lemmatizing 5 tokens")


class TestProcess:
    @patch('word_frequency.pipeline.gc')
    @patch('word_frequency.pipeline.tqdm')
    @patch('word_frequency.pipeline.lemmatize_text')
    @patch('word_frequency.pipeline.logger')
    def test_process_basic_functionality(self, mock_logger, mock_lemmatize, mock_tqdm, mock_gc):
        """Test the basic functionality of the process function."""
        # Setup mocks
        mock_nlp = Mock()
        mock_db = Mock()
        mock_text_gen = ['text1', 'text2', 'text3']
        
        # Mock nlp.pipe to return mock docs
        mock_docs = [Mock(), Mock(), Mock()]
        mock_nlp.pipe.return_value = mock_docs
        mock_tqdm.return_value = mock_docs  # tqdm returns the iterable
        
        # Mock lemmatize_text to return counters
        mock_counters = [
            Counter({'hello': 2, 'world': 1}),
            Counter({'test': 3, 'hello': 1}),
            Counter({'final': 1})
        ]
        mock_lemmatize.side_effect = mock_counters
        
        # Call function
        process(
            nlp=mock_nlp,
            db=mock_db,
            text_generator=iter(mock_text_gen),
            total_chunks=3,
            batch_size=10,
            n_process=2
        )
        
        # Verify nlp.pipe was called correctly
        mock_nlp.pipe.assert_called_once()
        call_args = mock_nlp.pipe.call_args
        assert call_args[1]['batch_size'] == 10
        assert call_args[1]['n_process'] == 2
        assert call_args[1]['as_tuples'] is False
        
        # Verify tqdm was called with correct total
        mock_tqdm.assert_called_once()
        tqdm_call_args = mock_tqdm.call_args
        assert tqdm_call_args[1]['total'] == 3
        
        # Verify lemmatize_text was called for each doc
        assert mock_lemmatize.call_count == 3
        
        # Verify db.bump_many was called for each counter
        assert mock_db.bump_many.call_count == 3
        expected_calls = [
            [('hello', 2), ('world', 1)],
            [('test', 3), ('hello', 1)],
            [('final', 1)]
        ]
        actual_calls = [call[0][0] for call in mock_db.bump_many.call_args_list]
        
        # Sort each call for comparison (order doesn't matter)
        for expected, actual in zip(expected_calls, actual_calls):
            assert sorted(expected) == sorted(actual)
        
        # Verify gc.collect was called after each iteration
        assert mock_gc.collect.call_count == 3
        
        # Verify logging
        mock_logger.info.assert_any_call("Processing 3 chunks with batch size 10, n_process 2")
        mock_logger.info.assert_any_call("Completed processing of all chunks")

    @patch('word_frequency.pipeline.gc')
    @patch('word_frequency.pipeline.tqdm')
    @patch('word_frequency.pipeline.lemmatize_text')
    @patch('word_frequency.pipeline.logger')
    def test_process_with_empty_generator(self, mock_logger, mock_lemmatize, mock_tqdm, mock_gc):
        """Test process function with empty text generator."""
        mock_nlp = Mock()
        mock_db = Mock()
        
        # Empty generator
        mock_nlp.pipe.return_value = []
        mock_tqdm.return_value = []
        
        process(
            nlp=mock_nlp,
            db=mock_db,
            text_generator=iter([]),
            total_chunks=0,
            batch_size=10,
            n_process=2
        )
        
        # Should not call lemmatize_text or db.bump_many
        mock_lemmatize.assert_not_called()
        mock_db.bump_many.assert_not_called()
        mock_gc.collect.assert_not_called()
        
        # Should still log start and completion
        mock_logger.info.assert_any_call("Processing 0 chunks with batch size 10, n_process 2")
        mock_logger.info.assert_any_call("Completed processing of all chunks")

    @patch('word_frequency.pipeline.gc')
    @patch('word_frequency.pipeline.tqdm')
    @patch('word_frequency.pipeline.lemmatize_text')
    def test_process_memory_cleanup(self, mock_lemmatize, mock_tqdm, mock_gc):
        """Test that process function calls garbage collection after each iteration."""
        mock_nlp = Mock()
        mock_db = Mock()
        
        # Setup for single iteration
        mock_doc = Mock()
        mock_nlp.pipe.return_value = [mock_doc]
        mock_tqdm.return_value = [mock_doc]
        mock_lemmatize.return_value = Counter({'word': 1})
        
        process(
            nlp=mock_nlp,
            db=mock_db,
            text_generator=iter(['text']),
            total_chunks=1,
            batch_size=5,
            n_process=1
        )
        
        # Verify gc.collect was called once (after the single iteration)
        mock_gc.collect.assert_called_once()