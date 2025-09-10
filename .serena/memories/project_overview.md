# Project Overview - word-frequency

## Purpose
word-frequency is a Python word frequency analyzer that processes large text files to generate word frequency databases and CSV exports. The tool is designed for efficient processing of large documents using spaCy's advanced NLP capabilities with transformer models.

## Core Functionality
- **Text Processing**: Reads large text files and chunks them for memory-efficient processing
- **Advanced NLP Pipeline**: Uses spaCy with `en_core_web_trf` transformer model for superior tokenization and lemmatization
- **Intelligent Word Filtering**: Applies sophisticated filtering rules to extract meaningful words
  - Removes punctuation, numbers, entities, and stop words
  - ASCII-only filtering with vowel requirements
  - Length constraints and custom exclusions
- **High-Performance Database Storage**: SQLite with WAL mode optimization for concurrent access
- **Flexible Export Options**: Generates frequency-sorted CSV files for analysis

## Key Features
- **Custom Enhanced Tokenizer**: Advanced prefix/suffix/infix patterns for better text processing
- **Memory-Efficient Architecture**: Chunked processing handles files of any size
- **Parallel Processing**: Multi-process support via spaCy's built-in parallelization
- **Optimized Database**: SQLite with WAL mode and bulk operations for performance
- **Comprehensive Filtering**: Multi-stage token filtering with ASCII, entity, and linguistic checks
- **Progress Tracking**: Visual progress bars with tqdm for long-running operations
- **Fallback Lemmatization**: Custom lemmatizer for edge cases where spaCy fails

## Technical Highlights
- **Modern Python Packaging**: Uses hatchling build backend with proper package structure
- **Console Script Integration**: Installed as `word-frequency` command-line tool
- **Comprehensive Testing**: Full pytest test suite covering all modules
- **Code Quality**: Ruff linting, mypy type checking, pre-commit hooks
- **Development Workflow**: UV package manager for fast dependency management

## Use Cases
- **Literary Analysis**: Vocabulary analysis of books, articles, and literary works
- **Corpus Linguistics**: Large-scale text corpus analysis and research
- **Content Analysis**: Blog posts, documentation, and web content analysis
- **Language Research**: Vocabulary extraction and frequency studies
- **Text Mining**: Data extraction from large document collections
- **Educational Tools**: Vocabulary learning and text complexity analysis

## Performance Characteristics
- **Scalable**: Handles files from KB to GB sizes efficiently
- **Fast**: Transformer-based NLP with parallel processing
- **Memory Efficient**: Generator-based streaming with configurable chunk sizes
- **Reliable**: Comprehensive error handling and logging
- **Portable**: Single Python package with minimal system dependencies
