# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup
```bash
uv sync --extra dev              # Install all dependencies including dev tools
uv run pre-commit install       # Install git pre-commit hooks
```

### Code Quality
```bash
uv run ruff check --fix          # Lint and auto-fix issues
uv run ruff format               # Format code
uv run mypy                      # Type checking
```

### Testing
```bash
uv run pytest                   # Run all tests
uv run pytest tests/test_pipeline.py  # Run specific test file
uv run pytest --cov=word_frequency --cov-report=html  # Generate coverage report
```

### Running the Application
```bash
# After uv sync, use the console script:
word-frequency --input_filepath="input.txt" --output_filepath="output.csv"

# For development:
uv run word-frequency --input_filepath="input.txt" --output_filepath="output.csv"

# Advanced usage with performance tuning:
word-frequency \
    --input_filepath="input.txt" \
    --output_filepath="output.csv" \
    --batch_size=8 \
    --n_process=4 \
    --chunk_size=1000000
```

## Architecture Overview

This is a high-performance word frequency analyzer built around a pipeline architecture with memory-efficient processing of large text files.

### Core Processing Pipeline
1. **Text Chunking** (`text_chunker.py`) - Memory-efficient text splitting into manageable chunks
2. **NLP Processing** (`nlp.py`) - spaCy model loading with custom tokenizer configuration
3. **Token Filtering** (`tokens.py`) - Multi-stage filtering to extract meaningful English words
4. **Lemmatization** (`pipeline.py`) - Transformer-based lemmatization with custom fallbacks
5. **Database Storage** (`db.py`) - SQLite with WAL mode for concurrent access
6. **CSV Export** - Frequency-sorted output generation

### Key Architectural Decisions

**Custom Tokenizer**: Disables spaCy's default tokenizer in favor of custom regex patterns optimized for English text processing.

**Transformer Model**: Uses `en_core_web_trf` (1.5GB model) for superior lemmatization accuracy compared to smaller models.

**Memory Management**: Generator-based chunked processing prevents OOM issues with large files. Memory footprint formula:
```
total_memory_gb = (n_process × batch_size × chunk_size × 300 × 4 / 1024³) + (n_process × 1.5)
```

**Database Strategy**: SQLite with WAL mode enables concurrent reads during processing. Bulk inserts with `executemany()` for performance.

### Module Responsibilities
- `cli.py` - Fire-based command-line interface
- `pipeline.py` - Core processing orchestration and lemmatization
- `nlp.py` - spaCy model management and custom tokenizer setup
- `tokens.py` - English language heuristics and token filtering logic
- `text_chunker.py` - Memory-efficient text processing utilities
- `db.py` - SQLite database operations with performance optimizations

### Performance Tuning
- Reduce `batch_size` first if experiencing OOM issues
- Adjust `chunk_size` second (default: 500,000 characters)
- `n_process` should account for 1.5GB per process for model loading
- Leave ~6GB system overhead for optimal performance

### Language Processing Features
- Removes non-English Unicode characters and diacritics
- Applies English word length limits (max 45 characters based on "Pneumonoultramicroscopicsilicovolcanoconiosis")
- Custom fallback lemmatizer for edge cases where spaCy fails
- Sophisticated token filtering removes entities, punctuation, numbers, and stop words
