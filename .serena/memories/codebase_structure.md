# Codebase Structure - word-frequency

## Project Organization
```
word_frequency/
├── src/word_frequency/          # Main package directory
│   ├── __init__.py             # Package initialization (empty)
│   ├── __main__.py             # Module execution entry point
│   ├── cli.py                  # Command-line interface with Fire integration
│   ├── pipeline.py             # Core processing pipeline functions
│   ├── nlp.py                  # NLP model loading and configuration
│   ├── tokens.py               # Token filtering and processing logic
│   ├── text_chunker.py         # Text chunking utilities
│   └── db.py                   # Database operations (CountsDB class)
├── tests/                      # Test directory with pytest
│   ├── test_cli.py            # CLI testing
│   ├── test_db.py             # Database testing
│   ├── test_nlp.py            # NLP functionality testing
│   ├── test_pipeline.py       # Pipeline testing
│   ├── test_text_chunker.py   # Text chunking testing
│   └── test_tokens.py         # Token processing testing
├── data/                      # Sample data directory
│   └── sample_ebook_word_freq.csv
├── pyproject.toml             # Modern Python project configuration
├── uv.lock                    # UV dependency lock file
├── README.md                  # Project documentation
├── CONTRIBUTING.md            # Contribution guidelines
├── .gitignore                 # Git ignore patterns
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
├── .coverage                  # Coverage report data
└── word_freq_*.{csv,db}       # Output files (multiple versions)
```

## Core Modules Detail

### cli.py
**Entry point with Fire CLI framework:**
- `run_word_frequency_analysis()` - Main processing function
- `main()` - CLI entry point (registered as console script)
- Handles command-line argument parsing and validation

### pipeline.py
**Core processing pipeline with 3 main functions:**
- `init_database()` - Database initialization wrapper
- `lemmatize_text()` - Document lemmatization with spaCy
- `process()` - Main processing loop coordinating all stages

### nlp.py
**NLP model management:**
- `load_model()` - spaCy model initialization with custom tokenizer
- `custom_fallback_lemmatizer()` - Fallback lemmatization logic
- Custom tokenizer configuration with enhanced patterns

### tokens.py
**Token filtering and processing:**
- Complex token filtering logic
- ASCII validation, entity removal
- Stop word filtering and length constraints

### text_chunker.py
**Text chunking utilities:**
- Memory-efficient text processing
- Character-based and line-based chunking
- Generator-based lazy evaluation

### db.py
**Database abstraction (24 lines):**
- `CountsDB` class with SQLite backend
- WAL mode optimization for performance
- Bulk word frequency updates and CSV export

## Package Structure
- **Package Name**: `word-frequency` (with hyphen)
- **Module Path**: `src/word_frequency` (with underscore)
- **Console Script**: `word-frequency = "word_frequency.cli:main"`
- **Build Backend**: hatchling
- **Python Requirements**: >=3.13

## Key Design Improvements
- **Proper Package Structure**: Follows modern Python packaging standards
- **Separation of Concerns**: Each module has a specific responsibility
- **Comprehensive Testing**: Full test coverage with pytest
- **Modern Tooling**: uv for dependency management, ruff for linting
- **Pre-commit Hooks**: Automated code quality checks
