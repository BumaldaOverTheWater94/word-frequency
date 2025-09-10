# Technology Stack - word-frequency

## Python Environment
- **Python Version**: 3.13+ (requires modern Python features)
- **Package Manager**: uv (modern, fast Python package manager)
- **Build System**: hatchling (modern Python build backend)
- **Project Configuration**: pyproject.toml (PEP 621 compliant)

## Core Dependencies
- **spaCy**: NLP library for tokenization, lemmatization, POS tagging
  - Uses `en_core_web_trf-3.8.0` transformer model (direct GitHub reference)
  - Custom tokenizer configuration with enhanced regex patterns
- **fire**: CLI framework for automatic command-line interfaces
- **loguru**: Modern logging library with colored, structured output
- **tqdm**: Progress bars for long-running operations

## Development Dependencies
- **ruff**: Ultra-fast Python linter and formatter
  - Replaces black, flake8, isort with single tool
  - Line length: 140 characters
  - Target: Python 3.13
  - Comprehensive rule set (E, W, F, I, B, C4, UP)
- **mypy**: Static type checker (configured but strict mode)
- **pytest**: Testing framework with async support
  - **pytest-cov**: Coverage reporting
  - **pytest-mock**: Mocking utilities
  - **pytest-asyncio**: Async test support
- **pre-commit**: Git hooks for code quality automation

## Release Dependencies
- **build**: Modern Python package building
- **twine**: PyPI package uploading

## Database & Storage
- **SQLite3**: Built-in Python module for local database storage
  - WAL mode for performance optimization
  - NORMAL synchronous mode for speed
- **CSV**: Standard library for data export
- **Collections.Counter**: Efficient word counting data structure

## System Integration
- **multiprocessing**: Via spaCy's `n_process` parameter for parallel processing
- **garbage collection**: Explicit `gc.collect()` for memory management
- **file I/O**: UTF-8 encoding with proper context managers

## Build Configuration
- **Console Scripts**: `word-frequency = "word_frequency.cli:main"`
- **Package Discovery**: `packages = ["src/word_frequency"]`
- **Direct References**: Allowed for spaCy model installation
- **Metadata**: PEP 621 compliant project metadata

## Code Quality Tools
- **Ruff Configuration**:
  - Line length: 140 characters
  - Target version: py313
  - Import sorting with `known-first-party = ["word_frequency"]`
  - Google-style docstring convention
- **MyPy Configuration**:
  - Strict mode enabled
  - Files: ["src", "tests"]
  - Comprehensive type checking rules
- **Pytest Configuration**:
  - Async mode: auto
  - Import mode: importlib
  - Excludes: examples directory

## Architecture Patterns
- **Generator-based**: Memory-efficient text chunking using Python generators
- **Pipeline Architecture**: Clear separation between processing stages
- **Modular Design**: Each module has single responsibility
- **Type Safety**: Comprehensive type hints throughout codebase
- **Performance**: Optimized for large file processing with minimal memory usage
