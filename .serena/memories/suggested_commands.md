# Suggested Commands - word-frequency

## Development Setup
```bash
# Install all dependencies including dev tools
uv sync

# Install only production dependencies
uv sync --no-dev

# Add new dependency
uv add package_name

# Add development dependency
uv add --dev package_name
```

## Running the Application
```bash
# Using the installed console script (after uv sync)
word-frequency input.txt output.csv

# Using uv run (recommended for development)
uv run word-frequency input.txt output.csv

# Direct module execution
uv run python -m word_frequency input.txt output.csv

# With custom parameters
uv run word-frequency input.txt output.csv --batch_size=8 --n_process=4 --chunk_size=1000000
```

## Code Quality (following pyproject.toml configuration)
```bash
# Lint and format with ruff (combined tool)
uv run ruff check .                    # Check for issues
uv run ruff check --fix .              # Auto-fix issues
uv run ruff format .                   # Format code

# Type checking
uv run mypy src tests                  # Static type analysis

# All quality checks together
uv run ruff check --fix . && uv run ruff format . && uv run mypy src tests
```

## Testing
```bash
# Run all tests with pytest
uv run pytest

# Run tests with coverage
uv run pytest --cov=word_frequency

# Run specific test file
uv run pytest tests/test_pipeline.py

# Run tests with verbose output
uv run pytest -v

# Run tests and show coverage report
uv run pytest --cov=word_frequency --cov-report=html
```

## Pre-commit Hooks
```bash
# Install pre-commit hooks (if configured)
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files

# Update pre-commit hook versions
uv run pre-commit autoupdate
```

## Package Building and Distribution
```bash
# Build package
uv run python -m build

# Check package
uv run twine check dist/*

# Upload to PyPI (requires authentication)
uv run twine upload dist/*
```

## Database Operations
```bash
# View SQLite database schema
sqlite3 word_freq_v16.db ".schema"

# Query top words
sqlite3 word_freq_v16.db "SELECT word, freq FROM wc ORDER BY freq DESC LIMIT 20;"

# Database info
sqlite3 word_freq_v16.db "SELECT COUNT(*) as total_words, SUM(freq) as total_occurrences FROM wc;"
```

## File Operations (macOS optimized)
```bash
# List project files
ls -la
find . -name "*.py" -type f            # Find Python files
find . -name "*.db" -type f            # Find database files

# macOS Spotlight search
mdfind -name "word_freq"               # Fast file search
mdfind "kind:csv word_freq"            # Search CSV files

# File sizes and disk usage
du -sh *.db *.csv                      # Size of output files
du -sh src/                            # Source code size
```

## Process and Performance Monitoring
```bash
# Monitor Python processes
ps aux | grep python
ps aux | grep word-frequency

# System resources (macOS)
top -l 1 | grep -E "^CPU|^PhysMem"     # CPU and memory usage
vm_stat                                # Virtual memory statistics

# File system monitoring
lsof +D .                              # Open files in directory
```

## Git Operations
```bash
git status
git add .
git commit -m "descriptive message"
git push

# Check for pre-commit hook results
git log --oneline -5                   # Recent commits
```

## Environment Information
```bash
# Python and package versions
python --version
uv --version
uv list                                # List installed packages
uv show spacy                          # Show specific package info

# Project information
uv run python -c "import word_frequency; print(word_frequency.__file__)"
uv run python -m word_frequency --help  # Show CLI help
```
