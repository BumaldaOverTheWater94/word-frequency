# Task Completion Checklist - freq-count

## Code Quality Checks
After completing any coding task, run these commands:

### 1. Linting and Formatting
```bash
# Check for linting issues
uv run ruff check .

# Auto-fix issues where possible
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### 2. Type Checking (if mypy is added in future)
```bash
# Currently no mypy configuration, but cache directory exists
# If mypy is added: uv run mypy .
```

### 3. Testing
```bash
# Run existing test file
uv run python test_tokenizer.py

# Manual testing with small input
python main.py small_input.txt test_output.csv
```

## Validation Steps

### 4. Functional Testing
- Test with sample input file
- Verify database creation (.db file)
- Verify CSV export with expected format
- Check log output for errors

### 5. Performance Verification
- Monitor memory usage with large files
- Verify multiprocessing works correctly
- Check database write performance

### 6. Code Review Checklist
- Type hints on all new functions
- Proper error handling
- Logging for debugging
- Memory-efficient patterns (generators, chunking)
- Database transactions handled properly
- File I/O with proper encoding (UTF-8)

## Pre-commit Considerations
Since there's no formal pre-commit hooks:
1. Manual ruff check
2. Ensure all imports are used
3. Verify function naming follows snake_case
4. Check that new code follows existing patterns
5. Test with both small and large input files

## Documentation Updates
- Update README.md if functionality changes
- Add docstrings for new public functions
- Update type hints if function signatures change
