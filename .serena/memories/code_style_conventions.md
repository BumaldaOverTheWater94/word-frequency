# Code Style and Conventions - freq-count

## Naming Conventions
- **Functions and Variables**: snake_case (e.g., `text_generator`, `chunk_size`)
- **Classes**: PascalCase (e.g., `CountsDB`)
- **Constants**: UPPER_SNAKE_CASE for module-level constants
- **Private/Internal**: Leading underscore (e.g., `_chunks()`)

## Type Hints
- **Comprehensive typing**: All function parameters and return types are typed
- **Modern syntax**: Uses `from typing import Generator` and other modern type imports
- **spaCy types**: Properly typed with spacy.Language, spacy.tokens.Doc, spacy.tokens.Token
- **Generic types**: Uses Counter[str], Generator[str, None, None]

## Function Design
- **Keyword-only arguments**: Uses `*` separator for optional parameters (e.g., `*, batch_size: int = 4`)
- **Default values**: Sensible defaults for all optional parameters
- **Single responsibility**: Each function has a clear, single purpose
- **Generator pattern**: Uses generators for memory-efficient processing

## Code Organization
- **Modular structure**: Database operations in separate db.py
- **Import organization**: Standard library imports first, then third-party, then local
- **Clean separation**: Processing logic separated from CLI interface

## Documentation Style
- **Descriptive names**: Function and variable names are self-documenting
- **Inline comments**: Used sparingly, mainly for complex regex patterns and business logic
- **Logging**: Comprehensive logging with loguru for debugging and monitoring

## Error Handling
- **Defensive programming**: Token filtering with multiple safety checks
- **Resource management**: Proper file handling with context managers
- **Memory management**: Explicit garbage collection in processing loops

## Performance Patterns
- **Lazy evaluation**: Generator-based processing
- **Batch processing**: Configurable batch sizes for spaCy processing
- **Database optimization**: WAL mode, bulk inserts with executemany()
- **Memory efficiency**: Chunked processing to handle large files
