# Design Patterns and Guidelines - freq-count

## Architectural Patterns

### 1. Pipeline Pattern
The codebase follows a clear pipeline architecture:
```
Input File → Text Chunks → spaCy Docs → Filtered Tokens → Lemmas → Database → CSV
```
- Each stage has a dedicated function
- Data flows unidirectionally
- Easy to test and debug individual stages

### 2. Generator Pattern for Memory Efficiency
- `text_generator()` yields chunks instead of loading entire file
- `chunk_chars()` provides lazy evaluation for large texts
- spaCy's `nlp.pipe()` processes in batches using generators
- Critical for handling large files without memory issues

### 3. Configuration Object Pattern
- Main parameters (batch_size, n_process, chunk_size) passed through pipeline
- Default values provided at entry point
- Easy to modify behavior without changing core logic

## Code Organization Principles

### 4. Separation of Concerns
- **main.py**: Processing pipeline and CLI
- **db.py**: Database operations only
- **test_tokenizer.py**: Testing utilities
- Clear module boundaries

### 5. Dependency Injection
- spaCy model passed to processing functions
- Database instance passed to processing functions
- Makes functions testable and flexible

## Performance Guidelines

### 6. Memory Management
- Use generators for large data processing
- Explicit `gc.collect()` in processing loops
- Chunked processing to avoid loading entire files
- SQLite WAL mode for concurrent reads

### 7. Batch Processing
- spaCy processes documents in configurable batches
- Database updates use `executemany()` for efficiency
- Multiprocessing via spaCy's built-in n_process

## Error Handling Patterns

### 8. Defensive Programming
- Comprehensive token filtering with multiple safety checks
- File operations with proper encoding specification
- Database operations with conflict resolution (UPSERT)

### 9. Logging Strategy
- Use loguru for structured, colored logging
- Log at key pipeline stages for debugging
- Include performance metrics (timing, counts)

## Code Style Guidelines

### 10. Type Safety
- Full type annotations on all functions
- Use specific types (spacy.Language, Counter[str])
- Modern typing imports (Generator, etc.)

### 11. Functional Style
- Pure functions where possible (filter_token, chunk_chars)
- Immutable data flow
- Side effects isolated to database and logging operations

## Testing Philosophy
- Simple test files over complex framework
- Focus on tokenizer correctness (most complex logic)
- Manual integration testing with real data
- Performance testing with large files
