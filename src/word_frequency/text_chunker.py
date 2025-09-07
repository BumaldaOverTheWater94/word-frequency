import math
from collections.abc import Iterator

from loguru import logger


def chunk_chars(t: str, chunk_size: int, overlap: int = 0) -> Iterator[str]:
    """
    Chunk text at word boundaries to avoid splitting words.

    Args:
        t: Input text to chunk
        chunk_size: Target chunk size in characters
        overlap: Number of characters to overlap between chunks

    Yields:
        Text chunks that end at word boundaries when possible
    """

    i, L = 0, len(t)

    while i < L:
        # Calculate target end position
        target_end = min(i + chunk_size, L)

        # If we're at the end of text, just take what's left
        if target_end >= L:
            yield t[i:]
            break

        # Look backwards from target_end to find a word boundary
        # Search backward until a word boundary is found
        # prioritize never splitting words over honoring the chunk size
        search_distance = min(chunk_size // 5, 500)
        actual_end = None

        # Start searching backwards for whitespace
        for pos in range(target_end, max(target_end - search_distance, i + 1), -1):
            if pos < L and t[pos - 1].isspace():
                actual_end = pos
                break

        # If still no boundary found, find the next word boundary by looking further
        if actual_end is None:
            # Look for the next whitespace from target_end forward (no distance limit)
            for pos in range(target_end, L):
                if t[pos].isspace():
                    actual_end = pos + 1
                    break
            
            # If no whitespace found at all, take the rest of the text
            if actual_end is None:
                actual_end = L

        # Ensure we make progress (avoid infinite loops)
        if actual_end <= i:
            # This should only happen with very unusual text - find next non-whitespace
            actual_end = i + 1
            while actual_end < L and not t[actual_end].isspace():
                actual_end += 1

        yield t[i:actual_end].rstrip()

        # Move to next position, skipping whitespace to avoid leading spaces
        next_i = actual_end - overlap
        while next_i < L and t[next_i].isspace():
            next_i += 1
        i = next_i


def text_generator(filepath: str, *, chunk_size: int = 500_000) -> tuple[Iterator[str], int]:
    with open(filepath, encoding="utf-8") as f:
        t = f.read().lower()
    total_chunks = math.ceil(len(t) / chunk_size)
    logger.info(f"Total chunks: {total_chunks}")

    def _chunks() -> Iterator[str]:
        yield from chunk_chars(t, chunk_size)

    return _chunks(), total_chunks
