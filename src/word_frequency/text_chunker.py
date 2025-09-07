import math
from collections.abc import Iterator

from loguru import logger


def chunk_chars(t: str, chunk_size: int) -> Iterator[str]:
    """
    Chunk text at word boundaries to avoid splitting words.
    Only looks backward to find word boundaries, never forward.
    Never splits words, even if they exceed chunk_size.
    Discards chunks containing malformed/gibberish words longer than 45 characters.

    Args:
        t: Input text to chunk
        chunk_size: Target chunk size in characters (minimum 45)

    Raises:
        ValueError: If chunk_size is less than 45

    Yields:
        Text chunks that end at word boundaries when possible
    """

    if chunk_size < 45:
        raise ValueError("chunk_size must be at least 45 characters")

    i, L = 0, len(t)

    while i < L:
        # Skip any leading whitespace
        while i < L and t[i].isspace():
            i += 1

        if i >= L:
            break

        # Calculate target end position
        target_end = min(i + chunk_size, L)

        # If we're at the end of text, just take what's left
        if target_end >= L:
            yield t[i:].rstrip()
            break

        # Check if we're at a natural word boundary
        if t[target_end].isspace():
            # Perfect - we're exactly at a word boundary
            actual_end = target_end
        else:
            # We're in the middle of a word
            # Look backward to see if there's a word boundary within reasonable distance
            found_boundary = False
            max_search_distance = min(45, target_end - i)

            for pos in range(target_end - 1, target_end - max_search_distance - 1, -1):
                if pos <= i:
                    break
                if t[pos].isspace():
                    # Found a word boundary by looking backward
                    # Complete the current word instead of breaking at the boundary
                    actual_end = target_end
                    while actual_end < L and not t[actual_end].isspace():
                        actual_end += 1
                    found_boundary = True
                    break

            if not found_boundary:
                # No word boundary found within 45 characters
                # If we're at the very beginning of text, include the entire first word
                if i == 0:
                    actual_end = 0
                    while actual_end < L and not t[actual_end].isspace():
                        actual_end += 1
                else:
                    # We're not at the beginning - this is a malformed/gibberish word
                    # Scan forward to find the next word boundary and discard this entire chunk
                    skip_to = target_end
                    while skip_to < L and not t[skip_to].isspace():
                        skip_to += 1
                    # Skip the whitespace too
                    while skip_to < L and t[skip_to].isspace():
                        skip_to += 1
                    i = skip_to
                    continue  # Skip this malformed chunk, don't yield anything

        # Extract the chunk and strip trailing whitespace
        chunk = t[i:actual_end].rstrip()
        if chunk:  # Only yield non-empty chunks
            yield chunk

        # Move to next position, skipping whitespace
        i = actual_end
        while i < L and t[i].isspace():
            i += 1  # Ensure we make progress  # Ensure we make progress  # Ensure we make progress  # Ensure we make progress  # Ensure we make progress  # Ensure we make progress  # Ensure we make progress  # Ensure we make progress


def text_generator(filepath: str, *, chunk_size: int = 500_000) -> tuple[Iterator[str], int]:
    with open(filepath, encoding="utf-8") as f:
        t = f.read().lower()
    total_chunks = math.ceil(len(t) / chunk_size)
    logger.info(f"Total chunks: {total_chunks}")

    def _chunks() -> Iterator[str]:
        yield from chunk_chars(t, chunk_size)

    return _chunks(), total_chunks
