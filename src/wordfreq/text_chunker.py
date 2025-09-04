import math
from loguru import logger

def chunk_chars(t, chunk_size: int, overlap: int = 0):
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
        # Search within reasonable distance (up to 20% of chunk_size or 500 chars max)
        search_distance = min(chunk_size // 5, 500)
        actual_end = target_end

        # Start searching backwards for whitespace
        for pos in range(target_end, max(target_end - search_distance, i + 1), -1):
            if pos < L and t[pos - 1].isspace():
                actual_end = pos
                break

        # If no whitespace found backwards, look forward for next whitespace
        if actual_end == target_end and target_end < L:
            for pos in range(target_end, min(target_end + search_distance, L)):
                if t[pos].isspace():
                    actual_end = pos + 1
                    break

        # Ensure we don't create empty chunks or infinite loops
        if actual_end <= i:
            actual_end = min(i + chunk_size, L)

        yield t[i:actual_end].rstrip()

        # Move to next position, skipping whitespace to avoid leading spaces
        next_i = actual_end - overlap
        while next_i < L and t[next_i].isspace():
            next_i += 1
        i = next_i


def text_generator(filepath: str, *, chunk_size: int = 500_000):
    with open(filepath, "r", encoding="utf-8") as f:
        t = f.read().lower()
    total_chunks = math.ceil(len(t) / chunk_size)
    logger.info(f"Total chunks: {total_chunks}")

    def _chunks():
        yield from chunk_chars(t, chunk_size)

    return _chunks(), total_chunks


