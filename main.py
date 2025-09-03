import sys
from typing import Generator
import spacy
from collections import Counter
from loguru import logger
import gc
import math
import time
from tqdm import tqdm
import fire
from db import CountsDB
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex
import re

# remove default sink
logger.remove()
logger.add(
    sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>"
)


def text_generator(filepath: str, *, chunk_size: int = 500_000):
    with open(filepath, "r", encoding="utf-8") as f:
        t = f.read().lower()
    total_chunks = math.ceil(len(t) / chunk_size)
    logger.info(f"Total chunks: {total_chunks}")

    def _chunks():
        yield from chunk_chars(t, chunk_size)

    return _chunks(), total_chunks


def load_model(max_length: int) -> spacy.Language:
    # need parser for NER and need tagger for lemmatizer
    # even attribute_ruler is needed for lemmatization
    nlp = spacy.load("en_core_web_trf", exclude=[])
    nlp.max_length = max_length

    # Enhanced custom tokenizer to fix problematic tokenization
    # Enhanced prefix patterns - split these at the beginning of tokens
    custom_prefixes = [
        r"[\.\,\;\:\!\?\[\]\(\)\{\}]",  # Punctuation at start
        r"[\"\'\`]",  # Quotes at start
        r"\([0-9]+\)",  # Standalone parentheses with numbers at start
        r"[~\-]",  # Tilde and dash at start
        r"[\"\=\+]",  # Quote-equals, plus at start
    ]

    # Enhanced suffix patterns - split these at the end of tokens
    custom_suffixes = [
        r"[\.\,\;\:\!\?\[\]\(\)\{\}]",  # Punctuation at end
        r"[\"\'\`]",  # Quotes at end
        r"(?<=[a-zA-Z])\d+",  # Numbers after letters
        r"(?<=[a-zA-Z])\([0-9]+\)",  # Parenthetical numbers after letters
        r"(?<=[0-9])\%",  # Percent after numbers
        r"\([0-9]+\)",  # Standalone parentheses with numbers at end
        r"(?<=[a-zA-Z])\.",  # Periods after single letters
        r"(?<=[a-zA-Z])[-~,]",  # Hyphens, tildes, and commas after letters
        r"[\"\'][^\"\']*[\"\']",  # Complex quote patterns
        r"[\"\=\+]",  # Quote-equals, plus at end
    ]

    # Enhanced infix patterns - split within tokens
    custom_infixes = [
        r"(?<=[A-Za-z])\(",  # Opening paren after letters
        r"(?<=[0-9])\)",  # Closing paren after numbers
        r"(?<=[A-Za-z0-9])\(",  # Opening paren after alphanumeric
        r"(?<=[0-9])(?=[A-Za-z])",  # Letters after numbers
        r"(?<=[A-Za-z])(?=[0-9])",  # Numbers after letters
        r"(?<=[0-9])%\(",  # Percent-paren combo after numbers
        r"(?<=[a-zA-Z])[!~-](?=[a-zA-Z])",  # Exclamation, tildes, and hyphens within compound words
        r"\([0-9]+\)",  # Standalone parentheses with numbers within tokens
        r"[\"\=\+]",  # Quote-equals, plus within tokens
    ]

    # Combine with existing spaCy defaults
    prefixes = list(nlp.Defaults.prefixes) + custom_prefixes
    suffixes = list(nlp.Defaults.suffixes) + custom_suffixes
    infixes = list(nlp.Defaults.infixes) + custom_infixes

    # Compile regex patterns
    prefix_regex = compile_prefix_regex(prefixes)
    suffix_regex = compile_suffix_regex(suffixes)
    infix_regex = compile_infix_regex(infixes)

    # Update tokenizer
    nlp.tokenizer.prefix_search = prefix_regex.search
    nlp.tokenizer.suffix_search = suffix_regex.search
    nlp.tokenizer.infix_finditer = infix_regex.finditer

    logger.info(f"Loaded model with enhanced custom tokenizer, max length {max_length}")
    return nlp


def init_database(output_filepath: str):
    db = CountsDB(output_filepath[:-4] + ".db")
    logger.info(f"Initialized database at {output_filepath[:-4] + '.db'}")
    return db


def filter_token(token: spacy.tokens.Token) -> bool:
    """
    This function provides comprehensive filtering to prevent problematic tokens
    like "head," from appearing in the final word frequency output.

    Args:
        token: spaCy Token object to filter

    Returns:
        bool: True if token should be included, False if it should be filtered out
    """

    basic_filters = (
        token.is_ascii
        and token.ent_type_ == ""
        and not token.is_space
        and not token.is_punct
        and not token.is_digit
        and not token.is_quote
        and not token.is_bracket
        and not token.is_currency
        and not token.like_num
        and not token.like_url
        and not token.like_email
        and not token.is_stop
        and token.pos_ not in ["SYM", "PUNCT", "INTJ", "NUM"]
    )

    if not basic_filters:
        return False

    lemma = token.lemma_.strip()

    # Define punctuation and symbols
    symbols = [
        "=",
        "+",
        "-",
        "~",
        ".",
        ",",
        "!",
        "?",
        '"',
        "'",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        ":",
        ";",
    ]

    if lemma in symbols or not lemma:
        return False

    if len(lemma) <= 3 and any(c in lemma for c in symbols):
        return False

    if any(c in lemma for c in symbols):
        return False

    excluded_unicode = [
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\u200e",  # Left-to-right mark
        "\u200f",  # Right-to-left mark
        "\ufeff",  # Zero-width no-break space (BOM)
        "\u00a0",  # Non-breaking space
    ]

    if any(char in lemma for char in excluded_unicode):
        return False

    # No English words repeat a letter more than twice (filters onomatopoeia)
    if re.search(r"(.)\\1\\1+", lemma):
        return False

    # Every English word has at least one vowel
    if not re.search(r"[aeiouy]", lemma):
        return False

    return True


def lemmatize_text(doc: spacy.tokens.Doc) -> Counter[str]:
    logger.info(f"Lemmatizing {len(doc)} tokens")
    return Counter(token.lemma_ for token in doc if filter_token(token))


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


def process(
    nlp: spacy.Language,
    db: CountsDB,
    text_generator: Generator[str, None, None],
    total_chunks: int,
    batch_size: int,
    n_process: int,
):
    logger.info(
        f"Processing {total_chunks} chunks with batch size {batch_size}, n_process {n_process}"
    )
    for doc in tqdm(
        nlp.pipe(
            text_generator,
            batch_size=batch_size,
            n_process=n_process,
            as_tuples=False,
        ),
        total=total_chunks,
    ):
        tokens = lemmatize_text(doc)
        db.bump_many(tokens.items())
        gc.collect()
    logger.info("Completed processing of all chunks")


def main(
    input_filepath: str,
    output_filepath: str,
    *,
    batch_size: int = 4,
    n_process: int = 2,
    chunk_size: int = 500_000,
    max_length: int = 550_000,
):
    logger.info(
        f"Starting to process {input_filepath} with batch size {batch_size}, n_process {n_process}, chunk size {chunk_size}, max length {max_length}"
    )
    start_time = time.time()
    nlp = load_model(max_length)
    db = init_database(output_filepath)
    chunks, total_chunks = text_generator(input_filepath, chunk_size=chunk_size)
    process(
        nlp,
        db,
        chunks,
        total_chunks,
        batch_size=batch_size,
        n_process=n_process,
    )
    db.export_csv(output_filepath)
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    fire.Fire(main)
