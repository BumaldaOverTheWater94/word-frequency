import sys
from typing import Generator
import spacy
import csv
from collections import Counter
import os
from loguru import logger
import gc
import math
import time
from tqdm import tqdm
import fire

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
    logger.info(f"Loaded model with max length {max_length}")
    return nlp


def filter_token(token: spacy.tokens.Token) -> bool:
    return (
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
    )


def lemmatize_text(doc: spacy.tokens.Doc) -> Counter[str]:
    logger.info(f"Lemmatizing {len(doc)} tokens")
    return Counter(token.lemma_ for token in doc if filter_token(token))


# accept minor chunking errors to simplify implementation
def chunk_chars(t, chunk_size: int, overlap: int = 0):
    i, L = 0, len(t)
    while i < L:
        j = min(i + chunk_size, L)
        yield t[i:j]
        i = j - overlap


def process(
    nlp: spacy.Language,
    text_generator: Generator[str, None, None],
    total_chunks: int,
    output_filepath: str,
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
        ),
        total=total_chunks,
        as_tuples=False,
        yield_when_ready=True,
    ):
        tokens = lemmatize_text(doc)
        update_csv(tokens, output_filepath)
        gc.collect()


def _read_existing_freqs(filepath: str) -> Counter[str]:
    c: Counter[str] = Counter()
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for word, freq in reader:
            try:
                c.update({word: int(freq)})
            except ValueError:
                logger.warning(f"Invalid frequency for word {word}")
                continue
    return c


# merge existing dataset with new word frequencies
def update_csv(tokens: Counter[str], filepath: str):
    if os.path.exists(filepath):
        c = _read_existing_freqs(filepath)
        c.update(tokens)
        logger.info(f"Updated existing frequencies for {len(c)} words")
        tokens = c

    rows = tokens.most_common()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main(
    input_filepath: str,
    output_filepath: str,
    *,
    batch_size: int = 6,
    n_process: int = 6,
    chunk_size: int = 500_000,
    max_length: int = 550_000,
):
    logger.info(
        f"Starting to process {input_filepath} with batch size {batch_size}, n_process {n_process}, chunk size {chunk_size}, max length {max_length}"
    )
    start_time = time.time()
    nlp = load_model(max_length)
    chunks, total_chunks = text_generator(input_filepath, chunk_size=chunk_size)
    process(
        nlp,
        chunks,
        total_chunks,
        output_filepath,
        batch_size=batch_size,
        n_process=n_process,
    )
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    fire.Fire(main)
