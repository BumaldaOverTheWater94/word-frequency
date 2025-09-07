import time

import fire
from loguru import logger

from word_frequency.nlp import load_model
from word_frequency.pipeline import init_database, process
from word_frequency.text_chunker import text_generator


def run_word_frequency_analysis(
    input_filepath: str,
    output_filepath: str,
    *,
    batch_size: int = 4,
    n_process: int = 2,
    chunk_size: int = 500_000,
    max_length: int = 550_000,
) -> None:
    if chunk_size < 45:
        raise ValueError("chunk_size must be at least 45 characters")

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


def main() -> None:
    fire.Fire(run_word_frequency_analysis)


if __name__ == "__main__":
    main()
