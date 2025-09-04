import spacy
from typing import Generator
from loguru import logger
from tqdm import tqdm
from wordfreq.db import CountsDB
import gc
from collections import Counter
from wordfreq.tokens import filter_token



def init_database(output_filepath: str):
    db = CountsDB(output_filepath[:-4] + ".db")
    logger.info(f"Initialized database at {output_filepath[:-4] + '.db'}")
    return db




def lemmatize_text(doc: spacy.tokens.Doc) -> Counter[str]:
    logger.info(f"Lemmatizing {len(doc)} tokens")
    return Counter(token.lemma_ for token in doc if filter_token(token))



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


