# Word Frequency CLI Tool
A CLI tool for producing an English word-frequency datasets as .csv files from local .txt files.

Existing solutions produced too many nonsense/malformed tokens and included unnecessary metadata in their outputs.
I wanted a "batteries-included", easy-to-use, word frequency counter that could be pointed at individual text files.


## Description
- Uses spaCy's transformer model (`en_core_web_trf`) for lemmatization, with a custom fallback for known lemmatization errors
- Relies on known English language heuristics (such as the fun fact that the longest English word is "Pneumonoultramicroscopicsilicovolcanoconiosis" at 45 letters)
- Removes all tokens with non-English unicode characters such as diacritics
- Support for batched and parallel processing
- On-disk aggregation using sqlite3

## Installation

### Prerequisites
- Python 3.13+

### Setup
```bash
pipx install word-frequency
```

## CLI Usage

Basic Usage:
```bash
word-frequency --input_filepath="input.txt" --output_filepath="output.csv"
```

There is an example word-frequency dataset at `data/sample_ebook_word_freq.csv` that has been derived from a Chinese webfiction novel.

Advanced Options:

```bash

word-frequency \
    --input_filepath="input.txt" \
    --output_filepath="output.csv" \
    --batch_size=8 \
    --n_process=4 \
    --chunk_size=1000000
```

Parameters:

- `input.txt`: Filepath of the text file to process
- `output.csv`: Filepath of the CSV output file
- `--chunk_size`: Size of text chunks in characters (default: 500,000)
- `--batch_size`: Number of text chunks to process as a single batch (default: 4)
- `--n_process`: Number of parallel processes (default: 2)

Note on memory pressure:
```
total_memory_footprint ~= n_process x batch_size x chunk_size = n_process x batch_size x max_seq_len x embedding_dim
```

If you find yourself going OOM, reduce `batch_size` first then `chunk_size`
