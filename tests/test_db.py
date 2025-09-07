import csv
import os
import tempfile

from word_frequency.db import CountsDB


def test_init_creates_database_and_table() -> None:
    """Test that initializing CountsDB creates database with proper table."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path: str = tmp.name

    try:
        db: CountsDB = CountsDB(db_path)

        # Check table exists
        cursor = db.con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='wc'")
        assert cursor.fetchone() is not None

        # Check table structure
        cursor = db.con.execute("PRAGMA table_info(wc)")
        columns = cursor.fetchall()
        assert len(columns) == 2
        assert columns[0][1] == "word"
        assert columns[1][1] == "freq"

    finally:
        os.unlink(db_path)


def test_bump_many_inserts_new_words() -> None:
    """Test that bump_many correctly inserts new words."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path: str = tmp.name

    try:
        db: CountsDB = CountsDB(db_path)

        # Insert some words
        words: list[tuple[str, int]] = [("hello", 3), ("world", 5), ("test", 1)]
        db.bump_many(words)

        # Verify insertion
        cursor = db.con.execute("SELECT word, freq FROM wc ORDER BY word")
        results = cursor.fetchall()

        expected: list[tuple[str, int]] = [("hello", 3), ("test", 1), ("world", 5)]
        assert results == expected

    finally:
        os.unlink(db_path)


def test_bump_many_updates_existing_words() -> None:
    """Test that bump_many correctly updates existing word frequencies."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path: str = tmp.name

    try:
        db: CountsDB = CountsDB(db_path)

        # Insert initial words
        db.bump_many([("hello", 3), ("world", 5)])

        # Update with overlapping words
        db.bump_many([("hello", 2), ("new", 1)])

        # Verify updates
        cursor = db.con.execute("SELECT word, freq FROM wc ORDER BY word")
        results = cursor.fetchall()

        expected: list[tuple[str, int]] = [("hello", 5), ("new", 1), ("world", 5)]
        assert results == expected

    finally:
        os.unlink(db_path)


def test_export_csv_creates_sorted_file() -> None:
    """Test that export_csv creates correctly formatted CSV file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path: str = tmp_db.name

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        csv_path: str = tmp_csv.name

    try:
        db: CountsDB = CountsDB(db_path)

        # Insert test data
        words: list[tuple[str, int]] = [("apple", 10), ("banana", 25), ("cherry", 5), ("date", 15)]
        db.bump_many(words)

        # Export to CSV
        db.export_csv(csv_path)

        # Read and verify CSV content
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            results: list[list[str]] = list(reader)

        # Should be sorted by frequency (descending)
        expected: list[list[str]] = [["banana", "25"], ["date", "15"], ["apple", "10"], ["cherry", "5"]]
        assert results == expected

    finally:
        os.unlink(db_path)
        os.unlink(csv_path)


def test_empty_database_export() -> None:
    """Test exporting an empty database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path: str = tmp_db.name

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        csv_path: str = tmp_csv.name

    try:
        db: CountsDB = CountsDB(db_path)
        db.export_csv(csv_path)

        # Verify empty CSV
        with open(csv_path, encoding="utf-8") as f:
            content: str = f.read()
        assert content == ""

    finally:
        os.unlink(db_path)
        os.unlink(csv_path)
