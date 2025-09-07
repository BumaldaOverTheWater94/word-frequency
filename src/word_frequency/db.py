import csv
import sqlite3


class CountsDB:
    def __init__(self, path: str) -> None:
        self.con = sqlite3.connect(path, isolation_level=None)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("CREATE TABLE IF NOT EXISTS wc(word TEXT PRIMARY KEY, freq INTEGER NOT NULL)")

    def bump_many(self, items: list[tuple[str, int]]) -> None:
        self.con.executemany(
            "INSERT INTO wc(word,freq) VALUES(?, ?) ON CONFLICT(word) DO UPDATE SET freq = wc.freq + EXCLUDED.freq",
            items,
        )

    def export_csv(self, csv_path: str) -> None:
        cur = self.con.execute("SELECT word, freq FROM wc ORDER BY freq DESC")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(cur)
