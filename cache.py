import sqlite3
import json
from contextlib import contextmanager
from state import PaperExtraction

DB_PATH = "extractions_cache.db"


def init_cache():
    """Call once at startup to ensure the table exists."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS extractions (
                arxiv_id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
        """)


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_cached(arxiv_id: str) -> PaperExtraction | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT data FROM extractions WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
    if row is None:
        return None
    return PaperExtraction(**json.loads(row[0]))


def save_to_cache(extraction: PaperExtraction):
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO extractions (arxiv_id, data) VALUES (?, ?)",
            (extraction.arxiv_id, extraction.model_dump_json()),
        )