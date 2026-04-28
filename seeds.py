"""
seeds.py
Seed URL management for Robin's Deep Crawl feature.

Seeds are stored in the same SQLite DB as investigations
(investigations/robin.db) in a dedicated `seeds` table.
This replaces the old seeds.json + add_seeds.py approach.

Schema
------
seeds
  id          INTEGER PRIMARY KEY AUTOINCREMENT
  url         TEXT UNIQUE NOT NULL
  hash        TEXT NOT NULL            -- SHA-256 of the URL
  name        TEXT NOT NULL DEFAULT '' -- human label
  status_code INTEGER                  -- last HTTP response code (nullable)
  crawled     INTEGER NOT NULL DEFAULT 0   -- 0/1 bool
  loaded      INTEGER NOT NULL DEFAULT 0   -- 0/1 bool (deep content extracted)
  content     TEXT    NOT NULL DEFAULT '' -- extracted plain-text content
  crawled_at  TEXT                        -- ISO-8601 of last crawl
  added_at    TEXT    NOT NULL            -- ISO-8601 timestamp
"""

import hashlib
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

DB_PATH = Path("investigations") / "robin.db"
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection helper (mirrors investigations.py)
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Schema init (safe to call multiple times)
# ---------------------------------------------------------------------------

def init_seeds_table() -> None:
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS seeds (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT    UNIQUE NOT NULL,
                hash        TEXT    NOT NULL,
                name        TEXT    NOT NULL DEFAULT '',
                status_code INTEGER,
                crawled     INTEGER NOT NULL DEFAULT 0,
                loaded      INTEGER NOT NULL DEFAULT 0,
                content     TEXT    NOT NULL DEFAULT '',
                crawled_at  TEXT,
                added_at    TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_seeds_crawled ON seeds(crawled);
            CREATE INDEX IF NOT EXISTS idx_seeds_loaded  ON seeds(loaded);
        """)
        # Idempotent migration for older DBs that pre-date `content`/`crawled_at`.
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(seeds)")}
        if "content" not in existing_cols:
            conn.execute("ALTER TABLE seeds ADD COLUMN content TEXT NOT NULL DEFAULT ''")
        if "crawled_at" not in existing_cols:
            conn.execute("ALTER TABLE seeds ADD COLUMN crawled_at TEXT")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def add_seed(url: str, name: str = "") -> dict:
    """
    Add a new seed URL. Returns the seed dict.
    If the URL already exists, returns the existing record unchanged.
    """
    init_seeds_table()
    url = url.strip().rstrip("/")
    if not url:
        raise ValueError("URL cannot be empty.")

    url_hash = _sha256(url)
    ts = datetime.now().isoformat()

    with _connect() as conn:
        try:
            conn.execute(
                """INSERT INTO seeds (url, hash, name, added_at)
                   VALUES (?, ?, ?, ?)""",
                (url, url_hash, name.strip() or "unknown", ts),
            )
        except sqlite3.IntegrityError:
            # Already exists — just return existing
            pass

    return get_seed_by_url(url)


def mark_crawled(seed_id: int, status_code: int = None, content: str = "") -> None:
    """
    Mark a seed as crawled. If `content` is provided, it's persisted into the
    `content` column and the seed is also flagged as loaded — so a single call
    is enough after a successful Deep Crawl.
    """
    ts = datetime.now().isoformat()
    with _connect() as conn:
        if content:
            conn.execute(
                """UPDATE seeds
                      SET crawled=1, loaded=1, status_code=?, content=?, crawled_at=?
                    WHERE id=?""",
                (status_code, content, ts, seed_id),
            )
        else:
            conn.execute(
                "UPDATE seeds SET crawled=1, status_code=?, crawled_at=? WHERE id=?",
                (status_code, ts, seed_id),
            )


def mark_loaded(seed_id: int, content: str = "") -> None:
    """Mark a seed as fully loaded (content extracted and ready for LLM)."""
    with _connect() as conn:
        if content:
            conn.execute(
                "UPDATE seeds SET loaded=1, content=? WHERE id=?",
                (content, seed_id),
            )
        else:
            conn.execute("UPDATE seeds SET loaded=1 WHERE id=?", (seed_id,))


def delete_seed(seed_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM seeds WHERE id=?", (seed_id,))


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def get_seed_by_url(url: str) -> dict:
    init_seeds_table()
    url = url.strip().rstrip("/")
    with _connect() as conn:
        row = conn.execute("SELECT * FROM seeds WHERE url=?", (url,)).fetchone()
        return _row_to_dict(row) if row else None


def get_all_seeds() -> list:
    init_seeds_table()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM seeds ORDER BY added_at DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_uncrawled(limit: int = 5) -> list:
    """Return up to `limit` seeds that haven't been crawled yet."""
    init_seeds_table()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM seeds WHERE crawled=0 ORDER BY added_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_unloaded(limit: int = 5) -> list:
    """Return up to `limit` seeds that are crawled but not yet loaded."""
    init_seeds_table()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM seeds WHERE crawled=1 AND loaded=0 ORDER BY added_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def seed_urls_from_sources(sources: list) -> None:
    """
    Convenience: bulk-add all source links from a Robin investigation's
    source list into the seeds table so they can be deep-crawled later.
    Each source dict must have a 'link' key and optionally 'title'.
    """
    for src in sources:
        url = src.get("link", "").strip()
        name = src.get("title", "")
        if url:
            try:
                add_seed(url, name)
            except Exception as exc:
                _logger.warning("Could not add seed %s: %s", url, exc)