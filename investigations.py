"""
investigations.py
Persistent SQLite storage for Robin investigations.

Replaces the flat investigation_*.json approach. Benefits:
  - Indexed queries (fast even with hundreds of investigations)
  - Status tracking: "active" | "pending" | "closed"
  - Free-form tags stored as a comma-separated string
  - One-time auto-migration of existing JSON files on first run

Schema
------
investigations
  id            INTEGER PRIMARY KEY AUTOINCREMENT
  timestamp     TEXT    ISO-8601 creation time
  query         TEXT    original user query
  refined_query TEXT    LLM-refined query
  model         TEXT    model used
  preset        TEXT    preset label
  summary       TEXT    full markdown summary
  status        TEXT    "active" | "pending" | "closed"  (default "active")
  tags          TEXT    comma-separated tags              (default "")

sources
  id                INTEGER PRIMARY KEY AUTOINCREMENT
  investigation_id  INTEGER → investigations(id) ON DELETE CASCADE
  title             TEXT
  link              TEXT
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH   = Path("investigations") / "robin.db"
LEGACY_DIR = Path("investigations")
VALID_STATUSES = ("active", "pending", "closed")

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Schema + migration
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if absent, then migrate any legacy JSON files."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS investigations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT NOT NULL,
                query         TEXT NOT NULL,
                refined_query TEXT NOT NULL DEFAULT '',
                model         TEXT NOT NULL DEFAULT '',
                preset        TEXT NOT NULL DEFAULT '',
                summary       TEXT NOT NULL DEFAULT '',
                status        TEXT NOT NULL DEFAULT 'active',
                tags          TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS sources (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                investigation_id  INTEGER NOT NULL
                                  REFERENCES investigations(id) ON DELETE CASCADE,
                title             TEXT NOT NULL DEFAULT '',
                link              TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_inv_timestamp ON investigations(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_inv_status    ON investigations(status);
        """)
    _migrate_legacy_json()


def _migrate_legacy_json() -> None:
    """Import existing investigation_*.json files into SQLite, then delete them."""
    json_files = sorted(LEGACY_DIR.glob("investigation_*.json"))
    if not json_files:
        return
    migrated = 0
    for path in json_files:
        try:
            data = json.loads(path.read_text())
            save_investigation(
                query=data.get("query", ""),
                refined_query=data.get("refined_query", ""),
                model=data.get("model", ""),
                preset_label=data.get("preset", ""),
                sources=data.get("sources", []),
                summary=data.get("summary", ""),
                timestamp=data.get("timestamp"),
            )
            path.unlink()
            migrated += 1
        except Exception as exc:
            _logger.warning("Failed to migrate %s: %s", path.name, exc)
    if migrated:
        _logger.info("Migrated %d legacy JSON investigation(s) to SQLite.", migrated)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_investigation(
    query: str,
    refined_query: str,
    model: str,
    preset_label: str,
    sources: list,
    summary: str,
    status: str = "active",
    tags: str = "",
    timestamp: str = None,
) -> int:
    """Save an investigation. Returns the new row id."""
    ts = timestamp or datetime.now().isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO investigations
               (timestamp, query, refined_query, model, preset, summary, status, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, query, refined_query, model, preset_label, summary, status, tags.strip()),
        )
        inv_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO sources (investigation_id, title, link) VALUES (?, ?, ?)",
            [(inv_id, s.get("title", ""), s.get("link", "")) for s in sources],
        )
    return inv_id


def update_status(inv_id: int, status: str) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Choose from {VALID_STATUSES}.")
    with _connect() as conn:
        conn.execute("UPDATE investigations SET status=? WHERE id=?", (status, inv_id))


def update_tags(inv_id: int, tags: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE investigations SET tags=? WHERE id=?", (tags.strip(), inv_id))


def delete_investigation(inv_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM investigations WHERE id=?", (inv_id,))


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row, sources: list) -> dict:
    d = dict(row)
    d["sources"] = sources
    d["tags_list"] = [t.strip() for t in d.get("tags", "").split(",") if t.strip()]
    return d


def load_all(
    status_filter: str = None,
    tag_filter: str = None,
    limit: int = 200,
) -> list:
    """Return investigations newest-first, with optional status/tag filters."""
    init_db()
    clauses, params = [], []
    if status_filter:
        clauses.append("status = ?")
        params.append(status_filter)
    if tag_filter:
        clauses.append("LOWER(tags) LIKE ?")
        params.append(f"%{tag_filter.lower()}%")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM investigations {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        results = []
        for row in rows:
            srcs = conn.execute(
                "SELECT title, link FROM sources WHERE investigation_id=?", (row["id"],)
            ).fetchall()
            results.append(_row_to_dict(row, [dict(s) for s in srcs]))
    return results


def load_one(inv_id: int) -> dict:
    """Load a single investigation by id. Returns None if not found."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM investigations WHERE id=?", (inv_id,)
        ).fetchone()
        if not row:
            return None
        srcs = conn.execute(
            "SELECT title, link FROM sources WHERE investigation_id=?", (inv_id,)
        ).fetchall()
        return _row_to_dict(row, [dict(s) for s in srcs])


def get_all_tags() -> list:
    """Return a sorted deduplicated list of every tag used across all investigations."""
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT tags FROM investigations WHERE tags != ''").fetchall()
    tags = set()
    for row in rows:
        for t in row["tags"].split(","):
            t = t.strip()
            if t:
                tags.add(t)
    return sorted(tags)