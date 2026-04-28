"""
presets.py
Persistent storage for user-defined research-domain presets.

The four built-in presets (threat_intel, ransomware_malware,
personal_identity, corporate_espionage) live in `llm.PRESET_PROMPTS` and
are NEVER touched here — this module only manages user-created domains.

Schema
------
custom_presets
  id            INTEGER PRIMARY KEY AUTOINCREMENT
  name          TEXT UNIQUE NOT NULL    -- human label, e.g. "Crypto Tracing"
  description   TEXT NOT NULL DEFAULT ''
  system_prompt TEXT NOT NULL           -- full system prompt template
  created_at    TEXT NOT NULL.
  updated_at    TEXT NOT NULL
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path

DB_PATH = Path("investigations") / "robin.db"
_logger = logging.getLogger(__name__)

# Prefix used to disambiguate a custom-preset key from a built-in one
# everywhere keys are passed around as strings.
CUSTOM_KEY_PREFIX = "custom:"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_presets_table() -> None:
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS custom_presets (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    UNIQUE NOT NULL,
                description   TEXT    NOT NULL DEFAULT '',
                system_prompt TEXT    NOT NULL,
                created_at    TEXT    NOT NULL,
                updated_at    TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_custom_presets_name
                ON custom_presets(name);
        """)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    # The string key the UI uses to identify this preset everywhere
    # (selectbox value, generate_summary `preset` argument, etc.).
    d["key"] = f"{CUSTOM_KEY_PREFIX}{d['id']}"
    return d


def is_custom_key(preset_key: str) -> bool:
    return isinstance(preset_key, str) and preset_key.startswith(CUSTOM_KEY_PREFIX)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def list_presets() -> list:
    init_presets_table()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM custom_presets ORDER BY name COLLATE NOCASE"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_preset(preset_id: int) -> dict | None:
    init_presets_table()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM custom_presets WHERE id=?", (preset_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_preset_by_key(preset_key: str) -> dict | None:
    """
    Look up a custom preset by the `custom:<id>` key the UI passes around.
    Returns None for built-in keys or unknown ids.
    """
    if not is_custom_key(preset_key):
        return None
    try:
        preset_id = int(preset_key[len(CUSTOM_KEY_PREFIX):])
    except ValueError:
        return None
    return get_preset(preset_id)


def create_preset(name: str, system_prompt: str, description: str = "") -> dict:
    """
    Insert a new custom preset. Raises ValueError on conflicting/empty input.
    """
    init_presets_table()
    name = (name or "").strip()
    system_prompt = (system_prompt or "").strip()
    description = (description or "").strip()

    if not name:
        raise ValueError("Preset name cannot be empty.")
    if not system_prompt:
        raise ValueError("System prompt cannot be empty.")

    ts = datetime.now().isoformat()
    with _connect() as conn:
        try:
            cur = conn.execute(
                """INSERT INTO custom_presets
                   (name, description, system_prompt, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, description, system_prompt, ts, ts),
            )
            preset_id = cur.lastrowid
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"A custom preset named '{name}' already exists."
            ) from exc

    return get_preset(preset_id)


def update_preset(
    preset_id: int,
    name: str | None = None,
    system_prompt: str | None = None,
    description: str | None = None,
) -> dict | None:
    """Partial update — only non-None fields are written."""
    init_presets_table()
    sets, params = [], []
    if name is not None:
        n = name.strip()
        if not n:
            raise ValueError("Preset name cannot be empty.")
        sets.append("name=?")
        params.append(n)
    if system_prompt is not None:
        sp = system_prompt.strip()
        if not sp:
            raise ValueError("System prompt cannot be empty.")
        sets.append("system_prompt=?")
        params.append(sp)
    if description is not None:
        sets.append("description=?")
        params.append(description.strip())

    if not sets:
        return get_preset(preset_id)

    sets.append("updated_at=?")
    params.append(datetime.now().isoformat())
    params.append(preset_id)

    with _connect() as conn:
        try:
            conn.execute(
                f"UPDATE custom_presets SET {', '.join(sets)} WHERE id=?",
                params,
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "Another preset already uses that name."
            ) from exc

    return get_preset(preset_id)


def delete_preset(preset_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM custom_presets WHERE id=?", (preset_id,))
