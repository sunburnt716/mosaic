"""SQLite-backed store tracking which documents have been ingested, powering L1/L2 dedup.

One table: seen(identity_key PK, content_hash, updated_at).

  contains_hash(content_hash) -> bool      L1 short-circuit (any key with this hash?)
  get_hash(identity_key) -> str | None     L2 lookup (what hash did we last see for this key?)
  set_hash(identity_key, content_hash)     write after ingest (called by engine, not dedup)

Consistency contract: set_hash must be called atomically with raw_store writes inside the
engine's transaction boundary. A partial write (doc saved, hash not recorded) would cause
re-ingestion on the next run.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class SeenStore:
    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen (
                identity_key TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def contains_hash(self, content_hash: str) -> bool:
        """L1 short-circuit: has this exact content_hash been seen for ANY identity_key?"""
        row = self._conn.execute(
            "SELECT 1 FROM seen WHERE content_hash = ? LIMIT 1", (content_hash,)
        ).fetchone()
        return row is not None

    def get_hash(self, identity_key: str) -> str | None:
        """L2 lookup: return the content_hash last stored for this identity_key, or None."""
        row = self._conn.execute(
            "SELECT content_hash FROM seen WHERE identity_key = ?", (identity_key,)
        ).fetchone()
        return row[0] if row else None

    def set_hash(self, identity_key: str, content_hash: str) -> None:
        """Record or update the content_hash for an identity_key after successful ingest."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO seen (identity_key, content_hash, updated_at) VALUES (?,?,?)",
            (identity_key, content_hash, now),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
