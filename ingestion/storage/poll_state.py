"""SQLite-backed store for per-source HTTP conditional-GET validators.

Implements the agreed caching strategy: every poll is a conditional GET (If-None-Match /
If-Modified-Since). Validators expire after 24 hours, forcing an unconditional refresh
that re-establishes ground truth even if a source stops sending 304s.

One table: poll_state(source_name PK, last_polled_at, etag, last_modified, etag_cached_at)

Ownership rule (one reader, one writer):
  run.py  reads poll state to build conditional headers and decide who is due.
  engine.py writes poll state after each poll attempt.
  No other component touches this store.

Scope of the 24h TTL: validators only — NOT raw_store, Chroma, or seen_store.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

_TTL = timedelta(hours=24)


@dataclass
class PollState:
    source_name: str
    last_polled_at: datetime | None
    etag: str | None
    last_modified: str | None
    etag_cached_at: datetime | None

    def validators_fresh(self) -> bool:
        """True if the cached ETag/Last-Modified are still within the 24h TTL."""
        if self.etag_cached_at is None:
            return False
        return (datetime.now(timezone.utc) - self.etag_cached_at) < _TTL

    def conditional_headers(self) -> dict:
        """Return HTTP headers for a conditional GET, or {} if validators are stale/absent."""
        if not self.validators_fresh():
            return {}
        headers = {}
        if self.etag:
            headers["If-None-Match"] = self.etag
        if self.last_modified:
            headers["If-Modified-Since"] = self.last_modified
        return headers


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class PollStateStore:
    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS poll_state (
                source_name    TEXT PRIMARY KEY,
                last_polled_at TEXT,
                etag           TEXT,
                last_modified  TEXT,
                etag_cached_at TEXT
            )
            """
        )
        self._conn.commit()

    def get(self, source_name: str) -> PollState:
        """Return the stored PollState for a source, or a blank one if never polled."""
        row = self._conn.execute(
            "SELECT last_polled_at, etag, last_modified, etag_cached_at "
            "FROM poll_state WHERE source_name = ?",
            (source_name,),
        ).fetchone()
        if row is None:
            return PollState(source_name, None, None, None, None)
        return PollState(
            source_name=source_name,
            last_polled_at=_parse_dt(row[0]),
            etag=row[1],
            last_modified=row[2],
            etag_cached_at=_parse_dt(row[3]),
        )

    def touch(self, source_name: str) -> None:
        """304 path: update last_polled_at only; leave validators untouched (TTL keeps aging)."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO poll_state (source_name, last_polled_at)
                VALUES (?, ?)
            ON CONFLICT(source_name) DO UPDATE SET last_polled_at = excluded.last_polled_at
            """,
            (source_name, now),
        )
        self._conn.commit()

    def update(
        self,
        source_name: str,
        etag: str | None,
        last_modified: str | None,
    ) -> None:
        """200 path: write new validators and reset the 24h TTL clock."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO poll_state
                (source_name, last_polled_at, etag, last_modified, etag_cached_at)
                VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_name) DO UPDATE SET
                last_polled_at = excluded.last_polled_at,
                etag           = excluded.etag,
                last_modified  = excluded.last_modified,
                etag_cached_at = excluded.etag_cached_at
            """,
            (source_name, now, etag, last_modified, now),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
