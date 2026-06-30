<<<<<<< HEAD
"""
PollStateStore — per-source runtime polling state.

Tracks when each source was last polled plus HTTP cache-control headers (ETag,
Last-Modified) so the engine can make conditional GET requests and avoid
re-processing unchanged feeds.

Separation of concerns:
  - sources.yaml (SourceConfig): static configuration — what to poll and how.
  - poll_state.json (PollStateStore): runtime state — when it was last polled.
  - seen_store: article-level dedup state — which article hashes we have seen.

run.py reads PollState to decide due-ness.
engine.py writes PollState after a successful fetch.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PollState value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PollState:
    """Immutable snapshot of a source's last-poll metadata.

    last_polled_at — UTC timestamp of the most recent successful fetch.
                     Must be timezone-aware (UTC). None if never polled.
    etag           — HTTP ETag from the last response, for conditional GETs.
    last_modified  — HTTP Last-Modified header value, for conditional GETs.
    """

    last_polled_at: datetime | None
    etag: str | None
    last_modified: str | None

    def __post_init__(self) -> None:
        # Enforce timezone-aware UTC so callers never accidentally compare
        # naive and aware datetimes when computing due-ness.
        if self.last_polled_at is not None and self.last_polled_at.tzinfo is None:
            raise ValueError(
                "PollState.last_polled_at must be timezone-aware (UTC). "
                "Use datetime.now(tz=timezone.utc) or datetime.fromisoformat() "
                "with a UTC offset."
            )


# ---------------------------------------------------------------------------
# Serialization helpers (private)
# ---------------------------------------------------------------------------


def _serialize_state(state: PollState) -> dict[str, str | None]:
    """Convert a PollState to a JSON-serialisable dict."""
    return {
        "last_polled_at": (
            state.last_polled_at.isoformat() if state.last_polled_at is not None else None
        ),
        "etag": state.etag,
        "last_modified": state.last_modified,
    }


def _deserialize_state(raw: dict[str, str | None]) -> PollState:
    """Reconstruct a PollState from a stored dict."""
    raw_lpa = raw.get("last_polled_at")
    last_polled_at: datetime | None = None
    if raw_lpa is not None:
        last_polled_at = datetime.fromisoformat(raw_lpa)
        # Older entries stored without timezone info get treated as UTC.
        if last_polled_at.tzinfo is None:
            last_polled_at = last_polled_at.replace(tzinfo=timezone.utc)
    return PollState(
        last_polled_at=last_polled_at,
        etag=raw.get("etag"),
        last_modified=raw.get("last_modified"),
    )


# ---------------------------------------------------------------------------
# PollStateStore
# ---------------------------------------------------------------------------


class PollStateStore:
    """JSON-file-backed store for per-source polling state.

    Writes are atomic (write-to-temp then rename) so a crash mid-write never
    leaves a corrupt state file. Reads degrade gracefully on a missing or
    corrupt file (returns None / empty state) rather than crashing the process.

    The store file is created on first write. Before any write, reads return
    None for every source.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, source_name: str) -> PollState | None:
        """Return the stored PollState for source_name, or None if never polled."""
        data = self._load()
        raw = data.get(source_name)
        if raw is None:
            return None
        return _deserialize_state(raw)

    def set(self, source_name: str, state: PollState) -> None:
        """Persist updated poll state for source_name.

        Called by engine.py after a successful fetch, not by run.py.
        """
        data = self._load()
        data[source_name] = _serialize_state(state)
        self._save(data)

    # ------------------------------------------------------------------
    # Private I/O
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, dict]:
        """Read the state file. Returns {} on missing file or parse errors."""
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # Log and degrade gracefully — a corrupt state file causes all
            # sources to be treated as never-polled, which is safe (they'll
            # just be re-fetched sooner than necessary).
            log.warning(
                "poll_state_read_error",
                extra={"path": str(self._path), "error": str(exc)},
            )
            return {}

    def _save(self, data: dict) -> None:
        """Write state atomically via a temp file + rename.

        Ensures readers never observe a partial write, even if the process
        is killed between the write and the rename.
        """
        tmp = self._path.with_suffix(".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except OSError as exc:
            log.error(
                "poll_state_write_error",
                extra={"path": str(self._path), "error": str(exc)},
            )
            raise
=======
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
>>>>>>> main
