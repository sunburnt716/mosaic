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
