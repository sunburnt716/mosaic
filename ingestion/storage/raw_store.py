"""SQLite-backed append-only store for raw payloads and normalized Documents.

Two tables:
  raw_payloads(doc_id PK, payload JSON, saved_at)  — written once, never mutated
  documents(doc_id PK, data JSON, saved_at)         — one row per content version

  save_raw(doc_id, raw_payload)        INSERT OR IGNORE (append-only; idempotent)
  save_document(doc)                   INSERT OR REPLACE (idempotent; re-ingesting
                                       the same doc_id is a no-op overwrite)
  get_document(doc_id) -> Document|None
  get_raw(doc_id) -> dict|None

doc_id is derived from (identity_key, content_hash), so an L2 update produces a new
doc_id and inserts a new row alongside the old version — both versions are retained.
Documents are serialized to JSON via dataclasses.asdict; datetimes stored as ISO-8601.
The raw_payload is stored as-is (JSON-encoded). Downstream stages re-run from this
store without re-fetching from external sources.
"""

import dataclasses
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ingestion.core.document import Document


def _doc_to_dict(doc: Document) -> dict:
    d = dataclasses.asdict(doc)
    d["published_date"] = doc.published_date.isoformat()
    d["fetched_at"] = doc.fetched_at.isoformat()
    return d


def _dict_to_doc(d: dict) -> Document:
    d["published_date"] = datetime.fromisoformat(d["published_date"])
    d["fetched_at"] = datetime.fromisoformat(d["fetched_at"])
    return Document(**d)


class RawStore:
    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS raw_payloads (
                doc_id   TEXT PRIMARY KEY,
                payload  TEXT NOT NULL,
                saved_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                doc_id   TEXT PRIMARY KEY,
                data     TEXT NOT NULL,
                saved_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def save_raw(self, doc_id: str, raw_payload) -> None:
        """Persist raw_payload verbatim. INSERT OR IGNORE — never overwrites."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR IGNORE INTO raw_payloads (doc_id, payload, saved_at) VALUES (?,?,?)",
            (doc_id, json.dumps(raw_payload), now),
        )
        self._conn.commit()

    def save_document(self, doc: Document) -> None:
        """Persist normalized Document. INSERT OR REPLACE — allows L2 update overwrites."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO documents (doc_id, data, saved_at) VALUES (?,?,?)",
            (doc.id, json.dumps(_doc_to_dict(doc)), now),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> Document | None:
        row = self._conn.execute(
            "SELECT data FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return _dict_to_doc(json.loads(row[0])) if row else None

    def get_raw(self, doc_id: str):
        row = self._conn.execute(
            "SELECT payload FROM raw_payloads WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self._conn.close()
