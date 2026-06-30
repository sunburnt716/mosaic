"""The single seam between the source registry and the ingestion engine.

`load_sources()` reads the makeshift registry at `config/sources.json` and returns typed
`SourceConfig` objects. Routing every read through this one function means the backing
store can later be swapped (DB, or a server written by the source_validation system)
without the engine or run loop changing — they only ever see `list[SourceConfig]`.

The engine never discovers or onboards sources at runtime; it only iterates whatever
this loader returns.
"""

import json
from pathlib import Path

from ingestion.core.source_config import SourceConfig

# Repo-root config/sources.json (this file lives in <repo>/ingestion/sources.py).
DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent / "config" / "sources.json"

# Default scheduler cadence when an entry omits poll_interval.
_DEFAULT_INTERVAL = "10m"


def load_sources(path: Path | str = DEFAULT_REGISTRY_PATH) -> list[SourceConfig]:
    """Read the source registry JSON and return typed SourceConfig objects.

    Entries may carry documentation-only keys prefixed with '_' (e.g. '_note'); those
    are ignored. Only the typed SourceConfig fields are read, so the registry can be
    annotated freely without affecting the engine.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    # Accept either a bare list or a {"sources": [...]} envelope (the envelope also
    # lets the file carry a top-level "_comment" without colliding with a source entry).
    entries = raw if isinstance(raw, list) else raw.get("sources", [])

    return [_to_config(entry) for entry in entries]


def _to_config(entry: dict) -> SourceConfig:
    """Build one SourceConfig from a registry entry. Unknown/`_`-prefixed keys are ignored."""
    return SourceConfig(
        name=entry["name"],
        adapter=entry["adapter"],
        tier=entry["tier"],
        url=entry["url"],
        enabled=entry.get("enabled", True),
        params=entry.get("params", {}),
        headers=entry.get("headers", {}),
        poll_interval=entry.get("poll_interval", _DEFAULT_INTERVAL),
        transform=entry.get("transform"),
        expects=entry.get("expects", {}),
        max_fallback_title_rate=entry.get("max_fallback_title_rate"),
        max_empty_body_rate=entry.get("max_empty_body_rate"),
        min_records=entry.get("min_records"),
    )
