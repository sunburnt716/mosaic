"""Adapter registry — maps the string adapter key in config/sources.json to a concrete Adapter class.

To register a new adapter:
  1. Implement the Adapter ABC in a new file under adapters/.
  2. Add one line to _REGISTRY below.
  No other files need to change.
"""

from ingestion.adapters.base import Adapter, ConfigError
from ingestion.adapters.rest_json import RestJsonAdapter
from ingestion.adapters.rss import RssAdapter

_REGISTRY: dict[str, type[Adapter]] = {
    "rss": RssAdapter,
    "rest_json": RestJsonAdapter,
}


def get_adapter(key: str) -> type[Adapter]:
    """Return the Adapter class for `key`, or raise ConfigError if unknown."""
    if key not in _REGISTRY:
        raise ConfigError(f"Unknown adapter {key!r}. Valid keys: {sorted(_REGISTRY)}")
    return _REGISTRY[key]
