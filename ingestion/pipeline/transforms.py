"""Per-source transform functions applied between adapter parse and generic field-mapping.

A transform takes one raw entry dict (as the adapter yielded it) plus its SourceConfig
and returns a corrected/augmented dict whose keys match what the normalizer expects.
Transforms are pure: no network, no disk, no clock — same input always yields same output.

Register a transform with @register("name"), then reference it in sources.yaml via the
`transform:` field. The normalizer resolves and applies it automatically.

Transform contract:
  (raw_entry: dict, config: SourceConfig) -> dict
"""

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.core.source_config import SourceConfig

# Registry: transform name -> callable
_REGISTRY: dict[str, Callable[[dict, "SourceConfig"], dict]] = {}

# Title pattern for EDGAR getcurrent Atom entries:
# "{form_type} - {entity_name} ({CIK}) (Filer)"
_EDGAR_TITLE_RE = re.compile(r"^(.+?)\s+-\s+(.+?)\s+\(\d+\)\s+\(Filer\)$")


def register(name: str):
    """Decorator that registers a function under the given transform name."""

    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_transform(name: str) -> Callable[[dict, "SourceConfig"], dict]:
    """Return the transform callable for `name`, or raise ValueError if unknown."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown transform {name!r}. Registered transforms: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


@register("edgar_filing_url")
def edgar_filing_url(raw: dict, config: "SourceConfig") -> dict:
    """Clean an EDGAR getcurrent Atom entry into a well-formed normalizer input.

    The getcurrent feed provides the canonical archive URL directly in the link field,
    so no URL construction is needed. This transform's job is to extract a clean title
    from the raw format "{form_type} - {entity_name} ({CIK}) (Filer)" and expose the
    form type as a separate field for downstream use.

    Guarantee: the produced `url` is taken unchanged from the RSS adapter (already
    canonical). The title never contains list reprs like ['...'] or doubled slashes.
    """
    raw = dict(raw)  # don't mutate the adapter's dict

    title_raw = raw.get("title", "")
    m = _EDGAR_TITLE_RE.match(title_raw)
    if m:
        form_type = m.group(1).strip()
        entity_name = m.group(2).strip()
        raw["title"] = f"{form_type} — {entity_name}"
        raw["form"] = form_type
    else:
        # Title didn't match expected format — leave as-is so the quality gate catches it.
        raw["form"] = ""

    return raw
