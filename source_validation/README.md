# source_validation/ (scaffold — not yet implemented)

The authoring-time system for **validating and onboarding sources** — confirming a feed
is real, healthy, and correctly mapped before it enters the ingestion registry.

## Relationship to ingestion
The ingestion engine **never discovers or onboards sources at runtime**; it only iterates
already-validated entries. This system is the producer of that registry.

Until it exists, **`config/sources.json` is the makeshift stand-in** — a hand-authored
file in the same shape this system will eventually write. Ingestion reads it through the
single `load_sources()` seam (`ingestion/sources.py`), so swapping this file for a
DB/server-backed store later requires no change to the engine.

## Responsibility (future)
- Validate a candidate source (resolves? returns the declared format? mappings produce
  conformant Documents?) at authoring time.
- Assist field-mapping authoring and assign trust `tier`.
- Persist validated sources to the registry ingestion consumes.

## Non-goals for now
No validation logic, UI, auto-detection, or assisted mapping is built here yet. Runtime
source health (did a *known* source's payload come back healthy?) is ingestion's concern,
not this system's.
