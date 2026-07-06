"""Per-source content-quality gate — the fourth, softest validation layer.

Where the lower layers ask "is *this record* valid?" and reject (transport/parse fail-closed;
per-record contract drops-and-counts), this gate asks "does this *batch of already-valid
records* look collectively wrong?" and **only warns — it never drops, classifies, or routes**.
A false positive that silently quarantined real breaking news would be worse than the silent
failure it guards against, so the gate's sole output is advisory: warnings + the statistics
it computed.

  check(docs, config) -> QualityReport

It runs on the normalized batch BEFORE dedup, so collapse/degeneracy signals aren't masked by
deduplication (on an all-duplicate re-poll the batch is still degenerate-or-not on its own terms).

Source-agnostic by design: the checks know *failure shapes* (empty bodies, collapsed URLs),
never specific sources. Per-source judgement comes from optional thresholds on SourceConfig,
applied only where set — never from branching on a source name here.

Red flags:
  - TITLE_FALLBACK    fraction of placeholder/empty titles over threshold
  - BODY_EMPTY        fraction of empty bodies over threshold (only when body is expected)
  - URL_COLLAPSE      every record in a non-trivial batch shares one URL
  - IDENTITY_COLLAPSE every record in a non-trivial batch shares one identity_key
  - HASH_COLLAPSE     every record in a non-trivial batch shares one content_hash
  - URL_MALFORMED     any URL carries chars that signal a botched parse (brackets, // in path)
  - EMPTY_BATCH       fewer usable records than the source's min_records (the only check that
                      must fire on an empty batch)
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.core.document import Document
    from ingestion.core.source_config import SourceConfig

# Source-agnostic default thresholds. Per-source SourceConfig fields override these
# where set; left None, these defaults apply.
DEFAULT_MAX_EMPTY_BODY_RATE = 0.80  # warn if >80% of docs have empty body (when body expected)
DEFAULT_MAX_FALLBACK_TITLE_RATE = 0.50  # warn if >50% of titles are placeholders/empty
MIN_BATCH_FOR_COLLAPSE = 5  # collapse checks need a non-trivial batch (a 1-doc batch is
# trivially "collapsed", so guarding avoids false positives on tiny/slow-news batches)

# A URL carrying these signals a botched parse (e.g. the old EDGAR list-repr bug:
# .../data/['001-39218']//-index.htm). (?<!:)// catches a doubled slash in the path
# while ignoring the scheme's "https://".
_BAD_URL_RE = re.compile(r"[\[\]']|(?<!:)//")

# A title ending in one of these is a placeholder the adapter fell back to.
_FALLBACK_TITLE_RE = re.compile(r"(?i)(unknown|untitled|none|n/?a)\s*$")


@dataclass
class QualityReport:
    """Structured gate output: advisory warnings plus the batch statistics computed.

    Both are surfaced in the run summary; neither changes what is stored.
    """

    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def check(docs: list["Document"], config: "SourceConfig") -> QualityReport:
    """Inspect a normalized batch from one source and return advisory warnings + stats."""
    warnings: list[str] = []
    expects = config.expects or {}
    n = len(docs)

    # EMPTY_BATCH is the one flag that must be evaluated on an empty batch, so it comes
    # first — before the early return below. Only active when the source sets min_records.
    if config.min_records is not None and n < config.min_records:
        warnings.append(
            f"EMPTY_BATCH: {n} usable records this poll (< min_records={config.min_records}); "
            f"source unexpectedly thin — check the feed or mappings"
        )

    if n == 0:
        # Nothing to compute rates/collapse over; EMPTY_BATCH (above) is the only signal.
        return QualityReport(warnings=warnings, stats={"records": 0})

    # --- compute batch statistics once, reused by every flag and surfaced in the summary ---
    empty_bodies = sum(1 for d in docs if not d.body.strip())
    fallback_titles = sum(1 for d in docs if not d.title or _FALLBACK_TITLE_RE.search(d.title))
    malformed_urls = [d.url for d in docs if _BAD_URL_RE.search(d.url)]
    unique_urls = len({d.url for d in docs})
    unique_identity_keys = len({d.identity_key for d in docs})
    unique_content_hashes = len({d.content_hash for d in docs})
    empty_body_rate = empty_bodies / n
    fallback_title_rate = fallback_titles / n

    stats = {
        "records": n,
        "empty_body_rate": round(empty_body_rate, 3),
        "fallback_title_rate": round(fallback_title_rate, 3),
        "unique_urls": unique_urls,
        "unique_identity_keys": unique_identity_keys,
        "unique_content_hashes": unique_content_hashes,
    }

    # --- rate flags: per-source threshold where set, else the source-agnostic default ---
    max_fallback = _or_default(config.max_fallback_title_rate, DEFAULT_MAX_FALLBACK_TITLE_RATE)
    if fallback_title_rate > max_fallback:
        warnings.append(
            f"TITLE_FALLBACK: {fallback_titles}/{n} titles are placeholders/empty "
            f"(rate {fallback_title_rate:.2f} > {max_fallback:.2f}); check title mapping"
        )

    if expects.get("body", True):  # EDGAR sets body=false: discovery is metadata-only
        max_empty = _or_default(config.max_empty_body_rate, DEFAULT_MAX_EMPTY_BODY_RATE)
        if empty_body_rate > max_empty:
            warnings.append(
                f"BODY_EMPTY: {empty_bodies}/{n} docs have empty body "
                f"(rate {empty_body_rate:.2f} > {max_empty:.2f}); check body mapping"
            )

    # --- collapse flags: a whole non-trivial batch degenerating to a single value ---
    if n >= MIN_BATCH_FOR_COLLAPSE:
        if unique_urls == 1:
            warnings.append(
                f"URL_COLLAPSE: all {n} records share one URL ({docs[0].url!r}); "
                f"likely a mapping pointing every record at the same link"
            )
        if unique_identity_keys == 1:
            warnings.append(
                f"IDENTITY_COLLAPSE: all {n} records share one identity_key "
                f"({docs[0].identity_key!r}); dedup would treat the whole batch as one article"
            )
        if unique_content_hashes == 1:
            warnings.append(
                f"HASH_COLLAPSE: all {n} records share one content_hash; "
                f"content is likely constant (check the body mapping)"
            )

    # --- malformed-URL flag: fires on a single bad record (a parse artifact, not a rate) ---
    if malformed_urls:
        warnings.append(
            f"URL_MALFORMED: {len(malformed_urls)} URLs contain suspicious chars "
            f"(brackets/quotes/'//' in path); first: {malformed_urls[0]!r}"
        )

    return QualityReport(warnings=warnings, stats=stats)


def _or_default(value: float | None, default: float) -> float:
    """Return the per-source override when set, otherwise the source-agnostic default."""
    return default if value is None else value
