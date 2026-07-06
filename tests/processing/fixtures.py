"""
Shared, fully-offline fixtures for processing-layer tests.

Builds normalized Documents with representative bodies (filing / article / tweet)
plus the degenerate shapes Phase 0 must catch. Bodies are synthesized from a small
pool of finance-flavored sentences so token counts are deterministic and sit clearly
on the intended side of each threshold — no live network or stored payloads needed.

Not a test module (no `test_` functions), so pytest does not collect it; the test
files import these builders directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import cycle, islice

from ingestion.core.document import Document

# A handful of neutral, news-flavored sentences (~12–15 words each). Cycled to pad
# bodies to a target length while reading like real prose rather than lorem ipsum.
_SENTENCES = (
    "The company reported quarterly results that the market had broadly anticipated this period.",
    "Revenue rose modestly across its core segments while operating margins held roughly steady.",
    "Management reaffirmed prior guidance and pointed to demand trends in its largest regions.",
    "Analysts noted that input costs eased somewhat compared with the prior reporting period.",
    "Shares moved only slightly in trading as investors weighed the figures against expectations.",
)


def _sentences(count: int) -> str:
    """Return `count` sentences joined by spaces (a single prose block)."""
    return " ".join(islice(cycle(_SENTENCES), count))


def _paragraphs(num_paragraphs: int, sentences_each: int) -> str:
    """Return blank-line-separated paragraphs of prose."""
    return "\n\n".join(_sentences(sentences_each) for _ in range(num_paragraphs))


# ---------------------------------------------------------------------------
# Representative bodies
# ---------------------------------------------------------------------------

# Filing: several distinct section markers + filing-scale length (> 500 tokens).
FILING_BODY = (
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n"
    "FORM 8-K\n\n"
    "Item 1.01 Entry into a Material Definitive Agreement.\n\n"
    + _paragraphs(4, 12)
    + "\n\nItem 9.01 Financial Statements and Exhibits.\n\n"
    + _paragraphs(4, 12)
    + "\n\nRisk Factors\n\n"
    + _paragraphs(2, 12)
    + "\n\nForward-Looking Statements\n\n"
    + _paragraphs(2, 12)
)

# Article: mid-length multi-paragraph prose, no filing markers.
ARTICLE_BODY = _paragraphs(3, 6)

# Tweet: tiny, single line, no markers.
TWEET_BODY = (
    "$AAPL ticks higher after hours on upbeat guidance; analysts watching margins "
    "into next quarter. #stocks #earnings"
)

# Degenerate: typed "filing" (e.g. via source advisory) but with zero section
# markers — filing-scale length so it is not mistaken for a tweet.
HEADERLESS_FILING_BODY = _paragraphs(8, 12)

# Degenerate: an "article" far too short to be meaningful prose (~30 tokens, < 50).
SHORT_ARTICLE_BODY = _sentences(2)

# Degenerate: a "tweet" that is unexpectedly long (≥ 280 tokens), single block.
OVERSIZED_TWEET_BODY = _sentences(40)

# Ambiguous: too short for an article, multi-paragraph so not a tweet, no markers —
# structure alone yields "unknown", leaving the advisory hint to decide.
AMBIGUOUS_BODY = _paragraphs(2, 3)


def make_document(**overrides) -> Document:
    """Build a Document with valid defaults; override any field via keyword.

    Defaults satisfy every required (ingest-time) field so tests only specify what
    they care about — typically `body`, `source_name`, or `tier`.
    """
    base = dict(
        id="doc-0001",
        content_hash="0" * 64,
        identity_key="test-source::0001",
        source_name="test-source",
        url="https://example.com/article/0001",
        tier=1,
        published_date=datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc),
        title="Test headline",
        body=ARTICLE_BODY,
        doc_type="article",
        raw_payload={},
        fetched_at=datetime(2026, 6, 30, 12, 0, 5, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return Document(**base)
