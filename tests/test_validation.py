"""Tests for ingestion/pipeline/validation.py — the fail-closed structural layers.

All offline. check_transport / check_feed_well_formed are pure functions over bytes /
a parsed-feed object, so they're tested directly. The challenge-page body is loaded from
tests/fixtures/challenge_page.html (a Cloudflare-style HTML interstitial).
"""

from pathlib import Path

import feedparser
import pytest

from ingestion.adapters.base import TransportError
from ingestion.pipeline.validation import check_feed_well_formed, check_transport

FIXTURES = Path(__file__).parent / "fixtures"
_CHALLENGE_HTML = (FIXTURES / "challenge_page.html").read_bytes()

_VALID_ATOM = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><title>T</title><link href="https://x/1"/><id>1</id>
  <updated>2026-01-01T00:00:00Z</updated></entry>
</feed>"""

_VALID_RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>C</title>
<item><title>T</title><link>https://x/1</link></item></channel></rss>"""


# ---------------------------------------------------------------------------
# check_transport — XML expected
# ---------------------------------------------------------------------------


class TestCheckTransportXml:
    def test_html_challenge_page_rejected(self):
        with pytest.raises(TransportError, match="HTML"):
            check_transport(_CHALLENGE_HTML, "text/html; charset=UTF-8", "xml")

    def test_valid_atom_passes(self):
        check_transport(_VALID_ATOM, "application/atom+xml", "xml")  # no raise

    def test_valid_rss_passes(self):
        check_transport(_VALID_RSS, "application/rss+xml", "xml")  # no raise

    def test_feed_served_as_text_html_still_passes(self):
        """A real feed mislabeled text/html must pass — body-sniffing is primary."""
        check_transport(_VALID_RSS, "text/html", "xml")  # starts with <?xml -> ok

    def test_empty_body_rejected(self):
        with pytest.raises(TransportError, match="empty"):
            check_transport(b"", "application/atom+xml", "xml")

    def test_whitespace_only_body_rejected(self):
        with pytest.raises(TransportError, match="empty"):
            check_transport(b"   \n\t  ", "application/atom+xml", "xml")

    def test_bom_prefixed_feed_passes(self):
        with_bom = b"\xef\xbb\xbf" + _VALID_ATOM
        check_transport(with_bom, "application/xml", "xml")  # no raise


# ---------------------------------------------------------------------------
# check_transport — JSON expected
# ---------------------------------------------------------------------------


class TestCheckTransportJson:
    def test_html_challenge_page_rejected(self):
        with pytest.raises(TransportError):
            check_transport(_CHALLENGE_HTML, "text/html", "json")

    def test_valid_json_object_passes(self):
        check_transport(b'{"items": []}', "application/json", "json")

    def test_valid_json_array_passes(self):
        check_transport(b"[{}]", "application/json", "json")

    def test_markup_where_json_expected_rejected(self):
        with pytest.raises(TransportError):
            check_transport(b"<?xml version='1.0'?><rss/>", "application/xml", "json")

    def test_empty_body_rejected(self):
        with pytest.raises(TransportError, match="empty"):
            check_transport(b"", "application/json", "json")


# ---------------------------------------------------------------------------
# check_feed_well_formed
# ---------------------------------------------------------------------------


class TestCheckFeedWellFormed:
    def test_valid_feed_passes(self):
        check_feed_well_formed(feedparser.parse(_VALID_ATOM))  # no raise

    def test_malformed_xml_with_no_entries_rejected(self):
        parsed = feedparser.parse(b"<feed><entry><title>unclosed")
        # feedparser flags bozo; if it also yielded no entries this is a real parse failure.
        if parsed.entries:
            pytest.skip("feedparser salvaged entries from this input on this version")
        with pytest.raises(TransportError):
            check_feed_well_formed(parsed)

    def test_wellformed_but_empty_feed_passes(self):
        """An empty-but-valid feed is NOT refused here (left for the soft quality gate)."""
        empty = feedparser.parse(
            b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        )
        check_feed_well_formed(empty)  # no raise
