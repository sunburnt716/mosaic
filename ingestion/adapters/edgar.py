"""SEC EDGAR adapter — RETIRED from the discovery stage.

EDGAR discovery now runs through the generic RSS adapter pointed at the
getcurrent Atom feed (config/sources.json: adapter: rss, transform: edgar_filing_url).
The efts full-text search endpoint this file previously used had field-name
mismatches and returned no bodies; see fixtures/README.md for the analysis.

TODO(processing-stage): filing body/structure parsing (10-K sections, 8-K items)
  belongs here once the processing stage is built. The adapter should accept a
  filing index URL, fetch the primary document, and return section-chunked content.
  Until then this file is a placeholder so the intent is not lost.
"""
