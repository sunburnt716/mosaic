# Specialized adapter for the SEC EDGAR full-text search and filing retrieval API.
#
# EDGAR is the crown-jewel Tier-0 source: the primary regulatory record for all US public companies.
# It warrants its own adapter (rather than rest_json.py) because its API shape, filing structure,
# and document model are unique enough that a generic adapter would require excessive config complexity.
#
# Responsibilities:
#   - Query the EDGAR full-text search API (efts.sec.gov) or the submissions API to discover
#     recent filings by form type (10-K, 10-Q, 8-K, S-1, etc.).
#   - For each discovered filing, retrieve the filing index and locate the primary document
#     (the human-readable HTML or text exhibit, not just the wrapper SGML).
#   - Yield one raw dict per filing containing at minimum:
#       url (the SEC EDGAR viewer URL), cik, accession_number, form_type, filed_at (date),
#       company_name, raw_body (full text of the primary document), raw_payload (full JSON response).
#   - Respect EDGAR's stated rate limit (10 requests/second max; use the standard User-Agent header
#     required by the SEC: "Company Name contact@example.com").
#   - Handle SGML/HTML stripping to yield clean plain text suitable for chunking downstream.
#
# doc_type for all EDGAR output: "filing" — this signals the chunker to split by section
# (Item 1, Item 1A, MD&A, etc.) rather than by paragraph.
#
# What this adapter does NOT do:
#   - Parse individual filing sections — the downstream chunker owns section-level splitting.
#   - Decide materiality or relevance — that is a retrieval/re-ranking concern.
#   - Cover non-US regulators (FSA, ESMA) — those would be separate adapters if needed.
