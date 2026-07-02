# Test Fixtures

Raw payloads captured from live sources. Each file documents the actual fields
observed in one real response — all field mappings are written against these,
not against assumed names.

## sec-edgar.xml

**Source:** `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom`
**Captured:** 2026-06-27 | **Format:** Atom feed (feedparser)

Feedparser key → value observed for one entry:

| feedparser key | value / pattern |
|---|---|
| `link` | `https://www.sec.gov/Archives/edgar/data/{CIK}/{ACCESSION_NODASHES}/{ACCESSION}-index.htm` |
| `title` | `"{form_type} - {entity_name} ({CIK}) (Filer)"` |
| `summary` | HTML: `<b>Filed:</b> {date} <b>AccNo:</b> {accession} <b>Size:</b> {N} KB<br/>Item N.NN: ...` |
| `updated` | RFC-3339 e.g. `"2026-06-26T17:30:25-04:00"` |
| `published` | `None` (not present in this feed) |
| `id` | `"urn:tag:sec.gov,2008:accession-number={ACCESSION}"` |
| `tags` | `[{'term': '8-K', 'scheme': 'https://www.sec.gov/', 'label': 'form type'}]` |

Key observations:
- The `link` is already a canonical archive index URL — no URL construction needed.
- `published` is absent; use `updated` (RSS adapter's fallback already handles this).
- Form type is available in both `tags[0].term` and the title prefix.
- `summary` contains real content (non-empty body), unlike the efts search endpoint.
- The `edgar_filing_url` transform cleans the title to `"{form_type} — {entity_name}"`.

## ft-rss.xml

**Source:** `https://www.ft.com/rss/home`
**Captured:** 2026-06-27 | **Format:** RSS 2.0 (feedparser)

Feedparser key → value observed for one entry:

| feedparser key | value / pattern |
|---|---|
| `link` | `https://www.ft.com/content/{uuid}` |
| `title` | plain article headline |
| `summary` | short teaser (1-2 sentences, plain text) |
| `published` | RFC-2822 e.g. `"Sat, 27 Jun 2026 02:17:50 GMT"` |
| `id` | UUID string (bare, no `urn:` prefix) |

Key observations:
- No transform needed — all standard fields present and well-formed.
- `summary` is a brief teaser, not the full article body (expected for RSS).
- `id` is a bare UUID; `source_article_id` will resolve to this via RSS adapter.

## rss_reuters_sample.json / rest_json_sample.json

Pre-existing fixtures for the generic normalizer tests. Format: the dict shape
that a parsed adapter entry yields (not raw HTTP response XML/JSON).
