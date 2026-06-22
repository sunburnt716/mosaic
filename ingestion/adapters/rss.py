# Generic RSS/Atom adapter — handles any source whose feed URL delivers an RSS or Atom feed.
#
# This adapter is format-driven: one implementation covers all RSS/Atom sources regardless
# of outlet (Reuters, AP, WSJ, FT, etc.). Per-source differences (URL, auth headers, schedule)
# live entirely in sources.yaml, not here.
#
# Responsibilities:
#   - Fetch the feed at config.url using config.headers (auth, user-agent).
#   - Parse the XML with a standard feed library (e.g. feedparser); handle both RSS 2.0 and Atom.
#   - Yield one raw dict per <item>/<entry> containing at minimum:
#       url, title, raw_body (summary or content), published (raw timestamp string), raw_payload.
#   - Handle pagination if the feed supports it (e.g. link rel="next") via config.params.
#   - Raise FetchError (from adapters/base.py) on HTTP errors, malformed XML, or empty feeds
#     rather than silently swallowing failures.
#
# What this adapter does NOT do:
#   - Parse or validate timestamps — normalizer.py owns timestamp coercion.
#   - Strip HTML or clean text — normalizer.py owns content cleaning.
#   - Decide the tier — that comes from SourceConfig and is stamped in normalizer.py.
