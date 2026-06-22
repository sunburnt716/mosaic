# Generic REST-JSON adapter — handles any source that exposes articles via a JSON REST API.
#
# Like rss.py, this is format-driven: one implementation covers all JSON API sources.
# Per-source behavior (endpoint, auth headers, pagination style, response field names)
# is expressed entirely through SourceConfig params, not branching code.
#
# Responsibilities:
#   - Make HTTP GET (or POST) requests to config.url with config.headers and config.params.
#   - Traverse the JSON response to locate the array of article/item objects using a
#     configurable path key (e.g. params.items_path = "data.articles").
#   - Yield one raw dict per item, preserving the full original JSON object as raw_payload.
#   - Handle cursor-based and page-based pagination using configurable param names
#     (e.g. params.next_cursor_field, params.page_param) so no per-source pagination code exists.
#   - Respect rate limits via configurable backoff (params.rate_limit_delay).
#   - Raise FetchError on non-2xx responses, JSON parse failures, or unexpected response shapes.
#
# What this adapter does NOT do:
#   - Map response fields to Document fields — normalizer.py owns that translation.
#   - Decide which fields constitute the "body" — controlled via params.body_field in config.
#   - Filter items by date or keyword — that is a pipeline concern, not an adapter concern.
