# Adapter registry — maps the string adapter key in sources.yaml to its concrete Adapter class.
#
# When the engine loads a SourceConfig entry, it reads config.adapter (e.g. "rss", "rest_json",
# "edgar") and looks it up here to get the class to instantiate. This indirection means the
# YAML never references Python import paths, and swapping an adapter implementation requires
# only changing this mapping.
#
# Registry contents (key -> class):
#   "rss"       -> RssAdapter       (adapters/rss.py)
#   "rest_json" -> RestJsonAdapter  (adapters/rest_json.py)
#   "edgar"     -> EdgarAdapter     (adapters/edgar.py)
#
# Responsibilities:
#   - Expose a get_adapter(key: str) -> type[Adapter] function that returns the class
#     for a given key, raising a clear ConfigError if the key is unknown.
#   - Keep the mapping exhaustive: every key that appears in sources.yaml must have an entry here.
#
# To register a new adapter:
#   1. Implement the Adapter ABC in a new file under adapters/.
#   2. Add a single line to the mapping dict here.
#   No other files need to change.
