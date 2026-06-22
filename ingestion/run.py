# Entrypoint and scheduler: starts the ingestion engine and drives periodic polling runs.
#
# This is the outermost shell of the ingestion subsystem — the file you point a cron job,
# a container CMD, or a developer at. It owns no business logic; it only wires config
# loading and scheduling to engine.run().
#
# Responsibilities:
#   - Parse CLI arguments: --config (path to sources.yaml), --once (run once and exit),
#     --source (run only the named source, for debugging), --log-level.
#   - Load and validate sources.yaml against the SourceConfig schema; fail fast with a
#     clear error if any entry is malformed, before any network calls are made.
#   - Instantiate the RawStore and SeenStore backends.
#   - If --once: call engine.run() once, print summary, exit.
#   - If scheduled mode: loop, calling engine.run() on the schedule defined per-source
#     in sources.yaml, sleeping between runs. Each source may have a different interval;
#     the scheduler must handle heterogeneous cadences without a fixed global tick.
#   - Handle SIGTERM/SIGINT gracefully: finish the in-progress source fetch if possible,
#     flush any pending writes, then exit cleanly.
#   - Emit structured logs (JSON lines) so runs can be monitored by external tooling.
#
# The run: command in CLAUDE.md will point here once the entrypoint is stable.
