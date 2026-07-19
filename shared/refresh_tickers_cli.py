"""CLI entry point: refresh the local ticker cache once and exit.

Run daily via OS cron / Windows Task Scheduler / cloud scheduler — SEC republishes
company_tickers.json nightly at ~3am ET, so a daily cadence matches the source. Mirrors
ingestion/run.py's --once mode at a fraction of the size: no scheduler loop, since this
runs once per cron invocation rather than as a long-lived process.
"""

from __future__ import annotations

import os
import sys

from shared.tickers import refresh_tickers


def main() -> int:
    email = os.environ.get("SEC_USER_AGENT_EMAIL")
    if not email:
        print("SEC_USER_AGENT_EMAIL is not set.", file=sys.stderr)
        return 1
    refresh_tickers(email)
    return 0


if __name__ == "__main__":
    sys.exit(main())
