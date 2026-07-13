"""Query-time read path — orchestration that ties retrieval to generation for one query.

Sibling of ingestion/, extraction/, retrieval/, generation/. `engine.answer()` is the
composition root the interfaces layer will call; `run.py` is the operator CLI harness.
"""
