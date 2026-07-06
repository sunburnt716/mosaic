# generation/ (scaffold — not yet implemented)

The final pipeline stage:

```
ingestion → extraction → [generation] → interfaces
```

## Responsibility (future)
- Retrieval over Chroma: semantic search + re-rank (blend semantic score with recency,
  source credibility/tier, user profile) + cross-outlet clustering (preserve L3 corroboration).
- Synthesis (Gemini Flash) that **cites source + timestamp on every claim**.

## Mission guardrail (binding)
**Inform, not advise.** No buy/sell/hold calls, ever. Every surfaced claim carries
`source_name`, `url`, `tier`, and `published_date` so the user can verify it.

## Non-goals for now
Nothing here is built yet. Depends on the extraction stage existing first.
