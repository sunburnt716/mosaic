# generation/

The final pipeline stage — synthesis + citation over the ranked, clustered chunks retrieval
produces:

```
ingestion → extraction → retrieval → [generation] → interfaces
```

Consumes `RetrievalOutput` (from `retrieval/`, a separate top-level package — semantic search,
re-rank, and cross-outlet clustering live there, not here) and produces a `GeneratedAnswer`
for the user-facing surface.

## Built: all five phases of the Generation Pipeline spec

```
prompt_builder.py  — PromptBuilder    (RetrievalOutput + query + lens -> Gemini prompt)
synthesizer.py      — Synthesizer      (Gemini Flash call, retry + fail-closed)
claim_parser.py      — ClaimParser      (structured text -> ParsedClaim)
validator.py          — CitationValidator (grounding gate: direct lookup + semantic fallback)
formatter.py           — AnswerFormatter  (deterministic citations, deep links, rejection policy)
contracts.py            — LensDoc, ParsedClaim, ValidatedClaim, Citation, GeneratedAnswer
```

See `.claude/CLAUDE.md`'s "Generation engine" section for the locked decisions as implemented,
including where the spec's literal phase boundaries needed a documented, necessary deviation
(e.g. the "reject, don't repair" rejection policy executes in `formatter.py`, not
`validator.py` — see that module's docstring for why).

## Mission guardrail (binding)
**Inform, not advise.** No buy/sell/hold calls, ever. Every claim that survives citation
validation carries its source, tier, and a deep link back to the exact sentence that grounds
it — an ungrounded claim is dropped, never backfilled from the model's parametric memory.

## Non-goals
- No fine-tuning — base Gemini Flash + prompt engineering only.
- No regenerate-on-failure loop; no self-graded grounding.
- No model-emitted quotes or spans anywhere — citation sentence selection is a deterministic
  word-overlap heuristic, never a model call.
