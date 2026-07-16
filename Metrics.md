# Mosaic — Metrics Log

**Purpose:** a running record of measurable outcomes as each phase is built, so
they can be turned into resume bullets, an ADR trail, and a technical blog post
later. Claude Code should append real numbers here as it implements each phase —
not aspirations, measured results.

> Convention: every entry gets a date, the phase, the metric, the measured value,
> and how it was measured. Prefer honest ranges over invented precision. If a
> metric wasn't measured, write `not measured` rather than guessing.

---

## How to log (for Claude Code)

When you complete or benchmark a phase, append a row under the relevant section:

```
### <date> — <engine>/<phase>
- metric: <what was measured>
- value: <measured result>
- method: <how it was measured; test name / dataset / n>
- notes: <caveats, surprises, follow-ups>
```

Keep raw benchmark output out of this file — link to the test or commit instead.

---

## Retrieval Pipeline

### Query Routing (Phase 1)
- routing latency (p50 / p95): _not measured_
- ticker/sector extraction accuracy: _not measured_ (define eval set of N labeled queries)
- intent classification accuracy: _not measured_

### Vector Search (Phase 2)
- search latency with metadata filter (p50 / p95): _not measured_
- candidate set size (avg n_results actually useful): _not measured_
- filter selectivity (corpus → filtered subset ratio): _not measured_

### Re-rank (Phase 3)
- rank stability vs. raw ANN order (Kendall tau or % reordered): _not measured_
- recency-decay sanity (does newest-relevant surface top?): _not measured_

### Cluster (Phase 4)
- clustering precision/recall on labeled duplicate stories: _not measured_
- avg outlets per corroborated cluster: _not measured_

---

## Generation Pipeline

### Synthesis (Phase 2)
- Gemini call latency (p50 / p95): _not measured_
- avg prompt tokens / completion tokens: _not measured_

### Citation Validation (Phase 4) — the headline quality metric
- **claim grounding rate** (% claims grounded via direct lookup): _not measured_
- semantic-fallback rate (% needing fuzzy grounding): _not measured_
- **hallucination catch rate** (% ungrounded claims correctly dropped): _not measured_
- answer rejection rate (% queries hitting the >30% drop warning): _not measured_

### Formatting / Citations (Phase 5)
- deep-link resolution rate (fragment actually highlights in browser): _not measured_
- citation fidelity (% claims with sentence-level vs. doc-level link): _not measured_

---

## Cost

- avg cost per query (Groq route + Gemini synthesis + optional validation): _not measured_
- cost breakdown by stage: _not measured_
- $ / 1k queries at current settings: _not measured_

---

## End-to-end

- full-pipeline latency, query → GeneratedAnswer (p50 / p95): _not measured_
- % queries answered vs. honestly declined ("insufficient sources"): _not measured_

---

## Answerability eval (`evals/`)

The measured basis for the broaden-sources decision. Run: `python -m evals.run --json ...`
(needs GEMINI_API_KEY + google-genai for the headline rates; retrieval-only otherwise).
Labeled set: `evals/questions.yaml` (~30 questions). Log one dated block per run so deltas
after adding feeds are visible.

- eval set size / composition: 31 questions — news-synthesis, point-in-time-statistic,
  ticker-specific, out-of-scope (answer / abstain / redirect)
- **answerable-in-scope rate** (of in-scope questions, % with a citable answer): _not run yet_
- **out-of-scope-abstention rate** (of out-of-scope questions, % correctly declined): _not run yet_
- bucket counts (working / in-scope-but-thin / out-of-scope-router-missed): _not run yet_
- avg top1 similarity, in-scope (max-pooled, not mean): _not run yet_

```
### <date> — evals/answerability (baseline, N sources)
- answerable-in-scope: __%   out-of-scope-abstention: __%
- buckets: working __ / in-scope-but-thin __ / out-of-scope-router-missed __
- avg top1 (in-scope): ____   method: python -m evals.run, collection=<...>, synthesis=Gemini
- notes: <which intents fell into in-scope-but-thin => which feeds to add next>
```

### 2026-07-14 — citation path fix (0% → 38% answerable-in-scope)
Two stacked bugs made **every** query return zero citations (`baseline.json`/`run-01.json`:
0/21 cited, `citation_path_suspect: true`):
1. **Opaque chunk IDs in the prompt.** Gemini was asked to echo the 64-hex `chunk_id` verbatim
   as `SOURCE_CHUNK_ID`; it can't, so grounding missed 100%. Fixed by showing short handles
   (S1, S2, …) and translating back to the real chunk_id before validation.
2. **Dead Gemini model id.** `gemini-2.0-flash` returns `429 RESOURCE_EXHAUSTED, limit: 0`
   (no free-tier quota) → the fail-closed synthesizer swallowed it into the INSUFFICIENT_DATA
   marker on every call, so synthesis never once succeeded. Fixed by `gemini-flash-latest`.

- **answerable-in-scope: 0% → 38%** (0/21 → 8/21 cited), `run-03.json`. out-of-scope
  meaningful-abstention: 100% (2/2). avg top1 (in-scope): 0.442. buckets: cited 8 /
  strong-uncited 13 / thin 10.
- Method: `python -m evals.run`, collection=`data/chroma` (`mosaic_minilm-l6-v2`, 529 chunks),
  router=Groq, synthesis=`gemini-flash-latest`.
- **Residual strong-uncited is NOT a code bug** — proven by trace: tk-05 (cited in run-03)
  and pt-02 both re-trace to the *same* INSUFFICIENT_DATA marker; the same question flips
  outcome across runs. The variance is Gemini **free-tier availability** (transient 503
  "high demand"; hard 429 once the per-key quota is spent by repeated eval bursts), not
  retrieval strength or ID mechanics. A paid key / spaced runs would raise the ceiling.
- Separately capped by **thin content**: FT-RSS stores headlines, SEC-EDGAR stores index
  snippets (bodies mean ~72 chars) — the coverage follow-up, tracked in CLAUDE.md.

### 2026-07-14 — eval instrument: `synth-failed` bucket (stop mislabeling API failures)
Follow-up to the row above. The eval could not tell a **failed Gemini call** (429/503 →
fail-closed marker → 0 citations) apart from a **broken citation path** — both landed in
`strong-uncited`, so a flaky free-tier key read as a code bug and sent debugging chasing a
phantom chunk-ID mismatch.
- **Proved the plumbing is intact, no Gemini:** `tools/check_citation_plumbing.py` — for
  pt-02 + tk-05, all 20 offered handles per question resolve to a `chunk_id` present in BOTH
  the validator's `chunks_by_id` AND Chroma, byte-identical. Zero mismatches. So the ID chain
  is not the residual failure.
- **Fix:** `QueryResult.synthesis_failed` (raw text == INSUFFICIENT_DATA_MARKER) → new
  `synth-failed` eval bucket, peeled off *before* strong/thin and **excluded from
  `citation_path_suspect`**. `strong-uncited` now means only "synthesis succeeded yet nothing
  grounded" — a genuine citation bug. Re-run reclassifies the ~13 prior `strong-uncited` rows
  as `synth-failed`; `citation_path_suspect` clears. (Live re-run pending free-tier quota
  reset — the API returns 429 on every call as of this writing.)

### 2026-07-15 — content coverage: EDGAR filing-body enrichment
The thin-content lever from the row above. SEC-EDGAR ingested only the getcurrent Atom
snippet as `body`; `pipeline/body_enrichment.py` now fetches the real filing at ingest
(index page → primary document → `clean_html`), gated by `SourceConfig.body_fetch:
edgar_filing`, before `normalize()` so content_hash/chunk_id reflect it.
- **EDGAR body length: ~72 chars → 34,908 chars** on a live 8-K (index
  `…/0001628280-26-048308-index.htm` → primary `a2q26earningsslides_vf.htm`). Method: live
  two-hop fetch via the real strategy + `default_fetch_url` (SEC UA, throttled). ~500× more
  groundable text per filing.
- Best-effort (fetch failure → keeps snippet, never drops); offline-testable via injected
  fetcher (13 new tests, all green). Existing 529 thin chunks are replaced on re-ingest +
  re-extract via the existing identity_key chunk eviction — no manual purge.
- **Pending (operator, needs network + quota):** full `ingestion.run` re-ingest →
  `inspect_pipeline` body_len → `extraction.run` re-extract → `evals.run` to measure the
  answerable-in-scope delta on the `pt-*` filing questions.

---

## Resume-bullet candidates (draft from measured rows above)

Fill these in only once backed by a measured value in this file. Each should name
a system property and a number.

- [ ] "Built a citation-validation layer that grounds every generated claim to a
      retrieved source, dropping ungrounded claims and catching ___% of
      hallucinations on an adversarial test set."
- [ ] "Designed a metadata-filtered retrieval pipeline (filter-then-ANN) cutting
      search scope by ___x while keeping p95 latency under ___ms."
- [ ] "Kept per-query cost at ~$___ via tiered model routing (Llama 3.1 8B on
      Groq for classification, Gemini Flash for synthesis)."
- [ ] "Implemented cross-outlet corroboration clustering surfacing story
      confidence across ___ average sources."
- [ ] "Built a labeled answerability eval that turned source-coverage decisions into
      measured deltas — added ___ feeds and moved answerable-in-scope from ___% to ___%
      while out-of-scope abstention held at ___%."
