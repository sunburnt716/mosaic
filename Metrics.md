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
