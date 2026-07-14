"""
Query-time orchestration — the read-path composition root.

`answer()` is to the read path what `ingestion/run.py`'s `main()` is to the write path:
the one place the query-time stages are wired together in order. It ties retrieval
(router -> search -> rerank -> cluster -> output) to generation (prompt -> synthesis ->
parse -> validate -> format) into a single call, so callers — the CLI harness today,
the interfaces layer later — never re-wire the nine stages by hand.

Everything it depends on is injected, never constructed here: the Chroma `collection`, the
`router` (QueryRouter-shaped), and the `synthesizer` (Synthesizer-shaped). That keeps this
module pure orchestration — no network, no model loads of its own — and fully testable
against fakes, the same discipline the individual phases already follow.

Graceful degradation is deliberate: `synthesizer=None` runs retrieval only and returns a
`QueryResult` whose `.answer` is None. This lets an operator exercise the whole retrieval
half without a Gemini key, and keeps all orchestration in one place rather than duplicating
the retrieval steps in a CLI's no-Gemini branch.

  route_offline(query, profile)          -> RoutingResult   (no-LLM fallback router)
  OfflineRouter                          -> QueryRouter-shaped wrapper of the above
  answer(query, profile, *, collection, router, synthesizer=None, ...) -> QueryResult
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from extraction.utils.embedding import embed_text
from generation.claim_parser import ClaimParser
from generation.contracts import GeneratedAnswer, LensDoc, ValidatedClaim
from generation.formatter import AnswerFormatter
from generation.prompt_builder import PromptBuilder
from generation.validator import CitationValidator
from retrieval.cluster import StoryClusterer
from retrieval.contracts import RoutingResult, UserProfile
from retrieval.output import RetrievalOutput, assemble_retrieval_output
from retrieval.rerank import Ranker
from retrieval.search import DEFAULT_N_RESULTS, VectorSearch

# Mirrors retrieval.router.DEFAULT_TIME_WINDOW_DAYS; duplicated here so the offline router
# doesn't import the module whose whole point is the (optional) Groq dependency.
DEFAULT_TIME_WINDOW_DAYS = 30


@dataclass
class QueryResult:
    """The read path's full output for one query.

    `answer` is the user-facing `GeneratedAnswer`; it is None only in retrieval-only mode
    (`synthesizer=None`), where `routing` and `retrieval` are still populated so the caller
    can show what was retrieved even when synthesis was skipped.

    `validated_claims` is the raw grounding-gate output (before the formatter drops the
    ungrounded ones) — an observability hook for the eval harness and a future UI debug
    view, so callers can see grounded-vs-total without re-running generation. Also None in
    retrieval-only mode.

    `filter_fallback` records whether Phase 2 search had to drop its metadata `where` clause
    because the filtered set was empty (see retrieval/search.py) — so a caller can tell an
    `n` that climbed via the unfiltered fallback apart from a natively-filtered one.

    `prompt` and `raw_synthesis` are the assembled Gemini prompt and Gemini's verbatim
    response — the trace instrument's raw material for diagnosing the citation path (marker
    vs prose vs mangled IDs). Populated only when `answer(..., trace=True)`; None otherwise,
    since they can be large and normal callers don't need them.
    """

    routing: RoutingResult
    retrieval: RetrievalOutput
    answer: Optional[GeneratedAnswer]
    validated_claims: Optional[list[ValidatedClaim]] = None
    filter_fallback: bool = False
    prompt: Optional[str] = None
    raw_synthesis: Optional[str] = None


def route_offline(
    query: str,
    profile: UserProfile,
    *,
    embedder: Callable[[str], list[float]] = embed_text,
    time_window_days: int = DEFAULT_TIME_WINDOW_DAYS,
) -> RoutingResult:
    """Build a RoutingResult without an LLM: embed the query, take tickers/sectors from profile.

    The no-Groq fallback. It cannot infer intent or extract tickers from the query text —
    only the shared query embedding (which drives semantic search) and the profile's declared
    interests (which drive the metadata filter) are available. Intent is left "unknown".
    """
    return RoutingResult(
        intent="unknown",
        tickers=list(profile.tickers),
        sectors=list(profile.sectors),
        time_window_days=time_window_days,
        query_embedding=embedder(query),
    )


class OfflineRouter:
    """QueryRouter-shaped wrapper around `route_offline`, so `answer()` treats both alike."""

    def __init__(
        self,
        embedder: Callable[[str], list[float]] = embed_text,
        time_window_days: int = DEFAULT_TIME_WINDOW_DAYS,
    ) -> None:
        self._embedder = embedder
        self._time_window_days = time_window_days

    def route(self, query: str, profile: UserProfile) -> RoutingResult:
        return route_offline(
            query,
            profile,
            embedder=self._embedder,
            time_window_days=self._time_window_days,
        )


def answer(
    query: str,
    profile: UserProfile,
    *,
    collection: Any,
    router: Any,
    synthesizer: Any = None,
    lens: Optional[list[LensDoc]] = None,
    now: Optional[datetime] = None,
    n_results: int = DEFAULT_N_RESULTS,
    trace: bool = False,
) -> QueryResult:
    """Run the full read path for `query`, returning routing, retrieval, and the answer.

    `router` must expose `.route(query, profile) -> RoutingResult` (real `QueryRouter` or
    `OfflineRouter`). `synthesizer`, if given, must expose `.synthesize(prompt) -> str`
    (real `Synthesizer` or a fake); when None, synthesis is skipped and `QueryResult.answer`
    is None. `now` is injectable so recency scoring and tests are deterministic. `trace`
    populates `QueryResult.prompt`/`.raw_synthesis` for the diagnostic instrument (in
    retrieval-only mode it still builds the prompt so the offered sources can be inspected).
    """
    now = now or datetime.now(tz=timezone.utc)

    # --- Retrieval half ---
    routing = router.route(query, profile)
    search = VectorSearch(collection)
    retrieved = search.search(routing, n_results=n_results)
    ranked = Ranker().rank(retrieved, routing, now)
    clusters = StoryClusterer().cluster(ranked)
    retrieval_output = assemble_retrieval_output(clusters)

    if synthesizer is None:
        # Build the prompt anyway under trace, so the instrument can show what *would* be
        # sent (and the offered CHUNK_IDs) even without a Gemini key.
        prompt = (
            PromptBuilder().build(retrieval_output, query, lens or [], profile) if trace else None
        )
        return QueryResult(
            routing=routing,
            retrieval=retrieval_output,
            answer=None,
            filter_fallback=search.last_filter_fallback,
            prompt=prompt,
        )

    # --- Generation half ---
    # The candidate set for grounding is every ranked chunk, keyed by id — a claim may cite
    # any retrieved chunk, not only the cluster primaries.
    chunks_by_id = {chunk.chunk_id: chunk for chunk in ranked}

    prompt = PromptBuilder().build(retrieval_output, query, lens or [], profile)
    raw_text = synthesizer.synthesize(prompt)
    claims = ClaimParser().parse(raw_text)
    validated = CitationValidator().validate(claims, chunks_by_id)
    generated = AnswerFormatter().format(validated, chunks_by_id, clusters)

    return QueryResult(
        routing=routing,
        retrieval=retrieval_output,
        answer=generated,
        validated_claims=validated,
        filter_fallback=search.last_filter_fallback,
        prompt=prompt if trace else None,
        raw_synthesis=raw_text if trace else None,
    )
