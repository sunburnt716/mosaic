"""
Answerability eval harness — turns "should I broaden sources?" into a measured decision.

Runs each labeled question (evals/questions.yaml) through the read path and records, per
question:
  - top1_similarity / top3_similarity — the MAX chunk similarity at rank 1 and rank 3, from
    raw cosine similarity, NOT the within-query mean. (The mean is the retrieval_confidence
    gate that under-reported strong-but-narrow matches; max-pooling is the fix.)
  - synthesis_citable — did generation produce an answer with at least one citation?
  - validator_passed — did the reject-don't-repair grounding gate let at least one claim
    through? (Distinct from citable: claims can ground yet the formatter still surface none;
    in practice they coincide, but both are logged so a divergence is visible.)

Then each question sorts into a **behavior-first** bucket — keyed on `(n, top1, cited)`, NOT
on the question's own label (an earlier version keyed on `expected`, which made the buckets a
frozen re-print of the label distribution whenever citation was uniformly broken). The bucket
names what the *system did*, so a broken citation path can't hide as thinness:
  - no-candidates    n == 0. Nothing survived retrieval — a filter/retrieval bug (e.g. a
                     ticker filter that matched nothing), NEVER "thin". Distinct on purpose.
  - cited            produced an answer with >=1 citation.
  - synth-failed     synthesis ran but the Gemini *call itself* failed (fail-closed marker:
                     429 quota / 503 / etc.). NOT a citation bug and NOT thinness — an
                     infrastructure failure that produces the same zero-citation outcome, so
                     it must be named separately or it hides as `strong-uncited` and slanders
                     the citation path (exactly the misread this bucket exists to prevent).
  - strong-uncited   top1 >= floor, synthesis SUCCEEDED, but no citation — a STRONG match the
                     citation path failed to ground. A generation bug, not coverage. The
                     smoking gun (now genuinely so, with synth-failed peeled off).
  - thin             n > 0, top1 < floor, no citation — genuinely weak signal. ADD FEEDS.
  - retrieval-only   synthesis was skipped (no Gemini); citation is unobservable.

A separate `verdict` cross-references the bucket against `expected` (correct-answer /
correct-decline / missed-answer / over-answered), and the two headline rates fall out:
answerable-in-scope (of in-scope questions, how many were `cited`) and out-of-scope
abstention. Abstention is counted as **meaningful only where retrieval surfaced strong
content** (top1 >= floor) — a decline on `n == 0` is vacuous. If any `strong-uncited` rows
exist, `summarize` flags `citation_path_suspect`: the rates are not trustworthy until the
citation path is fixed. Those rates and their before/after deltas are the résumé line — log
them to Metrics.md (see CLAUDE.md "Metrics").

Synthesis needs a Gemini client; in retrieval-only mode (`synthesizer=None`) the synthesis
fields are None and questions land in the `retrieval-only` bucket (or `no-candidates` when
`n == 0`, which is observable without Gemini) — enough to catch filter starvation and eyeball
the corpus, not enough for the citation-dependent buckets or the headline rates.

  load_questions(path)                 -> list[Question]
  evaluate(questions, *, collection, router, synthesizer=None, ...) -> list[QuestionResult]
  summarize(results)                   -> EvalSummary
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from query.engine import answer
from retrieval.contracts import UserProfile

_DEFAULT_QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"

# Retrieval-only proxy threshold: top1 cosine at/above this counts as "signal present".
# STARTING GUESS, not tuned — the eval's own top1 distribution is what should set it. Only
# used when synthesis is skipped; with synthesis, the citable-answer outcome decides buckets.
DEFAULT_SIMILARITY_FLOOR = 0.30

# Behavior-first buckets — keyed on (n, top1, cited, synth_failed), never on the question's label.
BUCKET_NO_CANDIDATES = "no-candidates"
BUCKET_CITED = "cited"
BUCKET_SYNTH_FAILED = "synth-failed"
BUCKET_STRONG_UNCITED = "strong-uncited"
BUCKET_THIN = "thin"
BUCKET_RETRIEVAL_ONLY = "retrieval-only"

# Verdicts — the bucket cross-referenced against the question's `expected` label.
VERDICT_CORRECT_ANSWER = "correct-answer"
VERDICT_CORRECT_DECLINE = "correct-decline"
VERDICT_MISSED_ANSWER = "missed-answer"
VERDICT_OVER_ANSWERED = "over-answered"
VERDICT_UNKNOWN = "unknown-no-synthesis"

_OUT_OF_SCOPE_BEHAVIORS = frozenset({"abstain", "redirect"})


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    intent: str
    expected: str  # "answer" | "abstain" | "redirect"
    note: Optional[str] = None


@dataclass
class QuestionResult:
    id: str
    question: str
    intent: str
    expected: str
    # --- retrieval metrics (always populated) ---
    n_retrieved: int
    top1_similarity: Optional[float]
    top3_similarity: Optional[float]
    has_signal: bool
    # --- synthesis metrics (None in retrieval-only mode) ---
    synthesis_ran: bool
    synthesis_citable: Optional[bool]
    # True when the Gemini call failed and the synthesizer returned its fail-closed marker
    # (429/503/etc.) — distinct from "answered but nothing grounded".
    synthesis_failed: bool
    claims_total: Optional[int]
    claims_grounded: Optional[int]
    validator_passed: Optional[bool]
    confidence_warning: Optional[str]
    # whether Phase 2 search dropped its where-clause (empty filtered set) to get this `n`
    filter_fallback: bool
    # --- verdict ---
    bucket: str
    verdict: str


@dataclass
class EvalSummary:
    total: int
    synthesis_ran: bool
    bucket_counts: dict[str, int]
    verdict_counts: dict[str, int]
    in_scope_total: int
    in_scope_cited: int
    answerable_in_scope_rate: Optional[float]
    out_of_scope_total: int
    # out-of-scope questions where retrieval surfaced strong content (the temptation to
    # answer existed) — the only ones whose abstention is a meaningful measurement
    out_of_scope_tempted: int
    out_of_scope_tempted_declined: int
    meaningful_abstention_rate: Optional[float]
    # the smoking gun: strong retrieval, no citation. Counted overall and, more tellingly,
    # among IN-SCOPE questions (where a citation was expected — an out-of-scope strong-uncited
    # is a correct refusal, not a bug). A non-trivial `strong_uncited_in_scope` => the
    # citation path is likely broken and the headline rates above are not trustworthy.
    strong_uncited_count: int
    strong_uncited_in_scope: int
    # synthesis-infrastructure failures (Gemini call returned the fail-closed marker) — an API
    # health signal, deliberately NOT folded into strong-uncited or citation_path_suspect.
    synth_failed_count: int
    no_candidates_count: int
    filter_fallback_count: int
    citation_path_suspect: bool
    # average of the per-question MAX signals across in-scope questions (an aggregate of
    # maxes, not the within-query mean the gate fix warns against)
    avg_top1_in_scope: Optional[float]


def load_questions(path: Path | None = None) -> list[Question]:
    """Parse the labeled eval set. Missing/blank `questions` yields an empty list."""
    path = path or _DEFAULT_QUESTIONS_PATH
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return [
        Question(
            id=q["id"],
            question=q["question"],
            intent=q["intent"],
            expected=q["expected"],
            note=q.get("note"),
        )
        for q in (data.get("questions") or [])
    ]


def _similarity_at_ranks(result: Any) -> tuple[int, Optional[float], Optional[float]]:
    """Return (n_retrieved, top1_similarity, top3_similarity) from a QueryResult.

    Similarities come from every retrieved chunk's raw `similarity_score`, sorted high to
    low — the max at rank 1, and the value at rank 3 (a "is there corroborating depth?"
    read). Both are max-pooled per rank, never averaged.
    """
    chunks = [chunk for cluster in result.retrieval.clusters for chunk in cluster.chunks]
    sims = sorted((c.similarity_score for c in chunks), reverse=True)
    n = len(sims)
    top1 = sims[0] if n >= 1 else None
    top3 = sims[2] if n >= 3 else None
    return n, top1, top3


def _bucket(
    n_retrieved: int,
    has_signal: bool,
    synthesis_ran: bool,
    synthesis_citable: Optional[bool],
    synthesis_failed: bool,
) -> str:
    """Sort one question into a behavior-first bucket — from what the system DID, not its label.

    `no-candidates` is checked first and independent of synthesis: zero survivors is a
    retrieval/filter bug regardless of what generation would have done with them.
    """
    if n_retrieved == 0:
        return BUCKET_NO_CANDIDATES
    if not synthesis_ran:
        return BUCKET_RETRIEVAL_ONLY
    if synthesis_citable:
        return BUCKET_CITED
    # Not cited. A failed Gemini *call* (fail-closed marker) is infrastructure, not a citation
    # bug — peel it off before the strong/thin split so it can't masquerade as strong-uncited.
    if synthesis_failed:
        return BUCKET_SYNTH_FAILED
    # Synthesis genuinely answered but nothing cited: strong retrieval that didn't ground is a
    # citation bug; weak retrieval is genuine thinness.
    return BUCKET_STRONG_UNCITED if has_signal else BUCKET_THIN


def _verdict(expected: str, bucket: str) -> str:
    """Cross-reference the behavior bucket against what the label wanted."""
    if bucket == BUCKET_RETRIEVAL_ONLY:
        return VERDICT_UNKNOWN
    answered = bucket == BUCKET_CITED
    if expected in _OUT_OF_SCOPE_BEHAVIORS:
        return VERDICT_OVER_ANSWERED if answered else VERDICT_CORRECT_DECLINE
    return VERDICT_CORRECT_ANSWER if answered else VERDICT_MISSED_ANSWER


def evaluate(
    questions: list[Question],
    *,
    collection: Any,
    router: Any,
    synthesizer: Any = None,
    now: Optional[datetime] = None,
    n_results: int = 20,
    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
) -> list[QuestionResult]:
    """Run every question through the read path and score it. Pure over its injected deps."""
    now = now or datetime.now(tz=timezone.utc)
    results: list[QuestionResult] = []

    for q in questions:
        qr = answer(
            q.question,
            UserProfile(),
            collection=collection,
            router=router,
            synthesizer=synthesizer,
            now=now,
            n_results=n_results,
        )

        n_retrieved, top1, top3 = _similarity_at_ranks(qr)
        has_signal = top1 is not None and top1 >= similarity_floor

        synthesis_ran = qr.answer is not None or qr.validated_claims is not None
        synthesis_failed = synthesis_ran and qr.synthesis_failed
        if synthesis_ran:
            claims = qr.validated_claims or []
            claims_total = len(claims)
            claims_grounded = sum(1 for c in claims if c.is_grounded)
            validator_passed = claims_grounded > 0
            synthesis_citable = qr.answer is not None and len(qr.answer.citations) > 0
            confidence_warning = qr.answer.confidence_warning if qr.answer else None
        else:
            claims_total = claims_grounded = None
            validator_passed = synthesis_citable = None
            confidence_warning = None

        bucket = _bucket(
            n_retrieved, has_signal, synthesis_ran, synthesis_citable, synthesis_failed
        )
        results.append(
            QuestionResult(
                id=q.id,
                question=q.question,
                intent=q.intent,
                expected=q.expected,
                n_retrieved=n_retrieved,
                top1_similarity=top1,
                top3_similarity=top3,
                has_signal=has_signal,
                synthesis_ran=synthesis_ran,
                synthesis_citable=synthesis_citable,
                synthesis_failed=synthesis_failed,
                claims_total=claims_total,
                claims_grounded=claims_grounded,
                validator_passed=validator_passed,
                confidence_warning=confidence_warning,
                filter_fallback=qr.filter_fallback,
                bucket=bucket,
                verdict=_verdict(q.expected, bucket),
            )
        )

    return results


def summarize(results: list[QuestionResult]) -> EvalSummary:
    """Aggregate per-question results into the headline rates, bucket/verdict counts, and
    the citation-path-suspect guard."""
    bucket_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    for r in results:
        bucket_counts[r.bucket] = bucket_counts.get(r.bucket, 0) + 1
        verdict_counts[r.verdict] = verdict_counts.get(r.verdict, 0) + 1

    synthesis_ran = any(r.synthesis_ran for r in results)

    in_scope = [r for r in results if r.expected == "answer"]
    out_of_scope = [r for r in results if r.expected in _OUT_OF_SCOPE_BEHAVIORS]

    in_scope_cited = sum(1 for r in in_scope if r.bucket == BUCKET_CITED)
    answerable_rate = (in_scope_cited / len(in_scope)) if in_scope and synthesis_ran else None

    # Abstention is only meaningful where retrieval surfaced strong content — the temptation
    # to answer existed and was resisted. A decline on n=0 or thin retrieval is vacuous.
    tempted = [r for r in out_of_scope if r.has_signal]
    tempted_declined = sum(1 for r in tempted if r.bucket != BUCKET_CITED)
    abstention_rate = (tempted_declined / len(tempted)) if tempted and synthesis_ran else None

    strong_uncited = sum(1 for r in results if r.bucket == BUCKET_STRONG_UNCITED)
    strong_uncited_in_scope = sum(1 for r in in_scope if r.bucket == BUCKET_STRONG_UNCITED)
    synth_failed = sum(1 for r in results if r.bucket == BUCKET_SYNTH_FAILED)
    no_candidates = sum(1 for r in results if r.bucket == BUCKET_NO_CANDIDATES)
    filter_fallback_count = sum(1 for r in results if r.filter_fallback)

    in_scope_top1s = [r.top1_similarity for r in in_scope if r.top1_similarity is not None]
    avg_top1 = (sum(in_scope_top1s) / len(in_scope_top1s)) if in_scope_top1s else None

    return EvalSummary(
        total=len(results),
        synthesis_ran=synthesis_ran,
        bucket_counts=bucket_counts,
        verdict_counts=verdict_counts,
        in_scope_total=len(in_scope),
        in_scope_cited=in_scope_cited,
        answerable_in_scope_rate=answerable_rate,
        out_of_scope_total=len(out_of_scope),
        out_of_scope_tempted=len(tempted),
        out_of_scope_tempted_declined=tempted_declined,
        meaningful_abstention_rate=abstention_rate,
        strong_uncited_count=strong_uncited,
        strong_uncited_in_scope=strong_uncited_in_scope,
        synth_failed_count=synth_failed,
        no_candidates_count=no_candidates,
        filter_fallback_count=filter_fallback_count,
        citation_path_suspect=(strong_uncited_in_scope > 0),
        avg_top1_in_scope=avg_top1,
    )
