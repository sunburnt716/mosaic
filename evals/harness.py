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

Then each question sorts into a bucket the operator can act on:
  - working                      behaved as its `expected` label wanted.
  - in-scope-but-thin            expected `answer` but produced no citable answer => the
                                 corpus is too thin here. ADD FEEDS; re-run to measure the
                                 delta. No router change would help.
  - out-of-scope-router-missed   expected `abstain`/`redirect` but produced a citable answer
                                 anyway => ROUTER work. No new sources would help.

The two headline rates fall straight out: answerable-in-scope (of in-scope questions, how
many got a cited answer) and out-of-scope-abstention (of out-of-scope questions, how many
were correctly declined). Those are the résumé line — "moved answerable-in-scope from X% to
Y% after adding N feeds while abstention stayed at 100%". Log results to Metrics.md
(see CLAUDE.md "Metrics").

Synthesis needs a Gemini client; in retrieval-only mode (`synthesizer=None`) the synthesis
fields are None and questions land in the `retrieval-only` bucket with `has_signal` reported
instead — enough to eyeball the corpus, not enough for the full buckets.

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

BUCKET_WORKING = "working"
BUCKET_THIN = "in-scope-but-thin"
BUCKET_ROUTER_MISS = "out-of-scope-router-missed"
BUCKET_RETRIEVAL_ONLY = "retrieval-only"

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
    claims_total: Optional[int]
    claims_grounded: Optional[int]
    validator_passed: Optional[bool]
    confidence_warning: Optional[str]
    # --- verdict ---
    bucket: str


@dataclass
class EvalSummary:
    total: int
    synthesis_ran: bool
    bucket_counts: dict[str, int]
    in_scope_total: int
    in_scope_working: int
    answerable_in_scope_rate: Optional[float]
    out_of_scope_total: int
    out_of_scope_declined: int
    abstention_rate: Optional[float]
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
    expected: str, synthesis_ran: bool, synthesis_citable: Optional[bool], has_signal: bool
) -> str:
    """Sort one question into an actionable bucket (see module docstring)."""
    out_of_scope = expected in _OUT_OF_SCOPE_BEHAVIORS

    if not synthesis_ran:
        # Can't observe abstain-vs-answer without generation; report the retrieval proxy.
        return BUCKET_RETRIEVAL_ONLY

    if out_of_scope:
        # Correct behavior is to decline; a citable answer is a miss the router should catch.
        return BUCKET_ROUTER_MISS if synthesis_citable else BUCKET_WORKING

    # In-scope: a citable answer is success; its absence means the corpus was too thin.
    return BUCKET_WORKING if synthesis_citable else BUCKET_THIN


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
                claims_total=claims_total,
                claims_grounded=claims_grounded,
                validator_passed=validator_passed,
                confidence_warning=confidence_warning,
                bucket=_bucket(q.expected, synthesis_ran, synthesis_citable, has_signal),
            )
        )

    return results


def summarize(results: list[QuestionResult]) -> EvalSummary:
    """Aggregate per-question results into the two headline rates + bucket counts."""
    bucket_counts: dict[str, int] = {}
    for r in results:
        bucket_counts[r.bucket] = bucket_counts.get(r.bucket, 0) + 1

    synthesis_ran = any(r.synthesis_ran for r in results)

    in_scope = [r for r in results if r.expected == "answer"]
    out_of_scope = [r for r in results if r.expected in _OUT_OF_SCOPE_BEHAVIORS]

    in_scope_working = sum(1 for r in in_scope if r.bucket == BUCKET_WORKING)
    out_declined = sum(1 for r in out_of_scope if r.bucket == BUCKET_WORKING)

    answerable_rate = (in_scope_working / len(in_scope)) if in_scope and synthesis_ran else None
    abstention_rate = (out_declined / len(out_of_scope)) if out_of_scope and synthesis_ran else None

    in_scope_top1s = [r.top1_similarity for r in in_scope if r.top1_similarity is not None]
    avg_top1 = (sum(in_scope_top1s) / len(in_scope_top1s)) if in_scope_top1s else None

    return EvalSummary(
        total=len(results),
        synthesis_ran=synthesis_ran,
        bucket_counts=bucket_counts,
        in_scope_total=len(in_scope),
        in_scope_working=in_scope_working,
        answerable_in_scope_rate=answerable_rate,
        out_of_scope_total=len(out_of_scope),
        out_of_scope_declined=out_declined,
        abstention_rate=abstention_rate,
        avg_top1_in_scope=avg_top1,
    )
