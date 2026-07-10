"""
Contract + adversarial tests for Phase 5 Output Formatting (generation/formatter.py).

Covers: deterministic sentence selection, deep-link building/quoting, tier/skepticism source
labels, corroboration_summary derivation from clusters, and the "reject, don't repair" policy
(>30%-dropped warning, zero-survive honest empty state) that lives in this phase, not
validator.py.
"""

from __future__ import annotations

from urllib.parse import unquote

from generation.contracts import ValidatedClaim
from generation.formatter import (
    CONFIDENCE_WARNING_MESSAGE,
    EMPTY_STATE_MESSAGE,
    AnswerFormatter,
)
from tests.retrieval.fixtures import make_retrieved_chunk, make_story_cluster


def _claim(**overrides) -> ValidatedClaim:
    base = dict(
        claim_text="NVIDIA beat Q2 earnings expectations.",
        confidence="high",
        is_grounded=True,
        supporting_chunk_id="a#0",
        validation_confidence=1.0,
    )
    base.update(overrides)
    return ValidatedClaim(**base)


class TestEmptyState:
    def test_no_claims_at_all_returns_empty_state(self):
        answer = AnswerFormatter().format([], {}, [])
        assert answer.prose == EMPTY_STATE_MESSAGE
        assert answer.citations == []
        assert answer.confidence_warning is None
        assert answer.corroboration_summary == {}

    def test_all_ungrounded_returns_empty_state(self):
        claims = [_claim(is_grounded=False, supporting_chunk_id=None) for _ in range(3)]
        answer = AnswerFormatter().format(claims, {}, [])
        assert answer.prose == EMPTY_STATE_MESSAGE

    def test_grounded_claim_whose_chunk_is_missing_from_map_falls_to_empty_state(self):
        # Defensive path: claim.supporting_chunk_id isn't a key in chunks at all.
        claim = _claim(supporting_chunk_id="missing#0")
        answer = AnswerFormatter().format([claim], {}, [])
        assert answer.prose == EMPTY_STATE_MESSAGE


class TestBasicCitationBuilding:
    def test_single_grounded_claim_produces_one_citation(self):
        chunk = make_retrieved_chunk(
            chunk_id="a#0",
            source_name="Reuters",
            tier=1,
            url="https://example.com/a",
            text="NVIDIA beat estimates. Guidance was raised too.",
        )
        claim = _claim(supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert len(answer.citations) == 1
        assert answer.citations[0].tier == 1
        assert answer.citations[0].source == "Reuters · Tier 1"

    def test_prose_weaves_claim_texts_together(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", text="Some source text.")
        claims = [
            _claim(claim_text="First claim.", supporting_chunk_id="a#0"),
            _claim(claim_text="Second claim.", supporting_chunk_id="a#0"),
        ]
        answer = AnswerFormatter().format(claims, {"a#0": chunk}, [])
        assert "First claim." in answer.prose
        assert "Second claim." in answer.prose


class TestSentenceSelection:
    def test_picks_sentence_with_highest_word_overlap(self):
        chunk = make_retrieved_chunk(
            chunk_id="a#0",
            text=(
                "The weather was sunny today. NVIDIA beat Q2 earnings expectations "
                "handily. Investors reacted with enthusiasm."
            ),
        )
        claim = _claim(
            claim_text="NVIDIA beat Q2 earnings expectations.", supporting_chunk_id="a#0"
        )
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert answer.citations[0].text == "NVIDIA beat Q2 earnings expectations handily."

    def test_no_sentence_boundary_returns_whole_trimmed_text(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", text="no terminator at all here   ")
        claim = _claim(claim_text="anything", supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert answer.citations[0].text == "no terminator at all here"

    def test_empty_chunk_text_yields_empty_citation_text(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", text="")
        claim = _claim(supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert answer.citations[0].text == ""

    def test_overlap_scoring_ignores_punctuation(self):
        # "expectations." (claim) vs "expectations" (mid-sentence, no period) must still
        # count as the same word for overlap purposes.
        chunk = make_retrieved_chunk(
            chunk_id="a#0",
            text="Unrelated filler sentence here. Q2 expectations were exceeded by NVIDIA.",
        )
        claim = _claim(claim_text="NVIDIA exceeded Q2 expectations.", supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert answer.citations[0].text == "Q2 expectations were exceeded by NVIDIA."

    def test_zero_overlap_falls_back_to_first_sentence(self):
        chunk = make_retrieved_chunk(
            chunk_id="a#0", text="Completely unrelated content. More unrelated content."
        )
        claim = _claim(claim_text="xyzzy plugh quux", supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert answer.citations[0].text == "Completely unrelated content."


class TestDeepLinks:
    def test_deep_link_built_from_chunk_url_and_selected_sentence(self):
        chunk = make_retrieved_chunk(
            chunk_id="a#0", url="https://example.com/article", text="NVIDIA beat estimates."
        )
        claim = _claim(claim_text="NVIDIA beat estimates.", supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        link = answer.citations[0].url_with_fragment
        assert link.startswith("https://example.com/article#:~:text=")
        assert unquote(link.split("text=")[1]) == "NVIDIA beat estimates."

    def test_special_characters_in_sentence_are_quoted(self):
        chunk = make_retrieved_chunk(
            chunk_id="a#0",
            url="https://example.com/a",
            text="Revenue rose 20% & margins held; shares gained.",
        )
        claim = _claim(claim_text="Revenue rose 20%", supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        link = answer.citations[0].url_with_fragment
        assert "%20" in link or "+" in link  # space encoded, not left raw
        assert " " not in link.split("text=")[1]

    def test_empty_url_still_builds_a_fragment(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", url="", text="Some text.")
        claim = _claim(supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert answer.citations[0].url_with_fragment.startswith("#:~:text=")


class TestTierSkepticismLabels:
    def test_tier_0_no_skepticism_note(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", source_name="SEC", tier=0)
        claim = _claim(supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert answer.citations[0].source == "SEC · Tier 0"

    def test_tier_1_no_skepticism_note(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", source_name="Reuters", tier=1)
        claim = _claim(supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert "skepticism" not in answer.citations[0].source

    def test_tier_2_includes_skepticism_note(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", source_name="RandomBlog", tier=2)
        claim = _claim(supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert "skepticism" in answer.citations[0].source

    def test_tier_3_includes_skepticism_note(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", source_name="AnonForum", tier=3)
        claim = _claim(supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert "skepticism" in answer.citations[0].source

    def test_relevance_over_tier_a_tier_3_claim_still_gets_cited(self):
        # RAG-fitness guardrail: tier never filters, only labels.
        chunk = make_retrieved_chunk(chunk_id="a#0", tier=3)
        claim = _claim(supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert len(answer.citations) == 1


class TestConfidenceWarningThreshold:
    def _claims(self, grounded_count, total_count):
        grounded = [_claim(supporting_chunk_id=f"{i}#0") for i in range(grounded_count)]
        ungrounded = [
            _claim(is_grounded=False, supporting_chunk_id=None)
            for _ in range(total_count - grounded_count)
        ]
        return grounded + ungrounded

    def _chunks(self, grounded_count):
        return {f"{i}#0": make_retrieved_chunk(chunk_id=f"{i}#0") for i in range(grounded_count)}

    def test_no_drops_no_warning(self):
        claims = self._claims(10, 10)
        answer = AnswerFormatter().format(claims, self._chunks(10), [])
        assert answer.confidence_warning is None

    def test_exactly_thirty_percent_dropped_no_warning(self):
        claims = self._claims(7, 10)  # 3/10 = 30%, not > 30%
        answer = AnswerFormatter().format(claims, self._chunks(7), [])
        assert answer.confidence_warning is None

    def test_just_over_thirty_percent_dropped_triggers_warning(self):
        claims = self._claims(6, 10)  # 4/10 = 40%, > 30%
        answer = AnswerFormatter().format(claims, self._chunks(6), [])
        assert answer.confidence_warning == CONFIDENCE_WARNING_MESSAGE


class TestCorroborationSummary:
    def test_only_clusters_with_grounded_citations_included(self):
        cited_chunk = make_retrieved_chunk(chunk_id="cited#0")
        uncited_chunk = make_retrieved_chunk(chunk_id="uncited#0")
        cited_cluster = make_story_cluster(
            cluster_id="cited-story", chunks=[cited_chunk], outlet_count=4
        )
        uncited_cluster = make_story_cluster(
            cluster_id="uncited-story", chunks=[uncited_chunk], outlet_count=2
        )
        claim = _claim(supporting_chunk_id="cited#0")
        answer = AnswerFormatter().format(
            [claim], {"cited#0": cited_chunk}, [cited_cluster, uncited_cluster]
        )
        assert answer.corroboration_summary == {"cited-story": 4}

    def test_multiple_claims_from_same_cluster_do_not_duplicate_entries(self):
        chunk_a = make_retrieved_chunk(chunk_id="a#0")
        chunk_b = make_retrieved_chunk(chunk_id="a#1")
        cluster = make_story_cluster(cluster_id="story", chunks=[chunk_a, chunk_b], outlet_count=3)
        claims = [
            _claim(supporting_chunk_id="a#0"),
            _claim(supporting_chunk_id="a#1"),
        ]
        answer = AnswerFormatter().format(claims, {"a#0": chunk_a, "a#1": chunk_b}, [cluster])
        assert answer.corroboration_summary == {"story": 3}

    def test_empty_clusters_list_yields_empty_summary(self):
        chunk = make_retrieved_chunk(chunk_id="a#0")
        claim = _claim(supporting_chunk_id="a#0")
        answer = AnswerFormatter().format([claim], {"a#0": chunk}, [])
        assert answer.corroboration_summary == {}


class TestLargeInput:
    def test_many_claims_does_not_crash(self):
        chunks = {f"{i}#0": make_retrieved_chunk(chunk_id=f"{i}#0") for i in range(200)}
        claims = [_claim(supporting_chunk_id=f"{i}#0") for i in range(200)]
        answer = AnswerFormatter().format(claims, chunks, [])
        assert len(answer.citations) == 200
