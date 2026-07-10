"""
End-to-end integration test: query -> RetrievalOutput, wiring all five retrieval phases
together over a real (ephemeral, in-process) Chroma collection and real MiniLM/Groq calls.

Requires network (MiniLM model download, Groq API + GROQ_API_KEY) — skipped by default,
mirroring tests/test_adapter_contract.py's live-network convention. Run manually with
`pytest -m integration` once `requirements.txt` is installed and GROQ_API_KEY is set.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


@pytest.mark.integration
@pytest.mark.skip(
    reason="Requires live network access (MiniLM download, Groq API) — run manually with"
    " -m integration"
)
class TestEndToEndRetrieval:
    def test_query_to_retrieval_output_over_real_vectors(self):
        import chromadb

        from processing.utils.embedding import embed_text
        from retrieval.cluster import StoryClusterer
        from retrieval.contracts import UserProfile
        from retrieval.output import assemble_retrieval_output
        from retrieval.rerank import Ranker
        from retrieval.router import QueryRouter
        from retrieval.search import VectorSearch

        collection = chromadb.Client().create_collection("mosaic-retrieval-integration-test")

        now_epoch = int(datetime.now(timezone.utc).timestamp())
        seed_chunks = [
            (
                "doc-0#0",
                "NVIDIA reported record quarterly revenue driven by AI chip demand.",
                "Reuters",
                1,
                "NVDA",
            ),
            (
                "doc-1#0",
                "Nvidia posted blowout earnings as data-center GPU sales surged.",
                "Bloomberg",
                1,
                "NVDA",
            ),
            (
                "doc-2#0",
                "The Federal Reserve held interest rates steady at this week's meeting.",
                "AP",
                0,
                None,
            ),
        ]
        for chunk_id, text, source_name, tier, ticker in seed_chunks:
            collection.add(
                ids=[chunk_id],
                embeddings=[embed_text(text)],
                documents=[text],
                metadatas=[
                    {
                        "source_name": source_name,
                        "tier": tier,
                        "published_epoch": now_epoch,
                        "ticker": ticker,
                        "url": f"https://example.com/{chunk_id}",
                        "section_label": None,
                        "ordinal": 0,
                    }
                ],
            )

        router = QueryRouter()  # real Groq client + real MiniLM embedder
        routing = router.route("What's the latest on NVDA earnings?", UserProfile())

        chunks = VectorSearch(collection).search(routing, n_results=10)
        ranked = Ranker().rank(chunks, routing, datetime.now(timezone.utc))
        clusters = StoryClusterer().cluster(ranked)
        output = assemble_retrieval_output(clusters)

        assert output.chunk_count > 0
        assert "Reuters" in output.outlets_represented or "Bloomberg" in output.outlets_represented
        assert 0.0 <= output.retrieval_confidence <= 1.0
