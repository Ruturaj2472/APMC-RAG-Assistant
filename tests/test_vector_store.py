"""
Tests for the Qdrant vector store (local in-process mode, TF-IDF vectors).
Skipped when qdrant-client isn't installed.
"""
import json

import pytest

pytest.importorskip("qdrant_client")

from data.generate_receipts import to_document  # noqa: E402
from src.embeddings import TfidfEmbedder  # noqa: E402
from src.retriever import HybridRetriever  # noqa: E402
from src.vector_store import InMemoryVectorStore, QdrantVectorStore  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    receipts = json.load(open("data/receipts.json"))
    return receipts, [to_document(r) for r in receipts]


@pytest.fixture(scope="module")
def retrievers(corpus):
    receipts, documents = corpus
    memory_embedder = TfidfEmbedder()
    memory = HybridRetriever(receipts, documents, embedder=memory_embedder,
                             vector_store=InMemoryVectorStore(memory_embedder))
    qdrant_embedder = TfidfEmbedder().fit(documents)
    qdrant = HybridRetriever(receipts, documents, embedder=qdrant_embedder,
                             vector_store=QdrantVectorStore(qdrant_embedder, documents, receipts))
    return memory, qdrant


def test_qdrant_scores_match_in_memory_scores(retrievers):
    memory, qdrant = retrievers
    query = "Trades handled by Shivshakti Traders"
    mem = memory.vector_store.scores(query, {})
    qd = qdrant.vector_store.scores(query, {})
    assert set(mem) == set(qd)
    for i in mem:
        assert qd[i] == pytest.approx(mem[i], abs=1e-4)


def test_qdrant_payload_filter_returns_only_matching_receipts(retrievers, corpus):
    receipts, _ = corpus
    _, qdrant = retrievers
    filters = {"apmc": "Khed APMC", "commodity": "Potato"}
    ids = qdrant.vector_store.scores("potato rates at Khed", filters)
    expected = {i for i, r in enumerate(receipts)
                if r["apmc"] == "Khed APMC" and r["commodity"] == "Potato"}
    assert set(ids) == expected


def test_qdrant_and_memory_retrieve_the_same_receipts(retrievers):
    memory, qdrant = retrievers
    for query in ["What was the rate for Onion at Junnar APMC?", "Receipts paid by Bank Transfer"]:
        mem_ids = [r["receipt"]["receipt_no"] for r in memory.retrieve(query)[0]]
        qd_ids = [r["receipt"]["receipt_no"] for r in qdrant.retrieve(query)[0]]
        assert mem_ids == qd_ids


def test_qdrant_zero_match_filters_return_nothing(retrievers):
    _, qdrant = retrievers
    results, filters = qdrant.retrieve("What was the average onion rate at Manchar in 2022?")
    assert filters["year"] == "2022"
    assert results == []
