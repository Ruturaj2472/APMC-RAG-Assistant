"""
Tests for the rerankers. The cross-encoder test is skipped when
sentence-transformers isn't installed (as in CI).
"""
import json

import pytest

from data.generate_receipts import to_document
from src.reranker import KeywordOverlapReranker, get_reranker
from src.retriever import HybridRetriever

DOCS = [
    "Mode of payment: Cash. Trader: BKC.",
    "Mode of payment: Bank Transfer. Trader: Export House A.",
    "Mode of payment: Bank Transfer. Trader: Local Market.",
]


def test_keyword_reranker_orders_by_term_overlap():
    reranker = KeywordOverlapReranker(DOCS)
    assert reranker.rerank("paid by bank transfer to Export House A", [0, 1, 2], top_k=2) == [1, 2]


def test_explicit_keyword_reranker(monkeypatch):
    monkeypatch.setenv("RERANKER", "keyword")
    assert get_reranker(DOCS).name == "keyword"


def test_auto_falls_back_to_keyword_without_sentence_transformers(monkeypatch):
    import src.reranker as reranker_module

    class Unavailable:
        def __init__(self, *args, **kwargs):
            raise ImportError("sentence-transformers not installed")

    monkeypatch.delenv("RERANKER", raising=False)
    monkeypatch.setattr(reranker_module, "CrossEncoderReranker", Unavailable)
    assert get_reranker(DOCS).name == "keyword"


def test_retriever_uses_the_given_reranker():
    receipts = json.load(open("data/receipts.json"))
    documents = [to_document(r) for r in receipts]
    calls = []

    class RecordingReranker:
        def rerank(self, query, candidate_indices, top_k):
            calls.append(len(candidate_indices))
            return list(reversed(candidate_indices))[:top_k]

    retriever = HybridRetriever(receipts, documents, reranker=RecordingReranker())
    results, _ = retriever.retrieve("Show me tomato trades", top_k=3)
    assert calls and calls[0] <= 20
    assert len(results) == 3


def test_cross_encoder_prefers_the_matching_receipt():
    pytest.importorskip("sentence_transformers")
    from src.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(DOCS)
    assert reranker.rerank("Which trades were paid in cash?", [1, 2, 0], top_k=1) == [0]
