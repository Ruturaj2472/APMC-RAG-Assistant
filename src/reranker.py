"""
Rerankers for the fused candidate set.

    KeywordOverlapReranker -- rescores candidates by exact term overlap with the
                              question. No dependencies; the offline fallback.
    CrossEncoderReranker   -- scores each (question, receipt) pair jointly with a
                              cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`
                              by default). Requires `sentence-transformers`.

Choose with the RERANKER environment variable: `cross-encoder`, `keyword`, or
`auto` (default: the cross-encoder if sentence-transformers is installed,
else keyword overlap).
"""
import os

from src.bm25 import tokenize

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class KeywordOverlapReranker:
    name = "keyword"

    def __init__(self, documents: list[str]):
        self.documents = documents

    def rerank(self, query: str, candidate_indices: list[int], top_k: int) -> list[int]:
        query_terms = set(tokenize(query))
        scored = []
        for idx in candidate_indices:
            doc_terms = set(tokenize(self.documents[idx]))
            overlap = len(query_terms & doc_terms)
            scored.append((overlap, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in scored[:top_k]]


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, documents: list[str], model_name: str = CROSS_ENCODER_MODEL):
        from sentence_transformers import CrossEncoder  # noqa: local import

        self.documents = documents
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidate_indices: list[int], top_k: int) -> list[int]:
        if not candidate_indices:
            return []
        scores = self.model.predict([(query, self.documents[i]) for i in candidate_indices])
        ranked = sorted(zip(scores, candidate_indices), key=lambda x: x[0], reverse=True)
        return [idx for _, idx in ranked[:top_k]]


def get_reranker(documents: list[str], name: str | None = None):
    name = (name or os.environ.get("RERANKER", "auto")).lower()
    if name == "keyword":
        return KeywordOverlapReranker(documents)
    if name == "cross-encoder":
        return CrossEncoderReranker(documents)
    try:
        return CrossEncoderReranker(documents)
    except ImportError:
        print("[reranker] sentence-transformers not available - using keyword-overlap reranker.")
        return KeywordOverlapReranker(documents)
