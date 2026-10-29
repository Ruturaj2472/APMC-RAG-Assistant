"""
Hybrid retrieval over APMC receipts: combines lexical (BM25) and semantic
(embedding cosine similarity) scores, applies metadata filters extracted
from the query, and reranks the fused candidate set.

Vector search runs either in memory or in Qdrant (see src/vector_store.py).
With Qdrant, receipt vectors are stored with their metadata as payload and the
extracted filters are applied as Qdrant payload filters.

The top fused candidates are reranked by a keyword-overlap heuristic or a
cross-encoder (see src/reranker.py).
"""
from src.bm25 import BM25, tokenize
from src.embeddings import get_embedder
from src.vector_store import get_vector_store
from src.reranker import get_reranker
from src.filter_extractor import get_filter_extractor


class HybridRetriever:
    def __init__(self, receipts: list[dict], documents: list[str], alpha: float = 0.5,
                 embedder=None, vector_store=None, reranker=None, filter_extractor=None):
        """
        alpha: weight given to semantic score vs. BM25 score in fusion (0-1).
        embedder / vector_store / reranker / filter_extractor: defaults come from
        the EMBEDDINGS, VECTOR_STORE, RERANKER and FILTER_EXTRACTOR environment
        variables.
        """
        self.receipts = receipts
        self.documents = documents
        self.alpha = alpha

        tokenized_docs = [tokenize(doc) for doc in documents]
        self.bm25 = BM25(tokenized_docs)

        self.embedder = embedder or get_embedder()
        self.embedder.fit(documents)
        self.vector_store = vector_store or get_vector_store(self.embedder, documents, receipts)
        self.reranker = reranker or get_reranker(documents)
        self.filter_extractor = filter_extractor or get_filter_extractor(receipts)

    @staticmethod
    def _normalize(scores):
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [0.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def extract_filters(self, query: str) -> dict:
        """APMC / commodity / year / month filters (LLM or regex, see src/filter_extractor.py)."""
        return self.filter_extractor.extract(query)

    def apply_metadata_filters(self, filters: dict):
        """
        Return indices of receipts matching the extracted filters.

        No filters -> every receipt is a candidate. Filters that match nothing
        -> empty list, so the pipeline can say so instead of answering from
        unrelated receipts.
        """
        indices = []
        for i, r in enumerate(self.receipts):
            if "apmc" in filters and r["apmc"] != filters["apmc"]:
                continue
            if "commodity" in filters and r["commodity"] != filters["commodity"]:
                continue
            if "year" in filters and not r["date"].startswith(filters["year"]):
                continue
            if "month" in filters:
                date_month = r["date"].split("-")[1]
                if date_month != filters["month"]:
                    continue
            indices.append(i)
        if not filters:
            return list(range(len(self.receipts)))
        return indices

    def rerank(self, query: str, candidate_indices: list[int], top_k: int):
        return self.reranker.rerank(query, candidate_indices, top_k)

    def retrieve(self, query: str, top_k: int = 5, fusion_k: int = 20):
        filters = self.extract_filters(query)
        allowed_indices = set(self.apply_metadata_filters(filters))
        if not allowed_indices:
            return [], filters

        bm25_scores = self.bm25.get_scores(query)
        bm25_norm = self._normalize(bm25_scores)

        # Semantic scores are normalized over the filtered candidates only, so
        # in-memory and Qdrant search rank the same receipts the same way.
        sem_scores = {i: s for i, s in self.vector_store.scores(query, filters).items()
                      if i in allowed_indices}
        sem_ids = list(sem_scores)
        sem_norm = dict(zip(sem_ids, self._normalize([sem_scores[i] for i in sem_ids])))

        fused = []
        for i in range(len(self.documents)):
            if i not in allowed_indices:
                continue
            score = self.alpha * sem_norm.get(i, 0.0) + (1 - self.alpha) * bm25_norm[i]
            fused.append((score, i))

        fused.sort(key=lambda x: x[0], reverse=True)
        top_fused_indices = [i for _, i in fused[:fusion_k]]

        reranked_indices = self.rerank(query, top_fused_indices, top_k)

        results = []
        for idx in reranked_indices:
            results.append({
                "receipt": self.receipts[idx],
                "document": self.documents[idx],
            })
        return results, filters
