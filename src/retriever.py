"""
Hybrid retrieval over APMC receipts: combines lexical (BM25) and semantic
(embedding cosine similarity) scores, applies metadata filters extracted
from the query, and reranks the fused candidate set.

Production notes:
  - Vector storage here is in-memory. In production this maps directly onto
    Qdrant (or ChromaDB): store `doc_embeddings` as vectors with payload =
    the receipt's metadata fields, and use Qdrant's payload filters instead
    of the manual `apply_metadata_filters` function below.
  - The rerank step here is a lightweight keyword-overlap heuristic standing
    in for a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`),
    which would normally be used to rescore the top-k fused candidates.
"""
import re
from src.bm25 import BM25, tokenize
from src.embeddings import get_embedder


class HybridRetriever:
    def __init__(self, receipts: list[dict], documents: list[str], alpha: float = 0.5):
        """
        alpha: weight given to semantic score vs. BM25 score in fusion (0-1).
        """
        self.receipts = receipts
        self.documents = documents
        self.alpha = alpha

        tokenized_docs = [tokenize(doc) for doc in documents]
        self.bm25 = BM25(tokenized_docs)

        self.embedder = get_embedder()
        self.embedder.fit(documents)

    @staticmethod
    def _normalize(scores):
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [0.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def extract_filters(self, query: str) -> dict:
        """
        Rule-based metadata extraction from the query (APMC name, commodity,
        date range). In production this would typically be done via an LLM
        function-calling / structured-output call (e.g. OpenAI function
        calling) instead of regex matching.
        """
        filters = {}
        q_lower = query.lower()

        apmc_names = {r["apmc"].lower(): r["apmc"] for r in self.receipts}
        for name_lower, name_original in apmc_names.items():
            if name_lower.split()[0] in q_lower:  # match on first word, e.g. "manchar"
                filters["apmc"] = name_original
                break

        commodities = {r["commodity"].lower() for r in self.receipts}
        for commodity in commodities:
            if commodity in q_lower:
                filters["commodity"] = commodity.capitalize()
                break

        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match:
            filters["year"] = year_match.group(1)

        month_names = ["january", "february", "march", "april", "may", "june",
                       "july", "august", "september", "october", "november", "december"]
        for idx, month in enumerate(month_names, start=1):
            if month in q_lower:
                filters["month"] = f"{idx:02d}"
                break

        return filters

    def apply_metadata_filters(self, filters: dict):
        """Return indices of receipts matching the extracted filters."""
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
        return indices if (filters and indices) else list(range(len(self.receipts)))

    def rerank(self, query: str, candidate_indices: list[int], top_k: int):
        """Lightweight cross-encoder stand-in: rescore by exact term overlap."""
        query_terms = set(tokenize(query))
        scored = []
        for idx in candidate_indices:
            doc_terms = set(tokenize(self.documents[idx]))
            overlap = len(query_terms & doc_terms)
            scored.append((overlap, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in scored[:top_k]]

    def retrieve(self, query: str, top_k: int = 5, fusion_k: int = 20):
        filters = self.extract_filters(query)
        allowed_indices = set(self.apply_metadata_filters(filters))

        bm25_scores = self.bm25.get_scores(query)
        sem_scores = self.embedder.similarity_scores(query)

        bm25_norm = self._normalize(bm25_scores)
        sem_norm = self._normalize(list(sem_scores))

        fused = []
        for i in range(len(self.documents)):
            if i not in allowed_indices:
                continue
            score = self.alpha * sem_norm[i] + (1 - self.alpha) * bm25_norm[i]
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
