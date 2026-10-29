"""
End-to-end RAG pipeline: query -> hybrid retrieval -> generation -> answer + citations.

Aggregate questions ("average", "total", "how many", ...) are answered from
statistics computed over every receipt that matches the filters, not just the
top-k retrieved ones. The numbers are computed in code; the LLM only phrases them.
"""
import json
import statistics

from src.retriever import HybridRetriever
from src.llm_provider import get_llm_provider, TemplateProvider
from data.generate_receipts import to_document

AGGREGATE_KEYWORDS = (
    "average", "avg", "mean", "total", "sum", "how many", "count",
    "range", "minimum", "maximum", "lowest", "highest", "overall",
)
MAX_CONTEXT_DOCS = 20  # receipts shown to the LLM alongside the computed statistics


def is_aggregate_question(query: str) -> bool:
    q = query.lower()
    return any(keyword in q for keyword in AGGREGATE_KEYWORDS)


def compute_statistics(receipts: list[dict]) -> dict:
    rates = [r["rate_per_unit"] for r in receipts]
    amounts = [r["amount"] for r in receipts]
    return {
        "receipts": len(receipts),
        "average_rate": round(statistics.mean(rates), 2),
        "min_rate": min(rates),
        "max_rate": max(rates),
        "total_amount": round(sum(amounts), 2),
    }


class MarketIntelligencePipeline:
    def __init__(self, receipts_path="data/receipts.json", top_k=5):
        with open(receipts_path) as f:
            self.receipts = json.load(f)
        self.documents = [to_document(r) for r in self.receipts]
        self.retriever = HybridRetriever(self.receipts, self.documents)
        self.llm = get_llm_provider()
        self.top_k = top_k

    def ask(self, query: str) -> dict:
        results, filters = self.retriever.retrieve(query, top_k=self.top_k)
        if not results:
            # Nothing matched the filters: answer without calling the LLM, so it
            # can't fill the gap with receipts that don't match the question.
            return {
                "query": query,
                "extracted_filters": filters,
                "answer": self._no_match_answer(filters),
                "cited_receipts": [],
                "num_matched": 0,
            }

        stats = None
        if is_aggregate_question(query):
            indices = self.retriever.apply_metadata_filters(filters)
            matched_receipts = [self.receipts[i] for i in indices]
            context_docs = [self.documents[i] for i in indices[:MAX_CONTEXT_DOCS]]
            stats = compute_statistics(matched_receipts)
            context_docs.insert(0, self._statistics_line(stats))
        else:
            matched_receipts = [r["receipt"] for r in results]
            context_docs = [r["document"] for r in results]

        if isinstance(self.llm, TemplateProvider):
            answer = self.llm.answer(query, context_docs, matched_receipts)
        else:
            answer = self.llm.answer(query, context_docs)

        response = {
            "query": query,
            "extracted_filters": filters,
            "answer": answer,
            "cited_receipts": [r["receipt_no"] for r in matched_receipts],
            "num_matched": len(matched_receipts),
        }
        if stats:
            response["statistics"] = stats
        return response

    @staticmethod
    def _statistics_line(stats: dict) -> str:
        return (
            f"Computed over all {stats['receipts']} matching receipts: "
            f"average rate Rs. {stats['average_rate']} per unit "
            f"(range Rs. {stats['min_rate']}-{stats['max_rate']}), "
            f"total transaction value Rs. {stats['total_amount']}."
        )

    @staticmethod
    def _no_match_answer(filters: dict) -> str:
        if not filters:
            return "No matching receipts were found for this query."
        applied = ", ".join(f"{key}={value}" for key, value in filters.items())
        return f"No matching receipts were found for {applied}."
