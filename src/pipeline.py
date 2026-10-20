"""
End-to-end RAG pipeline: query -> hybrid retrieval -> generation -> answer + citations.
"""
import json
from src.retriever import HybridRetriever
from src.llm_provider import get_llm_provider, TemplateProvider
from data.generate_receipts import to_document


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
        context_docs = [r["document"] for r in results]
        matched_receipts = [r["receipt"] for r in results]

        if isinstance(self.llm, TemplateProvider):
            answer = self.llm.answer(query, context_docs, matched_receipts)
        else:
            answer = self.llm.answer(query, context_docs)

        return {
            "query": query,
            "extracted_filters": filters,
            "answer": answer,
            "cited_receipts": [r["receipt_no"] for r in matched_receipts],
            "num_matched": len(matched_receipts),
        }
