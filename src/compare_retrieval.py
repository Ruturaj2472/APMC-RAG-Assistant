"""
Compare retrieval setups on questions that metadata filters can't answer.

The main evaluation (src/evaluate.py) is dominated by metadata filtering, so
every embedder scores the same there. This benchmark uses questions about
fields the filters don't extract (commission agent, trader, payment mode,
grade mark, village), so ranking depends on BM25 and the embeddings:

  - exact_term:  the question uses the same words as the receipt text
  - paraphrase:  the question describes the value in other words

Metrics (top_k=5): precision@5 = share of the top 5 receipts that are relevant;
hit@1 = share of questions whose top receipt is relevant.

Setups: BM25 only, BM25 + TF-IDF, BM25 + BGE (in memory / Qdrant), BGE only
(Qdrant), each with the keyword-overlap reranker, plus the same setups with a
cross-encoder reranker. BGE and cross-encoder setups are skipped if
sentence-transformers isn't installed; Qdrant setups if qdrant-client isn't.

    python -m src.compare_retrieval            # prints a table, writes results/retrieval_comparison.json
"""
import json
import pathlib

from data.generate_receipts import to_document
from src.embeddings import BGEEmbedder, TfidfEmbedder
from src.reranker import CrossEncoderReranker, KeywordOverlapReranker
from src.filter_extractor import RegexFilterExtractor
from src.retriever import HybridRetriever
from src.vector_store import InMemoryVectorStore, QdrantVectorStore

TOP_K = 5

QUESTIONS = [
    # (type, question, field, value)
    ("exact_term", "Trades handled by Shivshakti Traders", "commission_agent", "Shivshakti Traders"),
    ("exact_term", "Receipts from Krishna Commission Agency", "commission_agent", "Krishna Commission Agency"),
    ("exact_term", "Sales to Export House A", "trader", "Export House A"),
    ("exact_term", "Purchases by the Regional Wholesaler", "trader", "Regional Wholesaler"),
    ("exact_term", "Receipts paid by Bank Transfer", "mode_of_payment", "Bank Transfer"),
    ("exact_term", "Produce marked grade-A", "mark", "grade-A"),
    ("exact_term", "Farmers from Village 3", "village", "Village 3"),
    ("exact_term", "Lots marked loose", "mark", "loose"),
    ("paraphrase", "Deals settled electronically", "mode_of_payment", "Bank Transfer"),
    ("paraphrase", "Goods bought for overseas shipment", "trader", "Export House A"),
    ("paraphrase", "Premium quality produce", "mark", "grade-A"),
    ("paraphrase", "Trades settled with physical currency", "mode_of_payment", "Cash"),
    ("paraphrase", "Bulk buyers who resell to retailers", "trader", "Regional Wholesaler"),
    ("paraphrase", "Unpacked produce sold without sorting", "mark", "loose"),
    ("paraphrase", "Second-tier quality lots", "mark", "grade-B"),
    ("paraphrase", "Buyers from the nearby town market", "trader", "Local Market"),
]


def build_setups(receipts, documents):
    """Yield (name, retriever) for every setup whose dependencies are installed."""
    regex = RegexFilterExtractor(receipts)  # deterministic, no API calls

    def build(**kwargs):
        return HybridRetriever(receipts, documents, filter_extractor=regex, **kwargs)

    keyword = KeywordOverlapReranker(documents)
    tfidf = TfidfEmbedder()
    tfidf_store = InMemoryVectorStore(tfidf)
    yield "BM25 only", build(alpha=0.0, embedder=tfidf, vector_store=tfidf_store, reranker=keyword)
    yield "BM25 + TF-IDF", build(embedder=tfidf, vector_store=tfidf_store, reranker=keyword)
    try:
        bge = BGEEmbedder().fit(documents)
        cross = CrossEncoderReranker(documents)
    except ImportError:
        print("[compare] sentence-transformers not installed - skipping BGE and cross-encoder setups.")
        return
    bge_store = InMemoryVectorStore(bge)
    yield "BM25 + TF-IDF + cross-encoder", build(embedder=tfidf, vector_store=tfidf_store, reranker=cross)
    yield "BM25 + BGE (in memory)", build(embedder=bge, vector_store=bge_store, reranker=keyword)
    yield "BM25 + BGE + cross-encoder (in memory)", build(embedder=bge, vector_store=bge_store, reranker=cross)
    try:
        qdrant = QdrantVectorStore(bge, documents, receipts)
    except ImportError:
        print("[compare] qdrant-client not installed - skipping Qdrant setups.")
        return
    yield "BM25 + BGE (Qdrant)", build(embedder=bge, vector_store=qdrant, reranker=keyword)
    yield "BM25 + BGE + cross-encoder (Qdrant)", build(embedder=bge, vector_store=qdrant, reranker=cross)
    yield "BGE only (Qdrant)", build(alpha=1.0, embedder=bge, vector_store=qdrant, reranker=keyword)


def score(retriever, receipts):
    per_type = {}
    for qtype, question, field, value in QUESTIONS:
        results, filters = retriever.retrieve(question, top_k=TOP_K)
        assert not filters, f"benchmark question triggered metadata filters: {question!r} -> {filters}"
        relevant = [r["receipt"][field] == value for r in results]
        bucket = per_type.setdefault(qtype, {"precision": [], "hit": []})
        bucket["precision"].append(sum(relevant) / TOP_K)
        bucket["hit"].append(1.0 if relevant and relevant[0] else 0.0)

    def avg(xs):
        return round(sum(xs) / len(xs), 3)

    out = {}
    for qtype, b in per_type.items():
        out[f"{qtype}_precision@5"] = avg(b["precision"])
        out[f"{qtype}_hit@1"] = avg(b["hit"])
    all_p = [p for b in per_type.values() for p in b["precision"]]
    out["overall_precision@5"] = avg(all_p)
    return out


def main(receipts_path="data/receipts.json", output="results/retrieval_comparison.json"):
    receipts = json.loads(pathlib.Path(receipts_path).read_text())
    documents = [to_document(r) for r in receipts]

    results = {name: score(retriever, receipts) for name, retriever in build_setups(receipts, documents)}

    cols = ["exact_term_precision@5", "paraphrase_precision@5", "overall_precision@5",
            "exact_term_hit@1", "paraphrase_hit@1"]
    print(f"| Setup | {' | '.join(cols)} |")
    print(f"|---|{'---|' * len(cols)}")
    for name, metrics in results.items():
        print(f"| {name} | {' | '.join(str(metrics[c]) for c in cols)} |")

    path = pathlib.Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"top_k": TOP_K, "questions": len(QUESTIONS), "results": results}, indent=2) + "\n")
    print(f"\nSaved to {path}")
    return results


if __name__ == "__main__":
    main()
