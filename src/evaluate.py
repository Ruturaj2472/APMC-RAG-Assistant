"""
Simplified RAGAS-style evaluation harness.

Production path (requires internet + `pip install ragas`):
    RAGAS computes faithfulness, context precision, and context recall using
    an LLM judge against a labeled test set.

Offline/demo version (this file): since we generated the receipts ourselves,
we can compute ground truth directly (e.g. "true" average rate for a given
APMC+commodity filter) and check whether the retrieved context set actually
contains the receipts needed to answer correctly. This gives two of the same
signals RAGAS reports, without requiring an LLM judge or internet access:

  - context_recall:    fraction of the *true* relevant receipts that were
                        actually retrieved (did we find everything we should have?)
  - context_precision: fraction of *retrieved* receipts that were actually
                        relevant (did we pull in noise?)
"""
import json
from src.retriever import HybridRetriever
from data.generate_receipts import to_document


def build_test_set(receipts):
    """
    Construct a small labeled test set: for each (apmc, commodity) pair with
    enough receipts, create a query and the ground-truth set of relevant
    receipt_nos (computed directly from the data, not from retrieval).
    """
    test_cases = []
    seen_pairs = set()
    for r in receipts:
        key = (r["apmc"], r["commodity"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        relevant = [x["receipt_no"] for x in receipts
                    if x["apmc"] == r["apmc"] and x["commodity"] == r["commodity"]]
        if len(relevant) < 3:
            continue

        query = f"What was the rate for {r['commodity']} at {r['apmc']}?"
        test_cases.append({"query": query, "relevant_receipt_nos": set(relevant)})

        if len(test_cases) >= 15:
            break
    return test_cases


def evaluate(receipts_path="data/receipts.json", top_k=5):
    with open(receipts_path) as f:
        receipts = json.load(f)
    documents = [to_document(r) for r in receipts]
    retriever = HybridRetriever(receipts, documents)

    test_cases = build_test_set(receipts)
    precisions, recalls = [], []

    for case in test_cases:
        results, _ = retriever.retrieve(case["query"], top_k=top_k)
        retrieved_ids = {r["receipt"]["receipt_no"] for r in results}
        relevant_ids = case["relevant_receipt_nos"]

        true_positives = retrieved_ids & relevant_ids
        precision = len(true_positives) / len(retrieved_ids) if retrieved_ids else 0
        recall = len(true_positives) / len(relevant_ids) if relevant_ids else 0

        precisions.append(precision)
        recalls.append(recall)

    return {
        "num_test_cases": len(test_cases),
        "avg_context_precision": round(sum(precisions) / len(precisions), 3) if precisions else 0,
        "avg_context_recall": round(sum(recalls) / len(recalls), 3) if recalls else 0,
    }


if __name__ == "__main__":
    results = evaluate()
    print("Evaluation results (offline RAGAS-style harness):")
    print(json.dumps(results, indent=2))
