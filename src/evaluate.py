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

It also checks aggregate answers ("What was the average rate for X at Y?"):
whether the computed average equals the true average over all matching
receipts, using only the top-k retrieved receipts vs. all filtered receipts.
"""
import json
import statistics

from src.retriever import HybridRetriever
from src.pipeline import compute_statistics
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

    aggregate = evaluate_aggregates(receipts, retriever, test_cases, top_k)
    return {
        "num_test_cases": len(test_cases),
        "avg_context_precision": round(sum(precisions) / len(precisions), 3) if precisions else 0,
        "avg_context_recall": round(sum(recalls) / len(recalls), 3) if recalls else 0,
        **aggregate,
    }


def evaluate_aggregates(receipts, retriever, test_cases, top_k):
    """
    For each (apmc, commodity) pair, ask for the average rate and compare the
    answer against the true average over all relevant receipts.
    """
    by_no = {r["receipt_no"]: r for r in receipts}
    exact_top_k = exact_all = 0
    recall_top_k, recall_all = [], []

    for case in test_cases:
        relevant = [by_no[n] for n in case["relevant_receipt_nos"]]
        true_avg = round(statistics.mean(r["rate_per_unit"] for r in relevant), 2)
        query = "What was the average " + case["query"][len("What was the "):]

        top_results, filters = retriever.retrieve(query, top_k=top_k)
        top_receipts = [r["receipt"] for r in top_results]
        all_receipts = [receipts[i] for i in retriever.apply_metadata_filters(filters)]

        exact_top_k += compute_statistics(top_receipts)["average_rate"] == true_avg
        exact_all += compute_statistics(all_receipts)["average_rate"] == true_avg
        relevant_ids = case["relevant_receipt_nos"]
        recall_top_k.append(len({r["receipt_no"] for r in top_receipts} & relevant_ids) / len(relevant_ids))
        recall_all.append(len({r["receipt_no"] for r in all_receipts} & relevant_ids) / len(relevant_ids))

    n = len(test_cases)
    return {
        "aggregate_exact_average_top_k": f"{exact_top_k}/{n}",
        "aggregate_exact_average_all_filtered": f"{exact_all}/{n}",
        "aggregate_receipt_recall_top_k": round(sum(recall_top_k) / n, 3),
        "aggregate_receipt_recall_all_filtered": round(sum(recall_all) / n, 3),
    }


if __name__ == "__main__":
    results = evaluate()
    print("Evaluation results (offline RAGAS-style harness):")
    print(json.dumps(results, indent=2))
