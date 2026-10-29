"""
Compare regex and LLM filter extraction on questions with informal phrasing:
local commodity names (kanda, aloo, gobi, brinjal), short market names,
relative dates (this month, last month, last year), abbreviated months, and
"May" used as a word rather than the month.

A question counts as correct only if the extracted filters exactly equal the
expected ones. Relative dates are resolved against REFERENCE_DATE = 2023-12-31,
the end of the receipt data.

    python -m src.compare_filters     # regex only without OPENAI_API_KEY; writes results/filter_extraction.json
"""
import json
import os
import pathlib

from src.filter_extractor import LLMFilterExtractor, RegexFilterExtractor

REFERENCE_DATE = "2023-12-31"
MANCHAR = "Manchar Krishi Utpanna Bazar Samiti"

CASES = [
    ("What was the average onion rate at Manchar Krishi Utpanna Bazar Samiti?",
     {"apmc": MANCHAR, "commodity": "Onion"}),
    ("Tomato rates at the Manchar market", {"apmc": MANCHAR, "commodity": "Tomato"}),
    ("Brinjal prices at Khed", {"apmc": "Khed APMC", "commodity": "Eggplant"}),
    ("How did kanda sell at Junnar last month?",
     {"apmc": "Junnar APMC", "commodity": "Onion", "year": "2023", "month": "11"}),
    ("Aloo trades at Narayangaon in March 2023",
     {"apmc": "Narayangaon APMC", "commodity": "Potato", "year": "2023", "month": "03"}),
    ("Show me gobi trades in Junnar in April 2023",
     {"apmc": "Junnar APMC", "commodity": "Cauliflower", "year": "2023", "month": "04"}),
    ("Eggplant at Narayangaon in Dec 2023",
     {"apmc": "Narayangaon APMC", "commodity": "Eggplant", "year": "2023", "month": "12"}),
    ("May I see tomato trades at Khed?", {"apmc": "Khed APMC", "commodity": "Tomato"}),
    ("Onion rates at Junnar this month",
     {"apmc": "Junnar APMC", "commodity": "Onion", "year": "2023", "month": "12"}),
    ("Potato trades last year", {"commodity": "Potato", "year": "2022"}),
    ("Trades handled by Shivshakti Traders", {}),
    ("What was the onion rate at Narayangaon in the first month of 2023?",
     {"apmc": "Narayangaon APMC", "commodity": "Onion", "year": "2023", "month": "01"}),
]


def run(extractor):
    rows = []
    for question, expected in CASES:
        got = extractor.extract(question)
        rows.append({"question": question, "expected": expected, "got": got, "correct": got == expected})
    return rows


def main(receipts_path="data/receipts.json", output="results/filter_extraction.json"):
    receipts = json.loads(pathlib.Path(receipts_path).read_text())
    extractors = {"regex": RegexFilterExtractor(receipts)}
    if os.environ.get("OPENAI_API_KEY"):
        extractors["llm"] = LLMFilterExtractor(receipts, reference_date=REFERENCE_DATE)
    else:
        print("[compare] OPENAI_API_KEY not set - skipping the LLM extractor.")

    results = {name: run(ex) for name, ex in extractors.items()}

    names = list(results)
    print(f"| Question | {' | '.join(names)} |")
    print(f"|---|{'---|' * len(names)}")
    for i, (question, _) in enumerate(CASES):
        marks = " | ".join("✅" if results[n][i]["correct"] else "❌" for n in names)
        print(f"| {question} | {marks} |")
    summary = {n: f"{sum(r['correct'] for r in rows)}/{len(rows)}" for n, rows in results.items()}
    print("\nExact-match accuracy:", summary)

    report = {"reference_date": REFERENCE_DATE, "accuracy": summary,
              "llm_model": extractors["llm"].model if "llm" in extractors else None,
              "results": results}
    path = pathlib.Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved to {path}")
    return report


if __name__ == "__main__":
    main()
