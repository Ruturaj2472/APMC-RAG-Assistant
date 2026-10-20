"""
Runs the RAG pipeline directly (no FastAPI/uvicorn needed) so it's demonstrable
in environments without internet access to install those packages.
The underlying pipeline (src/pipeline.py) is identical to what app.py serves.
"""
from src.pipeline import MarketIntelligencePipeline

SAMPLE_QUERIES = [
    "What was the average onion rate at Manchar Krishi Utpanna Bazar Samiti?",
    "Show me tomato trades in April",
    "What is the total transaction value for potato in 2023?",
]


def main():
    pipeline = MarketIntelligencePipeline()
    print(f"Loaded {len(pipeline.receipts)} synthetic receipts.\n")

    for query in SAMPLE_QUERIES:
        print("=" * 80)
        print(f"Q: {query}")
        result = pipeline.ask(query)
        print(f"Extracted filters: {result['extracted_filters']}")
        print(f"Matched {result['num_matched']} receipts")
        print(f"A: {result['answer']}")
        print()


if __name__ == "__main__":
    main()
