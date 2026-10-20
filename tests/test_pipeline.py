"""
Basic tests for the retrieval pipeline. Run with:
    python3 -m pytest tests/ -v
"""
from src.pipeline import MarketIntelligencePipeline


def test_pipeline_loads_receipts():
    pipeline = MarketIntelligencePipeline()
    assert len(pipeline.receipts) > 0


def test_filter_extraction_apmc_and_commodity():
    pipeline = MarketIntelligencePipeline()
    query = "What was the average onion rate at Manchar Krishi Utpanna Bazar Samiti?"
    result = pipeline.ask(query)
    assert result["extracted_filters"].get("commodity") == "Onion"
    assert "Manchar" in result["extracted_filters"].get("apmc", "")


def test_retrieval_returns_matching_commodity():
    pipeline = MarketIntelligencePipeline()
    query = "Show me tomato trades"
    result = pipeline.ask(query)
    assert result["num_matched"] > 0
    for receipt_no in result["cited_receipts"]:
        matching = [r for r in pipeline.receipts if r["receipt_no"] == receipt_no]
        assert matching[0]["commodity"] == "Tomato"


def test_no_filter_query_still_returns_results():
    pipeline = MarketIntelligencePipeline()
    result = pipeline.ask("What trades happened recently?")
    assert result["num_matched"] > 0
