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


# Regression tests: filters that match no receipts must return nothing, not
# the full corpus. Receipt data covers 2023 only, so 2022 queries match nothing.

ZERO_MATCH_QUERY = "What was the average onion rate at Manchar in 2022?"


def test_zero_match_filters_return_no_receipts():
    pipeline = MarketIntelligencePipeline()
    result = pipeline.ask(ZERO_MATCH_QUERY)
    assert result["extracted_filters"].get("year") == "2022"
    assert result["num_matched"] == 0
    assert result["cited_receipts"] == []
    assert result["answer"].startswith("No matching receipts")


def test_zero_match_does_not_fall_back_to_full_corpus():
    pipeline = MarketIntelligencePipeline()
    retriever = pipeline.retriever
    assert retriever.apply_metadata_filters({"apmc": "Khed APMC", "year": "2022"}) == []
    assert len(retriever.apply_metadata_filters({})) == len(pipeline.receipts)


def test_zero_match_answer_does_not_call_the_llm():
    class FailingLLM:
        def answer(self, *args, **kwargs):
            raise AssertionError("LLM must not be called when no receipts match")

    pipeline = MarketIntelligencePipeline()
    pipeline.llm = FailingLLM()
    assert pipeline.ask(ZERO_MATCH_QUERY)["num_matched"] == 0


def test_filtered_results_match_every_filter():
    pipeline = MarketIntelligencePipeline()
    result = pipeline.ask("Onion trades at Manchar in March 2023")
    assert result["num_matched"] > 0
    by_no = {r["receipt_no"]: r for r in pipeline.receipts}
    for receipt_no in result["cited_receipts"]:
        receipt = by_no[receipt_no]
        assert receipt["commodity"] == "Onion"
        assert receipt["apmc"].startswith("Manchar")
        assert receipt["date"].startswith("2023-03")
