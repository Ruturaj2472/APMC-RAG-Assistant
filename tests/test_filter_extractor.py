"""
Tests for filter extraction. The LLM extractor is tested with a fake OpenAI
client, so no API calls are made.
"""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from data.generate_receipts import to_document
from src.filter_extractor import (
    LLMFilterExtractor,
    RegexFilterExtractor,
    get_filter_extractor,
)
from src.retriever import HybridRetriever

RECEIPTS = json.load(open("data/receipts.json"))


class FakeClient:
    """Mimics client.chat.completions.parse and records the messages sent."""

    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))

    def parse(self, model, messages, response_format, temperature):
        self.calls.append(messages)
        if self.error:
            raise self.error
        parsed = response_format(**self.response)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])


def llm(response=None, error=None, reference_date="2023-12-31"):
    client = FakeClient(response, error)
    return LLMFilterExtractor(RECEIPTS, client=client, reference_date=reference_date), client


NONE = {"apmc": None, "commodity": None, "year": None, "month": None}


def test_regex_extractor():
    filters = RegexFilterExtractor(RECEIPTS).extract("Onion trades at Manchar in March 2023")
    assert filters == {"apmc": "Manchar Krishi Utpanna Bazar Samiti", "commodity": "Onion",
                       "year": "2023", "month": "03"}


def test_llm_extractor_returns_canonical_filters():
    extractor, _ = llm(NONE | {"apmc": "Junnar APMC", "commodity": "Onion", "year": "2023", "month": "11"})
    assert extractor.extract("How did kanda sell at Junnar last month?") == {
        "apmc": "Junnar APMC", "commodity": "Onion", "year": "2023", "month": "11"}


def test_prompt_contains_reference_date_for_relative_dates():
    extractor, client = llm(NONE)
    extractor.extract("Onion rates at Junnar last month")
    assert "2023-12-31" in client.calls[0][0]["content"]


def test_null_fields_are_dropped():
    extractor, _ = llm(NONE | {"apmc": "Khed APMC"})
    assert extractor.extract("Trades at Khed") == {"apmc": "Khed APMC"}


def test_invalid_year_is_dropped():
    extractor, _ = llm(NONE | {"commodity": "Potato", "year": "last year"})
    assert extractor.extract("Potato trades last year") == {"commodity": "Potato"}


def test_schema_only_allows_values_in_the_data():
    extractor, _ = llm(NONE)
    with pytest.raises(ValidationError):
        extractor.schema(**(NONE | {"commodity": "Mango"}))
    with pytest.raises(ValidationError):
        extractor.schema(**(NONE | {"apmc": "Pune APMC"}))


def test_failed_llm_call_falls_back_to_regex():
    extractor, _ = llm(error=RuntimeError("API unavailable"))
    assert extractor.extract("Onion trades at Khed") == {"apmc": "Khed APMC", "commodity": "Onion"}


def test_repeated_questions_are_cached():
    extractor, client = llm(NONE | {"commodity": "Onion"})
    extractor.extract("Onion trades")
    extractor.extract("Onion trades")
    assert len(client.calls) == 1


def test_auto_uses_regex_without_an_api_key():
    assert get_filter_extractor(RECEIPTS).name == "regex"


def test_retriever_uses_the_given_extractor():
    documents = [to_document(r) for r in RECEIPTS]
    extractor, _ = llm(NONE | {"apmc": "Khed APMC", "commodity": "Eggplant"})
    retriever = HybridRetriever(RECEIPTS, documents, filter_extractor=extractor)
    results, filters = retriever.retrieve("Brinjal prices at Khed")
    assert filters == {"apmc": "Khed APMC", "commodity": "Eggplant"}
    assert results and all(r["receipt"]["commodity"] == "Eggplant" for r in results)
