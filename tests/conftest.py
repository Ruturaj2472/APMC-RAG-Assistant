import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Tests never call OpenAI: no API key, so generation and filters use the offline paths."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FILTER_EXTRACTOR", raising=False)
