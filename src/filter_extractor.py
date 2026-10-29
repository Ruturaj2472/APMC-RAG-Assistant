"""
Metadata filter extraction: turns a question into APMC / commodity / year /
month filters that match the receipt fields.

    RegexFilterExtractor -- rule-based matching on APMC first words, commodity
                            names, 4-digit years and full month names. No
                            dependencies; the offline fallback.
    LLMFilterExtractor   -- one OpenAI structured-output call. The APMC and
                            commodity fields are constrained to the values in
                            the data, relative dates ("last month") are resolved
                            against a reference date, and local names
                            ("kanda", "brinjal") are mapped to canonical ones.
                            Falls back to the regex extractor if the call fails.

Choose with FILTER_EXTRACTOR: `llm`, `regex`, or `auto` (default: LLM when
OPENAI_API_KEY is set, else regex). The LLM model is OPENAI_FILTER_MODEL
(default gpt-4o-mini); the reference date is REFERENCE_DATE (YYYY-MM-DD,
default today).
"""
import datetime as dt
import os
import re
from functools import lru_cache
from typing import Literal, Optional

MONTH_NAMES = ["january", "february", "march", "april", "may", "june",
               "july", "august", "september", "october", "november", "december"]


class RegexFilterExtractor:
    name = "regex"

    def __init__(self, receipts: list[dict]):
        self.apmc_names = {r["apmc"].lower(): r["apmc"] for r in receipts}
        self.commodities = sorted({r["commodity"].lower() for r in receipts})

    def extract(self, query: str) -> dict:
        filters = {}
        q_lower = query.lower()

        for name_lower, name_original in self.apmc_names.items():
            if name_lower.split()[0] in q_lower:  # match on first word, e.g. "manchar"
                filters["apmc"] = name_original
                break

        for commodity in self.commodities:
            if commodity in q_lower:
                filters["commodity"] = commodity.capitalize()
                break

        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match:
            filters["year"] = year_match.group(1)

        for idx, month in enumerate(MONTH_NAMES, start=1):
            if month in q_lower:
                filters["month"] = f"{idx:02d}"
                break

        return filters


SYSTEM_PROMPT = """You extract search filters from questions about APMC (agricultural market) auction receipts.

Today's date is {today}.

Return a value for a field only if the question restricts it; otherwise return null. Never guess.
- apmc: the market. Match short or informal names ("Manchar market", "Junnar mandi") to the allowed values.
- commodity: map local or alternative names to the allowed values (e.g. kanda -> Onion, aloo -> Potato, brinjal or baingan -> Eggplant, gobi -> Cauliflower).
- year: four digits. Resolve relative years ("this year", "last year") from today's date.
- month: two digits "01"-"12". Resolve relative months ("this month", "last month") from today's date, and also set the year then.
A word such as "may" is only a month when it refers to the month of May."""


class LLMFilterExtractor:
    name = "llm"

    def __init__(self, receipts: list[dict], client=None, model: str | None = None,
                 reference_date: dt.date | str | None = None):
        from pydantic import Field, create_model  # noqa: local import

        self.fallback = RegexFilterExtractor(receipts)
        self.model = model or os.environ.get("OPENAI_FILTER_MODEL", "gpt-4o-mini")
        reference_date = reference_date or os.environ.get("REFERENCE_DATE")
        if isinstance(reference_date, str):
            reference_date = dt.date.fromisoformat(reference_date)
        self.reference_date = reference_date or dt.date.today()

        if client is None:
            import openai  # noqa: local import
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.client = client

        apmcs = tuple(sorted({r["apmc"] for r in receipts}))
        commodities = tuple(sorted({r["commodity"] for r in receipts}))
        months = tuple(f"{m:02d}" for m in range(1, 13))
        # All fields required but nullable: the shape OpenAI strict structured outputs expect.
        self.schema = create_model(
            "QueryFilters",
            apmc=(Optional[Literal[apmcs]], Field(description="Market, or null")),
            commodity=(Optional[Literal[commodities]], Field(description="Commodity, or null")),
            year=(Optional[str], Field(description="Four-digit year, or null")),
            month=(Optional[Literal[months]], Field(description="Two-digit month, or null")),
        )
        self.extract = lru_cache(maxsize=256)(self._extract)

    def _extract(self, query: str) -> dict:
        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(today=self.reference_date.isoformat())},
                    {"role": "user", "content": query},
                ],
                response_format=self.schema,
            )
            parsed = completion.choices[0].message.parsed
        except Exception as exc:
            print(f"[filters] LLM extraction failed ({exc}); using regex fallback.")
            return self.fallback.extract(query)
        return self._clean(parsed.model_dump())

    @staticmethod
    def _clean(raw: dict) -> dict:
        filters = {k: v for k, v in raw.items() if v}
        if "year" in filters and not re.fullmatch(r"20\d{2}", str(filters["year"])):
            filters.pop("year")
        return filters


def get_filter_extractor(receipts: list[dict], name: str | None = None):
    name = (name or os.environ.get("FILTER_EXTRACTOR", "auto")).lower()
    if name == "regex":
        return RegexFilterExtractor(receipts)
    if name == "llm":
        return LLMFilterExtractor(receipts)
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return LLMFilterExtractor(receipts)
        except ImportError:
            print("[filters] openai package not available - using regex extractor.")
    return RegexFilterExtractor(receipts)
