"""
LLM provider abstraction for the generation step.

Production path (requires internet + `pip install openai` + OPENAI_API_KEY):
    OpenAIProvider -- calls gpt-4-turbo / gpt-3.5-turbo, both of which were
    widely available and production-grade throughout Jan-Aug 2024.
    Override the model with the OPENAI_MODEL environment variable.

Offline/demo fallback (used automatically if the `openai` package or an
API key isn't available, e.g. in this sandbox):
    TemplateProvider -- deterministically composes a grounded answer directly
    from the retrieved receipts (no hallucination possible, since there's no
    free-form generation) so the end-to-end pipeline is fully runnable and
    demonstrable without network access.
"""
import os
import statistics


class OpenAIProvider:
    def __init__(self, model=None):
        import openai  # noqa: local import, only required for production path
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4-turbo")

    def answer(self, query: str, context_docs: list[str]) -> str:
        context = "\n".join(f"- {doc}" for doc in context_docs)
        system_prompt = (
            "You are an assistant for an APMC (Agricultural Produce Market Committee) "
            "market intelligence system. Answer the user's question ONLY using the "
            "receipt data provided below. Cite receipt numbers where relevant. "
            "If the data doesn't contain the answer, say so plainly.\n\n"
            f"Receipt data:\n{context}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content


class TemplateProvider:
    """
    Deterministic, no-hallucination fallback: computes the actual answer
    from the retrieved receipts' structured fields rather than generating
    free text. Used automatically when no OpenAI access is available.
    """

    def answer(self, query: str, context_docs: list[str], receipts: list[dict] = None) -> str:
        if not receipts:
            return "No matching receipts were found for this query."

        rates = [r["rate_per_unit"] for r in receipts]
        amounts = [r["amount"] for r in receipts]
        commodities = {r["commodity"] for r in receipts}
        apmcs = {r["apmc"] for r in receipts}
        receipt_ids = [r["receipt_no"] for r in receipts]

        avg_rate = round(statistics.mean(rates), 2)
        min_rate = min(rates)
        max_rate = max(rates)
        total_amount = round(sum(amounts), 2)

        commodity_str = ", ".join(commodities)
        apmc_str = ", ".join(apmcs)

        return (
            f"Based on {len(receipts)} matching receipt(s) for {commodity_str} "
            f"at {apmc_str}: average rate was Rs. {avg_rate} per unit "
            f"(range: Rs. {min_rate}-{max_rate}), total transaction value "
            f"Rs. {total_amount}. "
            f"Source receipts: {', '.join(receipt_ids[:5])}"
            f"{' ...' if len(receipt_ids) > 5 else ''}."
        )


def get_llm_provider():
    """Return OpenAI provider if configured, else the offline template fallback."""
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("No OPENAI_API_KEY set")
        return OpenAIProvider()
    except Exception as e:
        print(f"[llm_provider] Falling back to offline TemplateProvider ({e}).")
        return TemplateProvider()
