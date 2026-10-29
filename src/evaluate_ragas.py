"""
LLM-judged faithfulness evaluation with RAGAS.

Answers each question with the OpenAI generation path, then RAGAS checks every
claim in the answer against the context that was given to the LLM
(faithfulness = share of claims supported by the context, 0-1).
Questions whose filters match no receipts are answered without the LLM; they
are reported separately as abstentions, not scored.

Requires OPENAI_API_KEY and:
    pip install -r requirements-demo.txt -r requirements-eval.txt
    python -m src.evaluate_ragas        # writes results/ragas_faithfulness.json

Models: generation uses OPENAI_MODEL (default here: gpt-4o-mini); the judge
uses RAGAS_JUDGE_MODEL (default: gpt-4o-mini).
"""
import json
import os
import pathlib
import sys

QUESTIONS = [
    # aggregate questions: answered from statistics over all filtered receipts
    "What was the average onion rate at Manchar Krishi Utpanna Bazar Samiti?",
    "What is the total transaction value for potato at Khed APMC?",
    "What was the average tomato rate at Junnar APMC in 2023?",
    "What was the highest cauliflower rate at Narayangaon APMC?",
    "How many eggplant trades happened at Khed APMC?",
    "What was the average potato rate in March?",
    # lookup questions: answered from the top retrieved receipts
    "Show me tomato trades in April",
    "Which traders bought onion at Junnar APMC in May?",
    "What was the onion rate at Narayangaon APMC in June 2023?",
    "Show me cauliflower receipts at Manchar paid by bank transfer",
    "Trades handled by Shivshakti Traders",
    # no matching receipts: the pipeline should abstain
    "What was the average onion rate at Manchar in 2022?",
    "Show me potato trades at Junnar in 2021",
]


def main(output="results/ragas_faithfulness.json"):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is required for the RAGAS evaluation.")
    os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")
    judge_model = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-4o-mini")

    import ragas
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    from src.llm_provider import OpenAIProvider
    from src.pipeline import MarketIntelligencePipeline

    pipeline = MarketIntelligencePipeline()
    if not isinstance(pipeline.llm, OpenAIProvider):
        sys.exit("The OpenAI generation path is not active; check OPENAI_API_KEY and the openai package.")

    samples, answered, abstained = [], [], []
    for question in QUESTIONS:
        result = pipeline.ask(question, include_context=True)
        if result["num_matched"] == 0:
            abstained.append({"question": question, "answer": result["answer"]})
            continue
        samples.append({
            "user_input": question,
            "retrieved_contexts": result["contexts"],
            "response": result["answer"],
        })
        answered.append({"question": question, "answer": result["answer"],
                         "num_matched": result["num_matched"]})

    judge = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    scores = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=[Faithfulness()],
        llm=judge,
        show_progress=False,
    ).to_pandas()["faithfulness"].tolist()

    for item, value in zip(answered, scores):
        item["faithfulness"] = round(float(value), 3)
    valid = [s for s in scores if s == s]  # drop NaN
    report = {
        "ragas_version": ragas.__version__,
        "generator_model": os.environ["OPENAI_MODEL"],
        "judge_model": judge_model,
        "questions": len(QUESTIONS),
        "scored": len(valid),
        "mean_faithfulness": round(sum(valid) / len(valid), 3) if valid else None,
        "abstained": abstained,
        "answers": answered,
    }

    path = pathlib.Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Mean faithfulness: {report['mean_faithfulness']} over {report['scored']} answers "
          f"(generator {report['generator_model']}, judge {judge_model})")
    print(f"Abstained on {len(abstained)} question(s) with no matching receipts.")
    for item in answered:
        print(f"  {item['faithfulness']:<6} {item['question']}")
    print(f"Saved to {path}")
    return report


if __name__ == "__main__":
    main()
