"""Eval harness — the answer to 'confidence in WHAT?'

We measure two things a regulated client actually cares about:
  - REFUSAL ACCURACY: when the docs can't answer, does it correctly refuse?
  - CITATION PRESENCE: does every answered question cite a source?

This is a starter harness using rule-based checks. The next step (documented
in the README) is to add LLM-as-judge faithfulness scoring — but ship the
measurable version first.

The eval SET is the real deliverable: note it includes 'should_refuse: True'
cases. Most naive RAG demos only test questions the docs CAN answer, which
hides the most dangerous failure mode — confident wrong answers.
"""
import json
from rag import answer

# Edit these to match the questions your real docs can/can't answer.
EVAL_SET = [
    {"q": "What is the policy on remote work?", "should_refuse": False},
    {"q": "How many annual leave days do employees get?", "should_refuse": False},
    {"q": "What is the company's stock price target for 2027?", "should_refuse": True},
    {"q": "Who won the 2024 cricket world cup?", "should_refuse": True},
]


def run():
    results = []
    for case in EVAL_SET:
        out = answer(case["q"])
        refused = out["refused"] or "don't have enough information" in out["answer"].lower()
        refusal_correct = (refused == case["should_refuse"])
        cited = bool(out["citations"]) or refused
        results.append({
            "q": case["q"],
            "should_refuse": case["should_refuse"],
            "refused": refused,
            "refusal_correct": refusal_correct,
            "has_citation_or_refused": cited,
        })

    n = len(results)
    refusal_acc = sum(r["refusal_correct"] for r in results) / n
    cite_rate = sum(r["has_citation_or_refused"] for r in results) / n

    print(json.dumps(results, indent=2))
    print(f"\nRefusal accuracy: {refusal_acc:.0%}")
    print(f"Citation/refusal coverage: {cite_rate:.0%}")
    return results


if __name__ == "__main__":
    run()
