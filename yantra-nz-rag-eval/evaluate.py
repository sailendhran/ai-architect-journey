"""
evaluate.py — DeepEval evaluation for the NZ-govt-AI RAG, using a LOCAL Ollama judge.

Two modes, both running entirely on the Spark (no data leaves the box):

  1. Reference-free (no answer key needed):
       - FaithfulnessMetric    : does the answer stay grounded in retrieved context
                                 (i.e. no hallucination beyond the sources)?
       - AnswerRelevancyMetric : does the answer actually address the question?

  2. Golden set (requires expected answers in goldens.yaml):
       - ContextualRecallMetric    : did retrieval fetch what's needed for the answer?
       - ContextualPrecisionMetric : is the relevant context ranked above noise?

The judge model is your local Granite (or any Ollama model). Note: LLM-as-judge is
itself a model, so treat scores as directional signal, not ground truth — which is
exactly why the golden set matters as you scale.

Usage in the notebook:

    from evaluate import RAGEvaluator
    ev = RAGEvaluator(trag, judge_model="granite4.1:30b")
    ev.run_reference_free()        # uses questions from goldens.yaml (answers optional)
    ev.run_with_goldens()          # uses questions + expected_output
    ev.save_results("eval_results.json")
"""
from __future__ import annotations

import json
import os
import datetime

import yaml

from deepeval.models import OllamaModel
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase


def load_goldens(path: str = "goldens.yaml") -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Create it from goldens.example.yaml.")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("goldens", [])


class RAGEvaluator:
    def __init__(self, traced_rag, judge_model: str = "granite4.1:30b",
                 base_url: str = "http://localhost:11434",
                 goldens_path: str = "goldens.yaml", threshold: float = 0.7):
        self.rag = traced_rag
        self.goldens_path = goldens_path
        self.threshold = threshold
        # Local judge — temperature 0 for stable scoring.
        self.judge = OllamaModel(model=judge_model, base_url=base_url, temperature=0)
        self.results: list[dict] = []

    # -- build a DeepEval test case by actually running the RAG -----------
    def _make_test_case(self, question: str, expected: str | None = None) -> LLMTestCase:
        res = self.rag.ask(question, verbose=False)
        return LLMTestCase(
            input=question,
            actual_output=res.answer,
            expected_output=expected,
            retrieval_context=res.contexts,
        )

    def _score(self, tc: LLMTestCase, metrics: list) -> dict:
        row = {"input": tc.input, "actual_output": tc.actual_output, "scores": {}}
        for m in metrics:
            try:
                m.measure(tc)
                row["scores"][m.__class__.__name__] = {
                    "score": round(float(m.score), 3),
                    "reason": getattr(m, "reason", None),
                    "passed": bool(m.score is not None and m.score >= self.threshold),
                }
            except Exception as e:
                row["scores"][m.__class__.__name__] = {"error": str(e)}
        return row

    # -- mode 1: reference-free ------------------------------------------
    def run_reference_free(self) -> list[dict]:
        goldens = load_goldens(self.goldens_path)
        metrics = [
            FaithfulnessMetric(model=self.judge, threshold=self.threshold),
            AnswerRelevancyMetric(model=self.judge, threshold=self.threshold),
        ]
        print(f"Reference-free eval over {len(goldens)} questions "
              f"(judge: {self.judge.model_name if hasattr(self.judge,'model_name') else 'ollama'})")
        out = []
        for g in goldens:
            print(f"  · {g['question'][:70]}…")
            tc = self._make_test_case(g["question"])
            out.append(self._score(tc, metrics))
        self.results.extend(out)
        self._print_summary(out)
        return out

    # -- mode 2: golden set ----------------------------------------------
    def run_with_goldens(self) -> list[dict]:
        goldens = [g for g in load_goldens(self.goldens_path) if g.get("expected_output")]
        if not goldens:
            print("No goldens with expected_output found — fill them in goldens.yaml first.")
            return []
        metrics = [
            ContextualRecallMetric(model=self.judge, threshold=self.threshold),
            ContextualPrecisionMetric(model=self.judge, threshold=self.threshold),
        ]
        print(f"Golden-set retrieval eval over {len(goldens)} questions")
        out = []
        for g in goldens:
            print(f"  · {g['question'][:70]}…")
            tc = self._make_test_case(g["question"], expected=g["expected_output"])
            out.append(self._score(tc, metrics))
        self.results.extend(out)
        self._print_summary(out)
        return out

    # -- reporting --------------------------------------------------------
    @staticmethod
    def _print_summary(rows: list[dict]):
        agg: dict[str, list[float]] = {}
        for r in rows:
            for metric, v in r["scores"].items():
                if "score" in v:
                    agg.setdefault(metric, []).append(v["score"])
        print("\n  Summary (mean score):")
        for metric, vals in agg.items():
            print(f"    {metric:28s} {sum(vals)/len(vals):.3f}  (n={len(vals)})")
        print()

    def save_results(self, path: str = "eval_results.json"):
        payload = {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "threshold": self.threshold,
            "results": self.results,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved {len(self.results)} eval rows to {path}")
        return path
