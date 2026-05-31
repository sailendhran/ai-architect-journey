"""The grounded answer layer — this is the whole point.

THREE grounding mechanisms, all visible here:

1. RELEVANCE THRESHOLD: chunks above RELEVANCE_THRESHOLD cosine distance are
   discarded. If nothing survives, we REFUSE before even calling the model.
   A client that 'cannot afford wrong answers' needs the system to say
   "I don't know" — that is a feature, not a failure.

2. CONTEXT-ONLY PROMPT: the system prompt forbids outside knowledge and
   instructs refusal when the context doesn't contain the answer.

3. CITATIONS: every chunk passed in is numbered; the model must cite [n],
   and we map [n] back to source+page for an audit trail.
"""
import json
import boto3
from config import (AWS_REGION, GEN_MODEL_ID, TOP_K, RELEVANCE_THRESHOLD)
from store import query

_bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

SYSTEM_PROMPT = """You answer questions about internal policy documents.

RULES — these are absolute:
- Use ONLY the numbered context passages provided. Do not use any outside knowledge.
- If the context does not contain the answer, reply exactly:
  "I don't have enough information in the provided documents to answer that."
- Cite every factual claim with the passage number(s) it came from, like [1] or [2,3].
- Do not guess, infer beyond the text, or fill gaps."""


def build_context(hits):
    """Number the surviving passages and format for the prompt."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        src = f'{h["meta"]["source"]}, p.{h["meta"]["page"]}'
        blocks.append(f"[{i}] (source: {src})\n{h['text']}")
    return "\n\n".join(blocks)


def answer(question):
    hits = query(question, top_k=TOP_K)

    # Mechanism 1: threshold-based refusal BEFORE calling the model
    relevant = [h for h in hits if h["distance"] <= RELEVANCE_THRESHOLD]
    if not relevant:
        return {
            "answer": "I don't have enough information in the provided documents to answer that.",
            "citations": [],
            "refused": True,
        }

    context = build_context(relevant)
    user_msg = f"Context passages:\n\n{context}\n\nQuestion: {question}"

    resp = _bedrock.invoke_model(
        modelId=GEN_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0,  # deterministic — no creative drift on policy answers
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_msg}],
        }),
    )
    body = json.loads(resp["body"].read())
    text = body["content"][0]["text"]

    citations = [
        {"n": i + 1, "source": h["meta"]["source"], "page": h["meta"]["page"]}
        for i, h in enumerate(relevant)
    ]
    return {"answer": text, "citations": citations, "refused": False}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is the policy on remote work?"
    result = answer(q)
    print("\nQ:", q)
    print("\nA:", result["answer"])
    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            print(f'  [{c["n"]}] {c["source"]}, p.{c["page"]}')
