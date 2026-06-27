"""
traced_rag.py — RAG query path instrumented with Langfuse tracing.

Wraps the retrieve→generate flow so every call produces a trace with nested
spans: the retrieval step (query, returned docs, scores) and the generation step
(prompt, model, output, latency, token usage where Ollama reports it). View these
in the self-hosted Langfuse dashboard (see docker-compose.langfuse.yml).

Everything runs locally. Langfuse self-hosted means traces never leave the Spark.

Usage in the notebook:

    from traced_rag import TracedRAG
    trag = TracedRAG(retriever, model, gen_model_name=GEN_MODEL)
    result = trag.ask("What principles underpin NZ's responsible AI approach?")
    # result.answer, result.contexts, result.sources

If Langfuse isn't configured (no env keys), tracing silently no-ops and the RAG
still works — so the notebook never breaks just because the dashboard is down.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

# Langfuse is optional at runtime. If not installed/configured, we degrade gracefully.
# v3/v4 expose get_client() + context-manager spans; we detect which API is present.
try:
    from langfuse import get_client  # v3 / v4
    _LF_AVAILABLE = True
    _LF_V3 = True
except Exception:  # pragma: no cover
    try:
        from langfuse import Langfuse  # legacy v2
        _LF_AVAILABLE = True
        _LF_V3 = False
    except Exception:
        _LF_AVAILABLE = False
        _LF_V3 = False


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    trace_id: str | None = None


RAG_PROMPT_TMPL = (
    "You are an assistant answering questions about New Zealand government AI "
    "policy. Use ONLY the context below. If the answer is not in the context, "
    "say so.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


class TracedRAG:
    def __init__(self, retriever, model, gen_model_name: str = "ollama",
                 prompt_template: str = RAG_PROMPT_TMPL, k: int = 5):
        self.retriever = retriever
        self.model = model
        self.gen_model_name = gen_model_name
        self.prompt_template = prompt_template
        self.k = k
        self.lf = None
        if _LF_AVAILABLE and os.environ.get("LANGFUSE_PUBLIC_KEY"):
            try:
                # Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from env.
                self.lf = get_client() if _LF_V3 else Langfuse()
            except Exception as e:
                print(f"[traced_rag] Langfuse init failed, tracing disabled: {e}")

    # -- internal helpers -------------------------------------------------
    def _retrieve(self, question: str):
        t0 = time.perf_counter()
        docs = self.retriever.invoke(question)
        dt = (time.perf_counter() - t0) * 1000
        contexts = [d.page_content for d in docs]
        sources = sorted({d.metadata.get("source", "?") for d in docs})
        return docs, contexts, sources, dt

    def _generate(self, question: str, contexts: list[str]):
        context = "\n\n".join(contexts)
        prompt = self.prompt_template.format(context=context, question=question)
        t0 = time.perf_counter()
        resp = self.model.invoke(prompt)
        dt = (time.perf_counter() - t0) * 1000
        answer = getattr(resp, "content", str(resp))
        # Ollama usage metadata, when present on the LangChain response.
        usage = {}
        meta = getattr(resp, "response_metadata", {}) or {}
        for key in ("prompt_eval_count", "eval_count", "total_duration"):
            if key in meta:
                usage[key] = meta[key]
        return answer, prompt, usage, dt

    # -- public API -------------------------------------------------------
    def ask(self, question: str, verbose: bool = True) -> RAGResult:
        if self.lf and _LF_V3:
            return self._ask_v3(question, verbose)
        if self.lf:
            return self._ask_v2(question, verbose)
        return self._ask_untraced(question, verbose)

    # -- v3 / v4: OpenTelemetry context-manager API ----------------------
    def _ask_v3(self, question: str, verbose: bool) -> RAGResult:
        with self.lf.start_as_current_observation(
            as_type="span", name="rag-query", input={"question": question}
        ) as root:
            # retrieval span
            with self.lf.start_as_current_observation(
                as_type="span", name="retrieval", input={"question": question, "k": self.k}
            ) as r_span:
                docs, contexts, sources, r_ms = self._retrieve(question)
                r_span.update(output={"sources": sources, "n_docs": len(docs)},
                              metadata={"retrieval_ms": round(r_ms, 1)})

            # generation span
            with self.lf.start_as_current_observation(
                as_type="generation", name="generation", model=self.gen_model_name
            ) as g_obs:
                answer, prompt, usage, g_ms = self._generate(question, contexts)
                g_obs.update(input=prompt, output=answer,
                             metadata={"generation_ms": round(g_ms, 1)})
                if usage:
                    try:
                        g_obs.update(usage_details={
                            "input": usage.get("prompt_eval_count"),
                            "output": usage.get("eval_count"),
                        })
                    except Exception:
                        pass

            root.update(output={"answer": answer, "sources": sources})
            try:
                trace_id = self.lf.get_current_trace_id()
            except Exception:
                trace_id = None

        try:
            self.lf.flush()
        except Exception:
            pass

        self._maybe_print(verbose, question, answer, sources, r_ms, g_ms, trace_id)
        return RAGResult(question=question, answer=answer, contexts=contexts,
                         sources=sources, retrieval_ms=r_ms, generation_ms=g_ms,
                         trace_id=trace_id)

    # -- v2: legacy imperative API (fallback) ----------------------------
    def _ask_v2(self, question: str, verbose: bool) -> RAGResult:
        trace = self.lf.trace(name="rag-query", input={"question": question},
                              tags=["yantra-nz-rag"])
        docs, contexts, sources, r_ms = self._retrieve(question)
        trace.span(name="retrieval", input={"question": question, "k": self.k},
                   output={"sources": sources, "n_docs": len(docs)},
                   metadata={"retrieval_ms": round(r_ms, 1)}).end()
        answer, prompt, usage, g_ms = self._generate(question, contexts)
        gen = trace.generation(name="generation", model=self.gen_model_name,
                               input=prompt, output=answer,
                               metadata={"generation_ms": round(g_ms, 1)})
        if usage:
            gen.update(usage={"input": usage.get("prompt_eval_count"),
                              "output": usage.get("eval_count")})
        gen.end()
        trace.update(output={"answer": answer, "sources": sources})
        trace_id = getattr(trace, "id", None)
        try:
            self.lf.flush()
        except Exception:
            pass
        self._maybe_print(verbose, question, answer, sources, r_ms, g_ms, trace_id)
        return RAGResult(question=question, answer=answer, contexts=contexts,
                         sources=sources, retrieval_ms=r_ms, generation_ms=g_ms,
                         trace_id=trace_id)

    # -- no tracing ------------------------------------------------------
    def _ask_untraced(self, question: str, verbose: bool) -> RAGResult:
        docs, contexts, sources, r_ms = self._retrieve(question)
        answer, prompt, usage, g_ms = self._generate(question, contexts)
        self._maybe_print(verbose, question, answer, sources, r_ms, g_ms, None)
        return RAGResult(question=question, answer=answer, contexts=contexts,
                         sources=sources, retrieval_ms=r_ms, generation_ms=g_ms,
                         trace_id=None)

    @staticmethod
    def _maybe_print(verbose, question, answer, sources, r_ms, g_ms, trace_id):
        if not verbose:
            return
        print("Q:", question, "\n")
        print("A:", answer, "\n")
        print("Sources:", sources)
        print(f"(retrieval {r_ms:.0f} ms · generation {g_ms:.0f} ms"
              + (f" · trace {trace_id}" if trace_id else "") + ")")
