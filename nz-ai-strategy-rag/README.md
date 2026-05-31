# RAG Policy Bot — Grounded Q&A over internal policy documents

A reference RAG pipeline for answering questions over internal policy documents
in a **regulated, cost-constrained** setting where wrong answers are unacceptable.

Built end-to-end on **AWS Bedrock** (Titan embeddings + Claude) with Chroma as the
vector store. The design choices below are the point of this repo — the code is the
proof.

## The problem this solves

> Architect a chatbot over ~10,000 internal policy documents for a cost-constrained
> client that cannot afford wrong answers.

"Cannot afford wrong answers" is the controlling constraint. Everything here serves
**grounding** — answering only from retrieved documents, and refusing when the
documents don't contain the answer.

## Architecture

```
PDFs ──> chunk (overlap) ──> Titan embeddings ──> Chroma (cosine)
                                                      │
                            question ──> embed ──> retrieve top-k
                                                      │
                                          relevance threshold filter
                                          │                       │
                                   nothing survives          chunks survive
                                          │                       │
                                       REFUSE          context-only prompt ──> Claude
                                                                    │
                                                          grounded answer + citations
```

## Three grounding mechanisms

1. **Relevance threshold (`RELEVANCE_THRESHOLD`)** — retrieved chunks above a cosine
   distance cutoff are discarded. If nothing survives, the system refuses *before*
   the model is ever called. Correct refusal is a feature, not a failure.
2. **Context-only system prompt** — the model is instructed to use only the supplied
   passages, refuse when they don't contain the answer, and never use outside knowledge.
   `temperature=0` for deterministic, non-creative policy answers.
3. **Citations** — every passage is numbered and the model must cite `[n]`; each `[n]`
   maps back to `source + page` for an audit trail.

## Why these decisions

- **Why AWS Bedrock end-to-end** — embeddings and generation both run in Bedrock
  (ap-southeast-2), one IAM boundary, no data leaving the account. Fits a regulated
  NZ-enterprise data-residency story.
- **Why overlapping chunks** — policy clauses that straddle a chunk boundary stay
  intact in at least one chunk. (Next step: structure-aware chunking — see below.)
- **Why temperature 0** — policy answers should be reproducible; creativity is a bug here.

## Evaluation — "confidence in *what*?"

A single "95% confident" number is meaningless. This harness measures specific
failure modes on a labelled set that **deliberately includes unanswerable questions**
to test refusal:

- **Refusal accuracy** — when docs can't answer, does it correctly refuse?
- **Citation coverage** — does every answered question cite a source?

Most naive RAG demos only test answerable questions, hiding the most dangerous
failure: confident wrong answers.

## Known gaps / roadmap (honest about what's not done)

- **Structure-aware chunking** — current chunking is sliding-window. Policy docs
  with sections, schedules, and cross-references need structure-aware splitting so
  "Clause 4.2 except where Schedule B overrides" doesn't fragment.
- **Reranking** — add a cross-encoder rerank step after retrieval for precision.
- **Hybrid search** — combine semantic (vector) with keyword (BM25) for exact terms
  like policy numbers and defined terms.
- **LLM-as-judge faithfulness scoring** — extend evals beyond rule-based checks.
- **FinOps model** — cost is per-query inference + vector store, not the one-time
  embedding. Tiered model selection (Haiku for retrieval-confident, Sonnet for hard
  queries) for a cost-constrained client.

## Run it

Pre-Requisites: 
- Python3.12.7 version
- AWS credentials stored / configured for usage of AWS Bedrock

```bash
pip3 install chromadb boto3 pypdf
# put PDFs in data/docs/, AWS creds configured for Bedrock in ap-southeast-2
python3 src/ingest.py          # verify chunking
python3 -c "from src.store import index_records; from src.ingest import ingest_folder; from src.config import DOCS_DIR; print(index_records(ingest_folder(DOCS_DIR)), 'chunks indexed')"
python3 src/rag.py "What is the policy on remote work?"
python3 evals/run_eval.py
```
