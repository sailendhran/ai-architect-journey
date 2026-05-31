# AI Architect Journey

Public log of building toward AI architecture consulting for NZ enterprises and government.

**Background:** 18+ years in cloud architecture, SRE, and platform engineering across AWS, GCP, and Azure. AWS Solutions Architect — Professional, CKA, Terraform Associate. Currently consulting at ASB Bank (Wellington, NZ).

**Why this repo:** Working in public as I layer agentic AI, RAG, and AI governance expertise on top of an enterprise cloud foundation. Six-month build. Real artifacts, not summaries.

## Active work
- Week 1: Production RAG on AWS Bedrock (Sydney region) :white_check_mark:

## Roadmap
- Weeks 1–2: Production-grade RAG with hybrid search, re-ranking, ADRs
- Weeks 3–4: Evaluation framework, guardrails, AI observability and cost tracking
- Weeks 5–6: Agentic systems — tool use, planning, multi-agent orchestration
- Months 3–6: Signature project for regulated NZ enterprise + public case study

## Projects

### [nz-ai-strategy-rag](./nz-ai-strategy-rag) — Grounded RAG over NZ's AI strategy & guidance
A production-pattern RAG system answering questions over New Zealand's
*Strategy for Artificial Intelligence* and *Responsible AI Guidance for
Businesses*. Built end-to-end on AWS Bedrock (ap-southeast-2).

- Semantic retrieval with a relevance threshold tuned against real embedding distances
- Grounded generation — answers only from retrieved context, with page-level citations across multiple source documents
- Refuses when the documents don't contain the answer (the behaviour that matters for regulated deployments)
- Inference scoped to the AU profile for data residency; model right-sized for cost

See the [project README](./nz-ai-strategy-rag/README.md) for the architecture decisions.


## Contact
Wellington, NZ · [LinkedIn](https://www.linkedin.com/in/sailendhran/) · [yantranz.com](https://yantranz.com/)
