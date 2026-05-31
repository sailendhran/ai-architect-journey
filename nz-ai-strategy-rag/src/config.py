"""Central config. One place to change models, region, and retrieval knobs."""

AWS_REGION = "ap-southeast-2"  # Sydney — your live Bedrock region

# Bedrock model IDs
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"   # 1024-dim embeddings
GEN_MODEL_ID = "au.anthropic.claude-sonnet-4-6"

# Chunking. Policy docs need overlap so a clause split across a boundary
# still appears whole in at least one chunk.
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200

# Retrieval
TOP_K = 5                  # chunks pulled per query
RELEVANCE_THRESHOLD = 0.6  # cosine distance cutoff; above this = "not relevant"
                            # this is the dial that powers REFUSAL-on-no-context

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DOCS_DIR = _os.path.join(_ROOT, "data", "docs")
CHROMA_DIR = _os.path.join(_ROOT, "data", "chroma")
COLLECTION = "policy_docs"
