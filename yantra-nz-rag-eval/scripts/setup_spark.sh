#!/usr/bin/env bash
# =============================================================================
# setup_spark.sh — prepare an NVIDIA DGX Spark (GB10, DGX OS / Ubuntu ARM64)
# to run the Yantra NZ multimodal RAG demo locally with Ollama.
#
# Run:  bash scripts/setup_spark.sh
#
# This installs Ollama, pulls the Granite generation + vision models, and
# creates a Python venv with the RAG dependencies. Everything runs on-device —
# no Replicate, no watsonx, no external API. Data never leaves the box.
# =============================================================================
set -euo pipefail

echo "==> 0. Sanity checks"
uname -m   # expect aarch64 on the Spark
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi || true; fi

echo "==> 1. Install Ollama (if not present)"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "    Ollama already installed: $(ollama --version || true)"
fi

echo "==> 2. Start the Ollama server (background) if not already running"
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  sleep 5
fi

echo "==> 3. Pull models"
# Generation model — Granite 4.1 30B (fits comfortably in the Spark's 128GB unified memory).
ollama pull granite4.1:30b || echo "    (set GEN_MODEL manually if tag differs)"
# Vision model for image captioning.
ollama pull granite3.2-vision || ollama pull llava:7b || echo "    (set VISION_MODEL manually if tag differs)"

echo "==> 4. Python environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install \
    docling \
    langchain-core \
    langchain-community \
    langchain-ollama \
    langchain-huggingface \
    langchain-chroma chromadb \
    sentence-transformers \
    transformers \
    pillow \
    reportlab matplotlib \
    deepeval langfuse python-dotenv pyyaml \
    jupyter

cat <<'EOF'

==> Done.
Next:
  source .venv/bin/activate
  jupyter notebook Granite_Multimodal_RAG_Spark.ipynb

If any corpus PDF failed to download (CDN block), open its URL in a browser,
save it into ./corpus/ with the filename shown, then re-run the notebook.

Tip: confirm models with `ollama list`. If your Granite tags differ from the
defaults above, set GEN_MODEL / VISION_MODEL at the top of the notebook.
EOF
