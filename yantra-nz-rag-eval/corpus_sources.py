"""
Yantra NZ — NZ Government AI Corpus
====================================
Source manifest for the multimodal RAG demo.

Each entry is a public New Zealand government document on AI policy / strategy /
guidance. URLs verified live (June 2026). If a download is blocked by a CDN
firewall in your environment, download the PDF manually in a browser and drop it
into ./corpus/ using the given `filename` — the pipeline picks up local files
automatically.

Maintained as the canonical Yantra knowledge corpus on the NZ public-sector AI
landscape — useful beyond this demo for bid positioning and capability briefs.
"""

# Browser-like UA: NZ govt CDNs reject non-browser user agents.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

CORPUS = [
    {
        "id": "nz-ai-strategy",
        "title": "New Zealand's Strategy for Artificial Intelligence — Investing with confidence",
        "publisher": "MBIE",
        "year": 2025,
        "filename": "nz-ai-strategy.pdf",
        "url": "https://www.mbie.govt.nz/assets/new-zealands-strategy-for-artificial-intelligence.pdf",
    },
    {
        "id": "responsible-ai-public-service-genai",
        "title": "Responsible AI Guidance for the Public Service: GenAI (Print)",
        "publisher": "Government Chief Digital Officer (GCDO) / digital.govt.nz",
        "year": 2025,
        "filename": "responsible-ai-public-service-genai.pdf",
        "url": "https://www.digital.govt.nz/assets/Standards-guidance/Technology-and-architecture/Generative-AI/Responsible-AI-Guidance-for-the-Public-Service-GenAI-Print.pdf",
    },
    {
        "id": "responsible-ai-businesses",
        "title": "Responsible AI Guidance for Businesses — Investing with confidence",
        "publisher": "MBIE",
        "year": 2025,
        "filename": "responsible-ai-businesses.pdf",
        "url": "https://www.mbie.govt.nz/assets/responsible-ai-guidance-for-businesses.pdf",
    },
    # --- Add more as you collect them. Examples linked from digital.govt.nz: ---
    # Public Service AI Framework A3 (PDF ~241KB)
    # Cabinet Paper — Approach to work on AI — MBIE (PDF ~155KB)
    # A3 Summary — Responsible AI Guidance for the Public Service: GenAI (PDF ~215KB)
    # Drop their PDFs into ./corpus/ and add entries here with a local filename.
]
