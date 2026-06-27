#!/usr/bin/env python3
"""
fetch_corpus.py — download the NZ govt AI corpus into ./corpus/

Run from the project root:  python scripts/fetch_corpus.py

Behaviour:
  - For each source, if ./corpus/<filename> already exists, skip (idempotent).
  - Otherwise download with a browser User-Agent.
  - If a download fails (e.g. CDN firewall / 403), print a clear instruction to
    download manually in a browser and place the file in ./corpus/, then continue.

This keeps the pipeline runnable even when some links are blocked in your network.
"""
import os
import sys
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from corpus_sources import CORPUS, USER_AGENT  # noqa: E402

CORPUS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus"
)


def fetch_one(src):
    dest = os.path.join(CORPUS_DIR, src["filename"])
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f"  ✓ already present: {src['filename']}")
        return True
    req = urllib.request.Request(src["url"], headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if not data.startswith(b"%PDF"):
            raise ValueError("response is not a PDF (got HTML/redirect)")
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  ✓ downloaded: {src['filename']} ({len(data)//1024} KB)")
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as e:
        print(f"  ✗ could not download {src['filename']}: {e}")
        print(f"      → open this in a browser and save into ./corpus/{src['filename']}:")
        print(f"        {src['url']}")
        return False


def main():
    os.makedirs(CORPUS_DIR, exist_ok=True)
    print(f"Fetching {len(CORPUS)} NZ govt AI documents into {CORPUS_DIR}\n")
    ok = sum(fetch_one(s) for s in CORPUS)
    print(f"\n{ok}/{len(CORPUS)} documents ready.")
    if ok == 0:
        print("No documents available yet — add PDFs to ./corpus/ manually, then re-run the notebook.")
    have = [f for f in os.listdir(CORPUS_DIR) if f.lower().endswith(".pdf")]
    print(f"PDFs currently in corpus/: {len(have)}")


if __name__ == "__main__":
    main()
