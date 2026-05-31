"""Ingestion: PDF -> chunks with source metadata.

Design note (the part that matters for regulated docs):
We chunk with overlap and carry source filename + page + chunk index as
metadata on every chunk. That metadata is what makes CITATIONS possible —
every answer can point back to 'policy_X.pdf, p.4'. No metadata, no audit trail,
no client trust.
"""
import os
import glob
from pypdf import PdfReader
from config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


def load_pdf_pages(path):
    """Return list of (page_number, text) — page-aware so citations name a page."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text, size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    """Sliding window with overlap. Overlap keeps clauses that straddle a
    boundary intact in at least one chunk."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def ingest_folder(folder):
    """Yield dicts: {id, text, source, page, chunk_index}."""
    records = []
    for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        source = os.path.basename(path)
        for page_no, page_text in load_pdf_pages(path):
            for ci, chunk in enumerate(chunk_text(page_text)):
                records.append({
                    "id": f"{source}::p{page_no}::c{ci}",
                    "text": chunk,
                    "source": source,
                    "page": page_no,
                    "chunk_index": ci,
                })
    return records


if __name__ == "__main__":
    from config import DOCS_DIR
    recs = ingest_folder(DOCS_DIR)
    print(f"Produced {len(recs)} chunks from data/docs")
    if recs:
        print("Sample chunk:", recs[0]["id"])
