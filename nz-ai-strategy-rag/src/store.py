"""Vector store: Chroma + Bedrock Titan embeddings.

We compute embeddings ourselves via Bedrock (not Chroma's default local model)
so the ENTIRE pipeline runs on AWS Bedrock — one IAM boundary, no data leaving
the account. This is the 'regulated-friendly, stay-on-AWS' story.
"""
import json
import boto3
import chromadb
from config import (AWS_REGION, EMBED_MODEL_ID, CHROMA_DIR, COLLECTION)

_bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def embed(text):
    """Titan Text Embeddings v2 -> 1024-dim vector."""
    resp = _bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text}),
    )
    return json.loads(resp["body"].read())["embedding"]


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # cosine space so RELEVANCE_THRESHOLD in config is meaningful
    return client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def index_records(records):
    col = get_collection()
    embeddings = [embed(r["text"]) for r in records]
    col.upsert(
        ids=[r["id"] for r in records],
        embeddings=embeddings,
        documents=[r["text"] for r in records],
        metadatas=[{"source": r["source"], "page": r["page"]} for r in records],
    )
    return col.count()


def query(question, top_k):
    col = get_collection()
    q_emb = embed(question)
    res = col.query(query_embeddings=[q_emb], n_results=top_k)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"text": doc, "meta": meta, "distance": dist})
    return hits
