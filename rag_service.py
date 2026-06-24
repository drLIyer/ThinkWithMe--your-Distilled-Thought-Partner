"""
Distill RAG Service — FastAPI microservice for FAISS retrieval

Exposes:
  GET  /health
  POST /retrieve
  POST /retrieve/with-context

Run:
  .venv312/bin/python3 -m uvicorn rag_service:app --host 127.0.0.1 --port 8083
"""

import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

INDEX_PATH  = Path("index.faiss")
CHUNKS_PATH = Path("chunks.pkl")
DATA_DIR    = Path(os.environ.get("DATA_DIR", "lennys-data"))
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 8

# ── Module-level state (loaded once at startup) ───────────────────────────────

_index:       Optional[faiss.Index]        = None
_chunks:      list                         = []
_embed_model: Optional[SentenceTransformer] = None


# ── Resource loading ──────────────────────────────────────────────────────────

def load_resources():
    global _index, _chunks, _embed_model
    print("RAG service: loading FAISS index...")
    _index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        _chunks = pickle.load(f)
    print(f"RAG service: loading embedding model ({EMBED_MODEL})...")
    _embed_model = SentenceTransformer(EMBED_MODEL)
    print(f"RAG service ready — {len(_chunks)} chunks")


# ── RAG functions ─────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    emb = _embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(emb)
    scores, indices = _index.search(emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = dict(_chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def compute_confidence(results: list[dict]) -> dict:
    if not results:
        return {"level": "low", "label": "No sources matched", "color": "#e74c3c"}
    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores)
    top = scores[0]
    if top >= 0.62 and avg >= 0.59:
        return {"level": "high",   "label": "High corpus coverage",    "color": "#27ae60"}
    elif top >= 0.55:
        return {"level": "medium", "label": "Moderate corpus coverage", "color": "#e67e22"}
    else:
        return {"level": "low",
                "label": "Limited corpus coverage — draws on general knowledge",
                "color": "#e74c3c"}


def _type_icon(doc_type: str) -> str:
    icons = {"podcast": "🎙️", "video": "🎬", "book": "📖", "newsletter": "📰"}
    return icons.get(doc_type, "📄")


def format_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        doc_type = r.get("type", "article").upper()
        guest = r.get("guest", "")
        guest_part = f" (guest: {guest})" if guest else ""
        header = f"[{doc_type}] {r['title']}{guest_part} ({r['date']})"
        parts.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def source_url(s: dict) -> str:
    # Prefer an explicit URL stored at ingest time
    return s.get("url", "")


def format_sources_md(sources: list[dict]) -> str:
    lines = []
    for s in sources:
        doc_type = s.get("type", "article")
        icon  = _type_icon(doc_type)
        kind  = doc_type.capitalize()
        label = s.get("guest") or s["title"]
        url   = source_url(s)
        score = f'{s["score"] * 100:.0f}% match' if s.get("score") else ""
        link  = f" · [Open ↗]({url})" if url else ""
        lines.append(f"{icon} **{kind}** · {label} · *{s['date']}* · `{score}`{link}")

        if s.get("text"):
            snippet = s["text"][:300].rsplit(" ", 1)[0] + " …"
            lines.append(f"> {snippet}\n")
    return "\n".join(lines)


# ── Pydantic models ───────────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class RetrieveWithContextRequest(BaseModel):
    query: str
    history: list[dict] = []


class ChunkResult(BaseModel):
    text: str
    title: str
    type: str
    date: str
    guest: str = ""
    filename: str = ""
    score: float


class ConfidenceResult(BaseModel):
    level: str
    label: str
    color: str


class RetrieveResponse(BaseModel):
    results: list[ChunkResult]
    confidence: ConfidenceResult


class RetrieveWithContextResponse(BaseModel):
    context_text: str
    sources_md: str
    confidence: ConfidenceResult
    results: list[ChunkResult]


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Distill RAG Service", version="1.0.0")


@app.on_event("startup")
def on_startup():
    load_resources()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(_chunks),
        "index_loaded": _index is not None,
        "embed_model_loaded": _embed_model is not None,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed_endpoint(req: EmbedRequest):
    embs = _embed_model.encode(req.texts, convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embs)
    return EmbedResponse(embeddings=embs.tolist())


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest):
    results = retrieve(req.query, req.top_k)
    confidence = compute_confidence(results)
    return RetrieveResponse(
        results=[ChunkResult(
            text=r.get("text", ""), title=r.get("title", ""), type=r.get("type", ""),
            date=r.get("date", ""), guest=r.get("guest", ""),
            filename=r.get("filename", ""), score=r["score"],
        ) for r in results],
        confidence=ConfidenceResult(**confidence),
    )


@app.post("/retrieve/with-context", response_model=RetrieveWithContextResponse)
def retrieve_with_context_endpoint(req: RetrieveWithContextRequest):
    # Prepend last user turn for better follow-up question retrieval
    retrieval_query = req.query
    if req.history:
        prev_user = next(
            (m["content"] for m in reversed(req.history)
             if m.get("role") == "user" and isinstance(m.get("content"), str)),
            None,
        )
        if prev_user:
            retrieval_query = f"{prev_user} {req.query}"

    results = retrieve(retrieval_query)
    confidence = compute_confidence(results)
    context_text = format_context(results)
    sources = [
        {"type": r["type"], "title": r["title"], "guest": r.get("guest", ""),
         "date": r["date"], "text": r["text"],
         "filename": r.get("filename", ""), "score": r["score"]}
        for r in results
    ]
    sources_md = format_sources_md(sources)

    return RetrieveWithContextResponse(
        context_text=context_text,
        sources_md=sources_md,
        confidence=ConfidenceResult(**confidence),
        results=[ChunkResult(
            text=r.get("text", ""), title=r.get("title", ""), type=r.get("type", ""),
            date=r.get("date", ""), guest=r.get("guest", ""),
            filename=r.get("filename", ""), score=r["score"],
        ) for r in results],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8083, log_level="info")
