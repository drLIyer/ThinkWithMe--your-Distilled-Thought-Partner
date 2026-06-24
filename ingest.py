"""
Ingest Lenny's Newsletter & Podcast data into a FAISS vector index.

Usage:
    python ingest.py

Outputs:
    index.faiss   — FAISS flat inner-product index (normalized = cosine sim)
    chunks.pkl    — list of chunk dicts with text + metadata
"""

import json
import pickle
from pathlib import Path

import frontmatter
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("/Users/liyer_1/Downloads/lennys-newsletterpodcastdata-all")
INDEX_PATH = Path("index.faiss")
CHUNKS_PATH = Path("chunks.pkl")

CHUNK_WORDS = 400
OVERLAP_WORDS = 50
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def word_chunks(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    chunks = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def load_documents() -> list[dict]:
    with open(DATA_DIR / "01-start-here/index.json") as f:
        index = json.load(f)

    all_chunks = []

    for item in index.get("newsletters", []):
        filepath = DATA_DIR / item["filename"]
        post = frontmatter.load(filepath)
        for i, chunk in enumerate(word_chunks(post.content)):
            all_chunks.append(
                {
                    "text": chunk,
                    "title": item["title"],
                    "type": "newsletter",
                    "date": item.get("date", ""),
                    "tags": item.get("tags", []),
                    "filename": item["filename"],
                    "chunk_idx": i,
                }
            )

    for item in index.get("podcasts", []):
        filepath = DATA_DIR / item["filename"]
        post = frontmatter.load(filepath)
        for i, chunk in enumerate(word_chunks(post.content)):
            all_chunks.append(
                {
                    "text": chunk,
                    "title": item["title"],
                    "type": "podcast",
                    "date": item.get("date", ""),
                    "tags": item.get("tags", []),
                    "filename": item["filename"],
                    "guest": item.get("guest", ""),
                    "chunk_idx": i,
                }
            )

    return all_chunks


def main():
    print("Loading documents...")
    chunks = load_documents()
    print(f"  {len(chunks)} chunks from {len(set(c['filename'] for c in chunks))} documents")

    print(f"Encoding with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Done. Saved {faiss_index.ntotal} vectors to {INDEX_PATH}")


if __name__ == "__main__":
    main()
