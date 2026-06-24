"""
Ingest any corpus of markdown files into a FAISS vector index.

Usage:
    python ingest.py

Reads all .md files under DATA_DIR recursively. Each file should have YAML
frontmatter with at minimum a `title` and `date` field. Optional fields:
  type    — any string label, e.g. "newsletter", "podcast", "article", "book"
             defaults to "article" if omitted
  url     — canonical URL for this piece of content
  guest   — for podcast/interview content
  tags    — list of strings

Outputs:
    index.faiss   — FAISS flat inner-product index (normalized = cosine sim)
    chunks.pkl    — list of chunk dicts with text + metadata
"""

import json
import os
import pickle
from pathlib import Path

import frontmatter
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR    = Path(os.environ.get("DATA_DIR", "data"))
INDEX_PATH  = Path("index.faiss")
CHUNKS_PATH = Path("chunks.pkl")

CHUNK_WORDS  = 400
OVERLAP_WORDS = 50
EMBED_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE   = 64


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
    md_files = sorted(DATA_DIR.rglob("*.md"))
    if not md_files:
        raise SystemExit(f"No markdown files found under {DATA_DIR}. Set DATA_DIR in .env.")

    all_chunks = []
    skipped = 0

    for filepath in md_files:
        try:
            post = frontmatter.load(filepath)
        except Exception as e:
            print(f"  skip {filepath.name}: {e}")
            skipped += 1
            continue

        title = post.metadata.get("title") or filepath.stem
        date  = str(post.metadata.get("date", ""))
        doc_type = str(post.metadata.get("type", "article")).lower()
        url   = post.metadata.get("url", "")
        guest = post.metadata.get("guest", "")
        tags  = post.metadata.get("tags", [])

        if not post.content.strip():
            skipped += 1
            continue

        for i, chunk in enumerate(word_chunks(post.content)):
            all_chunks.append({
                "text":      chunk,
                "title":     title,
                "type":      doc_type,
                "date":      date,
                "url":       url,
                "guest":     guest,
                "tags":      tags,
                "filename":  str(filepath.relative_to(DATA_DIR)),
                "chunk_idx": i,
            })

    print(f"  {len(md_files) - skipped} documents → {len(all_chunks)} chunks "
          f"({skipped} skipped)")
    return all_chunks


def main():
    print(f"Loading documents from {DATA_DIR}...")
    chunks = load_documents()

    print(f"Encoding with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
    )
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Done. {faiss_index.ntotal} vectors saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()
