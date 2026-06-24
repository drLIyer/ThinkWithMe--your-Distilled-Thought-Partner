"""
Interactive RAG CLI over Lenny's Newsletter & Podcast data.

Usage:
    python chat.py

Requires:
    - index.faiss and chunks.pkl (run ingest.py first)
    - ANTHROPIC_API_KEY environment variable
"""

import os
import pickle
from pathlib import Path

import anthropic
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("index.faiss")
CHUNKS_PATH = Path("chunks.pkl")

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 8
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_HISTORY_TURNS = 6  # user+assistant pairs to keep

SYSTEM_PROMPT = """\
You are a helpful assistant with deep knowledge of Lenny Rachitsky's newsletter and podcast — 349 newsletters and 289 podcasts on product, growth, leadership, and startups.

Always cite sources naturally in the text (e.g. "In the podcast with [Guest]..." or "In his newsletter '[Title]'...").

Adapt your response to the user's intent:

**Direct questions**: Answer concisely using the provided context. If the context is thin, say so — but share what is relevant.

**Draft or writing requests** ("draft an article", "write a post", "help me write X"):
Use the context to produce a concrete draft the user can act on immediately, even if no single source covers the topic perfectly. Present it under the heading "Here's a version you could start with:" and note which Lenny sources shaped it.

**How-to or brainstorm requests** ("how should I approach", "brainstorm ideas", "help me think through"):
1. Present a brief framework grounded in the sources.
2. Offer 2–3 specific directions or angles the user could take, each as a numbered option.
3. End with: "Which of these would you like to go deeper on?"

Never leave the user without something concrete to act on."""


def retrieve(query: str, index, chunks: list, model: SentenceTransformer, k: int = TOP_K) -> list[dict]:
    emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(emb)
    scores, indices = index.search(emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def format_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        if r["type"] == "podcast":
            source = f"[PODCAST] {r['title']} (guest: {r.get('guest', r['title'])}, {r['date']})"
        else:
            source = f"[NEWSLETTER] {r['title']} ({r['date']})"
        parts.append(f"{source}\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def chat():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        print("Error: index.faiss or chunks.pkl not found. Run `python ingest.py` first.")
        return

    print("Loading index and model...")
    faiss_index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    client = anthropic.Anthropic()

    print(f"\nLoaded {faiss_index.ntotal} vectors from {len(set(c['filename'] for c in chunks))} documents.")
    print("\nLenny's Newsletter & Podcast RAG")
    print("Ask anything about product, growth, leadership, and startups.")
    print("Commands: 'sources' to see last retrieved sources, 'quit' to exit.\n")

    history: list[dict] = []  # plain Q&A, no context stored here
    last_results: list[dict] = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if query.lower() == "sources":
            if not last_results:
                print("No sources yet.\n")
            else:
                print("\nLast retrieved sources:")
                for i, r in enumerate(last_results, 1):
                    if r["type"] == "podcast":
                        print(f"  {i}. [PODCAST] {r['title']} ({r['date']}) — score: {r['score']:.3f}")
                    else:
                        print(f"  {i}. [NEWSLETTER] {r['title']} ({r['date']}) — score: {r['score']:.3f}")
                print()
            continue

        last_results = retrieve(query, faiss_index, chunks, model)
        context = format_context(last_results)

        # Build messages: prior history + current query (with context injected)
        messages = list(history)
        messages.append(
            {
                "role": "user",
                "content": f"Context from Lenny's content:\n\n{context}\n\n---\n\nQuestion: {query}",
            }
        )

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        answer = response.content[0].text

        # Store only the plain Q&A in history (not the context blob)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    chat()
