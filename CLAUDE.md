# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

**Streamlit UI (local dev):**
```bash
ANTHROPIC_API_KEY=your_key python3 -m streamlit run app.py
```

**FastAPI server (production-style):**
```bash
ANTHROPIC_API_KEY=your_key uvicorn main:app --reload --port 8080
```

**CLI chat:**
```bash
ANTHROPIC_API_KEY=your_key python3 chat.py
```

**Rebuild the vector index** (only needed if the source data changes):
```bash
python3 ingest.py
```

## Architecture

There are two independent front-ends sharing the same vector index and retrieval logic:

- **`app.py`** — Streamlit UI. All-in-one: renders the chat, calls Claude directly via streaming, persists conversations to `conversations.json`. Run this for local development.
- **`main.py` + `static/index.html`** — FastAPI backend + vanilla JS SPA. The API streams SSE events (`sources` → `token`… → `done`). Deployed to Railway/Replit via `Procfile`. This is the production path.
- **`chat.py`** — terminal REPL, useful for quick testing without a browser.

The core RAG pipeline (duplicated across all three):
1. Embed the query with `all-MiniLM-L6-v2` (sentence-transformers)
2. Cosine similarity search via FAISS `IndexFlatIP` on L2-normalized vectors — `TOP_K=8` chunks
3. Format chunks as labeled context blocks (`[PODCAST]` / `[NEWSLETTER]`)
4. Call Claude with the context injected into the user turn — **context is NOT stored in message history**, only the plain Q&A is, capped at `MAX_HISTORY_TURNS=6` pairs

**Pre-built artifacts** (`index.faiss`, `chunks.pkl`) are committed to the repo. Source data lives at `/Users/liyer_1/Downloads/lennys-newsletterpodcastdata-all` and is read by `ingest.py` only — the app itself never reads the raw markdown files except to resolve podcast YouTube URLs at startup.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
| `CONVERSATIONS_DIR` | `.` (repo root) | Set to a Railway persistent volume path in production so conversations survive deploys |
| `DATA_DIR` | `/Users/liyer_1/Downloads/lennys-newsletterpodcastdata-all` | Only used by `main.py` at startup to build the podcast YouTube URL lookup |

## Data format

Source documents are markdown files with YAML frontmatter. The manifest at `DATA_DIR/01-start-here/index.json` lists all newsletters and podcasts with their filenames, titles, dates, and guests. `ingest.py` chunks each document at 400 words with 50-word overlap before embedding.
