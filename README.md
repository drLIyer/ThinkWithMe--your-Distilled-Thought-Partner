# Distill

A personal RAG chatbot that turns a corpus of newsletters and podcasts into a thought partner — grounded in real sources, personalized to your role and the products you manage.

Built with Claude, FAISS, sentence-transformers, and Chainlit. Runs locally.

![Distill chat interface](public/distill.css)

---

## What it does

- Answers questions grounded in your corpus with exact source attribution (title, date, match score, link)
- Adapts its style to your seniority — strategic framing for executives, practical how-tos for PMs
- Remembers what you work on across sessions via a lightweight user profile
- Saves answers you want to revisit (📌 bookmarks)
- Flags when the corpus doesn't cover a topic well (confidence scoring)
- Logs knowledge gaps — topics it couldn't answer well — so you know what's missing
- Health agent runs in the background, detects failures, and auto-restarts services
- Admin dashboard shows usage, feedback, and system health

---

## Architecture

Five FastAPI microservices + a Chainlit UI, all running locally:

| Service | Port | Purpose |
|---|---|---|
| Chainlit UI | 8081 | Chat interface |
| Admin dashboard | 8082 | Usage, health, knowledge gaps |
| RAG service | 8083 | FAISS retrieval + confidence scoring |
| Bookmarks service | 8084 | Saved answers |
| Users service | 8085 | Profile + memories |
| Feedback service | 8086 | Thumbs up/down + gap logging |

The RAG pipeline:
1. Embed the query with `all-MiniLM-L6-v2`
2. Cosine similarity search via FAISS — top 8 chunks
3. Chunks are labeled by type (`[PODCAST]` / `[NEWSLETTER]`) and injected into the Claude prompt
4. Response streams back with sources, confidence level, and follow-up suggestions

---

## Setup

### 1. Clone and create a virtualenv

```bash
git clone https://github.com/YOUR_USERNAME/distill.git
cd distill
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:

```
ANTHROPIC_API_KEY=your-key
CHAINLIT_USERNAME=your-username
CHAINLIT_PASSWORD=your-password
CHAINLIT_AUTH_SECRET=        # generate with: python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. Get your data and build the index

See [data/README.md](data/README.md) for how to get the Lenny archive (paid subscriber download) or use your own corpus.

```bash
# Point DATA_DIR at your data, then:
python ingest.py
```

### 4. Start everything

```bash
./start.sh
```

Open [http://localhost:8081](http://localhost:8081). Fill in your profile on first launch.

---

## Configuration

All configuration is via `.env`. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model for chat responses |
| `ANTHROPIC_HAIKU_MODEL` | `claude-haiku-4-5` | Model for background tasks (memory, profile) |
| `CHAINLIT_USERNAME` | — | Login username |
| `CHAINLIT_PASSWORD` | — | Login password |
| `CHAINLIT_AUTH_SECRET` | — | JWT secret for session tokens |
| `ADMIN_PASSWORD` | — | Admin dashboard password |
| `DATA_DIR` | `./data` | Path to your corpus of markdown files |
| `ASSISTANT_NAME` | `Distill` | Name shown in the UI |
| `CORPUS_DESCRIPTION` | Lenny description | Used in the system prompt |
| `SYSTEM_PROMPT_OVERRIDE` | — | Replace the entire system prompt |
| `HEALTH_EMAIL_TO` | — | Email for health agent alerts (requires `sendmail`) |

### Using a different corpus

Distill works with any collection of markdown files, not just Lenny's content. Set `DATA_DIR`, `ASSISTANT_NAME`, and `CORPUS_DESCRIPTION` in `.env`. See [data/README.md](data/README.md) for the required file format.

---

## Rebuilding the index

Run this any time your data changes:

```bash
python ingest.py
```

Then restart the RAG service (`pkill -f rag_service` and re-run `start.sh`, or restart via the admin dashboard).

---

## Weekly knowledge update

`run_update.sh` fetches new content, rebuilds the index, and restarts Distill. Wire it to a cron job to keep the corpus current:

```bash
# Run every Sunday at 7am
0 7 * * 0 /path/to/distill/run_update.sh
```

Set `HEALTH_EMAIL_TO` in `.env` to receive a summary email after each run.

---

## Stopping

```bash
pkill -f "chainlit run"
pkill -f uvicorn
```

---

## Built with

- [Claude](https://anthropic.com) — Sonnet for responses, Haiku for background tasks
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [sentence-transformers](https://www.sbert.net) — `all-MiniLM-L6-v2` for embeddings
- [Chainlit](https://chainlit.io) — chat UI
- [FastAPI](https://fastapi.tiangolo.com) — microservice backends
