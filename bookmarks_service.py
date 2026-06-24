"""
Distill Bookmarks Service — FastAPI microservice for saved posts

Exposes:
  GET  /health
  GET  /bookmarks
  POST /bookmarks
  GET  /bookmarks/render

Run:
  .venv312/bin/python3 -m uvicorn bookmarks_service:app --host 127.0.0.1 --port 8084
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from filelock import FileLock
from pydantic import BaseModel
import uvicorn

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

BOOKMARKS_PATH = Path(__file__).parent / "bookmarks.json"
_LOCK          = FileLock(str(BOOKMARKS_PATH) + ".lock")

# ── Data access ───────────────────────────────────────────────────────────────

def _load() -> list:
    if BOOKMARKS_PATH.exists():
        try:
            return json.loads(BOOKMARKS_PATH.read_text())
        except Exception:
            pass
    return []


def _save(bookmarks: list):
    with _LOCK:
        BOOKMARKS_PATH.write_text(json.dumps(bookmarks, indent=2))


def _render_md(bookmarks: list) -> str:
    import re
    if not bookmarks:
        return "_No saved posts yet. Click 📌 on any answer to save it here._"
    lines = []
    for b in reversed(bookmarks[-20:]):
        date       = b.get("timestamp", "")[:10]
        q          = b.get("question",  "")[:120]
        a          = b.get("answer",    "")[:500]
        sources_md = b.get("sources_md", "")

        # Extract all source links: titles + URLs
        src_blocks = re.findall(
            r'(🎙️|📰) \*\*(Podcast|Newsletter)\*\* · ([^·]+) · \*[^*]+\* · `[^`]*`([^\n]*)',
            sources_md
        )
        src_lines = []
        for icon, kind, label, link_part in src_blocks:
            label = label.strip()
            open_match = re.search(r'\[Open ↗\]\((https?://[^)]+)\)', link_part)
            yt_match   = re.search(r'\[Search on YouTube ↗\]\((https?://[^)]+)\)', link_part)
            url = (open_match or yt_match)
            if url:
                src_lines.append(f"{icon} [{label}]({url.group(1)})")
            else:
                src_lines.append(f"{icon} {label}")

        entry = f"**{q}**  \n*{date}*\n\n{a}…"
        if src_lines:
            entry += "\n\n" + "  \n".join(src_lines)
        lines.append(entry + "\n\n---")
    return "\n\n".join(lines)


# ── Pydantic models ───────────────────────────────────────────────────────────

class BookmarkCreate(BaseModel):
    question:   str
    answer:     str
    sources_md: str = ""
    thread_id:  str = ""
    thread_name: str = ""


class BookmarkEntry(BaseModel):
    id:          str
    timestamp:   str
    thread_id:   str
    thread_name: str
    question:    str
    answer:      str
    sources_md:  str = ""


class BookmarksResponse(BaseModel):
    bookmarks: list[BookmarkEntry]


class BookmarkCreateResponse(BaseModel):
    id:    str
    count: int


class RenderResponse(BaseModel):
    markdown: str
    count:    int


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Distill Bookmarks Service", version="1.0.0")


@app.on_event("startup")
def on_startup():
    count = len(_load())
    print(f"Bookmarks service ready — {count} bookmarks at {BOOKMARKS_PATH}")


@app.get("/health")
def health():
    return {"status": "ok", "count": len(_load())}


@app.get("/bookmarks", response_model=BookmarksResponse)
def get_bookmarks():
    entries = _load()
    return BookmarksResponse(bookmarks=[BookmarkEntry(**e) for e in entries])


@app.post("/bookmarks", response_model=BookmarkCreateResponse)
def create_bookmark(req: BookmarkCreate):
    with _LOCK:
        bookmarks = _load()
        new_id = str(uuid.uuid4())
        bookmarks.append({
            "id":           new_id,
            "timestamp":    datetime.utcnow().isoformat(),
            "thread_id":    req.thread_id,
            "thread_name":  req.thread_name or "Untitled",
            "question":     req.question,
            "answer":       req.answer,
            "sources_md":   req.sources_md,
        })
        BOOKMARKS_PATH.write_text(json.dumps(bookmarks, indent=2))
    return BookmarkCreateResponse(id=new_id, count=len(bookmarks))


@app.get("/bookmarks/render", response_model=RenderResponse)
def render_bookmarks():
    bookmarks = _load()
    return RenderResponse(markdown=_render_md(bookmarks), count=len(bookmarks))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8084, log_level="info")
