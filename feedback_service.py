"""
Distill Feedback Service — FastAPI microservice for feedback and knowledge gaps

Exposes:
  GET  /health
  POST /feedback
  GET  /feedback/recent
  POST /gaps
  GET  /gaps/recent

Run:
  .venv312/bin/python3 -m uvicorn feedback_service:app --host 127.0.0.1 --port 8086
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from filelock import FileLock
from pydantic import BaseModel
import uvicorn

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

FEEDBACK_PATH = Path("/Users/liyer_1/lennys-rag/feedback.json")
GAPS_PATH     = Path("/Users/liyer_1/lennys-rag/knowledge_gaps.json")

_FEEDBACK_LOCK = FileLock(str(FEEDBACK_PATH) + ".lock")
_GAPS_LOCK     = FileLock(str(GAPS_PATH)     + ".lock")


# ── Data access ───────────────────────────────────────────────────────────────

def _load_feedback() -> list:
    if FEEDBACK_PATH.exists():
        try:
            return json.loads(FEEDBACK_PATH.read_text())
        except Exception:
            pass
    return []


def _load_gaps() -> list:
    if GAPS_PATH.exists():
        try:
            return json.loads(GAPS_PATH.read_text())
        except Exception:
            pass
    return []


# ── Pydantic models ───────────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    id:            str
    question:      str
    answer:        str
    rating:        str
    feedback_text: str = ""


class FeedbackEntry(BaseModel):
    id:            str
    timestamp:     str
    question:      str
    answer:        str
    rating:        str
    feedback_text: str = ""


class FeedbackListResponse(BaseModel):
    feedback: list[FeedbackEntry]
    count:    int


class FeedbackCreateResponse(BaseModel):
    id:    str
    count: int


class GapCreate(BaseModel):
    query:       str
    top_score:   float
    avg_score:   float
    best_source: str = ""
    confidence:  str = ""


class GapEntry(BaseModel):
    timestamp:   str
    query:       str
    top_score:   float
    avg_score:   float
    best_source: str = ""
    confidence:  str = ""


class GapListResponse(BaseModel):
    gaps:  list[GapEntry]
    count: int


class GapCreateResponse(BaseModel):
    count: int


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Distill Feedback Service", version="1.0.0")


@app.on_event("startup")
def on_startup():
    fb = len(_load_feedback())
    gp = len(_load_gaps())
    print(f"Feedback service ready — {fb} feedback entries, {gp} knowledge gaps")


@app.get("/health")
def health():
    return {
        "status":         "ok",
        "feedback_count": len(_load_feedback()),
        "gaps_count":     len(_load_gaps()),
    }


@app.post("/feedback", response_model=FeedbackCreateResponse)
def create_feedback(req: FeedbackCreate):
    with _FEEDBACK_LOCK:
        records = _load_feedback()
        records.append({
            "id":            req.id,
            "timestamp":     datetime.utcnow().isoformat(),
            "question":      req.question,
            "answer":        req.answer,
            "rating":        req.rating,
            "feedback_text": req.feedback_text,
        })
        FEEDBACK_PATH.write_text(json.dumps(records, indent=2))
    return FeedbackCreateResponse(id=req.id, count=len(records))


@app.get("/feedback/recent", response_model=FeedbackListResponse)
def get_recent_feedback():
    records = _load_feedback()
    recent = records[-30:]
    return FeedbackListResponse(
        feedback=[FeedbackEntry(**r) for r in recent],
        count=len(records),
    )


@app.post("/gaps", response_model=GapCreateResponse)
def create_gap(req: GapCreate):
    with _GAPS_LOCK:
        records = _load_gaps()
        records.append({
            "timestamp":   datetime.utcnow().isoformat(),
            "query":       req.query,
            "top_score":   req.top_score,
            "avg_score":   req.avg_score,
            "best_source": req.best_source,
            "confidence":  req.confidence,
        })
        GAPS_PATH.write_text(json.dumps(records, indent=2))
    return GapCreateResponse(count=len(records))


@app.get("/gaps/recent", response_model=GapListResponse)
def get_recent_gaps():
    records = _load_gaps()
    recent = records[-20:]
    return GapListResponse(
        gaps=[GapEntry(**r) for r in recent],
        count=len(records),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8086, log_level="info")
