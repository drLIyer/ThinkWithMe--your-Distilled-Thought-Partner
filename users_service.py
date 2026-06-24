"""
Distill Users Service — FastAPI microservice for user profile and memories

Exposes:
  GET  /health
  GET  /profile
  PUT  /profile
  GET  /memories
  POST /memories
  GET  /system-prompt

Run:
  .venv312/bin/python3 -m uvicorn users_service:app --host 127.0.0.1 --port 8085
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from filelock import FileLock
from pydantic import BaseModel
import uvicorn

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

_BASE          = Path(__file__).parent
PROFILE_PATH   = _BASE / "user_profile.json"
MEMORIES_PATH  = _BASE / "user_memories.json"
ENGAGEMENT_PATH    = _BASE / "engagement.json"
ENGAGEMENT_SUMMARY = _BASE / "engagement_summary.txt"
MAX_MEMORIES       = 10
MAX_ENGAGEMENT     = 50
ENGAGEMENT_REGEN_EVERY = 5
HAIKU_MODEL   = os.environ.get("ANTHROPIC_HAIKU_MODEL", "claude-haiku-4-5")

_PROFILE_LOCK    = FileLock(str(PROFILE_PATH)    + ".lock")
_MEMORIES_LOCK   = FileLock(str(MEMORIES_PATH)   + ".lock")
_ENGAGEMENT_LOCK = FileLock(str(ENGAGEMENT_PATH) + ".lock")

SYSTEM_PROMPT = """\
You are Distill — a thought partner with deep knowledge of Lenny Rachitsky's newsletter and podcast: 600+ pieces on product, growth, leadership, and startups.

Always cite sources naturally in the text (e.g. "In the podcast with [Guest]..." or "In his newsletter '[Title]'...").

Adapt your response to the user's intent:

**Direct questions**: Answer concisely using the provided context. If the context is thin, say so — but share what is relevant.

**Draft or writing requests** ("draft an article", "write a post", "help me write X"):
Use the context to produce a concrete draft the user can act on immediately, even if no single source covers the topic perfectly. Present it under the heading "Here's a version you could start with:" and note which Lenny sources shaped it.

**How-to or brainstorm requests** ("how should I approach", "brainstorm ideas", "help me think through"):
1. Present a brief framework grounded in the sources.
2. Offer 2–3 specific directions or angles the user could take, each as a numbered option.
3. End with: "Which of these would you like to go deeper on?"

**Always end every response** with a section formatted exactly like this:

---
**Key Takeaways**
- [2–3 bullet takeaways from this response]

*As your thought partner — what would you like to explore next? [Ask one specific, relevant follow-up question]*

**Source transparency**: Always be honest about where your answer comes from.
- If the provided context strongly supports your answer, cite it naturally.
- If the context is thin or only partially relevant, say so explicitly — e.g. "The corpus doesn't cover this directly, but drawing on general product thinking…"
- Never silently blend general knowledge into a corpus-sourced answer without flagging it.

Never leave the user without something concrete to act on."""


# ── Data access ───────────────────────────────────────────────────────────────

def _load_profile() -> dict:
    if PROFILE_PATH.exists():
        try:
            return json.loads(PROFILE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_profile(profile: dict):
    with _PROFILE_LOCK:
        profile["updated_at"] = datetime.utcnow().isoformat()
        PROFILE_PATH.write_text(json.dumps(profile, indent=2))
    return profile


def _load_memories() -> list:
    if MEMORIES_PATH.exists():
        try:
            return json.loads(MEMORIES_PATH.read_text())
        except Exception:
            pass
    return []


def _save_memories(memories: list):
    with _MEMORIES_LOCK:
        MEMORIES_PATH.write_text(json.dumps(memories, indent=2))


def _detect_seniority(role: str) -> str:
    r = role.lower()
    if any(t in r for t in ["svp", "vp ", "vp,", "chief", "president", "head of", "evp", "gm "]):
        return "executive"
    if any(t in r for t in ["director", "sr. director", "senior director"]):
        return "director"
    if any(t in r for t in ["senior pm", "sr. pm", "principal", "lead pm", "staff pm", "group pm"]):
        return "senior_ic"
    if any(t in r for t in ["product manager", " pm", "pm,", "pm "]):
        return "pm"
    return "unknown"


_SENIORITY_GUIDANCE = {
    "executive": (
        "**Response style for this user's level:** Focus on strategic framing, org-level "
        "implications, business outcomes, and leadership decisions. Avoid step-by-step "
        "tactical guidance unless asked. Frame Lenny's frameworks in terms of what to "
        "prioritise and how to lead teams toward outcomes."
    ),
    "director": (
        "**Response style for this user's level:** Balance strategic context with "
        "cross-functional execution. Connect frameworks to team leadership, roadmap "
        "decisions, and stakeholder alignment. Address both 'what to decide' and "
        "'how to get the org moving'."
    ),
    "senior_ic": (
        "**Response style for this user's level:** Focus on product craft, stakeholder "
        "influence, and driving outcomes without direct authority. Assume strong execution "
        "skills — emphasise judgment calls, navigating ambiguity, and growing scope of impact."
    ),
    "pm": (
        "**Response style for this user's level:** Be practical and concrete. Provide "
        "step-by-step frameworks the user can act on immediately. Explain the 'why' behind "
        "recommendations. Focus on discovery, prioritisation, and delivery."
    ),
}


def _load_engagement() -> list:
    if ENGAGEMENT_PATH.exists():
        try:
            return json.loads(ENGAGEMENT_PATH.read_text())
        except Exception:
            pass
    return []


def _load_engagement_summary() -> str:
    if ENGAGEMENT_SUMMARY.exists():
        try:
            return ENGAGEMENT_SUMMARY.read_text().strip()
        except Exception:
            pass
    return ""


def _regenerate_engagement_summary(events: list):
    try:
        recent = events[-20:]
        lines  = "\n".join(f"- [{e['type']}] {e['content'][:120]}" for e in recent)
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=80,
            messages=[{"role": "user", "content": (
                "Based on these recent user interactions with an AI assistant, write ONE sentence "
                "describing their engagement pattern — what topics they revisit, what level of detail "
                "they seek, what they tend to save or follow up on. Start with 'This user tends to'.\n\n"
                f"{lines}"
            )}],
        )
        summary = resp.content[0].text.strip()
        if summary:
            ENGAGEMENT_SUMMARY.write_text(summary)
    except Exception:
        pass


def _build_system_prompt_from_files() -> str:
    profile  = _load_profile()
    memories = _load_memories()
    prefix   = ""

    if any(v for k, v in profile.items() if k != "updated_at" and v):
        goals = profile.get("goals", "")
        prefix += (
            "## About this user\n"
            f"- Role: {profile.get('role', '')}\n"
            f"- Team / Business unit: {profile.get('team', '')}\n"
            f"- Products / Portfolios: {profile.get('products', '')}\n"
            f"- Current projects: {profile.get('projects', '')}\n"
            + (f"- What they want from this thought partner: {goals}\n" if goals else "")
            + "\nAlways frame answers in the context of their specific role and the products they manage. "
            "When relevant, connect Lenny's frameworks to their actual situation."
            + (f" Keep in mind their stated goal: {goals}." if goals else "")
            + "\n\n"
        )

    if prefix:
        level    = _detect_seniority(profile.get("role", ""))
        guidance = _SENIORITY_GUIDANCE.get(level, "")
        if guidance:
            prefix += guidance + "\n\n"

    if memories:
        prefix += "## Context from your recent sessions\n"
        prefix += "\n".join(f"- {m['summary']}" for m in memories[-MAX_MEMORIES:])
        prefix += "\n\n"

    eng_summary = _load_engagement_summary()
    if eng_summary and prefix:
        prefix += f"## Engagement pattern\n{eng_summary}\n\n"

    return (prefix + SYSTEM_PROMPT) if prefix else SYSTEM_PROMPT


# ── Pydantic models ───────────────────────────────────────────────────────────

class ProfileUpdate(BaseModel):
    role:     str = ""
    team:     str = ""
    products: str = ""
    projects: str = ""
    goals:    str = ""


class MemoryCreate(BaseModel):
    query:         str
    response_text: str
    thread_id:     str = ""


class MemoryEntry(BaseModel):
    timestamp: str
    thread_id: str
    summary:   str


class MemoriesResponse(BaseModel):
    memories: list[MemoryEntry]


class MemoryCreateResponse(BaseModel):
    summary: str
    count:   int


class SystemPromptResponse(BaseModel):
    system_prompt: str


class EngagementCreate(BaseModel):
    type:      str
    content:   str
    thread_id: str = ""


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Distill Users Service", version="1.0.0")


@app.on_event("startup")
def on_startup():
    has_profile = PROFILE_PATH.exists()
    count = len(_load_memories())
    eng_count = len(_load_engagement())
    print(f"Users service ready — profile={'yes' if has_profile else 'no'}, {count} memories, {eng_count} engagement events")


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "has_profile":  PROFILE_PATH.exists(),
        "memory_count": len(_load_memories()),
    }


@app.get("/profile")
def get_profile():
    return _load_profile()


@app.put("/profile")
def update_profile(req: ProfileUpdate):
    profile = req.model_dump()
    _save_profile(profile)
    return _load_profile()


@app.get("/memories", response_model=MemoriesResponse)
def get_memories():
    entries = _load_memories()
    return MemoriesResponse(memories=[MemoryEntry(**e) for e in entries])


@app.post("/memories", response_model=MemoryCreateResponse)
def create_memory(req: MemoryCreate):
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=80,
            messages=[{"role": "user", "content": (
                "Summarize in one sentence the key insight or topic from this exchange, "
                "from the user's perspective. Start with 'You explored', 'You noted', or 'You asked about'.\n"
                f"Q: {req.query[:200]}\nA: {req.response_text[:400]}"
            )}],
        )
        summary = resp.content[0].text.strip()
        if not summary:
            raise HTTPException(status_code=500, detail="Empty summary returned")

        memories = _load_memories()
        memories.append({
            "timestamp": datetime.utcnow().isoformat(),
            "thread_id": req.thread_id,
            "summary":   summary,
        })
        if len(memories) > MAX_MEMORIES:
            memories = memories[-MAX_MEMORIES:]
        _save_memories(memories)
        return MemoryCreateResponse(summary=summary, count=len(memories))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system-prompt", response_model=SystemPromptResponse)
def get_system_prompt():
    return SystemPromptResponse(system_prompt=_build_system_prompt_from_files())


@app.post("/engagement")
def log_engagement(req: EngagementCreate):
    with _ENGAGEMENT_LOCK:
        events = _load_engagement()
        events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type":      req.type,
            "content":   req.content,
            "thread_id": req.thread_id,
        })
        if len(events) > MAX_ENGAGEMENT:
            events = events[-MAX_ENGAGEMENT:]
        ENGAGEMENT_PATH.write_text(json.dumps(events, indent=2))
        count = len(events)

    if count % ENGAGEMENT_REGEN_EVERY == 0:
        _regenerate_engagement_summary(events)

    return {"count": count}


@app.get("/engagement/summary")
def get_engagement_summary():
    return {
        "summary":       _load_engagement_summary(),
        "event_count":   len(_load_engagement()),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8085, log_level="info")
