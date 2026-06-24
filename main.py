"""
Ask Lenny — RAG Chat API with Azure AD authentication

Run locally:
    uvicorn main:app --reload --port 8080

Required env vars:
    ANTHROPIC_API_KEY
    AZURE_TENANT_ID
    AZURE_CLIENT_ID
    AZURE_CLIENT_SECRET
    SESSION_SECRET      (any long random string)
    APP_BASE_URL        (e.g. http://localhost:8080 or https://your-app.railway.app)
"""

import json
import os
import pickle
import uuid
from pathlib import Path

import anthropic
import faiss
import frontmatter
import msal
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from starlette.middleware.sessions import SessionMiddleware

# ── Config ────────────────────────────────────────────────────────────────────

INDEX_PATH = Path("index.faiss")
CHUNKS_PATH = Path("chunks.pkl")
CONVERSATIONS_DIR = Path(os.environ.get("CONVERSATIONS_DIR", "conversations"))
CONVERSATIONS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(os.environ.get("DATA_DIR", "/Users/liyer_1/Downloads/lennys-newsletterpodcastdata-all"))

EMBED_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-6"
TOP_K = 8
FETCH_K = 20  # fetch more candidates before deduplicating by source
MAX_HISTORY_TURNS = 6

AZURE_TENANT_ID = os.environ.get("AZURE_TENANT_ID", "")
AZURE_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-secret-change-in-production")
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:8080")
REDIRECT_URI = f"{APP_BASE_URL}/auth/callback"
SCOPES = ["User.Read"]

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

# ── Startup: load resources ───────────────────────────────────────────────────

print("Loading FAISS index...")
_index = faiss.read_index(str(INDEX_PATH))

print("Loading chunks...")
with open(CHUNKS_PATH, "rb") as f:
    _chunks: list[dict] = pickle.load(f)

print(f"Loading embedding model ({EMBED_MODEL})...")
_embed_model = SentenceTransformer(EMBED_MODEL)

# Build filename → youtube_url lookup for podcasts
_podcast_urls: dict[str, str] = {}
_index_json = DATA_DIR / "01-start-here/index.json"
if _index_json.exists():
    idx = json.loads(_index_json.read_text())
    for item in idx.get("podcasts", []):
        try:
            post = frontmatter.load(str(DATA_DIR / item["filename"]))
            yt = post.metadata.get("youtube_url", "")
            if yt:
                _podcast_urls[item["filename"]] = yt
        except Exception:
            pass
    print(f"Loaded {len(_podcast_urls)} podcast YouTube URLs")

print("Ready.")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Ask Lenny")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Auth ──────────────────────────────────────────────────────────────────────

def _msal_app():
    return msal.ConfidentialClientApplication(
        AZURE_CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}",
        client_credential=AZURE_CLIENT_SECRET,
    )


def get_current_user(request: Request) -> dict:
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@app.get("/signin", response_class=HTMLResponse)
def signin_page():
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sign in — Ask Lenny</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f0f0f; color: #ececec;
    height: 100dvh; display: flex; align-items: center; justify-content: center;
  }
  .card {
    text-align: center; max-width: 380px; width: 100%; padding: 48px 40px;
    background: #1a1a1a; border: 1px solid #2e2e2e; border-radius: 16px;
  }
  h1 { font-size: 26px; font-weight: 600; margin-bottom: 8px; }
  p { color: #888; font-size: 14px; line-height: 1.6; margin-bottom: 32px; }
  .ms-btn {
    display: inline-flex; align-items: center; gap: 10px;
    background: #0078d4; color: #fff; border: none; border-radius: 8px;
    padding: 12px 24px; font-size: 14px; font-weight: 500; cursor: pointer;
    text-decoration: none; transition: background 0.15s;
  }
  .ms-btn:hover { background: #106ebe; }
  .ms-btn svg { width: 18px; height: 18px; flex-shrink: 0; }
</style>
</head>
<body>
<div class="card">
  <h1>Ask Lenny</h1>
  <p>Search 349 newsletters and 289 podcasts on product, growth, and leadership.</p>
  <a href="/login" class="ms-btn">
    <svg viewBox="0 0 21 21" xmlns="http://www.w3.org/2000/svg">
      <rect x="1" y="1" width="9" height="9" fill="#f25022"/>
      <rect x="11" y="1" width="9" height="9" fill="#7fba00"/>
      <rect x="1" y="11" width="9" height="9" fill="#00a4ef"/>
      <rect x="11" y="11" width="9" height="9" fill="#ffb900"/>
    </svg>
    Sign in with Microsoft
  </a>
</div>
</body>
</html>""")


@app.get("/login")
async def login(request: Request):
    if not AZURE_CLIENT_ID:
        raise HTTPException(500, "Azure AD not configured — set AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET.")
    flow = _msal_app().initiate_auth_code_flow(SCOPES, redirect_uri=REDIRECT_URI)
    request.session["auth_flow"] = flow
    return RedirectResponse(flow["auth_uri"])


@app.get("/auth/callback")
async def auth_callback(request: Request):
    flow = request.session.pop("auth_flow", {})
    result = _msal_app().acquire_token_by_auth_code_flow(flow, dict(request.query_params))
    if "error" in result:
        raise HTTPException(400, f"Auth failed: {result.get('error_description', result['error'])}")
    claims = result.get("id_token_claims", {})
    request.session["user"] = {
        "oid": claims.get("oid", str(uuid.uuid4())),
        "name": claims.get("name", ""),
        "email": claims.get("preferred_username", ""),
    }
    return RedirectResponse("/")


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/signin")


@app.get("/me")
def me(user: dict = Depends(get_current_user)):
    return user

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str) -> list[dict]:
    emb = _embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(emb)
    scores, indices = _index.search(emb, FETCH_K)
    seen_files: set[str] = set()
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = dict(_chunks[idx])
        fname = chunk.get("filename", "")
        if fname in seen_files:
            continue
        seen_files.add(fname)
        chunk["score"] = float(score)
        results.append(chunk)
        if len(results) >= TOP_K:
            break
    return results


def build_retrieval_query(message: str, history: list[dict]) -> str:
    """Prepend the previous user turn so follow-up questions embed with context."""
    if not history:
        return message
    prev_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    if not prev_user:
        return message
    return f"{prev_user} {message}"


def format_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        if r["type"] == "podcast":
            header = f"[PODCAST] {r['title']} (guest: {r.get('guest', r['title'])}, {r['date']})"
        else:
            header = f"[NEWSLETTER] {r['title']} ({r['date']})"
        parts.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def source_url(s: dict) -> str:
    filename = s.get("filename", "")
    if s["type"] == "podcast":
        yt = _podcast_urls.get(filename, "")
        if yt:
            return yt
        query = (s.get("guest") or s.get("title", "")).replace(" ", "+")
        return f"https://www.youtube.com/results?search_query=Lenny+podcast+{query}"
    else:
        slug = Path(filename).stem
        return f"https://www.lennysnewsletter.com/p/{slug}" if slug else ""


def build_sources(results: list[dict]) -> list[dict]:
    return [
        {
            "type": r["type"],
            "title": r["title"],
            "guest": r.get("guest", ""),
            "date": r["date"],
            "snippet": r["text"][:400].rsplit(" ", 1)[0] + " …",
            "url": source_url(r),
            "filename": r.get("filename", ""),
        }
        for r in results
    ]

# ── Persistence (per user) ────────────────────────────────────────────────────

def _conv_path(user_oid: str) -> Path:
    return CONVERSATIONS_DIR / f"{user_oid}.json"


def load_user_conversations(user_oid: str) -> dict:
    path = _conv_path(user_oid)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    conv = {"id": str(uuid.uuid4()), "title": "New chat", "messages": [], "history": []}
    return {"active_id": conv["id"], "conversations": [conv]}


def save_user_conversations(user_oid: str, data: dict):
    _conv_path(user_oid).write_text(json.dumps(data, indent=2))

# ── Routes ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    history: list[dict]


@app.post("/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    retrieval_query = build_retrieval_query(req.message, req.history)
    results = retrieve(retrieval_query)
    context = format_context(results)
    sources = build_sources(results)

    messages = list(req.history)
    messages.append({
        "role": "user",
        "content": f"Context from Lenny's content:\n\n{context}\n\n---\n\nQuestion: {req.message}",
    })

    def generate():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        client = anthropic.Anthropic()
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/conversations")
def get_conversations(user: dict = Depends(get_current_user)):
    return load_user_conversations(user["oid"])


class SaveRequest(BaseModel):
    active_id: str
    conversations: list[dict]


@app.post("/conversations")
def post_conversations(req: SaveRequest, user: dict = Depends(get_current_user)):
    save_user_conversations(user["oid"], {"active_id": req.active_id, "conversations": req.conversations})
    return {"ok": True}


# ── Static files (must be last) ───────────────────────────────────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")
