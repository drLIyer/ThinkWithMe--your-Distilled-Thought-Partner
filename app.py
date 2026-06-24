"""
Lenny's Newsletter & Podcast — RAG Chat Interface

Run with:
    streamlit run app.py
"""

import base64
import io
import json
import mimetypes
import pickle
import time
import uuid
from datetime import datetime
from pathlib import Path

import anthropic
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

INDEX_PATH        = Path("index.faiss")
CHUNKS_PATH       = Path("chunks.pkl")
FEEDBACK_PATH     = Path("feedback.json")
CONVERSATIONS_PATH = Path("conversations.json")
EMBED_MODEL       = "all-MiniLM-L6-v2"
CLAUDE_MODEL      = "align-aws-sonnet-4-6"
HAIKU_MODEL       = "align-aws-haiku-4-5"
TOP_K             = 8
MAX_HISTORY_TURNS = 20
TURN_WARNING      = 15   # warn at this many assistant turns

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

**Always end every response** with a section formatted exactly like this:

---
**Key Takeaways**
- [2–3 bullet takeaways from this response]

*As your thought partner — what would you like to explore next? [Ask one specific, relevant follow-up question]*

**Source transparency**: Always be honest about where your answer comes from.
- If the provided context strongly supports your answer, cite it naturally.
- If the context is thin or only partially relevant, say so explicitly — e.g. "Lenny's content doesn't cover this directly, but drawing on general product thinking…"
- Never silently blend general knowledge into a Lenny-sourced answer without flagging it.

Never leave the user without something concrete to act on."""

FOLLOWUP_PROMPT = """\
Based on this conversation exchange, suggest exactly 3 short follow-up questions the user might want to ask next.
Return ONLY a JSON array of 3 strings, no other text. Each question should be under 10 words.
Example: ["How do I measure PMF?", "What metrics matter most?", "Can you give an example?"]"""

# ── Page setup (must be first Streamlit call) ─────────────────────────────────

st.set_page_config(page_title="Ask Lenny", page_icon="💬", layout="centered")

st.markdown("""
<style>
[data-testid="stToolbar"] { display: none !important; }

/* Sidebar conversation buttons */
div[data-testid="stSidebar"] .conv-btn button {
    text-align: left !important;
    justify-content: flex-start !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border-radius: 8px;
    border: none;
    background: transparent;
    color: inherit;
    font-size: 0.875rem;
    padding: 6px 10px;
}
div[data-testid="stSidebar"] .conv-btn-active button {
    background: rgba(255,255,255,0.12) !important;
    font-weight: 500;
}
div[data-testid="stSidebar"] .conv-btn button:hover {
    background: rgba(255,255,255,0.08) !important;
}
div[data-testid="stSidebar"] .del-btn button {
    padding: 4px 6px;
    border: none;
    background: transparent;
    color: rgba(255,255,255,0.4);
    border-radius: 6px;
    font-size: 0.8rem;
}
div[data-testid="stSidebar"] .del-btn button:hover {
    background: rgba(255,0,0,0.15) !important;
    color: #ff6b6b !important;
}

/* Follow-up chips */
.followup-chip { display: inline-block; }

/* File uploader compact */
div[data-testid="stFileUploader"] { margin-bottom: -8px; }
div[data-testid="stFileUploader"] section {
    padding: 4px 10px !important;
    border-radius: 10px 10px 0 0 !important;
    border-bottom: none !important;
    min-height: unset !important;
}
div[data-testid="stFileUploader"] section > div {
    flex-direction: row !important;
    align-items: center !important;
    gap: 8px !important;
}
div[data-testid="stFileUploader"] section small { display: none !important; }

/* Action button row under messages */
.action-row button {
    font-size: 0.75rem !important;
    padding: 2px 8px !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────

DATA_DIR = Path("/Users/liyer_1/Downloads/lennys-newsletterpodcastdata-all")


@st.cache_resource(show_spinner="Loading index…")
def load_resources():
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    podcast_urls: dict[str, str] = {}
    index_path = DATA_DIR / "01-start-here/index.json"
    if index_path.exists():
        import json as _json
        import frontmatter as _fm
        idx = _json.loads(index_path.read_text())
        for item in idx.get("podcasts", []):
            try:
                post = _fm.load(str(DATA_DIR / item["filename"]))
                yt = post.metadata.get("youtube_url", "")
                if yt:
                    podcast_urls[item["filename"]] = yt
            except Exception:
                pass
    return index, chunks, model, podcast_urls


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, index, chunks: list, model) -> list[dict]:
    emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(emb)
    scores, indices = index.search(emb, TOP_K)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = dict(chunks[idx])
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
        return {"level": "high",   "label": "High Lenny coverage",     "color": "#27ae60"}
    elif top >= 0.55:
        return {"level": "medium", "label": "Moderate Lenny coverage",  "color": "#e67e22"}
    else:
        return {"level": "low",    "label": "Limited Lenny coverage — draws on general knowledge", "color": "#e74c3c"}


def render_confidence(conf: dict):
    c, label = conf["color"], conf["label"]
    st.markdown(
        f'<div style="display:inline-flex;align-items:center;gap:6px;margin:2px 0 6px 0;">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{c};display:inline-block;flex-shrink:0"></span>'
        f'<span style="font-size:0.78em;color:{c};font-weight:500">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if conf["level"] == "low":
        st.caption("⚠️ Lenny's content had limited coverage here — verify key claims independently.")


def format_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        if r["type"] == "podcast":
            header = f"[PODCAST] {r['title']} (guest: {r.get('guest', r['title'])}, {r['date']})"
        else:
            header = f"[NEWSLETTER] {r['title']} ({r['date']})"
        parts.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(parts)


# ── Generation ────────────────────────────────────────────────────────────────

def stream_response(query: str, history: list[dict], context: str,
                    attachments: list[dict] | None = None, max_retries: int = 2,
                    model: str = CLAUDE_MODEL):
    client = anthropic.Anthropic()
    messages = list(history)
    preamble = f"Context from Lenny's content:\n\n{context}\n\n---\n\n"
    if attachments:
        content = attachments_to_claude_blocks(attachments, preamble + "Question: " + query)
    else:
        content = preamble + "Question: " + query
    messages.append({"role": "user", "content": content})

    for attempt in range(max_retries + 1):
        try:
            with client.messages.stream(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text
            return
        except anthropic.APIStatusError as e:
            if attempt < max_retries and e.status_code in (429, 500, 502, 503, 529):
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise


def get_followup_suggestions(question: str, answer: str) -> list[str]:
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"{FOLLOWUP_PROMPT}\n\nQ: {question}\nA: {answer[:600]}"
            }],
        )
        raw = resp.content[0].text.strip()
        suggestions = json.loads(raw)
        if isinstance(suggestions, list):
            return suggestions[:3]
    except Exception:
        pass
    return []


# ── Attachment handling ───────────────────────────────────────────────────────

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
SUPPORTED_TEXT_TYPES  = {"text/plain", "text/markdown", "text/csv", "application/json"}


def _mime(file) -> str:
    guessed, _ = mimetypes.guess_type(file.name)
    return guessed or file.type or "application/octet-stream"


def process_attachment(file) -> dict:
    mime = _mime(file)
    raw = file.read()
    if mime in SUPPORTED_IMAGE_TYPES:
        return {"kind": "image", "name": file.name, "mime": mime,
                "data": base64.b64encode(raw).decode()}
    if mime in SUPPORTED_TEXT_TYPES or file.name.endswith((".md", ".txt", ".csv")):
        return {"kind": "text", "name": file.name, "mime": mime,
                "content": raw.decode("utf-8", errors="replace")}
    if mime == "application/pdf" or file.name.lower().endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            text = "[PDF uploaded but pypdf is not installed — run: pip install pypdf]"
        return {"kind": "text", "name": file.name, "mime": "application/pdf", "content": text}
    try:
        content = raw.decode("utf-8", errors="replace")
    except Exception:
        content = "[Binary file — cannot display content]"
    return {"kind": "text", "name": file.name, "mime": mime, "content": content}


def attachments_to_claude_blocks(attachments: list[dict], query: str) -> list:
    blocks = []
    for att in attachments:
        if att["kind"] == "image":
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": att["mime"], "data": att["data"]}})
        else:
            blocks.append({"type": "text",
                           "text": f"[Attached file: {att['name']}]\n\n{att['content']}"})
    blocks.append({"type": "text", "text": query})
    return blocks


def render_attachments(attachments: list[dict]):
    for att in attachments:
        if att["kind"] == "image":
            st.image(base64.b64decode(att["data"]), caption=att["name"], use_container_width=True)
        else:
            with st.expander(f"📎 {att['name']}"):
                preview = att["content"][:2000]
                if len(att["content"]) > 2000:
                    preview += "\n\n… (truncated)"
                st.text(preview)


# ── Source rendering ──────────────────────────────────────────────────────────

def source_url(s: dict, podcast_urls: dict) -> str:
    filename = s.get("filename", "")
    if s["type"] == "podcast":
        yt = podcast_urls.get(filename, "")
        if yt:
            return yt
        query = s.get("guest") or s.get("title", "")
        return f"https://www.youtube.com/results?search_query=Lenny+podcast+{query.replace(' ', '+')}"
    else:
        slug = Path(filename).stem
        if slug:
            return f"https://www.lennysnewsletter.com/p/{slug}"
    return ""


def render_sources(sources: list[dict], podcast_urls: dict):
    for s in sources:
        if s["type"] == "podcast":
            icon, kind = "🎙️", "Podcast"
            label = s.get("guest") or s["title"]
        else:
            icon, kind = "📰", "Newsletter"
            label = s["title"]
        url = source_url(s, podcast_urls)
        link = f' · [Open ↗]({url})' if url else ""
        score_pct = f'{s["score"] * 100:.0f}% match' if s.get("score") else ""
        score_tag = f' · <small style="color:#888">{score_pct}</small>' if score_pct else ""
        header = f"{icon} **{kind}** · {label} · <small>{s['date']}</small>{score_tag}{link}"
        st.markdown(header, unsafe_allow_html=True)
        if s.get("text"):
            snippet = s["text"][:400].rsplit(" ", 1)[0] + " …"
            st.markdown(
                f"<blockquote style='margin:4px 0 12px 0;padding:6px 12px;"
                f"border-left:3px solid #ccc;color:#555;font-size:0.85em;"
                f"white-space:pre-wrap;'>{snippet}</blockquote>",
                unsafe_allow_html=True,
            )


# ── Feedback ─────────────────────────────────────────────────────────────────

def save_feedback(msg_id: str, question: str, answer: str, rating: str, text: str = ""):
    records = []
    if FEEDBACK_PATH.exists():
        try:
            records = json.loads(FEEDBACK_PATH.read_text())
        except Exception:
            records = []
    records.append({
        "id": msg_id,
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "rating": rating,
        "feedback_text": text,
    })
    FEEDBACK_PATH.write_text(json.dumps(records, indent=2))


def render_feedback(msg_id: str, question: str, answer: str):
    fb_key       = f"fb_{msg_id}"
    text_key     = f"fbt_{msg_id}"
    submitted_key = f"fbs_{msg_id}"

    if st.session_state.get(submitted_key):
        st.caption("✅ Thanks for your feedback!")
        return

    cols = st.columns([2, 2, 6])
    with cols[0]:
        if st.button("👍", key=f"up_{msg_id}", help="Helpful"):
            save_feedback(msg_id, question, answer, "up")
            st.session_state[submitted_key] = True
            st.toast("Thanks for the feedback!", icon="👍")
            st.rerun()
    with cols[1]:
        if st.button("👎", key=f"dn_{msg_id}", help="Not helpful"):
            st.session_state[fb_key] = "down"

    if st.session_state.get(fb_key) == "down":
        feedback_text = st.text_area(
            "What could be better?", key=text_key, height=80,
            label_visibility="collapsed", placeholder="Tell us what could be better…"
        )
        if st.button("Submit feedback", key=f"fbsubmit_{msg_id}"):
            save_feedback(msg_id, question, answer, "down", feedback_text)
            st.session_state[submitted_key] = True
            st.session_state.pop(fb_key, None)
            st.toast("Feedback submitted — thank you!", icon="✅")
            st.rerun()


# ── Copy button ───────────────────────────────────────────────────────────────

def render_copy_button(msg_id: str, text: str):
    copy_key = f"copy_open_{msg_id}"
    if st.button("📋 Copy", key=f"copy_{msg_id}", help="Copy response text"):
        st.session_state[copy_key] = not st.session_state.get(copy_key, False)
    if st.session_state.get(copy_key):
        st.text_area("Select all (Cmd+A) then copy (Cmd+C):",
                     value=text, height=120, key=f"copy_ta_{msg_id}",
                     label_visibility="visible")


# ── Export conversation ───────────────────────────────────────────────────────

def export_conversation(conv: dict) -> str:
    lines = [f"# {conv['title']}\n"]
    for msg in conv["messages"]:
        role = "**You**" if msg["role"] == "user" else "**Ask Lenny**"
        lines.append(f"{role}\n\n{msg['content']}\n")
        if msg["role"] == "assistant" and msg.get("sources"):
            lines.append("*Sources: " + ", ".join(
                s.get("title", "") for s in msg["sources"][:3]
            ) + "*\n")
        lines.append("---\n")
    return "\n".join(lines)


# ── Follow-up chips ───────────────────────────────────────────────────────────

def render_followup_chips(msg_id: str, suggestions: list[str]):
    if not suggestions:
        return
    st.markdown("<div style='margin:6px 0 2px 0;font-size:0.78em;color:#888'>Suggested follow-ups:</div>",
                unsafe_allow_html=True)
    cols = st.columns(len(suggestions))
    for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(suggestion, key=f"chip_{msg_id}_{i}",
                         use_container_width=True, type="secondary"):
                st.session_state["prefill_prompt"] = suggestion
                st.rerun()


# ── Conversation helpers ──────────────────────────────────────────────────────

def new_conversation() -> dict:
    return {"id": str(uuid.uuid4()), "title": "New chat", "messages": [], "history": []}


def load_conversations() -> tuple[list, str]:
    if CONVERSATIONS_PATH.exists():
        try:
            data = json.loads(CONVERSATIONS_PATH.read_text())
            convs = data.get("conversations", [])
            active_id = data.get("active_id", "")
            if convs:
                return convs, active_id
        except Exception:
            pass
    conv = new_conversation()
    return [conv], conv["id"]


def save_conversations():
    CONVERSATIONS_PATH.write_text(json.dumps({
        "active_id": st.session_state.active_id,
        "conversations": st.session_state.conversations,
    }, indent=2))


def active_conv() -> dict:
    cid = st.session_state.active_id
    for c in st.session_state.conversations:
        if c["id"] == cid:
            return c
    return st.session_state.conversations[0]


def set_active(cid: str):
    st.session_state.active_id = cid


def delete_conv(cid: str):
    st.session_state.conversations = [
        c for c in st.session_state.conversations if c["id"] != cid
    ]
    if not st.session_state.conversations:
        conv = new_conversation()
        st.session_state.conversations.append(conv)
        st.session_state.active_id = conv["id"]
    elif st.session_state.active_id == cid:
        st.session_state.active_id = st.session_state.conversations[0]["id"]
    save_conversations()


# ── Session state ─────────────────────────────────────────────────────────────

if "conversations" not in st.session_state:
    convs, active_id = load_conversations()
    st.session_state.conversations = convs
    st.session_state.active_id = (
        active_id if any(c["id"] == active_id for c in convs) else convs[0]["id"]
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Ask Lenny")
    st.caption("349 newsletters · 289 podcasts")
    st.divider()

    if st.button("✏️  New chat", use_container_width=True, type="secondary"):
        conv = new_conversation()
        st.session_state.conversations.insert(0, conv)
        set_active(conv["id"])
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    for c in st.session_state.conversations:
        is_active = c["id"] == st.session_state.active_id
        cols = st.columns([11, 1], gap="small")
        active_class = "conv-btn-active" if is_active else "conv-btn"
        with cols[0]:
            st.markdown(f'<div class="conv-btn {active_class}">', unsafe_allow_html=True)
            if st.button(c["title"], key=f"btn_{c['id']}", use_container_width=True):
                set_active(c["id"])
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown('<div class="del-btn">', unsafe_allow_html=True)
            if st.button("×", key=f"del_{c['id']}"):
                delete_conv(c["id"])
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Export active conversation
    conv_for_export = active_conv()
    if conv_for_export["messages"]:
        md = export_conversation(conv_for_export)
        st.download_button(
            "⬇️ Export chat",
            data=md,
            file_name=f"{conv_for_export['title'][:40]}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.caption("sentence-transformers · FAISS · Claude")

# ── Load resources ────────────────────────────────────────────────────────────

index, chunks, embed_model, podcast_urls = load_resources()
conv = active_conv()

# ── Render conversation ───────────────────────────────────────────────────────

if not conv["messages"]:
    st.markdown(
        "<div style='text-align:center;margin-top:80px;color:#888;'>"
        "<h3 style='font-size:1.5em;margin-bottom:8px;'>Ask Lenny anything</h3>"
        "<p>349 newsletters · 289 podcasts on product, growth, and leadership</p>"
        "</div>",
        unsafe_allow_html=True,
    )

for i, msg in enumerate(conv["messages"]):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg.get("attachments"):
            render_attachments(msg["attachments"])
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("confidence"):
                render_confidence(msg["confidence"])

            # Action row: copy + regenerate
            action_cols = st.columns([2, 2, 6])
            with action_cols[0]:
                render_copy_button(msg.get("id", f"legacy_{i}"), msg["content"])
            with action_cols[1]:
                if st.button("🔁", key=f"regen_{msg.get('id', i)}", help="Regenerate response"):
                    st.session_state["regenerate_idx"] = i
                    st.rerun()

            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})", expanded=False):
                    render_sources(msg["sources"], podcast_urls)

            prev_question = conv["messages"][i - 1]["content"] if i > 0 else ""
            render_feedback(msg.get("id", f"legacy_{i}"), prev_question, msg["content"])

            if msg.get("followups"):
                render_followup_chips(msg.get("id", f"legacy_{i}"), msg["followups"])

# ── Handle regeneration ───────────────────────────────────────────────────────

if "regenerate_idx" in st.session_state:
    regen_idx = st.session_state.pop("regenerate_idx")
    # Find the user message just before this assistant message
    if regen_idx > 0:
        user_msg = conv["messages"][regen_idx - 1]
        re_prompt = user_msg["content"]
        re_attachments = user_msg.get("attachments")

        # Remove the old assistant message
        conv["messages"] = conv["messages"][:regen_idx]
        # Rebuild history up to that point
        conv["history"] = []
        for m in conv["messages"]:
            if m["role"] == "user":
                conv["history"].append({"role": "user", "content": m["content"]})
            else:
                conv["history"].append({"role": "assistant", "content": m["content"]})
        if len(conv["history"]) > MAX_HISTORY_TURNS * 2:
            conv["history"] = conv["history"][-(MAX_HISTORY_TURNS * 2):]

        results = retrieve(re_prompt, index, chunks, embed_model)
        context = format_context(results)
        confidence = compute_confidence(results)
        sources = [{"type": r["type"], "title": r["title"], "guest": r.get("guest", ""),
                    "date": r["date"], "text": r["text"], "filename": r.get("filename", ""),
                    "score": r["score"]} for r in results]

        msg_id = str(uuid.uuid4())
        with st.chat_message("assistant"):
            try:
                response_text = st.write_stream(
                    stream_response(re_prompt, conv["history"], context, re_attachments)
                )
            except Exception as e:
                st.error(f"Failed to regenerate response: {e}")
                st.stop()

            render_confidence(confidence)
            action_cols = st.columns([1, 1, 8])
            with action_cols[0]:
                render_copy_button(msg_id, response_text)
            with st.expander(f"Sources ({len(sources)})", expanded=False):
                render_sources(sources, podcast_urls)
            render_feedback(msg_id, re_prompt, response_text)

        followups = get_followup_suggestions(re_prompt, response_text)
        conv["messages"].append({
            "role": "assistant", "content": response_text,
            "sources": sources, "confidence": confidence,
            "id": msg_id, "followups": followups,
        })
        save_conversations()
        st.rerun()

# ── Turn count & model selection ─────────────────────────────────────────────

def count_turns(conv: dict) -> int:
    return sum(1 for m in conv["messages"] if m["role"] == "assistant")

def active_model(conv: dict) -> str:
    turns = count_turns(conv)
    if turns >= TURN_WARNING:
        return HAIKU_MODEL
    return CLAUDE_MODEL

def render_turn_warning(conv: dict):
    turns = count_turns(conv)
    if turns == TURN_WARNING:
        st.warning(
            f"**You've reached {TURN_WARNING} turns.** Long conversations reduce response quality "
            f"as earlier context gets deprioritized. We recommend starting a new chat to keep answers sharp.",
            icon="⚠️",
        )
        if st.button("✏️ Start a new chat", key="warn_new_chat", type="primary"):
            new_conv = new_conversation()
            st.session_state.conversations.insert(0, new_conv)
            set_active(new_conv["id"])
            st.rerun()
    elif turns > TURN_WARNING:
        st.info(
            f"**Switched to Claude Haiku** (`{HAIKU_MODEL}`) to keep this conversation running. "
            f"Haiku is faster and cheaper but has limitations: shorter context window, less nuanced "
            f"reasoning, may miss subtle connections across Lenny's content, and less detailed responses. "
            f"**Start a new chat** for best results.",
            icon="ℹ️",
        )

# ── Handle new input ──────────────────────────────────────────────────────────

render_turn_warning(conv)

uploaded_files = st.file_uploader(
    "➕ Attach",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg", "gif", "webp", "pdf", "txt", "md", "csv", "json"],
    key=f"upload_{st.session_state.active_id}",
)

# Support follow-up chip pre-fill
default_prompt = st.session_state.pop("prefill_prompt", "")

prompt = st.chat_input("Ask about product, growth, strategy, leadership…")
if default_prompt and not prompt:
    prompt = default_prompt

if prompt:
    attachments = [process_attachment(f) for f in (uploaded_files or [])]

    if not conv["messages"]:
        conv["title"] = prompt[:45] + ("…" if len(prompt) > 45 else "")

    with st.chat_message("user"):
        if attachments:
            render_attachments(attachments)
        st.markdown(prompt)

    conv["messages"].append({"role": "user", "content": prompt, "attachments": attachments})

    results = retrieve(prompt, index, chunks, embed_model)
    context = format_context(results)
    confidence = compute_confidence(results)
    sources = [{"type": r["type"], "title": r["title"], "guest": r.get("guest", ""),
                "date": r["date"], "text": r["text"], "filename": r.get("filename", ""),
                "score": r["score"]} for r in results]

    model_to_use = active_model(conv)
    msg_id = str(uuid.uuid4())
    with st.chat_message("assistant"):
        try:
            response_text = st.write_stream(
                stream_response(prompt, conv["history"], context, attachments or None,
                                model=model_to_use)
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}. Please try again.")
            conv["messages"].pop()  # remove the user message we just added
            st.stop()

        render_confidence(confidence)
        action_cols = st.columns([1, 1, 8])
        with action_cols[0]:
            render_copy_button(msg_id, response_text)
        with st.expander(f"Sources ({len(sources)})", expanded=False):
            render_sources(sources, podcast_urls)
        render_feedback(msg_id, prompt, response_text)

    followups = get_followup_suggestions(prompt, response_text)
    if followups:
        with st.chat_message("assistant"):
            render_followup_chips(msg_id, followups)

    # Update history
    history_content: list = []
    for att in attachments:
        if att["kind"] == "text":
            history_content.append({"type": "text",
                                    "text": f"[File: {att['name']}]\n{att['content'][:1000]}"})
    history_content.append({"type": "text", "text": prompt})
    conv["history"].append({"role": "user",
                            "content": history_content if len(history_content) > 1 else prompt})
    conv["history"].append({"role": "assistant", "content": response_text})
    if len(conv["history"]) > MAX_HISTORY_TURNS * 2:
        conv["history"] = conv["history"][-(MAX_HISTORY_TURNS * 2):]

    conv["messages"].append({
        "role": "assistant", "content": response_text,
        "sources": sources, "confidence": confidence,
        "id": msg_id, "followups": followups,
    })
    save_conversations()
