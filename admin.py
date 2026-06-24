"""
Distill — Admin Dashboard

Run with:
    streamlit run admin.py --server.port 8082
"""

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH      = Path("asklenny.db")
GAPS_PATH    = Path("knowledge_gaps.json")
FEEDBACK_PATH = Path("feedback.json")
UPDATE_LOG   = Path("update_log.txt")

st.set_page_config(page_title="Distill Admin", page_icon="⚙️", layout="wide")

# ── Auth ──────────────────────────────────────────────────────────────────────
if "admin_auth" not in st.session_state:
    st.session_state.admin_auth = False

if not st.session_state.admin_auth:
    st.markdown("## Distill Admin")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if pwd == "distill2024":
            st.session_state.admin_auth = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_db():
    conn = sqlite3.connect(DB_PATH)

    threads = pd.read_sql("SELECT * FROM threads ORDER BY createdAt DESC", conn)
    steps   = pd.read_sql("SELECT * FROM steps", conn)
    feedbacks = pd.read_sql("SELECT * FROM feedbacks", conn)
    conn.close()

    # Parse props JSON on steps
    def parse_props(p):
        try:
            return json.loads(p) if p else {}
        except Exception:
            return {}

    steps["_props"] = steps["props"].apply(parse_props)
    steps["model"]      = steps["_props"].apply(lambda x: x.get("model", ""))
    steps["confidence"] = steps["_props"].apply(lambda x: x.get("confidence", ""))
    steps["turns"]      = steps["_props"].apply(lambda x: x.get("turns", 0))
    steps["top_score"]  = steps["_props"].apply(lambda x: x.get("top_score", 0))

    return threads, steps, feedbacks


@st.cache_data(ttl=30)
def load_gaps():
    if GAPS_PATH.exists():
        try:
            return json.loads(GAPS_PATH.read_text())
        except Exception:
            pass
    return []


@st.cache_data(ttl=30)
def load_feedback_file():
    if FEEDBACK_PATH.exists():
        try:
            return json.loads(FEEDBACK_PATH.read_text())
        except Exception:
            pass
    return []


threads, steps, feedbacks_db = load_db()
gaps      = load_gaps()
feedback  = load_feedback_file()

assistant_steps = steps[steps["type"] == "assistant_message"].copy()
user_steps      = steps[steps["type"] == "user_message"].copy()

# Per-thread stats
thread_turns = steps.groupby("threadId").apply(
    lambda g: (g["type"] == "user_message").sum()
).reset_index(columns=["turns"]) if not steps.empty else pd.DataFrame(columns=["threadId","turns"])
thread_turns.columns = ["threadId", "turns"]

threads = threads.merge(thread_turns, left_on="id", right_on="threadId", how="left")
threads["turns"] = threads["turns"].fillna(0).astype(int)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# ⚙️ Distill Admin Dashboard")
st.caption("Only visible to you. Refreshes every 30 seconds.")
st.divider()

# ── Top metrics ───────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

total_threads    = len(threads)
total_questions  = int(user_steps.shape[0])
avg_turns        = round(threads["turns"].mean(), 1) if total_threads else 0
thumbs_up        = len([f for f in feedback if f.get("rating") == "up"])
thumbs_down      = len([f for f in feedback if f.get("rating") == "down"])

col1.metric("Total Conversations", total_threads)
col2.metric("Total Questions Asked", total_questions)
col3.metric("Avg Turns / Conversation", avg_turns)
col4.metric("👍 Helpful", thumbs_up)
col5.metric("👎 Not Helpful", thumbs_down)

st.divider()

# ── Row 2: Coverage + Model ───────────────────────────────────────────────────

col_a, col_b, col_c = st.columns(3)

# Coverage breakdown
with col_a:
    st.subheader("Coverage Distribution")
    conf_data = assistant_steps[assistant_steps["confidence"] != ""]
    if not conf_data.empty:
        counts = conf_data["confidence"].value_counts().reindex(["high","medium","low"], fill_value=0)
        total_conf = counts.sum()
        for level, color, emoji in [("high","#27ae60","🟢"),("medium","#e67e22","🟠"),("low","#e74c3c","🔴")]:
            n = counts.get(level, 0)
            pct = round(100 * n / total_conf) if total_conf else 0
            st.markdown(
                f'{emoji} **{level.title()}** — {n} responses ({pct}%)'
            )
            st.progress(pct / 100)
    else:
        st.info("No coverage data yet — data populates after new conversations.")

# Model usage
with col_b:
    st.subheader("Model Usage")
    model_data = assistant_steps[assistant_steps["model"] != ""]
    if not model_data.empty:
        model_counts = model_data["model"].value_counts()
        for model_name, count in model_counts.items():
            short = "Sonnet" if "sonnet" in model_name.lower() else "Haiku" if "haiku" in model_name.lower() else model_name
            pct = round(100 * count / len(model_data))
            st.markdown(f"**{short}** — {count} responses ({pct}%)")
            st.progress(pct / 100)
        # Model switches (haiku used after sonnet in same thread)
        haiku_threads = set(model_data[model_data["model"].str.contains("haiku", case=False)]["threadId"])
        if haiku_threads:
            st.warning(f"⚠️ Model switched to Haiku in {len(haiku_threads)} conversation(s) — users hit the 15-turn limit.")
    else:
        st.info("No model data yet.")

# Knowledge gaps
with col_c:
    st.subheader("Knowledge Gaps")
    if gaps:
        st.caption(f"{len(gaps)} low-confidence queries logged")
        gap_df = pd.DataFrame(gaps)
        gap_df["timestamp"] = pd.to_datetime(gap_df["timestamp"])
        gap_df = gap_df.sort_values("timestamp", ascending=False)
        st.dataframe(
            gap_df[["timestamp","query","top_score","best_source"]].head(10),
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("When", format="MMM D, HH:mm"),
                "query":     st.column_config.TextColumn("Question", width="large"),
                "top_score": st.column_config.NumberColumn("Best Match", format="%.2f"),
                "best_source": st.column_config.TextColumn("Closest Source"),
            }
        )
    else:
        st.info("No gaps logged yet.")

st.divider()

# ── Conversations table ───────────────────────────────────────────────────────

st.subheader("All Conversations")

display_threads = threads[["name","createdAt","turns"]].copy()
display_threads.columns = ["Title","Started","Turns"]
display_threads["Started"] = pd.to_datetime(display_threads["Started"]).dt.strftime("%b %d %Y, %H:%M")

st.dataframe(
    display_threads,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Title":  st.column_config.TextColumn(width="large"),
        "Turns":  st.column_config.NumberColumn(width="small"),
    }
)

st.divider()

# ── Feedback detail ───────────────────────────────────────────────────────────

st.subheader("Feedback Detail")
if feedback:
    fb_df = pd.DataFrame(feedback)
    fb_df["timestamp"] = pd.to_datetime(fb_df["timestamp"])
    fb_df = fb_df.sort_values("timestamp", ascending=False)
    fb_df["rating"] = fb_df["rating"].map({"up": "👍", "down": "👎"})
    st.dataframe(
        fb_df[["timestamp","rating","question","feedback_text"]].head(50),
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp":     st.column_config.DatetimeColumn("When", format="MMM D, HH:mm"),
            "rating":        st.column_config.TextColumn("Rating", width="small"),
            "question":      st.column_config.TextColumn("Question Asked", width="large"),
            "feedback_text": st.column_config.TextColumn("User Feedback"),
        }
    )
else:
    st.info("No feedback submitted yet.")

st.divider()

# ── Update agent ──────────────────────────────────────────────────────────────

st.subheader("Knowledge Base Update")

col_btn, col_log = st.columns([1, 3])

with col_btn:
    st.caption("Fetches latest Lenny content, updates the index, restarts Distill, and emails you a summary.")
    if st.button("🔄 Run Update Now", type="primary", use_container_width=True):
        with st.spinner("Running update agent…"):
            result = subprocess.run(
                ["/bin/bash", "/Users/liyer_1/lennys-rag/run_update.sh"],
                capture_output=True, text=True, cwd="/Users/liyer_1/lennys-rag"
            )
        if result.returncode == 0:
            st.success("✅ Update complete — Distill restarted and email sent.")
        else:
            st.error(f"❌ Update failed:\n{result.stderr[:500]}")
        st.cache_data.clear()
        st.rerun()

with col_log:
    st.caption("Update log (last 30 lines)")
    if UPDATE_LOG.exists():
        lines = UPDATE_LOG.read_text().splitlines()[-30:]
        st.code("\n".join(lines), language="bash")
    else:
        st.info("No update log yet.")

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────

st.caption(f"Distill Admin · Last loaded: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} · Data refreshes every 30s")
if st.button("🔁 Refresh now"):
    st.cache_data.clear()
    st.rerun()
