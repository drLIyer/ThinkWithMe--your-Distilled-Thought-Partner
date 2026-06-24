"""
Distill Admin Dashboard — FastAPI

Run with:
    .venv312/bin/python3 admin_server.py

Access at: http://localhost:8082/admin
Only accessible from this machine. Password protected.
"""

import json
import secrets
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request, Response, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
import uvicorn

DB_PATH              = Path("/Users/liyer_1/lennys-rag/asklenny.db")
GAPS_PATH            = Path("/Users/liyer_1/lennys-rag/knowledge_gaps.json")
FEEDBACK_PATH        = Path("/Users/liyer_1/lennys-rag/feedback.json")
UPDATE_LOG           = Path("/Users/liyer_1/lennys-rag/update_log.txt")
UPDATE_SCRIPT        = Path("/Users/liyer_1/lennys-rag/run_update.sh")
HEALTH_EVENTS_PATH   = Path("/Users/liyer_1/lennys-rag/health_events.json")
HEALTH_LOG_PATH      = Path("/Users/liyer_1/lennys-rag/health_agent.log")
BOOKMARKS_PATH       = Path("/Users/liyer_1/lennys-rag/bookmarks.json")

ADMIN_PASSWORD = "distill-admin-2024"
SESSION_TOKEN  = secrets.token_hex(32)   # generated fresh each server start

app = FastAPI()


# ── Auth helpers ──────────────────────────────────────────────────────────────

def is_authed(request: Request) -> bool:
    return request.cookies.get("admin_session") == SESSION_TOKEN


def require_auth(request: Request):
    if not is_authed(request):
        raise HTTPException(status_code=303, headers={"Location": "/admin/login"})


# ── Data helpers ──────────────────────────────────────────────────────────────

def db_query(sql: str) -> list[dict]:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def load_metrics() -> dict:
    threads   = db_query("SELECT * FROM threads ORDER BY createdAt DESC")
    steps     = db_query("SELECT * FROM steps")
    feedbacks = db_query("SELECT * FROM feedbacks")

    fb_file = []
    if FEEDBACK_PATH.exists():
        try:
            fb_file = json.loads(FEEDBACK_PATH.read_text())
        except Exception:
            pass

    gaps = []
    if GAPS_PATH.exists():
        try:
            gaps = json.loads(GAPS_PATH.read_text())
        except Exception:
            pass

    # Per-thread turn counts
    turn_map: dict[str, int] = {}
    model_map: dict[str, str] = {}
    conf_counts = {"high": 0, "medium": 0, "low": 0}
    model_counts: dict[str, int] = {}
    model_switch_threads: list[str] = []

    for s in steps:
        tid = s.get("threadId", "")
        if s.get("type") == "user_message":
            turn_map[tid] = turn_map.get(tid, 0) + 1
        if s.get("type") == "assistant_message":
            try:
                props = json.loads(s.get("props") or "{}")
            except Exception:
                props = {}
            conf = props.get("confidence", "")
            model = props.get("model", "")
            if conf in conf_counts:
                conf_counts[conf] += 1
            if model:
                model_counts[model] = model_counts.get(model, 0) + 1
                cur = model_map.get(tid)
                if cur and cur != model:
                    model_switch_threads.append(tid)
                model_map[tid] = model

    for t in threads:
        t["turns"] = turn_map.get(t["id"], 0)

    total_turns = sum(t["turns"] for t in threads)
    avg_turns   = round(total_turns / len(threads), 1) if threads else 0

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_threads = [t for t in threads if (t.get("createdAt") or "").startswith(today_str)]

    thumbs_up   = len([f for f in fb_file if f.get("rating") == "up"])
    thumbs_down = len([f for f in fb_file if f.get("rating") == "down"])

    update_log = ""
    if UPDATE_LOG.exists():
        lines = UPDATE_LOG.read_text().splitlines()
        update_log = "\n".join(lines[-40:])

    health_events = []
    if HEALTH_EVENTS_PATH.exists():
        try:
            health_events = json.loads(HEALTH_EVENTS_PATH.read_text())[-50:]
        except Exception:
            pass

    health_log = ""
    if HEALTH_LOG_PATH.exists():
        lines = HEALTH_LOG_PATH.read_text().splitlines()
        health_log = "\n".join(lines[-40:])

    bookmarks = []
    if BOOKMARKS_PATH.exists():
        try:
            bookmarks = json.loads(BOOKMARKS_PATH.read_text())[-30:][::-1]
        except Exception:
            pass

    return {
        "total_threads":    len(threads),
        "today_threads":    len(today_threads),
        "total_questions":  sum(turn_map.values()),
        "avg_turns":        avg_turns,
        "thumbs_up":        thumbs_up,
        "thumbs_down":      thumbs_down,
        "conf_counts":      conf_counts,
        "model_counts":     model_counts,
        "model_switches":   len(set(model_switch_threads)),
        "threads":          threads[:50],
        "gaps":             gaps[-20:][::-1],
        "feedback":         fb_file[-30:][::-1],
        "update_log":       update_log,
        "health_events":    health_events,
        "health_log":       health_log,
        "bookmarks":        bookmarks,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/admin/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(LOGIN_HTML)


@app.post("/admin/login")
async def login(response: Response, password: str = Form(...)):
    if password == ADMIN_PASSWORD:
        resp = RedirectResponse("/admin", status_code=302)
        resp.set_cookie("admin_session", SESSION_TOKEN, httponly=True, samesite="strict")
        return resp
    return HTMLResponse(LOGIN_HTML.replace("</form>", '<p style="color:#e74c3c">Incorrect password</p></form>'))


@app.get("/admin/logout")
async def logout(response: Response):
    resp = RedirectResponse("/admin/login", status_code=302)
    resp.delete_cookie("admin_session")
    return resp


@app.get("/admin", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not is_authed(request):
        return RedirectResponse("/admin/login", status_code=302)
    m = load_metrics()
    return HTMLResponse(render_dashboard(m))


@app.post("/admin/update")
async def run_update(request: Request):
    if not is_authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        result = subprocess.run(
            ["/bin/bash", str(UPDATE_SCRIPT)],
            capture_output=True, text=True,
            cwd="/Users/liyer_1/lennys-rag", timeout=300
        )
        return JSONResponse({"ok": result.returncode == 0, "output": result.stdout[-2000:]})
    except Exception as e:
        return JSONResponse({"ok": False, "output": str(e)})


@app.get("/admin/metrics")
async def metrics_json(request: Request):
    if not is_authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return JSONResponse(load_metrics())


# ── HTML templates ────────────────────────────────────────────────────────────

LOGIN_HTML = """<!DOCTYPE html>
<html>
<head><title>Distill Admin</title>
<meta charset="UTF-8">
<style>
  body{font-family:-apple-system,sans-serif;background:#0f0f0f;color:#eee;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
  .box{background:#1a1a1a;border-radius:12px;padding:40px;width:320px;text-align:center}
  h2{margin:0 0 24px;font-size:1.4rem}
  input{width:100%;padding:10px;border-radius:8px;border:1px solid #333;background:#111;color:#eee;font-size:1rem;box-sizing:border-box;margin-bottom:12px}
  button{width:100%;padding:10px;background:#6366f1;color:#fff;border:none;border-radius:8px;font-size:1rem;cursor:pointer}
  button:hover{background:#4f46e5}
</style>
</head>
<body>
<div class="box">
  <h2>⚙️ Distill Admin</h2>
  <form method="post" action="/admin/login">
    <input type="password" name="password" placeholder="Password" autofocus>
    <button type="submit">Sign in</button>
  </form>
</div>
</body>
</html>"""


def bar(value: int, total: int, color: str) -> str:
    pct = round(100 * value / total) if total else 0
    return f'<div style="background:#222;border-radius:4px;height:8px;margin:4px 0 12px"><div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div></div>'


def render_dashboard(m: dict) -> str:
    conf  = m["conf_counts"]
    total_conf = sum(conf.values()) or 1
    models = m["model_counts"]
    total_models = sum(models.values()) or 1

    # ── Bookmarks section data ────────────────────────────────────────────────
    bookmark_rows = "".join(
        f'<tr>'
        f'<td style="color:#aaa;white-space:nowrap">{(b.get("timestamp") or "")[:10]}</td>'
        f'<td>{(b.get("question") or "")[:80]}</td>'
        f'<td style="color:#aaa">{(b.get("answer") or "")[:120]}…</td>'
        f'<td style="color:#555">{(b.get("thread_name") or "")[:40]}</td>'
        f'</tr>'
        for b in m.get("bookmarks", []) if b
    ) or '<tr><td colspan="4" style="color:#555;text-align:center">No saved responses yet</td></tr>'

    # ── Health section data ────────────────────────────────────────────────────
    health_events = m.get("health_events", [])
    last_event = health_events[-1] if health_events else None
    last_check = last_event["timestamp"][:16].replace("T", " ") + " UTC" if last_event else "Never"
    overall_sev = "ok"
    for ev in health_events[-5:]:
        s = ev.get("severity", "ok")
        if s == "critical":
            overall_sev = "critical"
            break
        if s == "warning":
            overall_sev = "warning"
    sev_badge = {"ok": "🟢 OK", "warning": "🟠 Warning", "critical": "🔴 Critical"}[overall_sev]
    sev_color = {"ok": "#27ae60", "warning": "#e67e22", "critical": "#e74c3c"}[overall_sev]

    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    cutoff = (_dt.now(_tz.utc) - _td(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
    fixes_24h = sum(len(ev.get("fixes", [])) for ev in health_events if ev.get("timestamp", "") >= cutoff)

    health_event_rows = ""
    for ev in reversed(health_events[-50:]):
        ts   = ev.get("timestamp", "")[:16].replace("T", " ")
        sev  = ev.get("severity", "ok")
        row_bg = {"ok": "#1a2e1a", "warning": "#2e2a1a", "critical": "#2e1a1a"}.get(sev, "#1a1a1a")
        badge  = {"ok": "🟢", "warning": "🟠", "critical": "🔴"}.get(sev, "")
        fixes_col  = "<br>".join(f"+ {f}" for f in ev.get("fixes", [])) or "—"
        alerts_col = "<br>".join(f"! {a}" for a in ev.get("alerts", [])) or "—"
        info_col   = "<br>".join(f". {i}" for i in ev.get("info", [])) or "—"
        health_event_rows += (
            f'<tr style="background:{row_bg}">'
            f'<td style="color:#aaa;white-space:nowrap">{ts}</td>'
            f'<td style="text-align:center">{badge}</td>'
            f'<td style="color:#4ade80;font-size:0.8rem">{fixes_col}</td>'
            f'<td style="color:#fbbf24;font-size:0.8rem">{alerts_col}</td>'
            f'<td style="color:#94a3b8;font-size:0.8rem">{info_col}</td>'
            f'</tr>'
        )
    if not health_event_rows:
        health_event_rows = '<tr><td colspan="5" style="color:#555;text-align:center">No health events yet — agent may not be running</td></tr>'

    model_rows = ""
    for mod, cnt in sorted(models.items(), key=lambda x: -x[1]):
        short = "Sonnet" if "sonnet" in mod.lower() else "Haiku" if "haiku" in mod.lower() else mod
        pct = round(100 * cnt / total_models)
        color = "#6366f1" if "sonnet" in mod.lower() else "#f59e0b"
        model_rows += f'<div class="stat-label">{short} <span style="float:right;color:#aaa">{cnt} ({pct}%)</span></div>{bar(cnt,total_models,color)}'

    gap_rows = "".join(
        f'<tr><td>{(g.get("timestamp") or "")[:16]}</td><td>{(g.get("query") or "")[:70]}</td>'
        f'<td style="color:#e74c3c">{g.get("top_score") or 0:.2f}</td>'
        f'<td style="color:#aaa">{(g.get("best_source") or "")[:40]}</td></tr>'
        for g in m["gaps"] if g
    ) or '<tr><td colspan="4" style="color:#555;text-align:center">No gaps logged yet</td></tr>'

    fb_rows = "".join(
        f'<tr><td>{(f.get("timestamp") or "")[:16]}</td>'
        f'<td>{"👍" if f.get("rating")=="up" else "👎"}</td>'
        f'<td>{(f.get("question") or "")[:60]}</td>'
        f'<td style="color:#aaa">{(f.get("feedback_text") or "")[:60]}</td></tr>'
        for f in m["feedback"]
    ) or '<tr><td colspan="4" style="color:#555;text-align:center">No feedback yet</td></tr>'

    thread_rows = "".join(
        f'<tr><td>{(t.get("name") or "")[:55]}</td>'
        f'<td style="color:#aaa">{(t.get("createdAt") or "")[:16]}</td>'
        f'<td style="text-align:center">{t.get("turns") or 0}</td></tr>'
        for t in m["threads"] if t
    )

    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    return f"""<!DOCTYPE html>
<html>
<head>
<title>Distill Admin</title>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif;background:#0f0f0f;color:#eee;padding:24px}}
  h1{{font-size:1.5rem;margin-bottom:4px}}
  .sub{{color:#666;font-size:0.85rem;margin-bottom:24px}}
  .grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:24px}}
  .card{{background:#1a1a1a;border-radius:10px;padding:16px}}
  .card .val{{font-size:2rem;font-weight:700;margin:4px 0}}
  .card .lbl{{font-size:0.78rem;color:#888;text-transform:uppercase;letter-spacing:.05em}}
  .section{{background:#1a1a1a;border-radius:10px;padding:20px;margin-bottom:16px}}
  .section h2{{font-size:1rem;margin-bottom:14px;color:#ccc}}
  .row2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
  .row3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}}
  .stat-label{{font-size:0.88rem;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;font-size:0.83rem}}
  th{{text-align:left;color:#666;font-weight:500;padding:6px 8px;border-bottom:1px solid #2a2a2a}}
  td{{padding:6px 8px;border-bottom:1px solid #1e1e1e;vertical-align:top}}
  tr:hover td{{background:#1f1f1f}}
  pre{{background:#111;border-radius:8px;padding:12px;font-size:0.78rem;overflow-x:auto;color:#aaa;max-height:200px;overflow-y:auto}}
  .btn{{background:#6366f1;color:#fff;border:none;border-radius:8px;padding:10px 20px;cursor:pointer;font-size:0.9rem;font-weight:500}}
  .btn:hover{{background:#4f46e5}}
  .btn-sm{{background:#222;color:#aaa;border:1px solid #333;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:0.8rem}}
  .btn-sm:hover{{background:#2a2a2a}}
  #update-out{{margin-top:12px;display:none}}
  .logout{{float:right;color:#555;font-size:0.8rem;text-decoration:none}}
  .logout:hover{{color:#aaa}}
  .switch-warn{{color:#f59e0b;font-size:0.82rem;margin-top:6px}}
</style>
</head>
<body>

<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
  <div>
    <h1>⚙️ Distill Admin Dashboard</h1>
    <div class="sub">Only visible to you · Auto-refreshes every 30s · Last loaded: {now}</div>
  </div>
  <a href="/admin/logout" class="logout">Sign out</a>
</div>

<!-- Top metrics -->
<div class="grid">
  <div class="card"><div class="lbl">Total Conversations</div><div class="val">{m["total_threads"]}</div></div>
  <div class="card"><div class="lbl">Today</div><div class="val">{m["today_threads"]}</div></div>
  <div class="card"><div class="lbl">Questions Asked</div><div class="val">{m["total_questions"]}</div></div>
  <div class="card"><div class="lbl">Avg Turns / Chat</div><div class="val">{m["avg_turns"]}</div></div>
  <div class="card"><div class="lbl">Feedback</div><div class="val">👍 {m["thumbs_up"]} &nbsp; 👎 {m["thumbs_down"]}</div></div>
</div>

<div class="row3">
  <!-- Coverage -->
  <div class="section">
    <h2>Coverage Distribution</h2>
    <div class="stat-label">🟢 High &nbsp;<span style="float:right;color:#aaa">{conf["high"]} ({round(100*conf["high"]/total_conf)}%)</span></div>
    {bar(conf["high"],total_conf,"#27ae60")}
    <div class="stat-label">🟠 Moderate &nbsp;<span style="float:right;color:#aaa">{conf["medium"]} ({round(100*conf["medium"]/total_conf)}%)</span></div>
    {bar(conf["medium"],total_conf,"#e67e22")}
    <div class="stat-label">🔴 Limited &nbsp;<span style="float:right;color:#aaa">{conf["low"]} ({round(100*conf["low"]/total_conf)}%)</span></div>
    {bar(conf["low"],total_conf,"#e74c3c")}
    {"<div class='sub' style='margin-top:8px;color:#555'>Data populates after new conversations</div>" if total_conf==1 else ""}
  </div>

  <!-- Model usage -->
  <div class="section">
    <h2>Model Usage</h2>
    {model_rows or '<div style="color:#555;font-size:0.85rem">No data yet</div>'}
    {f'<div class="switch-warn">⚠️ Model switched to Haiku in {m["model_switches"]} conversation(s) — 15-turn limit hit</div>' if m["model_switches"] else ""}
  </div>

  <!-- Knowledge update -->
  <div class="section">
    <h2>Knowledge Base Update</h2>
    <p style="font-size:0.83rem;color:#888;margin-bottom:12px">Fetches latest Lenny content, updates the index, restarts Distill, emails you a summary.</p>
    <button class="btn" onclick="runUpdate()">🔄 Run Update Now</button>
    <div id="update-out">
      <pre id="update-log-out"></pre>
    </div>
  </div>
</div>

<!-- Knowledge Gaps -->
<div class="section">
  <h2>Knowledge Gaps &nbsp;<span style="color:#555;font-weight:400;font-size:0.85rem">— queries where Lenny's content didn't have a strong answer</span></h2>
  <table>
    <tr><th>When</th><th>Question Asked</th><th>Best Match Score</th><th>Closest Source</th></tr>
    {gap_rows}
  </table>
</div>

<!-- Feedback -->
<div class="section">
  <h2>User Feedback</h2>
  <table>
    <tr><th>When</th><th>Rating</th><th>Question</th><th>Comment</th></tr>
    {fb_rows}
  </table>
</div>

<!-- Bookmarks -->
<div class="section">
  <h2>📌 Saved Responses</h2>
  <table>
    <tr><th>Date</th><th>Question</th><th>Answer snippet</th><th>Conversation</th></tr>
    {bookmark_rows}
  </table>
</div>

<!-- Conversations -->
<div class="section">
  <h2>All Conversations</h2>
  <table>
    <tr><th>Title</th><th>Started</th><th style="text-align:center">Turns</th></tr>
    {thread_rows}
  </table>
</div>

<!-- Update log -->
<div class="section">
  <h2>Update Log</h2>
  <pre>{m["update_log"] or "No updates run yet."}</pre>
</div>

<!-- Health Agent -->
<div class="section">
  <h2>🩺 Health Agent</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px">
    <div class="card"><div class="lbl">Last Check</div><div style="font-size:1rem;font-weight:600;margin-top:6px">{last_check}</div></div>
    <div class="card"><div class="lbl">Status (last 5 cycles)</div><div style="font-size:1.2rem;font-weight:700;margin-top:6px;color:{sev_color}">{sev_badge}</div></div>
    <div class="card"><div class="lbl">Auto-Fixes (24h)</div><div class="val">{fixes_24h}</div></div>
  </div>

  <table style="margin-bottom:16px">
    <tr>
      <th style="width:120px">Time (UTC)</th>
      <th style="width:40px;text-align:center">Status</th>
      <th>Auto-Fixed</th>
      <th>Alerts</th>
      <th>Info</th>
    </tr>
    {health_event_rows}
  </table>

  <h2 style="margin-top:4px;margin-bottom:8px;font-size:0.9rem;color:#888">Health Agent Log (last 40 lines)</h2>
  <pre>{m.get("health_log") or "health_agent.log not found — is the agent running?"}</pre>
</div>

<script>
async function runUpdate() {{
  document.getElementById('update-out').style.display = 'block';
  document.getElementById('update-log-out').textContent = 'Running… (this takes ~30 seconds)';
  try {{
    const r = await fetch('/admin/update', {{method:'POST'}});
    const d = await r.json();
    document.getElementById('update-log-out').textContent = d.output || 'Done.';
    if (d.ok) setTimeout(() => location.reload(), 2000);
  }} catch(e) {{
    document.getElementById('update-log-out').textContent = 'Error: ' + e;
  }}
}}
</script>

</body>
</html>"""


if __name__ == "__main__":
    print("Distill Admin running at http://localhost:8082/admin")
    uvicorn.run(app, host="127.0.0.1", port=8082, log_level="warning")
