"""
Distill Health Agent — background daemon

Runs every 60 seconds. Detects and auto-fixes common errors, writes structured
events to health_events.json (for the admin dashboard), logs to health_agent.log,
and sends an email digest at most once per hour when issues are found.

Start manually:  python3 health_agent.py
Daemon:          managed by com.distill.health.plist (LaunchAgent)
"""

import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR           = Path(__file__).parent
DB_PATH            = BASE_DIR / "asklenny.db"
FAISS_PATH         = BASE_DIR / "index.faiss"
FEEDBACK_PATH      = BASE_DIR / "feedback.json"
GAPS_PATH          = BASE_DIR / "knowledge_gaps.json"
LAUNCH_LOG         = BASE_DIR / "distill_launch.log"
HEALTH_LOG_PATH    = BASE_DIR / "health_agent.log"
HEALTH_EVENTS_PATH = BASE_DIR / "health_events.json"
VENV_CHAINLIT      = BASE_DIR / ".venv312/bin/chainlit"
VENV_PYTHON        = BASE_DIR / ".venv312/bin/python3"
CHAINLIT_APP       = BASE_DIR / "chainlit_app.py"
ADMIN_SERVER       = BASE_DIR / "admin_server.py"
ADMIN_LOG          = BASE_DIR / "admin.log"
RAG_SERVICE_LOG         = BASE_DIR / "rag_service.log"
BOOKMARKS_SERVICE_LOG   = BASE_DIR / "bookmarks_service.log"
USERS_SERVICE_LOG       = BASE_DIR / "users_service.log"
FEEDBACK_SERVICE_LOG    = BASE_DIR / "feedback_service.log"

# ── Tuning constants ──────────────────────────────────────────────────────────

POLL_INTERVAL         = 60      # seconds between cycles
EMAIL_COOLDOWN        = 3600    # max one email per hour
FAISS_STALE_DAYS      = 8
LOW_CONF_WINDOW       = 20      # last N gap entries to evaluate ratio
LOW_CONF_THRESHOLD    = 0.50    # >50% low-confidence → alert
CONTEXT_WINDOW_WARN   = 20      # threads with this many user turns
FEEDBACK_SPIKE_WINDOW = 10      # last N feedback entries
FEEDBACK_SPIKE_RATIO  = 0.60    # >60% thumbs-down → anomaly
API_ERROR_WINDOW      = 100     # last N log lines to scan
API_ERROR_THRESHOLD   = 5       # >N API errors in window → alert
DB_SIZE_LIMIT_MB      = 100
MAX_HEALTH_EVENTS     = 200     # rolling window kept in health_events.json
CHAINLIT_PORT         = 8081
ADMIN_PORT            = 8082
RAG_SERVICE_PORT      = 8083
BOOKMARKS_SERVICE_PORT = 8084
USERS_SERVICE_PORT     = 8085
FEEDBACK_SERVICE_PORT  = 8086
EMAIL_TO              = os.environ.get("HEALTH_EMAIL_TO", "")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class HealthRecord:
    timestamp: str = ""
    fixes:  list = field(default_factory=list)
    alerts: list = field(default_factory=list)
    info:   list = field(default_factory=list)

    @property
    def severity(self) -> str:
        if any(kw in a.lower() for a in self.alerts for kw in ("process", "down", "missing", "unload")):
            return "critical"
        if self.alerts or self.fixes:
            return "warning"
        return "ok"

    def has_findings(self) -> bool:
        return bool(self.fixes or self.alerts or self.info)


# ── Health Agent ──────────────────────────────────────────────────────────────

class HealthAgent:

    def __init__(self):
        logging.basicConfig(
            filename=str(HEALTH_LOG_PATH),
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Also echo to stdout so LaunchAgent captures it
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        self.log = logging.getLogger("health")
        self.last_email_time: float = 0.0
        self._seen_errors: set = set()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.log.info("Health agent started (poll interval %ds)", POLL_INTERVAL)
        while True:
            try:
                self.run_cycle()
            except Exception as exc:
                self.log.error("Unexpected error in cycle: %s", exc, exc_info=True)
            time.sleep(POLL_INTERVAL)

    def run_cycle(self):
        record = HealthRecord(timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))

        # Run all checks
        self.check_schema_columns(record)
        self.check_orphaned_parent_ids(record)
        self.check_empty_output_steps(record)
        self.check_duplicate_feedback_ids(record)
        self.check_chainlit_process(record)
        self.check_admin_process(record)
        self.check_rag_service_process(record)
        self.check_bookmarks_service_process(record)
        self.check_users_service_process(record)
        self.check_feedback_service_process(record)
        self.check_api_errors(record)
        self.check_embedding_model(record)
        self.check_knowledge_base_staleness(record)
        self.check_low_confidence_rate(record)
        self.check_context_window_exhaustion(record)
        self.check_feedback_anomaly(record)
        self.check_db_size(record)

        # Log findings
        for msg in record.fixes:
            self.log.info("FIX: %s", msg)
        for msg in record.alerts:
            self.log.warning("ALERT: %s", msg)
        for msg in record.info:
            self.log.info("INFO: %s", msg)
        if not record.has_findings():
            self.log.info("OK — no issues found")

        # Write structured events file (always, even for OK)
        self._write_health_events(record)

    # ── Category 1: Database Health ───────────────────────────────────────────

    def check_schema_columns(self, record: HealthRecord):
        required = {
            "defaultOpen":  "BOOLEAN DEFAULT FALSE",
            "autoCollapse": "BOOLEAN DEFAULT FALSE",
        }
        try:
            conn = sqlite3.connect(DB_PATH)
            existing = {row[1] for row in conn.execute("PRAGMA table_info(steps)").fetchall()}
            for col, typedef in required.items():
                if col not in existing:
                    conn.execute(f'ALTER TABLE steps ADD COLUMN "{col}" {typedef}')
                    record.fixes.append(f"Added missing DB column steps.{col}")
            conn.commit()
            conn.close()
        except Exception as e:
            record.alerts.append(f"DB schema check failed: {e}")

    def check_orphaned_parent_ids(self, record: HealthRecord):
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.execute(
                'UPDATE steps SET "parentId" = NULL '
                'WHERE "parentId" IS NOT NULL '
                'AND "parentId" NOT IN (SELECT id FROM steps)'
            )
            fixed = cur.rowcount
            conn.commit()
            conn.close()
            if fixed:
                record.fixes.append(f"Nulled {fixed} orphaned parentId(s) in steps table")
        except Exception as e:
            record.alerts.append(f"Orphaned parentId check failed: {e}")

    def check_empty_output_steps(self, record: HealthRecord):
        try:
            conn = sqlite3.connect(DB_PATH)
            count = conn.execute(
                "SELECT COUNT(*) FROM steps WHERE output IS NULL OR output = ''"
            ).fetchone()[0]
            conn.close()
            if count:
                record.info.append(f"{count} step(s) have empty output (harmless)")
        except Exception as e:
            record.alerts.append(f"Empty output check failed: {e}")

    def check_duplicate_feedback_ids(self, record: HealthRecord):
        if not FEEDBACK_PATH.exists():
            return
        try:
            data = json.loads(FEEDBACK_PATH.read_text())
            from collections import Counter
            counts = Counter(f.get("id", "") for f in data)
            dupes = {k: v for k, v in counts.items() if v > 1 and k}
            if dupes:
                seen, deduped = set(), []
                for entry in reversed(data):
                    eid = entry.get("id", "")
                    if eid not in seen:
                        seen.add(eid)
                        deduped.append(entry)
                deduped.reverse()
                FEEDBACK_PATH.write_text(json.dumps(deduped, indent=2))
                removed = sum(v - 1 for v in dupes.values())
                record.fixes.append(f"Removed {removed} duplicate feedback entry/entries")
        except Exception as e:
            record.alerts.append(f"Feedback dedup check failed: {e}")

    # ── Category 2: Process Health ────────────────────────────────────────────

    def _is_process_running(self, pattern: str) -> bool:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True)
        return result.returncode == 0

    def _is_port_responding(self, port: int) -> bool:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=5)
            return True
        except urllib.error.HTTPError:
            return True   # 4xx/5xx means the server is alive
        except Exception:
            return False

    def _restart_chainlit(self, record: HealthRecord):
        subprocess.run(["pkill", "-f", "chainlit run"], capture_output=True)
        time.sleep(2)
        with open(LAUNCH_LOG, "a") as log_fh:
            subprocess.Popen(
                [str(VENV_CHAINLIT), "run", str(CHAINLIT_APP), "--port", str(CHAINLIT_PORT)],
                cwd=str(BASE_DIR),
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
                env=os.environ.copy(),
            )
        record.fixes.append("Restarted Chainlit process on port 8081")

    def _restart_admin(self, record: HealthRecord):
        subprocess.run(["pkill", "-f", "admin_server.py"], capture_output=True)
        time.sleep(2)
        with open(ADMIN_LOG, "a") as log_fh:
            subprocess.Popen(
                [str(VENV_PYTHON), str(ADMIN_SERVER)],
                cwd=str(BASE_DIR),
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
            )
        record.fixes.append("Restarted admin_server.py on port 8082")

    def _restart_rag_service(self, record: HealthRecord):
        subprocess.run(["pkill", "-f", "rag_service:app"], capture_output=True)
        time.sleep(2)
        with open(RAG_SERVICE_LOG, "a") as log_fh:
            subprocess.Popen(
                [str(VENV_PYTHON), "-m", "uvicorn", "rag_service:app",
                 "--host", "127.0.0.1", "--port", str(RAG_SERVICE_PORT)],
                cwd=str(BASE_DIR),
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
            )
        record.fixes.append(f"Restarted RAG service on port {RAG_SERVICE_PORT}")

    def _restart_bookmarks_service(self, record: HealthRecord):
        subprocess.run(["pkill", "-f", "bookmarks_service:app"], capture_output=True)
        time.sleep(2)
        with open(BOOKMARKS_SERVICE_LOG, "a") as log_fh:
            subprocess.Popen(
                [str(VENV_PYTHON), "-m", "uvicorn", "bookmarks_service:app",
                 "--host", "127.0.0.1", "--port", str(BOOKMARKS_SERVICE_PORT)],
                cwd=str(BASE_DIR),
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
            )
        record.fixes.append(f"Restarted bookmarks service on port {BOOKMARKS_SERVICE_PORT}")

    def _restart_users_service(self, record: HealthRecord):
        subprocess.run(["pkill", "-f", "users_service:app"], capture_output=True)
        time.sleep(2)
        with open(USERS_SERVICE_LOG, "a") as log_fh:
            subprocess.Popen(
                [str(VENV_PYTHON), "-m", "uvicorn", "users_service:app",
                 "--host", "127.0.0.1", "--port", str(USERS_SERVICE_PORT)],
                cwd=str(BASE_DIR),
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
            )
        record.fixes.append(f"Restarted users service on port {USERS_SERVICE_PORT}")

    def _restart_feedback_service(self, record: HealthRecord):
        subprocess.run(["pkill", "-f", "feedback_service:app"], capture_output=True)
        time.sleep(2)
        with open(FEEDBACK_SERVICE_LOG, "a") as log_fh:
            subprocess.Popen(
                [str(VENV_PYTHON), "-m", "uvicorn", "feedback_service:app",
                 "--host", "127.0.0.1", "--port", str(FEEDBACK_SERVICE_PORT)],
                cwd=str(BASE_DIR),
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
            )
        record.fixes.append(f"Restarted feedback service on port {FEEDBACK_SERVICE_PORT}")

    def check_rag_service_process(self, record: HealthRecord):
        if not self._is_process_running("rag_service:app"):
            self._restart_rag_service(record)
            return
        if not self._is_port_responding(RAG_SERVICE_PORT):
            record.alerts.append(f"RAG service up but port {RAG_SERVICE_PORT} not responding")
            self._restart_rag_service(record)

    def check_bookmarks_service_process(self, record: HealthRecord):
        if not self._is_process_running("bookmarks_service:app"):
            self._restart_bookmarks_service(record)
            return
        if not self._is_port_responding(BOOKMARKS_SERVICE_PORT):
            record.alerts.append(f"Bookmarks service up but port {BOOKMARKS_SERVICE_PORT} not responding")
            self._restart_bookmarks_service(record)

    def check_users_service_process(self, record: HealthRecord):
        if not self._is_process_running("users_service:app"):
            self._restart_users_service(record)
            return
        if not self._is_port_responding(USERS_SERVICE_PORT):
            record.alerts.append(f"Users service up but port {USERS_SERVICE_PORT} not responding")
            self._restart_users_service(record)

    def check_feedback_service_process(self, record: HealthRecord):
        if not self._is_process_running("feedback_service:app"):
            self._restart_feedback_service(record)
            return
        if not self._is_port_responding(FEEDBACK_SERVICE_PORT):
            record.alerts.append(f"Feedback service up but port {FEEDBACK_SERVICE_PORT} not responding")
            self._restart_feedback_service(record)

    def check_chainlit_process(self, record: HealthRecord):
        if not self._is_process_running("chainlit run"):
            self._restart_chainlit(record)
            return
        if not self._is_port_responding(CHAINLIT_PORT):
            record.alerts.append("Chainlit process up but port 8081 not responding")
            self._restart_chainlit(record)
            record.fixes[-1] = "Killed unresponsive Chainlit, restarted on port 8081"

    def check_admin_process(self, record: HealthRecord):
        if not self._is_process_running("admin_server.py"):
            self._restart_admin(record)

    # ── Category 3: API Health ────────────────────────────────────────────────

    def check_api_errors(self, record: HealthRecord):
        if not LAUNCH_LOG.exists():
            return
        try:
            lines = LAUNCH_LOG.read_text(errors="replace").splitlines()[-API_ERROR_WINDOW:]
            errors = [l for l in lines if
                      ("API error" in l and any(c in l for c in ["4", "5"])) or
                      "400 Bad Request" in l or "500 Internal" in l or
                      "503 Service" in l or "529" in l]
            if len(errors) >= API_ERROR_THRESHOLD:
                fp = f"api_errors_{len(errors)}"
                if fp not in self._seen_errors:
                    self._seen_errors.add(fp)
                    record.alerts.append(
                        f"{len(errors)} Claude API errors in recent log — check connectivity"
                    )
        except Exception as e:
            record.alerts.append(f"API error scan failed: {e}")

    def check_embedding_model(self, record: HealthRecord):
        if not FAISS_PATH.exists():
            record.alerts.append("FAISS index file is missing — app cannot answer questions")
            return
        try:
            import faiss as _faiss
            _faiss.read_index(str(FAISS_PATH))
        except Exception as e:
            record.alerts.append(f"FAISS index unloadable: {e}")

    # ── Category 4: Data Quality ──────────────────────────────────────────────

    def check_knowledge_base_staleness(self, record: HealthRecord):
        if not FAISS_PATH.exists():
            return
        age_days = (time.time() - FAISS_PATH.stat().st_mtime) / 86400
        if age_days > FAISS_STALE_DAYS:
            fp = f"faiss_stale_{int(age_days)}"
            if fp not in self._seen_errors:
                self._seen_errors.add(fp)
                record.alerts.append(
                    f"Knowledge base is {age_days:.1f} days old (>{FAISS_STALE_DAYS}d) — "
                    f"trigger update from admin dashboard"
                )

    def check_low_confidence_rate(self, record: HealthRecord):
        if not GAPS_PATH.exists():
            return
        try:
            gaps = json.loads(GAPS_PATH.read_text())
            recent_gaps = gaps[-LOW_CONF_WINDOW:]
            if len(recent_gaps) < 5:
                return
            conn = sqlite3.connect(DB_PATH)
            total = conn.execute(
                "SELECT COUNT(*) FROM steps WHERE type='user_message'"
            ).fetchone()[0]
            conn.close()
            if total == 0:
                return
            ratio = len(recent_gaps) / min(total, LOW_CONF_WINDOW)
            if ratio > LOW_CONF_THRESHOLD:
                fp = f"low_conf_{int(ratio * 100)}"
                if fp not in self._seen_errors:
                    self._seen_errors.add(fp)
                    record.alerts.append(
                        f"High low-confidence rate: ~{ratio*100:.0f}% of recent queries "
                        f"have limited Lenny coverage — consider expanding knowledge base"
                    )
        except Exception as e:
            record.alerts.append(f"Confidence rate check failed: {e}")

    # ── Category 5: Chatbot Patterns ──────────────────────────────────────────

    def check_context_window_exhaustion(self, record: HealthRecord):
        try:
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute("""
                SELECT threadId, COUNT(*) as turns
                FROM steps WHERE type = 'user_message'
                GROUP BY threadId HAVING turns >= ?
            """, (CONTEXT_WINDOW_WARN,)).fetchall()
            conn.close()
            if rows:
                record.info.append(
                    f"{len(rows)} thread(s) have {CONTEXT_WINDOW_WARN}+ turns "
                    f"(quality may degrade — users should start new chats)"
                )
        except Exception as e:
            record.alerts.append(f"Context window check failed: {e}")

    def check_feedback_anomaly(self, record: HealthRecord):
        if not FEEDBACK_PATH.exists():
            return
        try:
            data = json.loads(FEEDBACK_PATH.read_text())
            if len(data) < FEEDBACK_SPIKE_WINDOW:
                return
            recent = data[-FEEDBACK_SPIKE_WINDOW:]
            down = sum(1 for f in recent if f.get("rating") == "down")
            ratio = down / len(recent)
            if ratio > FEEDBACK_SPIKE_RATIO:
                fp = f"feedback_spike_{down}"
                if fp not in self._seen_errors:
                    self._seen_errors.add(fp)
                    record.alerts.append(
                        f"Feedback anomaly: {down}/{len(recent)} recent responses got "
                        f"thumbs-down ({ratio*100:.0f}%) — review answer quality"
                    )
        except Exception as e:
            record.alerts.append(f"Feedback anomaly check failed: {e}")

    def check_db_size(self, record: HealthRecord):
        if not DB_PATH.exists():
            return
        size_mb = DB_PATH.stat().st_size / (1024 * 1024)
        if size_mb > DB_SIZE_LIMIT_MB:
            fp = f"db_size_{int(size_mb)}"
            if fp not in self._seen_errors:
                self._seen_errors.add(fp)
                record.alerts.append(
                    f"Database is {size_mb:.1f} MB (>{DB_SIZE_LIMIT_MB} MB threshold) — "
                    f"consider archiving old threads"
                )

    # ── Persistence: health_events.json ──────────────────────────────────────

    def _write_health_events(self, record: HealthRecord):
        events = []
        if HEALTH_EVENTS_PATH.exists():
            try:
                events = json.loads(HEALTH_EVENTS_PATH.read_text())
            except Exception:
                events = []

        events.append({
            "timestamp": record.timestamp,
            "severity":  record.severity,
            "fixes":     record.fixes,
            "alerts":    record.alerts,
            "info":      record.info,
        })

        # Keep rolling window
        if len(events) > MAX_HEALTH_EVENTS:
            events = events[-MAX_HEALTH_EVENTS:]

        try:
            HEALTH_EVENTS_PATH.write_text(json.dumps(events, indent=2))
        except Exception as e:
            self.log.error("Failed to write health_events.json: %s", e)

    # ── Email digest ──────────────────────────────────────────────────────────

    @staticmethod
    def _escape_applescript(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def _send_email_digest(self, record: HealthRecord):
        lines = [f"Distill Health Agent Report", f"{record.timestamp}", "=" * 50, ""]
        if record.fixes:
            lines.append("AUTO-FIXED")
            lines.extend(f"  + {f}" for f in record.fixes)
            lines.append("")
        if record.alerts:
            lines.append("NEEDS ATTENTION")
            lines.extend(f"  ! {a}" for a in record.alerts)
            lines.append("")
        if record.info:
            lines.append("INFO")
            lines.extend(f"  . {i}" for i in record.info)
            lines.append("")
        lines += [
            "=" * 50,
            f"Distill: http://localhost:{CHAINLIT_PORT}",
            f"Admin:   http://localhost:{ADMIN_PORT}/admin",
            "",
            "— Distill Health Agent",
        ]
        body = "\n".join(lines)

        status = "ACTION NEEDED" if record.alerts else "Auto-fixed"
        subject = f"Distill Health — {status} ({record.timestamp[:16]})"

        esc_subject = self._escape_applescript(subject)
        esc_body    = self._escape_applescript(body)
        esc_email   = self._escape_applescript(EMAIL_TO)

        applescript = (
            f'tell application "Mail"\n'
            f'    set newMessage to make new outgoing message with properties '
            f'{{subject:"{esc_subject}", content:"{esc_body}", visible:false}}\n'
            f'    tell newMessage\n'
            f'        make new to recipient at end of to recipients with properties '
            f'{{address:"{esc_email}"}}\n'
            f'    end tell\n'
            f'    send newMessage\n'
            f'end tell'
        )

        result = subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True)
        if result.returncode == 0:
            self.log.info("Email digest sent: %s", subject)
            self.last_email_time = time.time()
            self._seen_errors.clear()
        else:
            self.log.error("Email send failed: %s", result.stderr.strip())


# ── Entry point ───────────────────────────────────────────────────────────────

def _shutdown(signum, frame):
    logging.getLogger("health").info("Health agent shutting down (signal %d)", signum)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    os.chdir(BASE_DIR)
    HealthAgent().run()
