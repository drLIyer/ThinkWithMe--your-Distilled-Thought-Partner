#!/bin/bash
# Run the knowledge update agent, restart Distill, and optionally email a summary.
#
# Set HEALTH_EMAIL_TO in .env to receive email digests.
# Uses the Python venv at .venv/bin/python3 by default; override with VENV_PYTHON.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$DIR/update_log.txt"
PYTHON="${VENV_PYTHON:-$DIR/.venv/bin/python3}"
EMAIL="${HEALTH_EMAIL_TO:-}"

echo "=== $(date) ===" >> "$LOG"

OUTPUT=$("$PYTHON" "$DIR/update_knowledge.py" 2>&1)
EXIT_CODE=$?
echo "$OUTPUT" >> "$LOG"

# Restart Distill
pkill -f "chainlit run" 2>/dev/null || true
sleep 2
nohup "$PYTHON" -m chainlit run "$DIR/chainlit_app.py" --port "${CHAINLIT_PORT:-8081}" >> "$LOG" 2>&1 &
echo "Distill restarted at $(date)" >> "$LOG"

# Send email summary if address is configured and sendmail is available
if [ -n "$EMAIL" ] && command -v sendmail &>/dev/null; then
    if [ $EXIT_CODE -eq 0 ]; then
        SUBJECT="Distill Update — $(date '+%B %d, %Y')"
    else
        SUBJECT="Distill Update — Error on $(date '+%B %d, %Y')"
    fi
    {
        echo "To: $EMAIL"
        echo "Subject: $SUBJECT"
        echo ""
        echo "$OUTPUT"
    } | sendmail "$EMAIL"
    echo "Email sent to $EMAIL" >> "$LOG"
fi
