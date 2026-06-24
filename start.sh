#!/bin/bash
# Start all Distill services.
# Run once: cp .env.example .env  (then fill in your values)

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${VENV_PYTHON:-$DIR/.venv/bin/python3}"

if [ ! -f "$DIR/.env" ]; then
    echo "Error: .env not found. Copy .env.example to .env and fill in your values."
    exit 1
fi

if [ ! -f "$DIR/index.faiss" ]; then
    echo "Error: index.faiss not found. Run 'python ingest.py' first (see data/README.md)."
    exit 1
fi

# Init DB if needed
"$PYTHON" "$DIR/init_db.py"

echo "Starting services..."

nohup "$PYTHON" -m uvicorn rag_service:app       --host 127.0.0.1 --port 8083 >> "$DIR/rag_service.log"       2>&1 &
nohup "$PYTHON" -m uvicorn bookmarks_service:app --host 127.0.0.1 --port 8084 >> "$DIR/bookmarks_service.log" 2>&1 &
nohup "$PYTHON" -m uvicorn users_service:app     --host 127.0.0.1 --port 8085 >> "$DIR/users_service.log"     2>&1 &
nohup "$PYTHON" -m uvicorn feedback_service:app  --host 127.0.0.1 --port 8086 >> "$DIR/feedback_service.log"  2>&1 &
nohup "$PYTHON" -m uvicorn admin_server:app      --host 127.0.0.1 --port 8082 >> "$DIR/admin.log"             2>&1 &

# Give services a moment to bind
sleep 2

echo "Starting Distill UI..."
"$PYTHON" -m chainlit run "$DIR/chainlit_app.py" --port "${CHAINLIT_PORT:-8081}"
