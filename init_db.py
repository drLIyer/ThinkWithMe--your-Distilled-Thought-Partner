"""
Create SQLite tables for Chainlit and migrate conversations.json history.

Run once:
    .venv312/bin/python3 init_db.py
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

DB_PATH = Path("asklenny.db")
CONVERSATIONS_PATH = Path("conversations.json")
ANON_USER_ID = str(uuid.uuid4())
ANON_IDENTIFIER = "local-user"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id"          TEXT PRIMARY KEY,
    "identifier"  TEXT NOT NULL UNIQUE,
    "createdAt"   TEXT,
    "metadata"    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS threads (
    "id"          TEXT PRIMARY KEY,
    "createdAt"   TEXT,
    "name"        TEXT,
    "userId"      TEXT REFERENCES users("id") ON DELETE CASCADE,
    "userIdentifier" TEXT,
    "tags"        TEXT NOT NULL DEFAULT '[]',
    "metadata"    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS steps (
    "id"          TEXT PRIMARY KEY,
    "name"        TEXT NOT NULL,
    "type"        TEXT NOT NULL,
    "threadId"    TEXT NOT NULL REFERENCES threads("id") ON DELETE CASCADE,
    "parentId"    TEXT,
    "streaming"   INTEGER NOT NULL DEFAULT 0,
    "waitForAnswer" INTEGER DEFAULT 0,
    "isError"     INTEGER DEFAULT 0,
    "metadata"    TEXT NOT NULL DEFAULT '{}',
    "tags"        TEXT NOT NULL DEFAULT '[]',
    "input"       TEXT DEFAULT '',
    "output"      TEXT DEFAULT '',
    "createdAt"   TEXT,
    "start"       TEXT,
    "end"         TEXT,
    "generation"  TEXT,
    "showInput"   TEXT DEFAULT 'false',
    "language"    TEXT,
    "indent"      INTEGER
);

CREATE TABLE IF NOT EXISTS elements (
    "id"          TEXT PRIMARY KEY,
    "threadId"    TEXT REFERENCES threads("id") ON DELETE CASCADE,
    "type"        TEXT,
    "url"         TEXT,
    "chainlitKey" TEXT,
    "name"        TEXT NOT NULL,
    "display"     TEXT,
    "objectKey"   TEXT,
    "size"        TEXT,
    "page"        INTEGER,
    "language"    TEXT,
    "forId"       TEXT,
    "mime"        TEXT
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id"          TEXT PRIMARY KEY,
    "forId"       TEXT NOT NULL,
    "threadId"    TEXT NOT NULL REFERENCES threads("id") ON DELETE CASCADE,
    "value"       INTEGER NOT NULL,
    "comment"     TEXT
);
"""


async def main():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()
        print(f"✓ Schema created in {DB_PATH}")

        # Create a local user
        ts = datetime.now(timezone.utc).isoformat()
        await db.execute(
            'INSERT OR IGNORE INTO users (id, identifier, "createdAt", metadata) VALUES (?,?,?,?)',
            (ANON_USER_ID, ANON_IDENTIFIER, ts, '{"source":"migration"}'),
        )
        await db.commit()
        print(f"✓ User created: {ANON_IDENTIFIER}")

        # Migrate conversations
        if not CONVERSATIONS_PATH.exists():
            print("No conversations.json found — skipping migration.")
            return

        data = json.loads(CONVERSATIONS_PATH.read_text())
        conversations = [c for c in data.get("conversations", []) if c.get("messages")]
        migrated = 0

        for conv in conversations:
            thread_id = conv.get("id") or str(uuid.uuid4())
            title = conv.get("title", "Imported chat")

            await db.execute(
                'INSERT OR IGNORE INTO threads (id, "createdAt", name, "userId", "userIdentifier", tags, metadata) '
                'VALUES (?,?,?,?,?,?,?)',
                (thread_id, ts, title, ANON_USER_ID, ANON_IDENTIFIER, "[]",
                 json.dumps({"migrated": True})),
            )

            for msg in conv.get("messages", []):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if not content:
                    continue

                step_type = "user_message" if role == "user" else "assistant_message"
                step_name = "You" if role == "user" else "Ask Lenny"

                await db.execute(
                    'INSERT OR IGNORE INTO steps '
                    '(id, name, type, "threadId", streaming, "waitForAnswer", "isError", '
                    'metadata, tags, input, output, "createdAt", start, "end") '
                    'VALUES (?,?,?,?,0,0,0,?,?,?,?,?,?,?)',
                    (str(uuid.uuid4()), step_name, step_type, thread_id,
                     "{}", "[]", "", content, ts, ts, ts),
                )

            await db.commit()
            print(f"  ✓ '{title[:60]}'")
            migrated += 1

        print(f"\nDone — migrated {migrated} conversations.")
        print(f"\nTo start the app:\n  cd lennys-rag && .venv312/bin/chainlit run chainlit_app.py")


if __name__ == "__main__":
    asyncio.run(main())
