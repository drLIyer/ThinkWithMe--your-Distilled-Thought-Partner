"""
Migrate conversations.json → SQLite for Chainlit.

Run once:
    .venv312/bin/python3 migrate_history.py
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.user import User

CONVERSATIONS_PATH = Path("conversations.json")
DB_URL = "sqlite+aiosqlite:///asklenny.db"
ANON_USER_ID = "migrated-user"


async def main():
    if not CONVERSATIONS_PATH.exists():
        print("conversations.json not found — nothing to migrate.")
        return

    data = json.loads(CONVERSATIONS_PATH.read_text())
    conversations = data.get("conversations", [])
    if not conversations:
        print("No conversations found.")
        return

    layer = SQLAlchemyDataLayer(DB_URL)

    # Create a placeholder user for all migrated threads
    user = await layer.create_user(User(identifier=ANON_USER_ID, metadata={"source": "migration"}))
    user_id = user.id if user else ANON_USER_ID
    print(f"User: {user_id}")

    migrated = 0
    for conv in conversations:
        messages = conv.get("messages", [])
        if not messages:
            continue

        thread_id = conv.get("id") or str(uuid.uuid4())
        title = conv.get("title", "Imported chat")

        # Create the thread
        await layer.update_thread(
            thread_id=thread_id,
            name=title,
            user_id=user_id,
            metadata={"migrated": True},
        )

        # Create steps for each message pair
        ts = datetime.now(timezone.utc).isoformat()
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue

            step_type = "user_message" if role == "user" else "assistant_message"
            step_id = str(uuid.uuid4())

            step: dict = {
                "id": step_id,
                "threadId": thread_id,
                "name": "You" if role == "user" else "Ask Lenny",
                "type": step_type,
                "output": content,
                "input": "",
                "createdAt": ts,
                "start": ts,
                "end": ts,
                "streaming": False,
                "waitForAnswer": False,
                "isError": False,
                "metadata": {},
                "tags": [],
                "parentId": None,
                "command": None,
                "modes": [],
                "showInput": False,
                "defaultOpen": False,
                "autoCollapse": False,
                "language": None,
                "icon": None,
                "generation": None,
                "feedback": None,
            }

            await layer.create_step(step)

        print(f"  ✓ '{title[:55]}' ({len(messages)} messages)")
        migrated += 1

    print(f"\nDone — migrated {migrated} conversations to asklenny.db")


if __name__ == "__main__":
    asyncio.run(main())
