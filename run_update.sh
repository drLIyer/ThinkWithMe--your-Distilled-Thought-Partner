#!/bin/bash
cd /Users/liyer_1/lennys-rag
LOG="/Users/liyer_1/lennys-rag/update_log.txt"
EMAIL="liyer@aligntech.com"

echo "=== $(date) ===" >> "$LOG"

# Run the update agent and capture output
OUTPUT=$(/Users/liyer_1/lennys-rag/.venv312/bin/python3 update_knowledge.py 2>&1)
EXIT_CODE=$?
echo "$OUTPUT" >> "$LOG"

# Restart Distill regardless of whether new content was found
pkill -f "chainlit run" 2>/dev/null
sleep 2
nohup /Users/liyer_1/lennys-rag/.venv312/bin/chainlit run chainlit_app.py --port 8081 >> "$LOG" 2>&1 &
echo "Distill restarted at $(date)" >> "$LOG"

# Send email summary via macOS Mail
if [ $EXIT_CODE -eq 0 ]; then
    SUBJECT="Distill Update — $(date '+%B %d, %Y')"
    BODY="Hi,

Your Distill knowledge base was updated on $(date '+%A, %B %d at %I:%M %p').

Update Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

$OUTPUT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Distill is running at http://localhost:8081

— Distill Auto-Update Agent"
else
    SUBJECT="Distill Update — Error on $(date '+%B %d, %Y')"
    BODY="Hi,

The Distill knowledge update ran into an issue on $(date '+%A, %B %d at %I:%M %p').

Error output:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

$OUTPUT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please check update_log.txt for details.

— Distill Auto-Update Agent"
fi

osascript <<APPLESCRIPT
tell application "Mail"
    set newMessage to make new outgoing message with properties {subject:"$SUBJECT", content:"$BODY", visible:false}
    tell newMessage
        make new to recipient at end of to recipients with properties {address:"$EMAIL"}
    end tell
    send newMessage
end tell
APPLESCRIPT

echo "Email sent to $EMAIL" >> "$LOG"
