#!/bin/bash
# Calls snapshot_stats.py every INTERVAL_SECONDS, forever. Read-only (just
# stats() calls) -- safe to run alongside any agent.main process, never
# touches or interrupts a running game.
#
# Usage: poll_snapshots.sh [interval_seconds]

set -euo pipefail
INTERVAL="${1:-180}"

set -a
source /Users/haydenarnold/Projects/GLEE/.env
set +a
cd /Users/haydenarnold/Projects/GLEE

while true; do
    .venv/bin/python -m scripts.snapshot_stats
    sleep "$INTERVAL"
done
