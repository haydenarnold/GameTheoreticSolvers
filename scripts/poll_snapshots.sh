#!/bin/bash
# Runs snapshot_stats.py every interval seconds (default 180), forever.
# This is done because the API does not provide a way to retrieve aggregate results of games for an agent.
# Read-only, so it is safe to run alongside a live agent.
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
