"""Appends the agent's current per-family rating and games played to a CSV.

Read-only against the server, so it is safe to run while the agent is playing.
"""

from __future__ import annotations

import csv
import datetime
import os
from pathlib import Path

from glee_sdk import GleeClient

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stats_history.csv"
FIELDS = ["timestamp", "family", "rating", "games_played"]


def snapshot() -> list[dict]:
    api_key = os.environ.get("GLEE_API_KEY", "")
    if not api_key:
        raise SystemExit("Set GLEE_API_KEY environment variable")
    base_url = os.environ.get("GLEE_API_URL")
    client = GleeClient(api_key=api_key, base_url=base_url) if base_url else GleeClient(api_key=api_key)

    stats = client.stats()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows = [
        {
            "timestamp": now,
            "family": family,
            "rating": scores["rating"],
            "games_played": scores["games_played"],
        }
        for family, scores in stats["scores"].items()
    ]

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not DATA_PATH.exists()
    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    for row in snapshot():
        print(f"{row['timestamp']}  {row['family']:<12} rating={row['rating']:<10} games_played={row['games_played']}")
