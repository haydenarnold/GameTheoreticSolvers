"""Entrypoint for the GLEE Competition agent.

Usage:
    pip install -r requirements.txt
    export GLEE_API_KEY=glee_...            # from your dashboard at glee-competition.com
    export LLM_MODEL=gemini/gemini-2.5-flash  # or gpt-5-mini, claude-sonnet-5, ... (any litellm model string)
    export GEMINI_API_KEY=...               # whichever provider key LLM_MODEL needs

    python -m agent.main
"""

from __future__ import annotations

import argparse
import logging
import os

from glee_sdk import GleeClient

from agent.dispatch import strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GLEE Competition agent.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-time", type=float, default=None, help="seconds")
    parser.add_argument(
        "--families", nargs="*", default=None,
        choices=["bargaining", "negotiation", "persuasion"],
        help="defaults to all three",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GLEE_API_KEY", "")
    if not api_key:
        raise SystemExit("Set GLEE_API_KEY environment variable")

    base_url = os.environ.get("GLEE_API_URL")
    client = GleeClient(api_key=api_key, base_url=base_url) if base_url else GleeClient(api_key=api_key)

    print(f"Agent stats: {client.stats()}")
    client.run(
        strategy,
        game_families=args.families,
        concurrency=args.concurrency,
        max_games=args.max_games,
        max_time=args.max_time,
    )


if __name__ == "__main__":
    main()
