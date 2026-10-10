"""Build/extend a persistent, deduped index of every completed LIVE
competition game we've played, for easy reference (not simulation --
see scripts/montecarlo_opponents.py for that).

Nothing durable tracked this before: data/stats_history.csv only has
periodic aggregate (rating, games_played) snapshots, not individual games.
This scans every known run-log location for game IDs (same discovery
approach as the recurring-check scratchpad's fetch_recent_games.py,
including its 2026-08-27 fix for the flat live-process log files that the
original scratchpad LOGDIRS list couldn't see), fetches full game_state for
any completed game not already in the index, and appends one JSON record
per line to data/games_index.jsonl.

Idempotent and crash-safe: existing game_ids are skipped, and each fetched
record is appended immediately (not batched), so a killed/interrupted run
loses at most the one in-flight fetch. Safe to re-run any time to pick up
newly-completed games.
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glee_sdk import GleeClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = f"{ROOT}/data/games_index.jsonl"

SCRATCH = "/private/tmp/claude-501/-Users-haydenarnold-Projects-GLEE/93e2f0d9-a30b-4e0f-8179-4702df9bf262/scratchpad"
LOGDIRS = [
    f"{SCRATCH}/logs/run_to_1000", f"{SCRATCH}/logs/run_to_1101", f"{SCRATCH}/logs/run_to_1105",
    f"{SCRATCH}/logs/run_to_1200", f"{SCRATCH}/logs/run_to_1300", f"{SCRATCH}/logs/run_to_2000",
    f"{SCRATCH}/logs/run_to_10000", f"{SCRATCH}/logs/run_100_persuasion", f"{SCRATCH}/logs/run_to_2200",
    f"{SCRATCH}/logs/run_to_3000",
]
FAMILIES = ["bargaining", "negotiation", "persuasion"]
# Flat log files written by manually-launched `agent.main` processes (not the
# {logdir}/{family}_full.log layout above) -- see fetch_recent_games.py's
# 2026-08-27 fix for why these need scanning unconditionally per family
# (cross-family log mixing is real and already handled downstream by
# checking each fetched game's actual game_family).
EXTRA_LOGFILES = [
    "/Users/haydenarnold/.claude/jobs/0059f6f8/tmp/live_bargaining_to3000_v2.log",
    "/Users/haydenarnold/.claude/jobs/0059f6f8/tmp/live_negotiation_to3000.log",
    "/Users/haydenarnold/.claude/jobs/0059f6f8/tmp/live_negotiation_to3000_v2.log",
    "/Users/haydenarnold/.claude/jobs/0059f6f8/tmp/live_persuasion_to3000.log",
]
SEED_TRANSCRIPT_FILES = [
    f"{SCRATCH}/transcripts/bargaining_recent.json",
    f"{SCRATCH}/transcripts/negotiation_recent.json",
    f"{SCRATCH}/transcripts/persuasion_recent.json",
]

ANY_GAME_RE = re.compile(r"Game (?P<id>[0-9a-f-]{36}) \(")

MAX_FETCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 200  # per run, to bound API time/rate-limit use


def load_existing_ids():
    # DERIVED (2026-08-27): entries written before `game_state` was added to
    # the record are treated as NOT yet indexed, so they get re-fetched and
    # a second, complete line gets appended for the same game_id. This file
    # is append-only, so a game_id can appear twice after this transition --
    # readers (classify_opponents.py) take the LAST occurrence per game_id,
    # standard dedupe-by-latest for an append-only log.
    ids = set()
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "game_state" in record and "game_id" in record:
                    ids.add(record["game_id"])
    return ids


def all_candidate_ids():
    ids = []
    for logdir in LOGDIRS:
        for family in FAMILIES:
            path = f"{logdir}/{family}_full.log"
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = ANY_GAME_RE.search(line)
                    if m:
                        ids.append(m.group("id"))
    for path in EXTRA_LOGFILES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = ANY_GAME_RE.search(line)
                if m:
                    ids.append(m.group("id"))
    seen = set()
    deduped = []
    for gid in ids:
        if gid not in seen:
            seen.add(gid)
            deduped.append(gid)
    return deduped


def normalize_record(gs: dict) -> dict:
    state = gs["game_state"]
    me = gs["your_player"]
    family = gs["game_family"]
    result = state.get("result") or gs.get("result") or {}
    my_key = "player_1_payoff" if me == "player_1" else "player_2_payoff"
    opp_key = "player_2_payoff" if me == "player_1" else "player_1_payoff"
    record = {
        "game_id": gs["game_id"],
        "family": family,
        "your_player": me,
        "outcome": result.get("outcome"),
        "my_payoff": result.get(my_key),
        "opponent_payoff": result.get(opp_key),
        "rounds_played": len(state.get("history", [])),
    }
    if family == "bargaining":
        record["money_to_divide"] = state.get("money_to_divide")
        record["complete_information"] = state.get("complete_information")
        record["agreed_round"] = result.get("agreed_round")
    elif family == "negotiation":
        role = state.get(f"{me}_role")
        record["role"] = role
        record["my_value"] = state.get(f"{me}_value")
        record["complete_information"] = state.get("complete_information")
        record["max_rounds"] = state.get("max_rounds")
    elif family == "persuasion":
        record["role"] = state.get(f"{me}_role") or state.get("your_role")
        record["price"] = state.get("product_price") or state.get("price")
        record["rounds_bought"] = result.get("rounds_bought")
    # DERIVED (2026-08-27): the summary fields above are enough for quick
    # scanning, but not for classify_opponents.py, which needs the full
    # round-by-round history (both our actual offers/signals and the
    # opponent's actual responses) to replay against each archetype's
    # deterministic rules. That's already sitting in `state` from the same
    # API call -- storing it here is free (no extra fetch), so every entry
    # carries it rather than needing a second pass later. Entries indexed
    # before this change lack `game_state` and are skipped by that script.
    record["game_state"] = state
    return record


def seed_from_cached_transcripts(existing_ids, out_f):
    seeded = 0
    for path in SEED_TRANSCRIPT_FILES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            try:
                games = json.load(f)
            except json.JSONDecodeError:
                continue
        for gs in games:
            gid = gs.get("game_id")
            if not gid or gid in existing_ids or gs.get("status") != "completed":
                continue
            try:
                record = normalize_record(gs)
            except Exception:
                continue
            out_f.write(json.dumps(record) + "\n")
            existing_ids.add(gid)
            seeded += 1
    return seeded


def main():
    existing_ids = load_existing_ids()
    print(f"index currently has {len(existing_ids)} games")

    with open(INDEX_PATH, "a", encoding="utf-8") as out_f:
        seeded = seed_from_cached_transcripts(existing_ids, out_f)
        if seeded:
            print(f"seeded {seeded} games from cached transcripts (no API calls)")

        candidates = all_candidate_ids()
        to_fetch = [gid for gid in candidates if gid not in existing_ids]
        print(f"{len(candidates)} candidate ids seen in logs, {len(to_fetch)} not yet indexed")

        api_key = os.environ["GLEE_API_KEY"]
        client = GleeClient(api_key=api_key)
        fetched, skipped_incomplete, errors = 0, 0, 0
        for gid in to_fetch:
            if fetched >= MAX_FETCHES:
                print(f"stopping at MAX_FETCHES={MAX_FETCHES}; re-run to continue")
                break
            try:
                gs = client.game_state(gid)
            except Exception as e:
                errors += 1
                continue
            time.sleep(0.9)
            if gs.get("status") != "completed":
                skipped_incomplete += 1
                continue
            try:
                record = normalize_record(gs)
            except Exception:
                errors += 1
                continue
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            existing_ids.add(gid)
            fetched += 1
            if fetched % 20 == 0:
                print(f"  fetched {fetched}/{min(len(to_fetch), MAX_FETCHES)}")

        print(f"done: {fetched} fetched, {skipped_incomplete} still in-progress, {errors} errors")
        print(f"index now has {len(existing_ids)} games total -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
