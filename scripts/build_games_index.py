"""Builds the persistent index of completed games used by the rhetoric analysis.

Game ids are scraped from agent run logs, then each unindexed game is fetched
from the server. The script is resumable: indexed ids are skipped, and an
optional argument caps how many games are fetched per run.
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

MAX_FETCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def load_existing_ids():
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
            # Spacing between calls keeps us under the API's rate limit.
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
