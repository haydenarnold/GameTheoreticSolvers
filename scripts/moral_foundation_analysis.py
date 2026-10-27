"""Tests whether the moral-foundation framing of our messages predicts success.

Each outgoing message is labeled with its dominant foundation using the same
lexicon the agent uses live, then a chi-square test of independence is run per
family.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import chi2_contingency

from agent.rhetoric.profile import MIN_FOUNDATION_EVIDENCE, _FOUNDATION_RES, _count_with_negation

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "data" / "games_index.jsonl"

FOUNDATIONS = ["care", "fairness", "loyalty", "authority", "purity"]

# Final games-played counts per family, used only to report sample coverage.
FINAL_GAMES_PLAYED = {"bargaining": 4631, "negotiation": 5750, "persuasion": 5325}


def dominant_foundation_for_message(text: str | None) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    scores = {name: _count_with_negation(lowered, pattern) for name, pattern in _FOUNDATION_RES.items()}
    if sum(scores.values()) < MIN_FOUNDATION_EVIDENCE:
        return None
    return max(scores, key=scores.get)


def load_latest_games() -> dict:
    latest = {}
    with open(INDEX_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "game_state" in d and "game_id" in d:
                latest[d["game_id"]] = d
    return latest


def collect_counts(games: dict, family: str) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for d in games.values():
        if d["family"] != family:
            continue
        state = d["game_state"]
        me = d["your_player"]

        # Success means a different thing per family: the buyer bought
        # (persuasion) or the opponent accepted our offer (negotiation, bargaining).
        if family == "persuasion":
            if state.get("seller_message_type") != "text":
                continue
            role = state.get(f"{me}_role") or state.get("your_role")
            if role != "seller":
                continue
            for h in state.get("history", []):
                msg = h.get("seller_message")
                bought = h.get("bought")
                if msg is None or bought is None:
                    continue
                fnd = dominant_foundation_for_message(msg)
                if fnd is None:
                    continue
                counts[fnd][0 if bought else 1] += 1

        elif family == "negotiation":
            for h in state.get("history", []):
                offer = h.get("offer") or {}
                if offer.get("from_player") != me:
                    continue
                msg = offer.get("message")
                decision = h.get("decision")
                if not msg or decision is None:
                    continue
                fnd = dominant_foundation_for_message(msg)
                if fnd is None:
                    continue
                success = decision == "AcceptOffer"
                counts[fnd][0 if success else 1] += 1

        elif family == "bargaining":
            for h in state.get("history", []):
                offer = h.get("offer") or {}
                if offer.get("proposer") != me:
                    continue
                msg = offer.get("message")
                decision = h.get("decision")
                if not msg or decision is None:
                    continue
                fnd = dominant_foundation_for_message(msg)
                if fnd is None:
                    continue
                success = decision == "accept"
                counts[fnd][0 if success else 1] += 1

        else:
            raise ValueError(f"unknown family: {family}")

    return counts


def report_family(games: dict, family: str) -> tuple[int, float, float]:
    counts = collect_counts(games, family)
    n_indexed = sum(1 for d in games.values() if d["family"] == family)
    coverage = n_indexed / FINAL_GAMES_PLAYED[family] if family in FINAL_GAMES_PLAYED else float("nan")

    table = []
    for fnd in FOUNDATIONS:
        succ, fail = counts[fnd]
        n = succ + fail
        rate = succ / n if n else float("nan")
        table.append((fnd, n, succ, rate))
    total = sum(n for _, n, _, _ in table)

    contingency = [[counts[fnd][0], counts[fnd][1]] for fnd in FOUNDATIONS]
    chi2, p, dof, _ = chi2_contingency(contingency)

    print(f"\n=== {family} ===")
    print(f"{n_indexed} indexed games ({coverage:.1%} of {FINAL_GAMES_PLAYED.get(family, '?')} final games_played)")
    print(f"{total} messages classified with a clear dominant foundation")
    print(f"{'foundation':<12} {'n':>6} {'success':>8} {'rate':>8}")
    for fnd, n, succ, rate in sorted(table, key=lambda t: -t[3] if t[3] == t[3] else 0):
        print(f"{fnd:<12} {n:>6} {succ:>8} {rate:>7.1%}")
    print(f"chi2({dof}) = {chi2:.1f}, p = {p:.3g}, n = {total}")

    return total, chi2, p


def main() -> None:
    games = load_latest_games()
    print(f"{len(games)} unique indexed games with game_state")

    results = {family: report_family(games, family) for family in ("persuasion", "bargaining", "negotiation")}

    print("\n=== summary ===")
    print(f"{'family':<12} {'n':>8} {'chi2':>8} {'p':>12}")
    for family, (n, chi2, p) in results.items():
        print(f"{family:<12} {n:>8} {chi2:>8.1f} {p:>12.3g}")


if __name__ == "__main__":
    main()
