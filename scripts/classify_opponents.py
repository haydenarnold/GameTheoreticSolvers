"""Classify each real completed live game's opponent against the 11
deterministic archetype behaviors already implemented in
montecarlo_opponents.py, using FULL per-round history -- both our own
actual historical moves and the opponent's actual real responses -- not
just aggregate outcome.

This is a best-fit SIMILARITY match, not a ground-truth label. A real
opponent (almost certainly another LLM-driven agent, not literally one of
these 11 hand-written rule-sets) is not guaranteed to resemble any of them
well. For each archetype, we replay our own real historical offers/signals
through that archetype's exact `propose()`/`decide()`/`signal()` rules
(imported directly from montecarlo_opponents.py, not reimplemented) and
score how closely its predictions match what the real opponent actually
did, round by round. The archetype with the lowest error is the game's
inferred label; the fit-quality gap between the best and second-best match
is reported too, since a poor best-fit score means "least-bad guess", not
a confident label.

Information caveats, stated up front rather than glossed over:
  - Bargaining: fully specified either way (money/max_rounds are always
    known to both sides); classification confidence does not depend on
    complete_information here.
  - Negotiation: the archetype rules need the opponent's OWN true
    reservation value. Under complete_information that's known exactly
    (high confidence). Under incomplete information it is NOT knowable --
    we substitute a proxy estimate (their most extreme revealed price) and
    mark these classifications lower-confidence explicitly.
  - Persuasion: price/p/v/u are always public, so classifying the BUYER
    (when we're seller) is always fully specified. Classifying the SELLER
    (when we're buyer) requires the true quality each round, which we only
    ever learn on rounds where we actually bought -- rounds we passed on
    contribute no evidence and are skipped, so seller classifications from
    fewer purchased rounds carry a smaller, explicitly reported sample size.

Requires data/games_index.jsonl entries built with `game_state` present
(see build_games_index.py's 2026-08-27 update) -- earlier summary-only
entries are skipped.

Usage: .venv/bin/python scripts/classify_opponents.py [family ...]
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from montecarlo_opponents import (
    ARCHETYPES, BargOpponent, NegOpponent, PersOpponentBuyer, PersOpponentSeller,
)
from agent.schema import opponent_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = f"{ROOT}/data/games_index.jsonl"
OUT_PATH = f"{ROOT}/data/opponent_classifications.jsonl"

FAMILIES = sys.argv[1:] or ["bargaining", "negotiation", "persuasion"]

# Games with fewer than this many scoreable (predict-vs-actual) rounds
# don't carry enough signal to distinguish 11 archetypes meaningfully --
# e.g. a round-1 instant agreement is one data point. Still classified
# (lowest error wins), but flagged low-confidence via `n_observations`
# in the output rather than silently presented as equally solid.
MIN_CONFIDENT_OBSERVATIONS = 3

# A guessed, fixed stand-in for `BargOpponent`/`NegOpponent`'s own internal
# `_bayes_delta_me` / patience parameter -- in the original Monte Carlo
# simulation this is drawn per-trial from a private RNG the archetype uses
# to model ITS OWN hidden patience, which has no real counterpart to
# recover from a single real game's revealed behavior. Only affects the
# bayesian_rational/noisy_rational archetypes' predicted curve; a fixed
# central estimate is a reasonable simplification for a best-fit match,
# not a claim about the real opponent's actual patience.
ASSUMED_OPPONENT_DELTA = 0.85


def load_index():
    """Dedupe by latest occurrence per game_id (see build_games_index.py's
    load_existing_ids docstring for why a game_id can appear twice)."""
    by_id = {}
    with open(INDEX_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "game_state" not in record:
                continue
            by_id[record["game_id"]] = record
    return list(by_id.values())


# ============================================================================
# BARGAINING
# ============================================================================

def classify_bargaining(record) -> dict | None:
    state = record["game_state"]
    me = record["your_player"]
    opp = opponent_name(me)
    money = state.get("money_to_divide")
    max_rounds = state.get("max_rounds") if state.get("horizon_known", state.get("max_rounds") is not None) else None
    history = state.get("history", [])
    if not money or not history:
        return None

    my_key = f"{me}_gain"
    opp_key = f"{opp}_gain"

    scores = {}
    for archetype in ARCHETYPES:
        rng = __import__("random").Random(0)  # deterministic seed; noise archetypes still vary run-to-run in
        # the original simulation, but a fixed seed here makes the fit score reproducible, which matters more
        # for a classifier than matching any one specific noisy draw.
        opponent = BargOpponent(archetype, money, max_rounds, rng)
        propose_errors = []
        decision_matches = []
        for h in history:
            proposer = h.get("proposer")
            offer = h.get("offer", {})
            round_no = h.get("round")
            if proposer == opp:
                actual_ask = offer.get(opp_key)
                if actual_ask is None:
                    continue
                predicted_ask = opponent.propose(round_no)
                propose_errors.append(abs(predicted_ask - actual_ask) / money)
            elif proposer == me:
                offered_to_them = offer.get(opp_key)
                if offered_to_them is None:
                    continue
                predicted_accept = opponent.decide(round_no, offered_to_them)
                opponent.record_our_offer(offered_to_them)
                actual_accept = h.get("decision") == "accept"
                decision_matches.append(predicted_accept == actual_accept)
        n_obs = len(propose_errors) + len(decision_matches)
        if n_obs == 0:
            continue
        propose_mae = sum(propose_errors) / len(propose_errors) if propose_errors else 0.0
        decision_mismatch = 1 - (sum(decision_matches) / len(decision_matches)) if decision_matches else 0.0
        scores[archetype] = (propose_mae + decision_mismatch, n_obs)

    if not scores:
        return None
    best = min(scores, key=lambda a: scores[a][0])
    ranked = sorted(scores, key=lambda a: scores[a][0])
    return {
        "best_fit": best,
        "best_fit_score": scores[best][0],
        "runner_up": ranked[1] if len(ranked) > 1 else None,
        "runner_up_score": scores[ranked[1]][0] if len(ranked) > 1 else None,
        "n_observations": scores[best][1],
        "confident": scores[best][1] >= MIN_CONFIDENT_OBSERVATIONS,
    }


# ============================================================================
# NEGOTIATION
# ============================================================================

def _estimate_opponent_value(state, me, opp, opp_role) -> tuple[float | None, bool]:
    """Returns (estimate, is_exact). Exact under complete_information (the
    real value is in state); otherwise a proxy from their own most extreme
    revealed price -- their lowest ask if they're a seller, highest if
    buyer -- which is the best available lower/upper-bound signal without
    ever knowing their true reservation value."""
    if state.get("complete_information"):
        val = state.get(f"{opp}_value")
        if val is not None:
            return val, True
    prices = [h["offer"]["price"] for h in state.get("history", [])
              if h.get("offer", {}).get("from_player") == opp and h["offer"].get("price") is not None]
    if not prices:
        return None, False
    return (min(prices) if opp_role == "seller" else max(prices)), False


def classify_negotiation(record) -> dict | None:
    state = record["game_state"]
    me = record["your_player"]
    opp = opponent_name(me)
    my_role = state.get(f"{me}_role")
    opp_role = "buyer" if my_role == "seller" else "seller"
    max_rounds = state.get("max_rounds") if state.get("horizon_known", state.get("max_rounds") is not None) else None
    history = state.get("history", [])
    if not history:
        return None

    opp_value, exact = _estimate_opponent_value(state, me, opp, opp_role)
    if opp_value is None or opp_value <= 0:
        return None

    scores = {}
    for archetype in ARCHETYPES:
        rng = __import__("random").Random(0)
        opponent = NegOpponent(archetype, opp_role, (opp_value, opp_value), rng)
        opponent.true_value = opp_value  # pin exactly rather than sampling from the (degenerate) range above
        propose_errors = []
        decision_matches = []
        for h in history:
            offer = h.get("offer", {})
            round_no = h.get("round")
            from_player = offer.get("from_player")
            price = offer.get("price")
            if price is None:
                continue
            if from_player == opp:
                predicted_price = opponent.propose(round_no, max_rounds)
                denom = opp_value if opp_value else 1.0
                propose_errors.append(abs(predicted_price - price) / denom)
            elif from_player == me:
                predicted_accept = opponent.decide(round_no, max_rounds, price)
                opponent.record_our_offer(price)
                actual_accept = h.get("decision") in ("AcceptOffer", "accept")
                decision_matches.append(predicted_accept == actual_accept)
        n_obs = len(propose_errors) + len(decision_matches)
        if n_obs == 0:
            continue
        propose_mae = sum(propose_errors) / len(propose_errors) if propose_errors else 0.0
        decision_mismatch = 1 - (sum(decision_matches) / len(decision_matches)) if decision_matches else 0.0
        scores[archetype] = (propose_mae + decision_mismatch, n_obs)

    if not scores:
        return None
    best = min(scores, key=lambda a: scores[a][0])
    ranked = sorted(scores, key=lambda a: scores[a][0])
    return {
        "best_fit": best,
        "best_fit_score": scores[best][0],
        "runner_up": ranked[1] if len(ranked) > 1 else None,
        "runner_up_score": scores[ranked[1]][0] if len(ranked) > 1 else None,
        "n_observations": scores[best][1],
        "confident": scores[best][1] >= MIN_CONFIDENT_OBSERVATIONS and exact,
        "opponent_value_exact": exact,
    }


# ============================================================================
# PERSUASION
# ============================================================================

def classify_persuasion(record) -> dict | None:
    state = record["game_state"]
    me = record["your_player"]
    my_role = state.get(f"{me}_role")
    history = state.get("history", [])
    p, v, u, price = state.get("p"), state.get("v"), state.get("u"), state.get("product_price") or state.get("price")
    if not history or price is None or p is None:
        return None

    scores = {}
    if my_role == "seller":
        # Classify the BUYER's archetype: always fully specified (we know
        # true quality every round when we're the seller).
        for archetype in ARCHETYPES:
            rng = __import__("random").Random(0)
            opp = PersOpponentBuyer(archetype, rng)
            matches = []
            for h in history:
                quality_high = h.get("quality") == "high"
                signal_buy = h.get("seller_message") == "yes"
                predicted_buy = opp.decide(h.get("round"), signal_buy, p, v, u, price)
                opp.observe(signal_buy, quality_high)
                actual_buy = bool(h.get("bought"))
                matches.append(predicted_buy == actual_buy)
            if matches:
                scores[archetype] = (1 - sum(matches) / len(matches), len(matches))
    else:
        # Classify the SELLER's archetype: only usable on rounds where we
        # (the buyer) actually learn true quality -- i.e. rounds we bought.
        # Rounds we passed on reveal nothing about the seller's honesty
        # that round and are skipped -- see module docstring.
        known_quality_rounds = [h for h in history if "quality" in h]
        if not known_quality_rounds:
            return None
        for archetype in ARCHETYPES:
            rng = __import__("random").Random(0)
            opp = PersOpponentSeller(archetype, rng)
            matches = []
            for h in known_quality_rounds:
                quality_high = h.get("quality") == "high"
                predicted_signal = opp.signal(h.get("round"), quality_high, p, v, u, price)
                actual_signal = h.get("seller_message") == "yes"
                matches.append(predicted_signal == actual_signal)
            if matches:
                scores[archetype] = (1 - sum(matches) / len(matches), len(matches))

    if not scores:
        return None
    best = min(scores, key=lambda a: scores[a][0])
    ranked = sorted(scores, key=lambda a: scores[a][0])
    return {
        "best_fit": best,
        "best_fit_score": scores[best][0],
        "runner_up": ranked[1] if len(ranked) > 1 else None,
        "runner_up_score": scores[ranked[1]][0] if len(ranked) > 1 else None,
        "n_observations": scores[best][1],
        "confident": scores[best][1] >= MIN_CONFIDENT_OBSERVATIONS,
    }


CLASSIFIERS = {
    "bargaining": classify_bargaining,
    "negotiation": classify_negotiation,
    "persuasion": classify_persuasion,
}


def our_normalized_score(record) -> float | None:
    """Same convention as the recurring check's rank_losses.py badness
    functions, sign-flipped so higher = better here."""
    state = record["game_state"]
    me = record["your_player"]
    family = record["family"]
    result = state.get("result") or record.get("result") or {}
    my_key = f"{me}_payoff"
    payoff = result.get(my_key)
    if payoff is None:
        return None
    if family == "bargaining":
        money = state.get("money_to_divide")
        return payoff / money if money else None
    if family == "negotiation":
        outcome = result.get("outcome")
        if outcome != "agreement":
            return -1.0
        my_value = state.get(f"{me}_value")
        return payoff / my_value if my_value else None
    if family == "persuasion":
        price = state.get("product_price") or state.get("price")
        rounds_total = result.get("rounds_total") or 20
        return payoff / (price * rounds_total) if price else None
    return None


def main():
    records = load_index()
    print(f"loaded {len(records)} indexed games with full game_state")

    results_by_family = defaultdict(list)
    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for record in records:
            family = record.get("family")
            if family not in FAMILIES or family not in CLASSIFIERS:
                continue
            try:
                classification = CLASSIFIERS[family](record)
            except Exception as e:
                continue
            if classification is None:
                continue
            score = our_normalized_score(record)
            row = {
                "game_id": record["game_id"], "family": family,
                "our_score": score, **classification,
            }
            out_f.write(json.dumps(row) + "\n")
            results_by_family[family].append(row)

    print(f"classified games written to {OUT_PATH}\n")

    for family in FAMILIES:
        rows = results_by_family.get(family, [])
        if not rows:
            print(f"=== {family}: no classifiable games ===\n")
            continue
        confident_rows = [r for r in rows if r["confident"]]
        print(f"=== {family}: {len(rows)} classified ({len(confident_rows)} confident) ===")
        by_archetype = defaultdict(list)
        for r in confident_rows:
            if r["our_score"] is not None:
                by_archetype[r["best_fit"]].append(r["our_score"])
        for archetype in sorted(by_archetype, key=lambda a: statistics_median(by_archetype[a])):
            scores = by_archetype[archetype]
            scores_sorted = sorted(scores)
            median = scores_sorted[len(scores_sorted) // 2]
            mean = sum(scores) / len(scores)
            print(f"  {archetype:20s} n={len(scores):3d}  median={median:+.3f}  mean={mean:+.3f}")
        print()


def statistics_median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


if __name__ == "__main__":
    main()
