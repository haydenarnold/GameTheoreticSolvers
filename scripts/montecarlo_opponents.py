"""Monte Carlo simulation: our real strategy code vs 11 reactive opponent
archetypes, across a randomized parameter grid, for bargaining, negotiation,
and persuasion. Investigation-only -- does not modify any strategy code.

Every opponent here is REACTIVE (responds to our actual offers each round),
unlike the static/pre-scripted opponents used in ad-hoc incident replays
earlier in the session.

Usage: .venv/bin/python montecarlo_opponents.py [family ...]
"""
from __future__ import annotations

import logging
import math
import random
import statistics
import sys
from collections import defaultdict

logging.disable(logging.WARNING)  # schema.check_fields warns on the real,
# expected "reject + bundled counteroffer" shape our synthetic
# valid_actions fixtures don't declare -- harmless, just noisy.

sys.path.insert(0, "/Users/haydenarnold/Projects/GLEE")

from agent.bargaining.strategy import bargaining_strategy
from agent.negotiation.strategy import negotiation_strategy
from agent.negotiation.solver import concession_target, time_fraction
from agent.persuasion.strategy import persuasion_strategy

ARCHETYPES = [
    "hardliner", "fair_splitter", "tit_for_tat", "deadline_exploiter", "random",
    "deceptive", "highly_impatient", "lowball_highball", "bayesian_rational",
    "llm_like", "noisy_rational",
]

TRIALS_PER_CELL = 400  # per (archetype, family, role) -- see config randomization inside each runner


# ============================================================================
# BARGAINING
# ============================================================================

class BargOpponent:
    """Reactive bargaining opponent. `propose()` returns the opponent's own
    ask (their gain); `decide()` returns True (accept) given what we just
    offered them (their gain if they accept)."""

    def __init__(self, archetype, money, max_rounds, rng):
        self.archetype = archetype
        self.money = money
        self.max_rounds = max_rounds
        self.rng = rng
        self.last_own_ask = None
        self.our_prev_offer_to_them = None  # what we offered them, previous round
        self.our_prev_prev_offer_to_them = None
        self._bayes_delta_me = rng.uniform(0.7, 0.98)  # opponent's own private patience (bayesian/noisy only)
        self._exploit_round = None
        self._llm_noise = rng.uniform(-0.03, 0.03)

    def _t(self, round_no):
        return time_fraction(round_no, self.max_rounds, unlimited_tau=30.0)

    def propose(self, round_no):
        m = self.money
        a = self.archetype
        if a == "hardliner":
            ask = 0.92 * m
        elif a == "fair_splitter":
            ask = 0.5 * m
        elif a == "tit_for_tat":
            if self.last_own_ask is None:
                ask = 0.8 * m
            else:
                our_concession = 0.0
                if self.our_prev_prev_offer_to_them is not None and self.our_prev_offer_to_them is not None:
                    our_concession = max(0.0, self.our_prev_offer_to_them - self.our_prev_prev_offer_to_them)
                ask = max(0.5 * m, self.last_own_ask - our_concession)
        elif a == "deadline_exploiter":
            exploit_round = self.max_rounds - 2 if self.max_rounds else 88
            if round_no < exploit_round:
                ask = 0.90 * m
            else:
                ask = 0.55 * m  # concede just enough to look reasonable right at the panic zone
        elif a == "random":
            ask = self.rng.uniform(0.3, 0.9) * m
        elif a == "deceptive":
            # erratic: mostly concedes like a Boulware curve but randomly
            # reverses upward, breaking any inferable trend.
            base = concession_target(0.9 * m, 0.5 * m, self._t(round_no), 0.6)
            if self.rng.random() < 0.35:
                base = min(0.95 * m, base + self.rng.uniform(0.05, 0.2) * m)
            ask = base
        elif a == "highly_impatient":
            ask = concession_target(0.7 * m, 0.35 * m, self._t(round_no), 2.5)  # conceder, fast
        elif a == "lowball_highball":
            ask = concession_target(0.95 * m, 0.55 * m, self._t(round_no), 0.35)  # extreme anchor, slow after
        elif a in ("bayesian_rational", "noisy_rational"):
            delta_opp_for_us = self._bayes_delta_me  # opponent's own delta (their patience)
            # they don't know OUR true delta; assume symmetric guess of 0.9
            share = m * (1 - 0.9) / max(1e-9, (1 - delta_opp_for_us * 0.9))
            ask = m - share  # their own ask = money - (our theoretical share)
            ask = min(max(ask, 0.0), m)
            if a == "noisy_rational":
                ask = min(max(ask + self.rng.gauss(0, 0.06 * m), 0.0), m)
        elif a == "llm_like":
            raw = concession_target(0.75 * m, 0.5 * m, self._t(round_no), 0.8)
            raw = round(raw / (m * 0.01)) * (m * 0.01)  # round-number offers
            ask = min(max(raw + self._llm_noise * m, 0.0), m)
        else:
            ask = 0.6 * m
        ask = min(max(ask, 0.0), m)
        self.last_own_ask = ask
        return ask

    def decide(self, round_no, offered_to_them):
        m = self.money
        a = self.archetype
        frac = offered_to_them / m if m else 0
        if a == "hardliner":
            accept = frac >= 0.85
        elif a == "fair_splitter":
            accept = frac >= 0.45
        elif a == "tit_for_tat":
            target_frac = 1.0 - (self.propose(round_no) / m if self.last_own_ask is None else self.last_own_ask / m)
            accept = frac >= max(0.4, target_frac - 0.05)
        elif a == "deadline_exploiter":
            exploit_round = self.max_rounds - 2 if self.max_rounds else 88
            accept = frac >= 0.85 if round_no < exploit_round else frac >= 0.35
        elif a == "random":
            accept = frac >= self.rng.uniform(0.15, 0.55)
        elif a == "deceptive":
            bar = 0.55 - 0.3 * self._t(round_no)
            if self.rng.random() < 0.35:
                bar += self.rng.uniform(0.05, 0.2)
            accept = frac >= bar
        elif a == "highly_impatient":
            accept = frac >= max(0.3, 0.55 - 0.4 * self._t(round_no))
        elif a == "lowball_highball":
            accept = frac >= max(0.4, 0.85 - 0.5 * self._t(round_no))
        elif a in ("bayesian_rational", "noisy_rational"):
            delta_opp_for_us = self._bayes_delta_me
            their_share = m * (1 - 0.9) / max(1e-9, (1 - 0.9 * delta_opp_for_us))
            # continuation value of rejecting ~ delta * their_share
            threshold = delta_opp_for_us * their_share / m
            if a == "noisy_rational":
                threshold = min(max(threshold + self.rng.gauss(0, 0.06), 0.0), 1.0)
            accept = frac >= threshold
        elif a == "llm_like":
            accept = frac >= max(0.35, 0.55 - 0.5 * self._t(round_no) + self._llm_noise)
        else:
            accept = frac >= 0.5
        return accept

    def record_our_offer(self, offer_to_them):
        self.our_prev_prev_offer_to_them = self.our_prev_offer_to_them
        self.our_prev_offer_to_them = offer_to_them


def _barg_offer_game(game_id, me, opp, round_no, money, delta_me, delta_opp, max_rounds, history_list, complete_info=True):
    delta_me_key = "delta_1" if me == "alice" else "delta_2"
    delta_opp_key = "delta_2" if me == "alice" else "delta_1"
    state = {
        "current_player": me, "proposer": me, "round": round_no,
        "max_rounds": max_rounds, "horizon_known": max_rounds is not None,
        "money_to_divide": money, delta_me_key: delta_me,
        "complete_information": complete_info, "messages_allowed": False,
        "last_offer": None, "history": history_list,
    }
    if complete_info:
        state[delta_opp_key] = delta_opp
    return {
        "game_id": game_id, "game_family": "bargaining", "your_player": me, "prompt": "x",
        "game_state": state,
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
    }


def _barg_decision_game(game_id, me, opp, round_no, money, delta_me, delta_opp, max_rounds, last_offer, history_list, complete_info=True):
    delta_me_key = "delta_1" if me == "alice" else "delta_2"
    delta_opp_key = "delta_2" if me == "alice" else "delta_1"
    state = {
        "current_player": me, "proposer": opp, "round": round_no,
        "max_rounds": max_rounds, "horizon_known": max_rounds is not None,
        "money_to_divide": money, delta_me_key: delta_me,
        "complete_information": complete_info, "messages_allowed": False,
        "last_offer": last_offer, "history": history_list,
    }
    if complete_info:
        state[delta_opp_key] = delta_opp
    return {
        "game_id": game_id, "game_family": "bargaining", "your_player": me, "prompt": "x",
        "game_state": state,
        "valid_actions": {"type": "decision", "fields": {"decision": "'accept' or 'reject'"}},
    }


def run_bargaining_trial(archetype, rng):
    # FIXED (2026-08-29): every _barg_offer_game/_barg_decision_game call
    # below used to pass a FRESH f"mc-{archetype}-{rng.random()}" id every
    # single round, not once per trial -- unlike run_persuasion_trial's
    # correct pattern (one game_id computed at the top, reused for all 20
    # rounds). Since bargaining_strategy's DeltaBelief, offer/take-history
    # (_LAST_SENT_TAKE), and StyleBandit persistence are all keyed by
    # game_id, a fresh id every round meant that state never survived past
    # a single round -- the belief was silently re-initialized to its
    # uniform prior (0.819) every round instead of accumulating evidence,
    # for the entire suite. Confirmed as the root cause of the paper's own
    # bargaining ablation reporting "zero measured effect" from disabling
    # DeltaBelief: with the bug in place, DISABLE_DELTA_BELIEF was already
    # a no-op in this harness (delta_opp was always the fresh uniform mean
    # either way), so that ablation could never have detected a real
    # effect one way or the other. One stable id per trial (still globally
    # unique across trials via rng.random(), so no cross-trial state
    # leakage) matches how a real live game's single persistent game_id
    # actually behaves.
    game_id = f"mc-{archetype}-{rng.random()}"
    me, opp = "alice", "bob"
    money = rng.choice([100.0, 10000.0, 1_000_000.0])
    delta_me = rng.uniform(0.5, 0.98)
    max_rounds = rng.choice([None, None, 8, 12, 20])
    complete_info = rng.random() < 0.7
    we_propose_first = rng.random() < 0.5
    opp_reported_delta = rng.uniform(0.5, 0.98) if complete_info else None

    opponent = BargOpponent(archetype, money, max_rounds, rng)
    history_list = []
    hard_cap = max_rounds if max_rounds else 100

    delta_opp_field = opp_reported_delta if complete_info else rng.uniform(0.5, 0.98)  # incomplete-info: strategy uses belief; feed a plausible starting value it can't see anyway (server wouldn't send this key)
    my_gain_final = None
    opp_gain_final = None
    agreed_round = None

    for round_no in range(1, hard_cap + 1):
        my_turn = (round_no % 2 == 1) == we_propose_first
        if my_turn:
            game = _barg_offer_game(game_id, me, opp, round_no, money, delta_me, delta_opp_field, max_rounds, list(history_list), complete_info)
            try:
                action = bargaining_strategy(game)
            except Exception:
                return None
            my_take = action.get("alice_gain") if me == "alice" else action.get("bob_gain")
            if my_take is None:
                my_take = money - (action.get("bob_gain") if me == "alice" else action.get("alice_gain"))
            my_take = min(max(my_take, 0.0), money)
            offer_to_opp = money - my_take
            opponent.record_our_offer(offer_to_opp)
            accept = opponent.decide(round_no, offer_to_opp)
            offer_dict = {"alice_gain": my_take if me == "alice" else offer_to_opp,
                          "bob_gain": offer_to_opp if me == "alice" else my_take,
                          "proposer": me, "round": round_no}
            history_list.append({"round": round_no, "proposer": me, "offer": offer_dict, "decision": "accept" if accept else "reject"})
            if accept:
                my_gain_final, opp_gain_final, agreed_round = my_take, offer_to_opp, round_no
                break
        else:
            opp_ask = opponent.propose(round_no)
            my_gain_if_accept = money - opp_ask
            offer_dict = {"alice_gain": my_gain_if_accept if me == "alice" else opp_ask,
                          "bob_gain": opp_ask if me == "alice" else my_gain_if_accept,
                          "proposer": opp, "round": round_no}
            game = _barg_decision_game(game_id, me, opp, round_no, money, delta_me, delta_opp_field, max_rounds, offer_dict, list(history_list), complete_info)
            try:
                action = bargaining_strategy(game)
            except Exception:
                return None
            accept = action.get("decision") == "accept"
            history_list.append({"round": round_no, "proposer": opp, "offer": offer_dict, "decision": "accept" if accept else "reject"})
            if accept:
                my_gain_final, opp_gain_final, agreed_round = my_gain_if_accept, opp_ask, round_no
                break

    if agreed_round is None:
        return 0.0  # no deal, both get 0 -- matches bargaining's own no-agreement payoff
    discounted = my_gain_final * (delta_me ** (agreed_round - 1))
    return discounted / money if money else 0.0


# ============================================================================
# NEGOTIATION
# ============================================================================

class NegOpponent:
    """Reactive negotiation opponent. Operates on price directly.
    `role` is the OPPONENT's role (seller/buyer, opposite of ours)."""

    def __init__(self, archetype, role, my_value_range, rng):
        self.archetype = archetype
        self.role = role  # opponent's role
        self.rng = rng
        self.true_value = rng.uniform(*my_value_range)
        self.last_own_price = None
        self.our_prev_offer = None
        self.our_prev_prev_offer = None
        self._llm_noise = rng.uniform(-0.03, 0.03)
        self._bayes_noise_scale = rng.uniform(0.03, 0.08)

    def _anchor(self):
        # opening anchor away from true value
        return self.true_value * (1.5 if self.role == "seller" else 0.6)

    def _t(self, round_no, max_rounds):
        return time_fraction(round_no, max_rounds, unlimited_tau=30.0)

    def propose(self, round_no, max_rounds):
        v = self.true_value
        a = self.archetype
        anchor = self._anchor()
        sign = 1 if self.role == "seller" else -1  # seller wants HIGH, buyer wants LOW
        if a == "hardliner":
            price = anchor
        elif a == "fair_splitter":
            price = v * (1.05 if self.role == "seller" else 0.95)
        elif a == "tit_for_tat":
            if self.last_own_price is None:
                price = anchor
            else:
                our_concession = 0.0
                if self.our_prev_prev_offer is not None and self.our_prev_offer is not None:
                    our_concession = abs(self.our_prev_offer - self.our_prev_prev_offer)
                price = self.last_own_price - sign * our_concession
        elif a == "deadline_exploiter":
            exploit_round = (max_rounds - 2) if max_rounds else 88
            price = anchor if round_no < exploit_round else v * (1.1 if self.role == "seller" else 0.9)
        elif a == "random":
            price = v * self.rng.uniform(0.7, 1.8) if self.role == "seller" else v * self.rng.uniform(0.5, 1.1)
        elif a == "deceptive":
            base = concession_target(anchor, v, self._t(round_no, max_rounds), 0.6)
            if self.rng.random() < 0.35:
                base = base + sign * self.rng.uniform(0.05, 0.2) * v
            price = base
        elif a == "highly_impatient":
            price = concession_target(anchor, v, self._t(round_no, max_rounds), 2.5)
        elif a == "lowball_highball":
            price = concession_target(anchor * 1.15, v, self._t(round_no, max_rounds), 0.35)
        elif a in ("bayesian_rational", "noisy_rational"):
            t = self._t(round_no, max_rounds)
            price = concession_target(anchor, v, min(1.0, t * 1.4), 1.0)
            if a == "noisy_rational":
                price = price + self.rng.gauss(0, self._bayes_noise_scale * v)
        elif a == "llm_like":
            raw = concession_target(anchor, v, self._t(round_no, max_rounds), 0.8)
            raw = round(raw / (v * 0.01)) * (v * 0.01) if v else raw
            price = raw + self._llm_noise * v
        else:
            price = anchor
        if self.role == "seller":
            price = max(price, 0.0)
        self.last_own_price = price
        return price

    def decide(self, round_no, max_rounds, offer_price):
        v = self.true_value
        a = self.archetype
        profitable = offer_price >= v if self.role == "seller" else offer_price <= v
        margin = (offer_price - v) / v if self.role == "seller" else (v - offer_price) / v if v else 0
        t = self._t(round_no, max_rounds)
        if a == "hardliner":
            accept = margin >= 0.30
        elif a == "fair_splitter":
            accept = profitable and margin >= 0.02
        elif a == "tit_for_tat":
            accept = profitable and margin >= max(0.02, 0.25 - 0.2 * t)
        elif a == "deadline_exploiter":
            exploit_round = (max_rounds - 2) if max_rounds else 88
            accept = margin >= (0.30 if round_no < exploit_round else 0.02) and profitable
        elif a == "random":
            accept = profitable and self.rng.random() < 0.5
        elif a == "deceptive":
            bar = 0.25 - 0.2 * t
            if self.rng.random() < 0.35:
                bar += self.rng.uniform(0.05, 0.15)
            accept = profitable and margin >= bar
        elif a == "highly_impatient":
            accept = profitable and margin >= max(0.0, 0.15 - 0.15 * t)
        elif a == "lowball_highball":
            accept = profitable and margin >= max(0.02, 0.30 - 0.28 * t)
        elif a in ("bayesian_rational", "noisy_rational"):
            bar = max(0.0, 0.20 - 0.20 * min(1.0, t * 1.4))
            if a == "noisy_rational":
                bar = max(0.0, bar + self.rng.gauss(0, 0.04))
            accept = profitable and margin >= bar
        elif a == "llm_like":
            accept = profitable and margin >= max(0.0, 0.15 - 0.15 * t + self._llm_noise)
        else:
            accept = profitable
        return accept

    def record_our_offer(self, price):
        self.our_prev_prev_offer = self.our_prev_offer
        self.our_prev_offer = price


def _neg_offer_game(game_id, me, my_role, my_value, round_no, max_rounds, complete_info, opp_value, history_list):
    opp = "player_2" if me == "player_1" else "player_1"
    opp_role = "buyer" if my_role == "seller" else "seller"
    state = {
        "current_player": me, f"{me}_role": my_role, f"{opp}_role": opp_role,
        f"{me}_value": my_value, "round": round_no, "max_rounds": max_rounds,
        "horizon_known": max_rounds is not None, "messages_allowed": False,
        "complete_information": complete_info, "history": history_list,
    }
    if complete_info:
        state[f"{opp}_value"] = opp_value
    return {
        "game_id": game_id, "game_family": "negotiation", "your_player": me, "prompt": "x",
        "game_state": state, "valid_actions": {"type": "offer", "fields": {"product_price": "number"}},
    }


def _neg_decision_game(game_id, me, my_role, my_value, round_no, max_rounds, complete_info, opp_value, last_offer, history_list):
    opp = "player_2" if me == "player_1" else "player_1"
    opp_role = "buyer" if my_role == "seller" else "seller"
    state = {
        "current_player": me, f"{me}_role": my_role, f"{opp}_role": opp_role,
        f"{me}_value": my_value, "round": round_no, "max_rounds": max_rounds,
        "horizon_known": max_rounds is not None, "messages_allowed": False,
        "complete_information": complete_info, "last_offer": last_offer, "history": history_list,
    }
    if complete_info:
        state[f"{opp}_value"] = opp_value
    return {
        "game_id": game_id, "game_family": "negotiation", "your_player": me, "prompt": "x",
        "game_state": state, "valid_actions": {"type": "decision", "fields": {"decision": "'AcceptOffer', 'RejectOffer', or 'WalkAway'"}},
    }


def run_negotiation_trial(archetype, rng):
    # FIXED (2026-08-29): identical bug to run_bargaining_trial's fix above
    # -- every _neg_offer_game/_neg_decision_game call used a fresh random
    # game_id per round instead of one stable id per trial. This is the
    # confirmed root cause of the paper's own appendix note that
    # "Negotiation's mechanism never fires against these archetypes (0
    # calls across 4,400 trials)": _LAST_SENT_PRICE.get(game_id) is read
    # inside `negotiation_strategy`'s decision branch to find OUR OWN prior
    # offer so ValueBounds.tighten_from_rejection can fire -- but with a
    # fresh id every round, that lookup was ALWAYS None, so the mechanism
    # was structurally prevented from ever running, not genuinely rare.
    # This wasn't an inconclusive ablation result; it was a harness bug
    # that made the ablation incapable of testing the mechanism at all.
    game_id = f"mc-{archetype}-{rng.random()}"
    me = "player_1"
    my_role = rng.choice(["seller", "buyer"])
    opp_role = "buyer" if my_role == "seller" else "seller"
    my_value = rng.choice([80.0, 8000.0, 1_000_000.0]) * rng.uniform(0.7, 1.3)
    max_rounds = rng.choice([None, None, 8, 10, 20])
    complete_info = rng.random() < 0.4
    we_propose_first = rng.random() < 0.5

    value_range = (my_value * 0.6, my_value * 1.6) if my_role == "seller" else (my_value * 0.3, my_value * 1.3)
    opponent = NegOpponent(archetype, opp_role, value_range, rng)
    hard_cap = max_rounds if max_rounds else 100
    history_list = []
    outcome_price = None
    agreed_round = None

    for round_no in range(1, hard_cap + 1):
        my_turn = (round_no % 2 == 1) == we_propose_first
        if my_turn:
            game = _neg_offer_game(game_id, me, my_role, my_value, round_no, max_rounds, complete_info, opponent.true_value, list(history_list))
            try:
                action = negotiation_strategy(game)
            except Exception:
                return None
            price = action["product_price"]
            opponent.record_our_offer(price)
            accept = opponent.decide(round_no, max_rounds, price)
            offer = {"price": price, "from_player": me, "message": "", "round": round_no}
            history_list.append({"round": round_no, "offer": offer, "decision": "AcceptOffer" if accept else "RejectOffer", "decided_by": "player_2"})
            if accept:
                outcome_price, agreed_round = price, round_no
                break
        else:
            price = opponent.propose(round_no, max_rounds)
            game = _neg_decision_game(game_id, me, my_role, my_value, round_no, max_rounds, complete_info, opponent.true_value,
                                       {"price": price, "from_player": "player_2", "round": round_no}, list(history_list))
            try:
                action = negotiation_strategy(game)
            except Exception:
                return None
            decision = action.get("decision")
            offer = {"price": price, "from_player": "player_2", "message": "", "round": round_no}
            history_list.append({"round": round_no, "offer": offer, "decision": decision, "decided_by": me})
            if decision == "AcceptOffer":
                outcome_price, agreed_round = price, round_no
                break
            if decision == "WalkAway":
                return -1.0

    if outcome_price is None:
        return -1.0
    payoff = (my_value - outcome_price) if my_role == "buyer" else (outcome_price - my_value)
    return payoff / my_value if my_value else -1.0


# ============================================================================
# PERSUASION
# ============================================================================

class PersOpponentBuyer:
    """Reactive buyer opponent (we are seller). Decides based on seller's
    signal history this game."""

    def __init__(self, archetype, rng):
        self.archetype = archetype
        self.rng = rng
        self.seller_correct = 0
        self.seller_total = 0
        self.last_bought = None

    def decide(self, round_no, signal_buy, true_high_prob, v, u, price):
        a = self.archetype
        prior_ev = true_high_prob * v + (1 - true_high_prob) * u
        if a == "hardliner":
            return False  # never buys, regardless of signal
        if a == "fair_splitter":
            return signal_buy and prior_ev >= price
        if a == "tit_for_tat":
            # mirrors seller's revealed honesty: buys once trust (correct rate) is decent
            trust = self.seller_correct / self.seller_total if self.seller_total else 0.5
            return signal_buy and (self.seller_total < 2 or trust > 0.5) and prior_ev >= price * 0.9
        if a == "deadline_exploiter":
            return signal_buy and round_no >= 15 and prior_ev >= price * 0.8
        if a == "random":
            return self.rng.random() < 0.4
        if a == "deceptive":
            # inconsistent: sometimes buys against signal, sometimes not with it
            base = signal_buy and prior_ev >= price
            if self.rng.random() < 0.25:
                base = not base
            return base
        if a == "highly_impatient":
            return signal_buy  # buys on any positive signal, doesn't wait for trust
        if a == "lowball_highball":
            return signal_buy and prior_ev >= price * 1.3  # demands a big margin
        if a in ("bayesian_rational", "noisy_rational"):
            trust = (self.seller_correct + 1) / (self.seller_total + 2)  # Bayesian-ish
            eff_p = trust if signal_buy else (1 - trust)
            ev = eff_p * v + (1 - eff_p) * u
            if a == "noisy_rational":
                ev += self.rng.gauss(0, 0.05 * price)
            return ev >= price
        if a == "llm_like":
            return signal_buy and prior_ev >= price * self.rng.uniform(0.85, 1.05)
        return signal_buy and prior_ev >= price

    def observe(self, signal_buy, quality_high):
        self.seller_total += 1
        if signal_buy == quality_high:
            self.seller_correct += 1


class PersOpponentSeller:
    """Reactive seller opponent (we are buyer). Decides its "yes"/"no"
    signal based on true quality and archetype-specific honesty."""

    def __init__(self, archetype, rng):
        self.archetype = archetype
        self.rng = rng

    def signal(self, round_no, quality_high, p, v, u, price):
        a = self.archetype
        if a == "hardliner":
            return True  # always says buy, regardless of quality (extreme anchor)
        if a == "fair_splitter":
            return quality_high  # perfectly honest
        if a == "tit_for_tat":
            return quality_high
        if a == "deadline_exploiter":
            return True if round_no < 18 else quality_high
        if a == "random":
            return self.rng.random() < 0.6
        if a == "deceptive":
            return True if quality_high else self.rng.random() < 0.5
        if a == "highly_impatient":
            return True  # always pushes to sell fast
        if a == "lowball_highball":
            return True
        if a in ("bayesian_rational", "noisy_rational"):
            # optimal-ish bluffing: honest on high, bluffs low at rate making buyer indifferent
            if quality_high:
                return True
            q_star = min(1.0, max(0.0, (price - u) / max(1e-9, (v - u)))) if v != u else 0.0
            noisy = q_star + (self.rng.gauss(0, 0.1) if a == "noisy_rational" else 0)
            return self.rng.random() < max(0.0, min(1.0, noisy))
        if a == "llm_like":
            return quality_high or self.rng.random() < 0.3
        return quality_high


def _pers_seller_game(game_id, round_no, quality, p, v, u, price):
    return {
        "game_id": game_id, "game_family": "persuasion", "your_player": "seller", "prompt": "x",
        "game_state": {
            "product_price": price, "p": p, "v": v, "u": u, "current_quality": quality,
            "round": round_no, "total_rounds": 20, "seller_total_payoff": 0.0,
            "buyer_total_payoff": 0.0, "is_seller_know_cv": True,
        },
        "valid_actions": {"type": "seller_recommendation", "fields": {"decision": "'yes' or 'no'"}},
    }


def _pers_buyer_game(game_id, round_no, seller_signal, p, v, u, price):
    return {
        "game_id": game_id, "game_family": "persuasion", "your_player": "buyer", "prompt": "x",
        "game_state": {
            "product_price": price, "p": p, "v": v, "u": u,
            "seller_message": "yes" if seller_signal else "no",
            "round": round_no, "total_rounds": 20, "seller_total_payoff": 0.0,
            "buyer_total_payoff": 0.0, "is_seller_know_cv": True, "seller_message_type": "binary", "history": [],
        },
        "valid_actions": {"type": "buyer_decision", "fields": {"decision": "'yes' or 'no'"}},
    }


def run_persuasion_trial(archetype, rng, we_are_seller):
    # FIXED (2026-08-29): agent/persuasion/strategy.py's seller-side bluff
    # decision (`signal_buy = quality_high or (random.random() < q)`) draws
    # from Python's global, process-wide `random` module -- correct for live
    # play (a bluff probability genuinely should be unpredictable against a
    # real opponent, not derived from a fixed per-game seed an opponent
    # could in principle exploit), but it means a seller-role trial's
    # outcome here depends on however many prior random.random()-family
    # calls happened anywhere else in this process, not just on `rng`.
    # Confirmed: two ablation passes run back-to-back in one process
    # produced different persuasion numbers for the SAME nominal trial than
    # a fresh, single-pass process did, even after fixing the separate
    # per-game_id cache-collision bug above -- purely from the global RNG's
    # accumulated state differing between the two. Reseeding Python's
    # global random module here, deterministically derived from this
    # trial's own already-seeded `rng`, makes the seller's bluff draw
    # reproducible given the trial's seed without changing anything about
    # how the deployed strategy code itself behaves live.
    random.seed(rng.random())
    p = rng.choice([0.3, 0.5, 0.8])
    price = rng.choice([50.0, 10000.0, 1_000_000.0])
    v = price * rng.uniform(1.1, 3.0)
    u = 0.0
    game_id = f"mc-{archetype}-{rng.random()}"
    total_payoff = 0.0
    n_bought = 0

    if we_are_seller:
        opp = PersOpponentBuyer(archetype, rng)
        for round_no in range(1, 21):
            quality_high = rng.random() < p
            quality = "high" if quality_high else "low"
            game = _pers_seller_game(game_id, round_no, quality, p, v, u, price)
            try:
                action = persuasion_strategy(game)
            except Exception:
                return None
            signal_buy = action.get("decision") == "yes"
            bought = opp.decide(round_no, signal_buy, p, v, u, price)
            opp.observe(signal_buy, quality_high)
            if bought:
                n_bought += 1
                total_payoff += price  # seller gets price regardless of quality (buyer bears risk)
        return total_payoff / (price * 20)
    else:
        opp = PersOpponentSeller(archetype, rng)
        for round_no in range(1, 21):
            quality_high = rng.random() < p
            seller_signal = opp.signal(round_no, quality_high, p, v, u, price)
            game = _pers_buyer_game(game_id, round_no, seller_signal, p, v, u, price)
            try:
                action = persuasion_strategy(game)
            except Exception:
                return None
            buy = action.get("decision") == "yes"
            if buy:
                n_bought += 1
                total_payoff += (v - price) if quality_high else (u - price)
        return total_payoff / (price * 20)


# ============================================================================
# RUNNER
# ============================================================================

def run_family(family, trials):
    rng_master = random.Random(12345)
    results = defaultdict(list)
    crashes = defaultdict(int)
    for archetype in ARCHETYPES:
        for i in range(trials):
            rng = random.Random(rng_master.random())
            if family == "bargaining":
                score = run_bargaining_trial(archetype, rng)
            elif family == "negotiation":
                score = run_negotiation_trial(archetype, rng)
            elif family == "persuasion":
                we_are_seller = i % 2 == 0
                score = run_persuasion_trial(archetype, rng, we_are_seller)
            else:
                raise ValueError(family)
            if score is None:
                crashes[archetype] += 1
            else:
                results[archetype].append(score)
    return results, crashes


def main():
    families = sys.argv[1:] or ["bargaining", "negotiation", "persuasion"]
    trials = TRIALS_PER_CELL
    for family in families:
        print(f"\n=== {family} (n={trials}/archetype) ===")
        results, crashes = run_family(family, trials)
        ranked = sorted(results.items(), key=lambda kv: statistics.mean(kv[1]) if kv[1] else -999)
        for archetype, scores in ranked:
            if not scores:
                print(f"  {archetype:20s} ALL CRASHED ({crashes[archetype]})")
                continue
            mean = statistics.mean(scores)
            median = statistics.median(scores)
            stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
            crash_note = f" ({crashes[archetype]} crashed)" if crashes.get(archetype) else ""
            print(f"  {archetype:20s} n={len(scores):3d} mean={mean:+.4f} median={median:+.4f} stdev={stdev:.4f}{crash_note}")


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] == "trace"):
    main()


# ============================================================================
# DIAGNOSTIC TRACE MODE (deep-dive on specific archetype/family pairs)
# ============================================================================

def trace_bargaining(archetype, seed, verbose=True):
    rng = random.Random(seed)
    me, opp = "alice", "bob"
    money = rng.choice([100.0, 10000.0, 1_000_000.0])
    delta_me = rng.uniform(0.5, 0.98)
    max_rounds = rng.choice([None, None, 8, 12, 20])
    complete_info = rng.random() < 0.7
    we_propose_first = rng.random() < 0.5
    opp_reported_delta = rng.uniform(0.5, 0.98) if complete_info else None
    opponent = BargOpponent(archetype, money, max_rounds, rng)
    history_list = []
    hard_cap = max_rounds if max_rounds else 100
    delta_opp_field = opp_reported_delta if complete_info else rng.uniform(0.5, 0.98)
    if verbose:
        print(f"config: money={money} delta_me={delta_me:.3f} max_rounds={max_rounds} complete_info={complete_info} we_propose_first={we_propose_first}")
    agreed_round = None
    my_gain_final = None
    for round_no in range(1, hard_cap + 1):
        my_turn = (round_no % 2 == 1) == we_propose_first
        if my_turn:
            game = _barg_offer_game(f"trace-{archetype}", me, opp, round_no, money, delta_me, delta_opp_field, max_rounds, list(history_list), complete_info)
            action = bargaining_strategy(game)
            my_take = action.get("alice_gain") if me == "alice" else action.get("bob_gain")
            if my_take is None:
                my_take = money - (action.get("bob_gain") if me == "alice" else action.get("alice_gain"))
            my_take = min(max(my_take, 0.0), money)
            offer_to_opp = money - my_take
            opponent.record_our_offer(offer_to_opp)
            accept = opponent.decide(round_no, offer_to_opp)
            if verbose:
                print(f"  r{round_no}: WE offer ourselves {my_take/money:.3f} of pot -> opp {'ACCEPTS' if accept else 'rejects'}")
            history_list.append({"round": round_no, "proposer": me, "offer": {"alice_gain": my_take, "bob_gain": offer_to_opp, "proposer": me, "round": round_no}, "decision": "accept" if accept else "reject"})
            if accept:
                agreed_round, my_gain_final = round_no, my_take
                break
        else:
            opp_ask = opponent.propose(round_no)
            my_gain_if_accept = money - opp_ask
            offer_dict = {"alice_gain": my_gain_if_accept, "bob_gain": opp_ask, "proposer": opp, "round": round_no}
            game = _barg_decision_game(f"trace-{archetype}", me, opp, round_no, money, delta_me, delta_opp_field, max_rounds, offer_dict, list(history_list), complete_info)
            action = bargaining_strategy(game)
            accept = action.get("decision") == "accept"
            if verbose:
                print(f"  r{round_no}: OPP asks for {opp_ask/money:.3f} of pot (offers us {my_gain_if_accept/money:.3f}) -> WE {'accept' if accept else 'reject'}")
            history_list.append({"round": round_no, "proposer": opp, "offer": offer_dict, "decision": "accept" if accept else "reject"})
            if accept:
                agreed_round, my_gain_final = round_no, my_gain_if_accept
                break
    if agreed_round is None:
        print(f"  NO DEAL after {hard_cap} rounds. score=0.0")
        return 0.0
    discounted = my_gain_final * (delta_me ** (agreed_round - 1))
    score = discounted / money
    print(f"  AGREED round {agreed_round}, nominal frac={my_gain_final/money:.3f}, discounted score={score:.4f}")
    return score


def trace_negotiation(archetype, seed, verbose=True):
    rng = random.Random(seed)
    me = "player_1"
    my_role = rng.choice(["seller", "buyer"])
    opp_role = "buyer" if my_role == "seller" else "seller"
    my_value = rng.choice([80.0, 8000.0, 1_000_000.0]) * rng.uniform(0.7, 1.3)
    max_rounds = rng.choice([None, None, 8, 10, 20])
    complete_info = rng.random() < 0.4
    we_propose_first = rng.random() < 0.5
    value_range = (my_value * 0.6, my_value * 1.6) if my_role == "seller" else (my_value * 0.3, my_value * 1.3)
    opponent = NegOpponent(archetype, opp_role, value_range, rng)
    hard_cap = max_rounds if max_rounds else 100
    history_list = []
    if verbose:
        zopa = (my_value > opponent.true_value) if my_role == "buyer" else (opponent.true_value > my_value)
        print(f"config: my_role={my_role} my_value={my_value:.1f} opp_true_value={opponent.true_value:.1f} max_rounds={max_rounds} complete_info={complete_info} ZOPA_exists={zopa}")
    outcome_price = None
    agreed_round = None
    for round_no in range(1, hard_cap + 1):
        my_turn = (round_no % 2 == 1) == we_propose_first
        if my_turn:
            game = _neg_offer_game(f"trace-{archetype}", me, my_role, my_value, round_no, max_rounds, complete_info, opponent.true_value, list(history_list))
            action = negotiation_strategy(game)
            price = action["product_price"]
            opponent.record_our_offer(price)
            accept = opponent.decide(round_no, max_rounds, price)
            if verbose:
                print(f"  r{round_no}: WE offer price={price:.1f} -> opp {'ACCEPTS' if accept else 'rejects'}")
            history_list.append({"round": round_no, "offer": {"price": price, "from_player": me, "message": "", "round": round_no}, "decision": "AcceptOffer" if accept else "RejectOffer", "decided_by": "player_2"})
            if accept:
                outcome_price, agreed_round = price, round_no
                break
        else:
            price = opponent.propose(round_no, max_rounds)
            game = _neg_decision_game(f"trace-{archetype}", me, my_role, my_value, round_no, max_rounds, complete_info, opponent.true_value, {"price": price, "from_player": "player_2", "round": round_no}, list(history_list))
            action = negotiation_strategy(game)
            decision = action.get("decision")
            if verbose:
                print(f"  r{round_no}: OPP offers price={price:.1f} -> WE {decision}")
            history_list.append({"round": round_no, "offer": {"price": price, "from_player": "player_2", "message": "", "round": round_no}, "decision": decision, "decided_by": me})
            if decision == "AcceptOffer":
                outcome_price, agreed_round = price, round_no
                break
            if decision == "WalkAway":
                print(f"  WE WALKED AWAY at round {round_no}. score=-1.0")
                return -1.0
    if outcome_price is None:
        print(f"  NO DEAL (round limit) after {hard_cap} rounds. score=-1.0")
        return -1.0
    payoff = (my_value - outcome_price) if my_role == "buyer" else (outcome_price - my_value)
    score = payoff / my_value if my_value else -1.0
    print(f"  AGREED round {agreed_round} at price={outcome_price:.1f}, score={score:.4f}")
    return score


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "trace":
    family, archetype = sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    for seed in range(n):
        print(f"--- {family} vs {archetype}, seed={seed} ---")
        if family == "bargaining":
            trace_bargaining(archetype, seed)
        elif family == "negotiation":
            trace_negotiation(archetype, seed)
        print()
