"""Persuasion strategy for both sides of the seller/buyer signaling game.

The seller bluffs at a dampened Kamenica-Gentzkow rate and not at all in the
opening rounds; the buyer weighs the seller's signal against expected value
with a safety margin. Messages are written by the rhetoric layer, with the
framing chosen by a bandit seeded from live success rates.
"""

from __future__ import annotations

import random

from agent import history
from agent.persuasion.signal import classify_discourage_signal
from agent.persuasion.solver import (
    BuyerTrustModel,
    SellerDisclosureAdapter,
    optimal_bluff_prob,
)
from agent.rhetoric.bandit import StyleBandit, get_style_bandit, persist_style_bandit
from agent.rhetoric.compose import compose_message

FALLBACK_LOW_QUALITY_BLUFF_PROB = 0.35

BUYER_SAFETY_MARGIN = 0.20
# Uses the game's prior p instead of the learned trust estimate once past the
# bootstrap purchases.
DISABLE_BUYER_TRUST_MODEL = False

MIN_ROUNDS_FOR_NO_INTEL_DETECTION = 3

EXPLORATION_EV_RATIO_THRESHOLD = 1.0 + BUYER_SAFETY_MARGIN

BOOTSTRAP_EV_RATIO_THRESHOLD = 0.95

DECLINE_TEST_UPSIDE_RATIO_THRESHOLD = 1.0 + BUYER_SAFETY_MARGIN

DECLINE_TRUST_PRIOR_P_HIGH = 0.15

# Initial Beta means per framing, set from observed buy rates (Table 4).
SELLER_FOUNDATION_PRIOR_MEANS = {
    "loyalty": 0.646,
    "purity": 0.578,
    "fairness": 0.533,
    "authority": 0.469,
    "care": 0.406,
}

# Honesty persists across rounds: one caught bluff can suppress purchasing for the
# rest of the game, so we don't bluff while the buyer is forming its opinion.
SELLER_TRUST_BUILDING_ROUNDS = 3
SELLER_TRUST_BUILDING_MAX_Q = 0.0

SELLER_BLUFF_DAMPENING = 0.7

DISCOURAGE_PHRASES = (
    "low quality", "low-quality", "wouldn't recommend", "would not recommend",
    "don't recommend", "do not recommend", "not recommend", "pass on it",
    "should pass", "hold off", "skip this", "steer clear", "not worth it",
    "poor quality", "subpar", "avoid this", "wouldn't buy", "would not buy",
    "ordinary", "mediocre", "middling", "nothing special", "run of the mill",
    "run-of-the-mill", "wait for the next", "wait for a better", "not this one",
    "not the one", "sit this one out", "sit this round out",
    "may want to pass", "want to pass today", "isn't my best",
    "below our usual quality", "below our usual",
)

DISCOURAGE_CONFIDENCE_THRESHOLD = 0.7

_SELLER_ADAPTERS: dict[str, SellerDisclosureAdapter] = {}
_LAST_SELLER_SIGNAL: dict[str, bool] = {}
_LAST_SELLER_PAYOFF: dict[str, float] = {}

_LAST_STYLE_ARM: dict[str, str] = {}

_BUYER_TRUST: dict[str, BuyerTrustModel] = {}
_LAST_BUYER_BOUGHT: dict[str, bool] = {}
_LAST_BUYER_PAYOFF: dict[str, float] = {}

_BUYER_NO_TRUST: dict[str, BuyerTrustModel] = {}
_LAST_BUYER_TESTED_NO: dict[str, bool] = {}


def _seller_adapter(game_id: str, q0: float) -> SellerDisclosureAdapter:
    if game_id not in _SELLER_ADAPTERS:
        _SELLER_ADAPTERS[game_id] = SellerDisclosureAdapter(q0)
    return _SELLER_ADAPTERS[game_id]


def _opponent_display_name(game: dict) -> str | None:
    return (game.get("opponent") or {}).get("name")


def _style_bandit(game: dict) -> StyleBandit:
    return get_style_bandit(game["game_id"], _opponent_display_name(game), prior_means=SELLER_FOUNDATION_PRIOR_MEANS)


def _update_seller_adapter_from_last_round(game: dict, state: dict, price: float) -> None:
    game_id = game["game_id"]
    if game_id not in _LAST_SELLER_SIGNAL:
        return
    signaled = _LAST_SELLER_SIGNAL.pop(game_id)
    prev_payoff = _LAST_SELLER_PAYOFF.get(game_id, 0.0)
    current_payoff = state.get("seller_total_payoff", 0.0)
    bought = (current_payoff - prev_payoff) > price / 2
    if signaled and game_id in _SELLER_ADAPTERS:
        _SELLER_ADAPTERS[game_id].record(bought)
    arm = _LAST_STYLE_ARM.pop(game_id, None)
    if arm is not None:
        _style_bandit(game).update(arm, bought)
        persist_style_bandit(game_id, _opponent_display_name(game))


def _buyer_trust(game_id: str, prior_p: float) -> BuyerTrustModel:
    if game_id not in _BUYER_TRUST:
        _BUYER_TRUST[game_id] = BuyerTrustModel(prior_p)
    return _BUYER_TRUST[game_id]


def _buyer_no_trust(game_id: str) -> BuyerTrustModel:
    if game_id not in _BUYER_NO_TRUST:
        _BUYER_NO_TRUST[game_id] = BuyerTrustModel(DECLINE_TRUST_PRIOR_P_HIGH, min_observations=1)
    return _BUYER_NO_TRUST[game_id]


# Free-text signals go to the LLM classifier first; the phrase list is a backstop
# when it fails or is unsure. Anything not clearly discouraging counts as a buy.
def _seller_signaled_buy(state: dict) -> bool:
    raw = state.get("seller_message")
    if state.get("seller_message_type") == "binary":
        if isinstance(raw, str):
            return raw.strip().lower() in ("yes", "true", "buy", "recommend")
        return bool(raw)
    if isinstance(raw, str) and raw.strip():
        llm_result = classify_discourage_signal(raw)
        if llm_result is not None:
            discourages, confidence = llm_result
            if confidence >= DISCOURAGE_CONFIDENCE_THRESHOLD:
                return not discourages
        lowered = raw.lower()
        if any(phrase in lowered for phrase in DISCOURAGE_PHRASES):
            return False
    return True


# An identical message every round carries no information about quality, so the
# buyer falls back to the prior instead of trusting it.
def _seller_message_repeated_every_round(state: dict) -> bool:
    current = state.get("seller_message")
    if current is None:
        return False
    prior = [h.get("seller_message") for h in state.get("history") or []]
    seen = prior + [current]
    if len(seen) < MIN_ROUNDS_FOR_NO_INTEL_DETECTION:
        return False
    return all(m == current for m in seen)


def _update_buyer_trust_from_last_round(game_id: str, state: dict, price: float, v: float, u: float) -> None:
    if game_id not in _LAST_BUYER_BOUGHT:
        return
    bought = _LAST_BUYER_BOUGHT.pop(game_id)
    tested_no = _LAST_BUYER_TESTED_NO.pop(game_id, False)
    if not bought:
        return
    prev_payoff = _LAST_BUYER_PAYOFF.get(game_id, 0.0)
    current_payoff = state.get("buyer_total_payoff", 0.0)
    delta = current_payoff - prev_payoff
    # Quality isn't revealed directly; infer it from which payoff the change matches.
    quality_was_high = abs(delta - (v - price)) < abs(delta - (u - price))
    if tested_no:
        if game_id in _BUYER_NO_TRUST:
            _BUYER_NO_TRUST[game_id].record(quality_was_high)
    elif game_id in _BUYER_TRUST:
        _BUYER_TRUST[game_id].record(quality_was_high)


def persuasion_strategy(game: dict) -> dict:
    state = game["game_state"]
    valid_actions = game["valid_actions"]
    game_id = game["game_id"]
    price = state["product_price"]
    p = state["p"]

    if valid_actions["type"] in ("seller_message", "seller_recommendation"):
        _update_seller_adapter_from_last_round(game, state, price)

        quality_high = state["current_quality"] == "high"
        know_cv = state.get("is_seller_know_cv", False) and "v" in state and "u" in state
        q_base = (
            optimal_bluff_prob(p, state["v"], state["u"], price)
            if know_cv
            else FALLBACK_LOW_QUALITY_BLUFF_PROB
        )
        q_base *= SELLER_BLUFF_DAMPENING
        q = _seller_adapter(game_id, q_base).adapted_q()
        if int(state.get("round", 1)) <= SELLER_TRUST_BUILDING_ROUNDS:
            q = min(q, SELLER_TRUST_BUILDING_MAX_Q)
        signal_buy = quality_high or (random.random() < q)

        if valid_actions["type"] == "seller_recommendation":
            action = {"decision": "yes" if signal_buy else "no"}
        else:
            style = _style_bandit(game).select_arm() if signal_buy else None
            message = compose_message(
                family="persuasion",
                situation=game["prompt"],
                stance={
                    "role": "seller",
                    "signaling_buy": signal_buy,
                    "fallback_message": (
                        "This is a strong unit — I'd recommend it."
                        if signal_buy
                        else "I'd hold off on this one."
                    ),
                },
                profile=None,
                style=style,
            )
            action = {"message": message}
            if style is not None:
                _LAST_STYLE_ARM[game_id] = style

        _LAST_SELLER_SIGNAL[game_id] = signal_buy
        _LAST_SELLER_PAYOFF[game_id] = state.get("seller_total_payoff", 0.0)
        history.record_own_action(game, action)
        return action

    if valid_actions["type"] == "buyer_decision":
        v, u = state["v"], state["u"]
        _update_buyer_trust_from_last_round(game_id, state, price, v, u)
        tested_no = False

        if _seller_message_repeated_every_round(state):
            effective_p = p
            expected_value = effective_p * v + (1 - effective_p) * u
            decision = "yes" if expected_value >= price * (1 + BUYER_SAFETY_MARGIN) else "no"
        elif _seller_signaled_buy(state):
            trust = _buyer_trust(game_id, p)
            prior_ev = p * v + (1 - p) * u
            bootstrap_ev_ratio_ok = price > 0 and prior_ev / price >= BOOTSTRAP_EV_RATIO_THRESHOLD
            if trust.buys < trust.min_observations and bootstrap_ev_ratio_ok:
                decision = "yes"
            elif DISABLE_BUYER_TRUST_MODEL:
                effective_p = p
                expected_value = effective_p * v + (1 - effective_p) * u
                decision = "yes" if expected_value >= price * (1 + BUYER_SAFETY_MARGIN) else "no"
            else:
                effective_p = trust.posterior_high_given_signal()
                expected_value = effective_p * v + (1 - effective_p) * u
                decision = "yes" if expected_value >= price * (1 + BUYER_SAFETY_MARGIN) else "no"
        else:
            # Declined signal: buy anyway only to test a seller when the upside
            # justifies the risk, and stop once those tests keep losing.
            downside = price - u
            upside = v - price
            unconditional_ev_ok = price > 0 and (p * v + (1 - p) * u) / price >= EXPLORATION_EV_RATIO_THRESHOLD
            worth_testing = (
                downside > 0 and upside > 0
                and upside / downside >= DECLINE_TEST_UPSIDE_RATIO_THRESHOLD
                and unconditional_ev_ok
            )
            if worth_testing:
                no_trust = _buyer_no_trust(game_id)
                if no_trust.buys < no_trust.min_observations:
                    decision, tested_no = "yes", True
                elif DISABLE_BUYER_TRUST_MODEL:
                    decision = "no"
                else:
                    effective_p_high = no_trust.posterior_high_given_signal()
                    expected_value = effective_p_high * v + (1 - effective_p_high) * u
                    if expected_value > price * (1 + BUYER_SAFETY_MARGIN):
                        decision, tested_no = "yes", True
                    else:
                        decision = "no"
            else:
                decision = "no"

        _LAST_BUYER_TESTED_NO[game_id] = tested_no
        _LAST_BUYER_BOUGHT[game_id] = decision == "yes"
        _LAST_BUYER_PAYOFF[game_id] = state.get("buyer_total_payoff", 0.0)
        action = {"decision": decision}
        history.record_own_action(game, action)
        return action

    return {}
