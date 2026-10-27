"""Negotiation strategy: Faratin time-dependent concession with acceptance rules.

Counters follow a concession curve from an anchor toward our reservation value,
and offers are accepted early when they clear a safety margin or when a
profitable offer has stopped moving. A language model writes the accompanying
message.
"""

from __future__ import annotations

from agent import history
from agent.negotiation.solver import ValueBounds, concession_target, deadline_pressure, time_fraction
from agent.rhetoric.bandit import StyleBandit, get_style_bandit, persist_style_bandit
from agent.rhetoric.compose import compose_message
from agent.rhetoric.profile import classify
from agent.schema import check_fields, opponent_name

CONCESSION_BETA = 0.85
DEFAULT_PRIOR_SPREAD = 1.5
# Skips tightening the opponent's value interval from our rejected prices, so the
# opening anchor stays at the prior bound.
DISABLE_VALUE_BOUNDS = False
UNLIMITED_TAU = 30.0
# A one-shot game has no later round to concede in, so the single offer is priced
# at a fixed point on the curve rather than at t = 0.
ONE_SHOT_ULTIMATUM_T = 0.75

ONE_SHOT_ULTIMATUM_COMPLETE_INFO_T = 0.03
STATIONARY_OPPONENT_ROUNDS = 3
STATIONARY_OPPONENT_TOLERANCE_FRAC = 0.005

EARLY_ACCEPT_MARGIN_FRAC = 0.30

STATIONARY_ACCEPT_MIN_MARGIN_FRAC = 0.10

OFFER_CONCESSION_DAMPING_FRAC = 0.3

_BOUNDS: dict[str, ValueBounds] = {}
_LAST_SENT_PRICE: dict[str, float] = {}


def _get_bounds(game_id: str, role: str, my_value: float, opponent_true_value: float | None) -> ValueBounds:
    if game_id not in _BOUNDS:
        if opponent_true_value is not None:
            seller_value = my_value if role == "seller" else opponent_true_value
            buyer_value = my_value if role == "buyer" else opponent_true_value
            _BOUNDS[game_id] = ValueBounds(lo=seller_value, hi=buyer_value)
        elif role == "seller":
            _BOUNDS[game_id] = ValueBounds(lo=my_value, hi=my_value * DEFAULT_PRIOR_SPREAD)
        else:
            _BOUNDS[game_id] = ValueBounds(lo=my_value / DEFAULT_PRIOR_SPREAD, hi=my_value)
    return _BOUNDS[game_id]


def _rounds_remaining(state: dict) -> int | None:
    max_rounds = state.get("max_rounds")
    if not state.get("horizon_known", max_rounds is not None) or max_rounds is None:
        return None
    return max(int(max_rounds) - int(state["round"]), 0)


# There is no per-round decay here, so a stuck negotiation can only end in a
# walk-away; a profitable offer that has stopped moving is worth taking.
def _opponent_offer_is_stationary(game: dict, opp: str, my_value: float, current_price: float) -> bool:
    if my_value <= 0:
        return False
    prior_opponent_prices = [
        h["offer"]["price"] for h in history.get_history(game)
        if history.is_opponent_entry(h, opp) and h.get("offer", {}).get("price") is not None
    ]
    recent = prior_opponent_prices[-(STATIONARY_OPPONENT_ROUNDS - 1):] + [current_price]
    if len(recent) < STATIONARY_OPPONENT_ROUNDS:
        return False
    return max(recent) - min(recent) <= STATIONARY_OPPONENT_TOLERANCE_FRAC * my_value


def _stationary_offer_clears_min_margin(
    role: str, my_value: float, offer_price: float, bounds: ValueBounds, complete_information: bool,
) -> bool:
    if not complete_information:
        return True
    zopa_width = bounds.hi - bounds.lo
    if zopa_width <= 0:
        return True
    margin = (offer_price - my_value) if role == "seller" else (my_value - offer_price)
    return margin >= STATIONARY_ACCEPT_MIN_MARGIN_FRAC * zopa_width


def _offer_clears_safety_margin(role: str, my_value: float, offer_price: float) -> bool:
    if my_value <= 0:
        return False
    if role == "seller":
        return (offer_price - my_value) / my_value >= EARLY_ACCEPT_MARGIN_FRAC
    return (my_value - offer_price) / my_value >= EARLY_ACCEPT_MARGIN_FRAC


_LAST_STYLE_ARM: dict[str, str] = {}


def _opponent_display_name(game: dict) -> str | None:
    return (game.get("opponent") or {}).get("name")


def _style_bandit(game: dict) -> StyleBandit:
    return get_style_bandit(game["game_id"], _opponent_display_name(game))


def _credit_style_bandit_from_opponent_response(
    game: dict, opp: str, game_id: str, role: str, current_price: float, my_value: float,
) -> None:
    arm = _LAST_STYLE_ARM.pop(game_id, None)
    if arm is None:
        return
    prior_opponent_prices = [
        h["offer"]["price"] for h in history.get_history(game)
        if history.is_opponent_entry(h, opp) and h.get("offer", {}).get("price") is not None
    ]
    if not prior_opponent_prices:
        return
    moved_by = (
        current_price - prior_opponent_prices[-1] if role == "seller"
        else prior_opponent_prices[-1] - current_price
    )
    success = moved_by > STATIONARY_OPPONENT_TOLERANCE_FRAC * my_value
    _style_bandit(game).update(arm, success)
    persist_style_bandit(game_id, _opponent_display_name(game))


def negotiation_strategy(game: dict) -> dict:
    state = game["game_state"]
    valid_actions = game["valid_actions"]
    me = game["your_player"]
    opp = opponent_name(me)
    role = state[f"{me}_role"]
    opponent_role = "buyer" if role == "seller" else "seller"
    my_value = state[f"{me}_value"]
    game_id = game["game_id"]

    opp_value_key = "player_1_value" if f"{me}_value" == "player_2_value" else "player_2_value"
    complete_information = state.get("complete_information", opp_value_key in state)

    rounds_remaining = _rounds_remaining(state)
    messages_allowed = state.get("messages_allowed", False)
    opponent_profile = classify(
        m["offer"].get("message", "")
        for m in history.get_history(game)
        if history.is_opponent_entry(m, opp) and m.get("offer", {}).get("message")
    ) if messages_allowed else None

    opponent_true_value = state.get(opp_value_key) if complete_information else None
    bounds = _get_bounds(game_id, role, my_value, opponent_true_value)

    if valid_actions["type"] == "decision":
        last_offer = state["last_offer"]
        history.record_opponent_offer(game, last_offer)
        offer_price = last_offer["price"]
        _credit_style_bandit_from_opponent_response(game, opp, game_id, role, offer_price, my_value)

        prior_price = _LAST_SENT_PRICE.get(game_id)
        if not DISABLE_VALUE_BOUNDS and prior_price is not None and not complete_information:
            bounds.tighten_from_rejection(opponent_role, prior_price)

        profitable = offer_price <= my_value if role == "buyer" else offer_price >= my_value

        if rounds_remaining == 0:
            action = {"decision": "AcceptOffer"} if profitable else {"decision": "WalkAway"}
            check_fields(valid_actions, action)
            history.record_own_action(game, action)
            return action

        if (
            profitable
            and _opponent_offer_is_stationary(game, opp, my_value, offer_price)
            and _stationary_offer_clears_min_margin(role, my_value, offer_price, bounds, complete_information)
        ):
            action = {"decision": "AcceptOffer"}
            check_fields(valid_actions, action)
            history.record_own_action(game, action)
            return action

        if profitable and _offer_clears_safety_margin(role, my_value, offer_price):
            action = {"decision": "AcceptOffer"}
            check_fields(valid_actions, action)
            history.record_own_action(game, action)
            return action

        t = time_fraction(state["round"], state.get("max_rounds"), unlimited_tau=UNLIMITED_TAU)
        anchor = bounds.hi if role == "seller" else bounds.lo
        target = concession_target(anchor, my_value, t, CONCESSION_BETA)

        meets_target = offer_price >= target if role == "seller" else offer_price <= target
        if meets_target and profitable:
            action = {"decision": "AcceptOffer"}
            check_fields(valid_actions, action)
            history.record_own_action(game, action)
            return action

        counter = target
        counter = max(counter, my_value) if role == "seller" else min(counter, my_value)

        # Against a stalled opponent, give up only part of the gap each round.
        if prior_price is not None:
            conceded_further = counter < prior_price if role == "seller" else counter > prior_price
            if conceded_further and _opponent_offer_is_stationary(game, opp, my_value, offer_price):
                counter = (
                    prior_price - OFFER_CONCESSION_DAMPING_FRAC * (prior_price - counter) if role == "seller"
                    else prior_price + OFFER_CONCESSION_DAMPING_FRAC * (counter - prior_price)
                )

        message = None
        if messages_allowed:
            style = _style_bandit(game).select_arm()
            message = compose_message(
                family="negotiation",
                situation=game["prompt"],
                stance={
                    "role": role,
                    "my_reservation_value": my_value,
                    "their_last_price": offer_price,
                    "my_counter_price": counter,
                    "deadline_pressure": deadline_pressure(rounds_remaining),
                    "fallback_message": "That price doesn't reflect the real value here — here's my counter.",
                },
                profile=opponent_profile,
                style=style,
            )
            _LAST_STYLE_ARM[game_id] = style
        action = {"decision": "RejectOffer", "product_price": counter}
        if message:
            action["message"] = message
        _LAST_SENT_PRICE[game_id] = counter
        check_fields(valid_actions, action)
        history.record_own_action(game, action)
        return action

    if valid_actions["type"] == "offer":
        if state.get("max_rounds") == 1:
            t = ONE_SHOT_ULTIMATUM_COMPLETE_INFO_T if complete_information else ONE_SHOT_ULTIMATUM_T
        else:
            t = time_fraction(state["round"], state.get("max_rounds"), unlimited_tau=UNLIMITED_TAU)
        anchor = bounds.hi if role == "seller" else bounds.lo
        price = concession_target(anchor, my_value, t, CONCESSION_BETA)
        price = max(price, my_value) if role == "seller" else min(price, my_value)

        message = None
        if messages_allowed:
            style = _style_bandit(game).select_arm()
            message = compose_message(
                family="negotiation",
                situation=game["prompt"],
                stance={
                    "role": role,
                    "my_reservation_value": my_value,
                    "opening_price": price,
                    "fallback_message": "Here's my opening price — grounded in the real value of this deal.",
                },
                profile=opponent_profile,
                style=style,
            )
            _LAST_STYLE_ARM[game_id] = style
        action = {"product_price": price}
        if message:
            action["message"] = message
        _LAST_SENT_PRICE[game_id] = price
        check_fields(valid_actions, action)
        history.record_own_action(game, action)
        return action

    return {}
