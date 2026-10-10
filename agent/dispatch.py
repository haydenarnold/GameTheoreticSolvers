from __future__ import annotations

import concurrent.futures
import logging
import time

from agent.bargaining.strategy import bargaining_strategy
from agent.negotiation.strategy import negotiation_strategy
from agent.persuasion.strategy import persuasion_strategy

logger = logging.getLogger("agent.dispatch")

STRATEGIES = {
    "bargaining": bargaining_strategy,
    "negotiation": negotiation_strategy,
    "persuasion": persuasion_strategy,
}

# CONFIRMED (2026-08-25), from the competition's own docs: "120 seconds per
# turn; miss it and the game ends as no-deal" -- and separately, "Abandoning a
# game (turn timeout or five invalid moves) scores it at the 5th percentile --
# the bottom of the scale." A missed turn isn't scored like an ordinary bad
# outcome; it's scored at the floor. Three consecutive own-turn timeouts also
# trigger a 30-minute matchmaking cooldown ("agent_cooldown"), compounding the
# rating loss with a throughput loss at a moment throughput is scarce.
#
# Every real strategy call finishes in low single digits of seconds except the
# LLM-backed message-composition/signal-classification path, which is already
# bounded by its own per-call `timeout=` passed to litellm (15s in
# agent/persuasion/signal.py, 20s in agent/rhetoric/compose.py). But a
# per-call timeout only protects against the provider responding slowly --
# not against a DNS stall, a hung connection before litellm's own timeout
# logic engages, or contention on the SDK's shared thread pool. This is a
# second, independent backstop, at the one seam that wraps every family: if
# the real strategy call hasn't returned within TURN_DEADLINE_SECONDS, submit
# a safe, always-schema-valid fallback move instead of risking the full 120s
# forfeit. 45s is comfortably clear of the ~20s worst-case real latency (wide
# margin for load/queueing) and far short of 120s, leaving room for the
# fallback's own negligible cost plus the SDK's move-submission round trip.
TURN_DEADLINE_SECONDS = 45.0

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=32, thread_name_prefix="strategy-deadline")


def _fallback_action(game: dict) -> dict:
    """A safe, always-legal move for when the real strategy didn't return in
    time. Deliberately unambitious -- it exists only to avoid the forfeit
    floor (5th percentile, per the docs), not to play well. Accepting
    whatever's on the table (or, on an opening move, proposing a plain even
    split / role-scaled anchor) reliably lands as an ordinary scored outcome
    instead of an abandonment, which is the entire point: almost any real
    outcome beats the forfeit floor by a wide margin."""
    valid_actions = game.get("valid_actions") or {}
    action_type = valid_actions.get("type")
    state = game.get("game_state") or {}
    family = game.get("game_family")

    if family == "bargaining":
        if action_type == "offer":
            money = state.get("money_to_divide", 0) or 0
            return {"alice_gain": money / 2, "bob_gain": money / 2}
        return {"decision": "accept"}

    if family == "negotiation":
        if action_type == "offer":
            me = state.get("current_player")
            role = state.get(f"{me}_role")
            my_value = state.get(f"{me}_value", 0) or 0
            price = my_value * (1.1 if role == "seller" else 0.9)
            return {"product_price": price}
        return {"decision": "AcceptOffer"}

    # persuasion
    if action_type == "seller_message":
        return {"message": "This product is worth it."}
    if action_type == "seller_recommendation":
        return {"decision": "yes"}
    return {"decision": "no"}  # buyer_decision fallback: decline rather than risk an unknown-EV blind buy


def strategy(game: dict) -> dict:
    handler = STRATEGIES.get(game["game_family"])
    if handler is None:
        raise ValueError(f"Unknown game family: {game['game_family']}")

    start = time.monotonic()
    future = _EXECUTOR.submit(handler, game)
    try:
        action = future.result(timeout=TURN_DEADLINE_SECONDS)
    except concurrent.futures.TimeoutError:
        elapsed = time.monotonic() - start
        logger.error(
            "Strategy call exceeded %.0fs deadline (%.1fs elapsed) for game %s (%s); submitting a safe fallback move.",
            TURN_DEADLINE_SECONDS, elapsed, game.get("game_id"), game.get("game_family"),
        )
        return _fallback_action(game)

    elapsed = time.monotonic() - start
    if elapsed > TURN_DEADLINE_SECONDS / 3:
        logger.warning("Slow strategy call: %.1fs for game %s (%s)", elapsed, game.get("game_id"), game.get("game_family"))
    else:
        logger.info("Strategy call took %.2fs for game %s (%s)", elapsed, game.get("game_id"), game.get("game_family"))
    return action
