"""Routes each game to its family's strategy under a hard per-turn deadline.

If a strategy overruns, a safe fallback move is submitted so the game is never
forfeited on a timeout.
"""

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

TURN_DEADLINE_SECONDS = 45.0

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=32, thread_name_prefix="strategy-deadline")


# Fallbacks favor a legal, low-risk move: an even split, a mild markup, and
# accepting when responding.
def _fallback_action(game: dict) -> dict:
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

    if action_type == "seller_message":
        return {"message": "This product is worth it."}
    if action_type == "seller_recommendation":
        return {"decision": "yes"}
    return {"decision": "no"}


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
