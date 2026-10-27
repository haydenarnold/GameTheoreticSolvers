"""Player-name and offer-format helpers shared by the bargaining code.

Games name the two players either alice/bob or player_1/player_2; these
helpers normalize between the two so solvers can ignore the difference.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("agent.schema")

_PAIRS = {
    "alice": "bob",
    "bob": "alice",
    "player_1": "player_2",
    "player_2": "player_1",
}

_TO_CANONICAL = {
    "alice": "alice",
    "bob": "bob",
    "player_1": "alice",
    "player_2": "bob",
}


def opponent_name(me: str) -> str:
    if me not in _PAIRS:
        raise ValueError(f"Unknown player name {me!r}")
    return _PAIRS[me]


def gain_key(player: str) -> str:
    return f"{player}_gain"


def read_gain(offer: dict, player: str) -> float:
    key = gain_key(player)
    if key in offer:
        return offer[key]
    alias = _PAIRS.get(player)
    aliased_alias = {"alice": "player_1", "bob": "player_2",
                      "player_1": "alice", "player_2": "bob"}.get(player)
    for candidate in (aliased_alias,):
        if candidate and gain_key(candidate) in offer:
            return offer[gain_key(candidate)]
    raise KeyError(f"Could not find gain for player {player!r} in offer {offer!r}")


def build_bargaining_offer(me: str, my_gain: float, money: float, message: str | None = None) -> dict:
    if me not in _TO_CANONICAL:
        raise ValueError(f"Unknown player name {me!r}")
    me_canonical = _TO_CANONICAL[me]
    opp_canonical = opponent_name(me_canonical)
    action = {gain_key(me_canonical): my_gain, gain_key(opp_canonical): money - my_gain}
    if message:
        action["message"] = message
    return action


def check_fields(valid_actions: dict, action: dict) -> None:
    fields = valid_actions.get("fields")
    if not isinstance(fields, dict):
        return
    unexpected = set(action) - set(fields) - {"message"}
    if unexpected:
        logger.warning("Action has fields not in valid_actions['fields']: %s", unexpected)
