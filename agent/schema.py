"""Defensive helpers for reading/writing the GLEE wire format.

CONFIRMED live (2026-08-22, server error message): a bargaining offer
action's gain keys are ALWAYS literally `alice_gain`/`bob_gain`, regardless
of whether `your_player`/`current_player` report `"alice"`/`"bob"` or
`"player_1"`/`"player_2"` for that same game. The two naming conventions
coexist: player-identity fields may use either form, but the offer action's
keys never do. `read_gain` already tolerated both forms when reading an
incoming offer, which is why decisions (accept/reject) never failed; the
bug was that `build_bargaining_offer` used `me` as given, unranslated, to
build the OUTGOING offer's keys — sending `player_2_gain` when the server
required `bob_gain`. That got every bargaining game where `your_player` was
reported in `player_1`/`player_2` form rejected 5 times running and
force-closed at the 5th percentile. `_canonical` below is the fix: always
normalize to alice/bob before building an offer.
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
    """Best-effort sanity check against the server's self-documented field
    list; logs rather than raises so a docs/format drift doesn't crash the
    agent mid-game."""
    fields = valid_actions.get("fields")
    if not isinstance(fields, dict):
        return
    unexpected = set(action) - set(fields) - {"message"}
    if unexpected:
        logger.warning("Action has fields not in valid_actions['fields']: %s", unexpected)
