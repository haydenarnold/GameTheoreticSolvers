"""Per-game move history, preferring the server's log over a local one.

Some games omit history from the state, so offers are mirrored locally and
deduplicated by round.
"""

from __future__ import annotations

from collections import defaultdict

_LOCAL_HISTORY: dict[str, list[dict]] = defaultdict(list)
_LAST_RECORDED_ROUND: dict[str, int] = {}


def get_history(game: dict) -> list[dict]:
    server_history = game["game_state"].get("history")
    if server_history is not None:
        return server_history
    return _LOCAL_HISTORY[game["game_id"]]


def record_opponent_offer(game: dict, offer: dict) -> None:
    game_id = game["game_id"]
    round_no = offer.get("round")
    if round_no is not None and _LAST_RECORDED_ROUND.get(game_id) == round_no:
        return
    _LOCAL_HISTORY[game_id].append({"speaker": "opponent", "round": round_no, "offer": offer})
    if round_no is not None:
        _LAST_RECORDED_ROUND[game_id] = round_no


def record_own_action(game: dict, action: dict) -> None:
    game_id = game["game_id"]
    round_no = game["game_state"].get("round")
    _LOCAL_HISTORY[game_id].append({"speaker": "me", "round": round_no, "action": action})


def forget(game_id: str) -> None:
    _LOCAL_HISTORY.pop(game_id, None)
    _LAST_RECORDED_ROUND.pop(game_id, None)


def is_opponent_entry(entry: dict, opponent: str) -> bool:
    if entry.get("proposer") == opponent or entry.get("speaker") == "opponent":
        return True
    offer = entry.get("offer")
    return isinstance(offer, dict) and offer.get("from_player") == opponent
