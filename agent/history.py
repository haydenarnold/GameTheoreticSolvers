"""Per-game move history.

`llm_agent.py`'s prompt-building comment implies `game_state` may carry a
server-provided `history` of past rounds; the SDK README's field table
doesn't confirm this explicitly. We prefer the server's copy when present
and fall back to a locally-accumulated log otherwise, so opponent-modeling
code has a `history` to read either way.

CONFIRMED live (2026-08-22): the server's own history entries never carry a
"speaker" key -- pulled a real game's `game_state` directly from the API and
its history is `{"round", "proposer", "offer", "decision", ...}`, with
"proposer" naming the actual player (e.g. "player_1"). "speaker":
"opponent"/"me" is a shape only OUR local fallback log uses (see
`record_opponent_offer`/`record_own_action` below). Since `get_history`
prefers the server's copy whenever present -- true in every live game
checked -- any caller filtering on `h.get("speaker") == "opponent"` was
silently getting an empty result in production. This bit hard: bargaining's
`_opponent_offer_is_stationary` (built specifically to stop grinding out a
holdout against a proposer who never moves) never fired on a real game,
including one where the opponent proposed an identical 60/40 split for 88+
straight rounds and we finally accepted at round 91 with our own delta
(0.8) having compounded the realized payoff down to ~$0. `is_opponent_entry`
below recognizes both shapes so this actually works against real games.

CONFIRMED live (2026-08-23): negotiation's real history entries use a
THIRD, different shape from bargaining's -- pulled a real negotiation
game_state and its entries are `{"round", "offer": {"price", "from_player",
...}, "decision", "decided_by", ...}`. There is no top-level "proposer" key
at all; the proposer identity is nested one level down, inside "offer", and
named "from_player" instead of "proposer". The bargaining-oriented fix above
didn't cover this -- `is_opponent_entry` was still silently returning False
for every entry in a real negotiation game, so negotiation's opponent-style
profiling was still dead even after the first fix. This was caught by
reviewing a real game (a seller offered an identical $118.80 for 3 straight
rounds and then walked away at round 12 rather than continuing, after we
kept pushing past the point they'd already proven immovable at) while
building the negotiation-side stationary-offer detection that needs
`is_opponent_entry` to actually work against this shape too.
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
    """True if a history entry (any of the three shapes seen live) was the
    opponent's move. Bargaining names the proposer at the top level
    (`entry["proposer"]`); negotiation nests it inside "offer" under a
    different key (`entry["offer"]["from_player"]`); the local fallback log
    uses the "opponent"/"me" `speaker` tag instead. See module docstring for
    why all three must be checked."""
    if entry.get("proposer") == opponent or entry.get("speaker") == "opponent":
        return True
    offer = entry.get("offer")
    return isinstance(offer, dict) and offer.get("from_player") == opponent
