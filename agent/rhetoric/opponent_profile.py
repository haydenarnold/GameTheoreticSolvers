"""Persistent cross-game "concession profile" per opponent, per family:
tracks how often a NAMED opponent has shown proven-unmovable/unresponsive
behavior across past games, so a strategy can reduce (never skip) its
in-game confirmation requirement when facing a name with a trusted track
record of exactly that behavior.

Mirrors `agent.rhetoric.bandit`'s persistence pattern deliberately (read
that module's docstring for the full rationale) -- disk-backed
(`data/opponent_concession_profile.json`), `fcntl`-locked for cross-process
safety, `threading.Lock` for cross-thread safety (the SDK's
`client.run(concurrency=N)` uses a real `ThreadPoolExecutor`, not asyncio --
two games sharing an opponent name really can run concurrently in the same
process), in-process cache to avoid re-hitting disk every round. File I/O
happens on the live, scored strategy path, so every read/write fails safe:
any I/O or parse error falls back to a fresh, untrusted profile rather than
raising out of a strategy call.

DERIVED (2026-08-25), from real data: a census of the existing
`opponent_style_bandits.json` persistent store (125 tracked names) found
only 3 with a single game's worth of evidence -- the other 122 show
accumulated trial counts far beyond what one game could produce (a single
persuasion game yields at most ~17 real feedback rounds; several tracked
names show 40-126+ accumulated trials, implying repeat encounters in the
double digits). Opponent names recur heavily in this competition's pool,
which is what makes this module worth building at all.

Scoped PER FAMILY deliberately, not globally: an opponent's behavior in one
game family (or role within a family) is not guaranteed to transfer to
another -- a bargaining hardliner might negotiate completely normally. Each
(name, family) pair gets an independent record.

CRITICAL DESIGN CONSTRAINT: this must NEVER replace in-game confirmation
(e.g. `_opponent_not_conceding` in agent/bargaining/strategy.py, or the
analogous persuasion signal-reliability checks) -- it exists ONLY to let a
caller REDUCE an existing confirmation-round requirement (e.g. halve
`HOPELESS_CONFIRMATION_ROUNDS`) once a name has a trusted prior record, never
to skip real in-game evidence-gathering entirely. `is_trusted_hardliner()`
requires at least `TRUST_THRESHOLD_GAMES` (2) independent prior games
showing the pattern before returning True -- deliberately not 1, since a
single prior game is too easy to misattribute to a specific matchup's
parameters (an unusually bad delta pairing, a genuinely-unreachable ZOPA)
rather than a real trait of the opponent. A caller must still run its own
full in-game confirmation logic even when `is_trusted_hardliner()` is True;
it should only use that signal to shorten how long that confirmation takes,
and should treat real in-game evidence of genuine concession as an
immediate override of any stale prior profile, exactly as the existing
same-game logic already does.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
from dataclasses import dataclass

logger = logging.getLogger("agent.rhetoric")

_LOCK = threading.Lock()

# See module docstring: 1 prior game is not enough to trust -- could be a
# one-off unfavorable matchup rather than a real trait of this opponent.
TRUST_THRESHOLD_GAMES = 2


@dataclass
class ConcessionProfile:
    """One (opponent name, family) pair's accumulated cross-game record.

    `games_observed` counts every past game this profile has evidence for
    (whether or not the pattern was confirmed); `games_confirmed_hopeless`
    counts how many of those showed proven-unmovable/unresponsive behavior
    (bargaining/negotiation: `_opponent_not_conceding`-confirmed
    non-concession that also crossed the caller's own floor/hopelessness
    bar; persuasion: an unresponsive buyer or an uninformatively-signaling
    seller, per that family's own definition -- this module doesn't judge
    what counts, only tallies what the caller reports)."""

    games_observed: int = 0
    games_confirmed_hopeless: int = 0

    def is_trusted_hardliner(self) -> bool:
        """True only once at least TRUST_THRESHOLD_GAMES prior games have
        ALL shown the pattern (strict equality, not merely a majority) --
        a single game showing genuine concession permanently disqualifies
        this name until confirmed-hopeless games catch back up to equal
        every observed game again. Deliberately conservative: this is a
        signal to REDUCE (e.g. halve), never eliminate, an existing in-game
        confirmation requirement -- see module docstring -- so a false
        negative here just costs a few extra in-game confirmation rounds
        next time, while a false positive risks under-confirming a name
        that isn't really a reliable hardliner."""
        return (
            self.games_observed >= TRUST_THRESHOLD_GAMES
            and self.games_confirmed_hopeless == self.games_observed
        )

    def to_dict(self) -> dict:
        return {"games_observed": self.games_observed, "games_confirmed_hopeless": self.games_confirmed_hopeless}

    @classmethod
    def from_dict(cls, data: dict) -> "ConcessionProfile":
        return cls(
            games_observed=int(data["games_observed"]),
            games_confirmed_hopeless=int(data["games_confirmed_hopeless"]),
        )


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STORE_PATH = os.path.join(_REPO_ROOT, "data", "opponent_concession_profile.json")

# Per-process cache, keyed by "<family>:<opponent_display_name>" -- avoids
# re-hitting disk on every single round of the same game.
_CACHE: dict[str, ConcessionProfile] = {}


def _store_key(opponent_display_name: str, family: str) -> str:
    return f"{family}:{opponent_display_name}"


def _load_store() -> dict:
    try:
        with open(_STORE_PATH, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                raw = f.read()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except (FileNotFoundError, OSError):
        return {}
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Opponent concession-profile store is corrupt; ignoring it for this read.")
        return {}


def _load_profile_from_store(opponent_display_name: str, family: str) -> ConcessionProfile | None:
    store = _load_store()
    entry = store.get(_store_key(opponent_display_name, family))
    if not entry:
        return None
    try:
        return ConcessionProfile.from_dict(entry)
    except (KeyError, TypeError, ValueError):
        logger.warning(
            "Corrupt opponent concession-profile entry for %r/%r; starting fresh.",
            opponent_display_name, family,
        )
        return None


def _save_profile_to_store(opponent_display_name: str, family: str, profile: ConcessionProfile) -> None:
    try:
        os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
        fd = os.open(_STORE_PATH, os.O_RDWR | os.O_CREAT, 0o644)
        with os.fdopen(fd, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # spans the full read-modify-write below
            try:
                f.seek(0)
                raw = f.read()
                try:
                    store = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    store = {}
                store[_store_key(opponent_display_name, family)] = profile.to_dict()
                f.seek(0)
                f.truncate()
                json.dump(store, f)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        # Never let a disk problem take down a live, scored strategy call --
        # the profile stays correct in-process for the rest of this game,
        # it just won't carry over to the opponent's next game.
        logger.warning(
            "Failed to persist opponent concession profile for %r/%r; continuing in-memory only.",
            opponent_display_name, family, exc_info=True,
        )


def get_concession_profile(opponent_display_name: str | None, family: str) -> ConcessionProfile:
    """The ConcessionProfile for this (opponent name, family) pair: loaded
    from disk once per process and cached thereafter, or a fresh
    (untrusted -- `is_trusted_hardliner()` is False until real evidence
    accumulates) profile for a never-seen name or hidden identity.

    Hidden-identity games (`opponent_display_name is None`) always get a
    fresh, never-persisted, in-memory-only profile -- there's no name to
    key a cross-game record on, matching `agent.rhetoric.bandit`'s
    identical handling."""
    if not opponent_display_name:
        return ConcessionProfile()
    key = _store_key(opponent_display_name, family)
    with _LOCK:
        if key not in _CACHE:
            loaded = _load_profile_from_store(opponent_display_name, family)
            _CACHE[key] = loaded if loaded is not None else ConcessionProfile()
        return _CACHE[key]


def record_concession_evidence(opponent_display_name: str | None, family: str, was_confirmed_hopeless: bool) -> None:
    """Call once, after a game against a named opponent concludes, with
    whether THIS game showed the proven-unmovable/unresponsive pattern (per
    the caller family's own in-game confirmation logic -- this module
    doesn't judge what counts, only tallies what's reported). No-op for
    hidden-identity games (nothing to key a persistent record on).

    This increments both `games_observed` and, if `was_confirmed_hopeless`,
    `games_confirmed_hopeless` -- there is deliberately no way to decrement
    or "unconfirm" a past game once recorded; a later game showing genuine
    concession from the same name should be recorded as its own
    `was_confirmed_hopeless=False` observation (lowering the fraction, not
    erasing history), consistent with `is_trusted_hardliner()` requiring
    ALL of the last `TRUST_THRESHOLD_GAMES` observations to show the
    pattern, not just a majority historically."""
    if not opponent_display_name:
        return
    key = _store_key(opponent_display_name, family)
    with _LOCK:
        profile = _CACHE.get(key)
        if profile is None:
            loaded = _load_profile_from_store(opponent_display_name, family)
            profile = loaded if loaded is not None else ConcessionProfile()
            _CACHE[key] = profile
        profile.games_observed += 1
        if was_confirmed_hopeless:
            profile.games_confirmed_hopeless += 1
    _save_profile_to_store(opponent_display_name, family, profile)
