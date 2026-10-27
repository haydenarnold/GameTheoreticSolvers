"""Persistent record of opponents who never concede, keyed by display name.

After enough games where an opponent was confirmed hopeless, strategies can
stop spending effort on them. Stored on disk so it carries across runs.
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

TRUST_THRESHOLD_GAMES = 2


@dataclass
class ConcessionProfile:

    games_observed: int = 0
    games_confirmed_hopeless: int = 0

    # Requires every observed game to be hopeless.
    def is_trusted_hardliner(self) -> bool:
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
            fcntl.flock(f, fcntl.LOCK_EX)
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
        logger.warning(
            "Failed to persist opponent concession profile for %r/%r; continuing in-memory only.",
            opponent_display_name, family, exc_info=True,
        )


def get_concession_profile(opponent_display_name: str | None, family: str) -> ConcessionProfile:
    if not opponent_display_name:
        return ConcessionProfile()
    key = _store_key(opponent_display_name, family)
    with _LOCK:
        if key not in _CACHE:
            loaded = _load_profile_from_store(opponent_display_name, family)
            _CACHE[key] = loaded if loaded is not None else ConcessionProfile()
        return _CACHE[key]


def record_concession_evidence(opponent_display_name: str | None, family: str, was_confirmed_hopeless: bool) -> None:
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
