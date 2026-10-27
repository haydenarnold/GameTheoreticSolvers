"""Thompson-sampling bandit over the five moral-foundation framings.

Each arm keeps a Beta posterior over its success rate. When the opponent's
identity is disclosed, the posterior is stored on disk and reloaded the next
time we meet them.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import random
import threading

logger = logging.getLogger("agent.rhetoric")

_LOCK = threading.Lock()

ARMS = ("care", "fairness", "loyalty", "authority", "purity")


class StyleBandit:

    ARMS = ARMS

    def __init__(self, prior_strength: float = 1.0, prior_means: dict[str, float] | None = None):
        if prior_means:
            total = 2.0 * prior_strength
            self.alpha = {arm: total * prior_means.get(arm, 0.5) for arm in self.ARMS}
            self.beta = {arm: total * (1.0 - prior_means.get(arm, 0.5)) for arm in self.ARMS}
        else:
            self.alpha = {arm: prior_strength for arm in self.ARMS}
            self.beta = {arm: prior_strength for arm in self.ARMS}

    def select_arm(self) -> str:
        with _LOCK:
            samples = {arm: random.betavariate(self.alpha[arm], self.beta[arm]) for arm in self.ARMS}
            return max(samples, key=samples.get)

    def update(self, arm: str, success: bool) -> None:
        with _LOCK:
            if arm not in self.alpha:
                return
            if success:
                self.alpha[arm] += 1.0
            else:
                self.beta[arm] += 1.0

    def to_dict(self) -> dict:
        return {"alpha": dict(self.alpha), "beta": dict(self.beta)}

    @classmethod
    def from_dict(cls, data: dict) -> "StyleBandit":
        bandit = cls()
        bandit.alpha = {arm: float(data["alpha"][arm]) for arm in cls.ARMS}
        bandit.beta = {arm: float(data["beta"][arm]) for arm in cls.ARMS}
        return bandit


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STORE_PATH = os.path.join(_REPO_ROOT, "data", "opponent_style_bandits.json")

_CACHE: dict[str, StyleBandit] = {}


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
        logger.warning("Opponent style-bandit store is corrupt; ignoring it for this read.")
        return {}


def _load_bandit_from_store(opponent_display_name: str) -> StyleBandit | None:
    store = _load_store()
    entry = store.get(opponent_display_name)
    if not entry:
        return None
    try:
        return StyleBandit.from_dict(entry)
    except (KeyError, TypeError, ValueError):
        logger.warning(
            "Corrupt opponent style-bandit entry for %r; starting fresh.", opponent_display_name
        )
        return None


def _save_bandit_to_store(opponent_display_name: str, bandit: StyleBandit) -> None:
    try:
        os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
        # Read-modify-write under an exclusive lock: concurrent games may persist
        # different opponents into the same file.
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
                store[opponent_display_name] = bandit.to_dict()
                f.seek(0)
                f.truncate()
                json.dump(store, f)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        logger.warning(
            "Failed to persist opponent style bandit for %r; continuing in-memory only.",
            opponent_display_name, exc_info=True,
        )


# Stronger prior for named opponents
PERSISTENT_PRIOR_STRENGTH = 4.0


def get_style_bandit(
    game_id: str, opponent_display_name: str | None, prior_means: dict[str, float] | None = None,
) -> StyleBandit:
    key = f"opponent:{opponent_display_name}" if opponent_display_name else f"game:{game_id}"
    with _LOCK:
        if key not in _CACHE:
            loaded = _load_bandit_from_store(opponent_display_name) if opponent_display_name else None
            if loaded is not None:
                _CACHE[key] = loaded
            elif opponent_display_name:
                _CACHE[key] = StyleBandit(prior_strength=PERSISTENT_PRIOR_STRENGTH, prior_means=prior_means)
            else:
                _CACHE[key] = StyleBandit(prior_means=prior_means)
        return _CACHE[key]


def persist_style_bandit(game_id: str, opponent_display_name: str | None) -> None:
    if not opponent_display_name:
        return
    key = f"opponent:{opponent_display_name}"
    with _LOCK:
        bandit = _CACHE.get(key)
    if bandit is not None:
        _save_bandit_to_store(opponent_display_name, bandit)
