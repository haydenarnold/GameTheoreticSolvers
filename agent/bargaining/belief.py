"""Online Bayesian belief over the bargaining opponent's hidden discount factor.

The belief is a weight vector over a grid of candidate factors, reweighted each
time the opponent proposes a split.
"""

from __future__ import annotations

import math

DEFAULT_GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99)
_KERNEL_SIGMA = 0.08
SHRINKAGE_PSEUDO_COUNT = 3.0


class DeltaBelief:
    def __init__(self, grid: tuple[float, ...] = DEFAULT_GRID):
        self.grid = grid
        self.weights = [1.0 / len(grid)] * len(grid)
        self.n_observations = 0

    def update(self, observed_take_fraction: float, predicted_fraction_fn) -> None:
        prev_weights = list(self.weights)

        for i, delta_hyp in enumerate(self.grid):
            predicted = predicted_fraction_fn(delta_hyp)
            err = predicted - observed_take_fraction
            likelihood = math.exp(-(err * err) / (2 * _KERNEL_SIGMA * _KERNEL_SIGMA))
            self.weights[i] *= likelihood
        total = sum(self.weights)
        if total > 0:
            raw_weights = [w / total for w in self.weights]
        else:
            raw_weights = [1.0 / len(self.grid)] * len(self.grid)

        self.n_observations += 1
        # Shrink toward the pre-update weights so one outlier offer can't swing the
        # posterior; the pull fades as observations accumulate.
        mix = SHRINKAGE_PSEUDO_COUNT / (SHRINKAGE_PSEUDO_COUNT + self.n_observations)
        self.weights = [(1 - mix) * w + mix * pw for w, pw in zip(raw_weights, prev_weights)]

    def mean(self) -> float:
        return sum(w * d for w, d in zip(self.weights, self.grid))


_BELIEFS: dict[str, DeltaBelief] = {}


def get_belief(game_id: str) -> DeltaBelief:
    if game_id not in _BELIEFS:
        _BELIEFS[game_id] = DeltaBelief()
    return _BELIEFS[game_id]


def forget(game_id: str) -> None:
    _BELIEFS.pop(game_id, None)
