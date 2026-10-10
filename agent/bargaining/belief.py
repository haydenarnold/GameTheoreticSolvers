"""Bayesian belief over an opponent's hidden discount factor.

Solving the true incomplete-information bargaining equilibrium (a
Rubinstein-Stahl screening game) is analytically hard. This is a tractable
certainty-equivalent approximation: keep a discretized posterior over the
opponent's delta, update it each time they act as proposer by comparing
their observed take (as a fraction of the pie) to what each hypothesis
would have predicted, and plug the posterior mean into the complete-
information formulas. It converges toward the opponent's true patience as
evidence accumulates, which is the whole point — a static guess never does.
"""

from __future__ import annotations

import math

DEFAULT_GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99)
_KERNEL_SIGMA = 0.08
# CONFIRMED live (2026-08-22): even after fixing the damped-vs-undamped
# solver mismatch (see strategy.py's docstring), a SINGLE observed offer
# still swung the posterior mean from the grid's uniform average (0.819) to
# 0.92 -- because a 75% self-ask genuinely doesn't fit moderate-delta
# hypotheses well on this grid even under the correct math, so one data
# point was enough to make the likelihood-weighted posterior nearly ignore
# every hypothesis below ~0.9. This showed up on BOTH sides live: an accept
# threshold that collapsed to ~26% by round 3, and (a separate game) an
# OFFER that collapsed from a fair 50.9%-for-us opening to giving the
# opponent 69.2% after just their single aggressive counter. Blending each
# update's raw Bayesian result back toward the uniform prior, at a rate
# that decays as observations accumulate (`SHRINKAGE_PSEUDO_COUNT`), keeps
# a lone data point from dominating while still letting genuinely
# consistent, repeated evidence move the belief substantially -- the same
# principle already applied to persuasion's BuyerTrustModel.
SHRINKAGE_PSEUDO_COUNT = 3.0


class DeltaBelief:
    def __init__(self, grid: tuple[float, ...] = DEFAULT_GRID):
        self.grid = grid
        # CONFIRMED live (2026-08-22): unnormalized weights here (summing to
        # len(grid), not 1.0) made `mean()` return the raw SUM of the grid
        # (8.19) instead of its average (0.819) on every fresh belief — i.e.
        # on the very first decision of nearly every incomplete-information
        # bargaining game, before any `update()` call has run. A "discount
        # factor" of 8.19 blew the backward-induction formula out to roughly
        # -$23.5M, which then got clamped to $0. `update()` already
        # renormalizes after the first call, so this only bit the opening
        # move — but the opening move is exactly where it matters most.
        self.weights = [1.0 / len(grid)] * len(grid)
        self.n_observations = 0

    def update(self, observed_take_fraction: float, predicted_fraction_fn) -> None:
        # CONFIRMED (2026-08-25), via scratchpad/delta_belief_fix_candidates.py:
        # shrinking every update back toward a fixed UNIFORM target forever
        # (not just the first one) turned out to fight genuine convergence,
        # not just resist single-sample noise. Against a rational opponent
        # whose true delta sits away from the grid's center, mean|error|
        # after 3 real, consistent observations was still 0.053 with the old
        # uniform target -- and against a noisy-but-honest opponent, the bias
        # at an extreme true delta (0.99) never fully closed even after 8
        # observations (-0.047 remaining). The uniform target re-dilutes
        # accumulated evidence on every single call, not just the first.
        #
        # Trust-region fix: shrink each update toward the belief's OWN
        # pre-update weights instead of a fixed uniform distribution -- this
        # damps the STEP SIZE of new evidence relative to where the belief
        # already stands, rather than perpetually pulling it back to an
        # arbitrary center. On a belief's first-ever update this is
        # mathematically IDENTICAL to the old behavior (prev weights are
        # still the initial uniform ones), so the exact 2026-08-22 incident
        # this shrinkage was built to prevent (a single offer swinging the
        # mean to 0.92) is verified unchanged. From the second update
        # onward it diverges favorably: mean|error| after 3 observations
        # dropped to 0.023 (rational opponent) in the same test, and the
        # persistent extreme-delta bias against a noisy opponent closed to
        # ~0.000 by 8 observations. Cost: modestly higher variance early on
        # (sd 0.022 vs 0.015 at n=3), which converges back down by n=8 --
        # judged an acceptable price for eliminating a bias that was
        # otherwise persistent for the entire game.
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
