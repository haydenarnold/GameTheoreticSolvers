"""Rubinstein alternating-offer bargaining, with per-player discount factors.

`delta_x` is a per-round *discount multiplier* (0.9 = 10% inflation/round),
so a payoff realized one round later is worth `delta` times as much to that
player. Both the finite-horizon (backward induction) and infinite/unknown-
horizon (closed form) subgame-perfect equilibria are standard textbook
results (Rubinstein 1982, Osborne & Rubinstein 1990 ch. 3); this module is a
direct transcription of those formulas, generalized to asymmetric deltas.
"""

from __future__ import annotations

from agent.schema import opponent_name


def infinite_horizon_share(money: float, delta_me: float, delta_opp: float) -> float:
    """My equilibrium take as proposer, stationary SPE, no deadline."""
    denom = 1 - delta_me * delta_opp
    if denom <= 0:
        return money  # degenerate (delta product >= 1); take everything
    return money * (1 - delta_opp) / denom


def finite_horizon_take(
    money: float,
    current_proposer: str,
    delta_of,
    rounds_remaining: int,
) -> float:
    """The current round's proposer's equilibrium take, via backward
    induction from the deadline.

    `rounds_remaining` counts additional rounds possible after the current
    one (0 means the current round IS the deadline: reject -> no-deal -> 0
    for both, so the proposer takes the whole pie).
    """
    if rounds_remaining < 0:
        raise ValueError("rounds_remaining must be >= 0")

    proposer_seq = []
    p = current_proposer
    for _ in range(rounds_remaining + 1):
        proposer_seq.append(p)
        p = opponent_name(p)

    take_at = {rounds_remaining: money}
    for k in range(rounds_remaining - 1, -1, -1):
        proposer_k = proposer_seq[k]
        responder_k = opponent_name(proposer_k)
        take_at[k] = money - delta_of(responder_k) * take_at[k + 1]
    return take_at[0]


def damped_finite_horizon_take(
    money: float,
    current_proposer: str,
    delta_of,
    rounds_remaining: int,
) -> float:
    """`finite_horizon_take`, damped toward the converged infinite-horizon
    value.

    CONFIRMED (2026-08-22): the pure finite backward induction oscillates
    wildly by round PARITY even with real discounting, because it assumes
    the deadline proposer takes literally everything (a knife-edge that
    flips who's favored every single round). At 11 rounds remaining with
    symmetric delta=0.9, the proposer gets only 37.8% -- when the true
    converged (infinite-horizon) value is 52.6% -- purely because 11 is odd,
    not because of anything strategically meaningful. This blends the two:
    weight on the exact finite value decays geometrically as
    `(delta_me*delta_opp)^rounds_remaining`, so it dominates only near a
    real deadline (where "who proposes last" is actually informative) and
    fades out for anything more than a handful of rounds away, where hanging
    the entire offer on that knife-edge is not a realistic assumption about
    how the game will actually be played.
    """
    other = opponent_name(current_proposer)
    finite = finite_horizon_take(money, current_proposer, delta_of, rounds_remaining)
    infinite = infinite_horizon_share(money, delta_of(current_proposer), delta_of(other))
    alpha = (delta_of(current_proposer) * delta_of(other)) ** rounds_remaining
    return alpha * finite + (1 - alpha) * infinite


def reject_continuation_value(
    money: float,
    me: str,
    delta_of,
    rounds_remaining: int,
) -> float:
    """My payoff if I reject the current offer: I become proposer next
    round, discounted once by my own delta. `rounds_remaining` is the
    remaining-rounds count for the CURRENT (offer) round, i.e. the same
    value passed to `finite_horizon_take` for this decision."""
    if rounds_remaining <= 0:
        return 0.0  # this decision is at (or past) the deadline
    my_take_next_round = finite_horizon_take(money, me, delta_of, rounds_remaining - 1)
    return delta_of(me) * my_take_next_round
