"""Rubinstein alternating-offers solutions for the bargaining game.

Provides the infinite-horizon share, the backward-induction take for a known
deadline, and a damped blend of the two (Section 3.1 and Appendix A).
"""

from __future__ import annotations

from agent.schema import opponent_name


def infinite_horizon_share(money: float, delta_me: float, delta_opp: float) -> float:
    denom = 1 - delta_me * delta_opp
    if denom <= 0:
        return money
    return money * (1 - delta_opp) / denom


def finite_horizon_take(
    money: float,
    current_proposer: str,
    delta_of,
    rounds_remaining: int,
) -> float:
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
    other = opponent_name(current_proposer)
    finite = finite_horizon_take(money, current_proposer, delta_of, rounds_remaining)
    infinite = infinite_horizon_share(money, delta_of(current_proposer), delta_of(other))
    # Raw backward induction flips with round parity when deltas are near 1, so
    # the exact result only dominates close to the deadline.
    alpha = (delta_of(current_proposer) * delta_of(other)) ** rounds_remaining
    return alpha * finite + (1 - alpha) * infinite


def reject_continuation_value(
    money: float,
    me: str,
    delta_of,
    rounds_remaining: int,
) -> float:
    if rounds_remaining <= 0:
        return 0.0
    my_take_next_round = finite_horizon_take(money, me, delta_of, rounds_remaining - 1)
    return delta_of(me) * my_take_next_round
