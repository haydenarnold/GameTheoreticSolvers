"""Faratin time-dependent concession and opponent value bounds for negotiation.

The target price moves from our anchor to our reservation value as time
elapses, and the opponent's value interval tightens only on hard rejections.
"""

from __future__ import annotations


def concession_target(anchor: float, reservation: float, t: float, beta: float) -> float:
    t = min(max(t, 0.0), 1.0)
    frac = t ** (1.0 / beta) if beta > 0 else 1.0
    return reservation + (1 - frac) * (anchor - reservation)


def time_fraction(round_no: int, max_rounds: int | None, unlimited_tau: float = 6.0) -> float:
    if max_rounds:
        # Elapsed rounds, not the round index, so a one-shot ultimatum isn't
        # treated as fully time-elapsed on its only move.
        return (round_no - 1) / max_rounds
    import math
    return 1 - math.exp(-round_no / unlimited_tau)


def deadline_pressure(rounds_remaining: int | None) -> float:
    if rounds_remaining is None:
        return 0.0
    if rounds_remaining <= 0:
        return 1.0
    return 1.0 / (1 + rounds_remaining)


class ValueBounds:

    def __init__(self, lo: float, hi: float):
        self.lo = lo
        self.hi = hi

    # A rejected price is a verified bound; counteroffers are biased signals and
    # are deliberately not used here.
    def tighten_from_rejection(self, opponent_role: str, rejected_price: float) -> None:
        if opponent_role == "seller":
            self.lo = max(self.lo, rejected_price)
        else:
            self.hi = min(self.hi, rejected_price)

    def midpoint(self) -> float:
        return (self.lo + self.hi) / 2
