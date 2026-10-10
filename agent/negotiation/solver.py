"""Negotiation tactics: no inflation field exists in this family's state —
only the round cap creates urgency, and (per the docs) a rejection on the
final capped round takes no counteroffer, i.e. degenerates to walk-away.
Backward induction with no discounting is a knife-edge, parity-driven result
(whoever proposes on the literal last round can claim the whole surplus) —
correct in theory, but fragile against any opponent who isn't playing the
exact same equilibrium (most humans and baseline LLMs won't). So we use it
only as a directional signal (see `deadline_pressure`) and lean on a
standard time-dependent concession tactic (Faratin, Sierra & Jennings 1998)
for the actual offer, which degrades gracefully against imperfect play.
"""

from __future__ import annotations


def concession_target(anchor: float, reservation: float, t: float, beta: float) -> float:
    """`t` in [0, 1] is how far through the negotiation we are.
    `beta < 1` concedes slowly then fast near the end (Boulware);
    `beta > 1` concedes fast up front then holds (Conceder); `beta == 1` is
    linear. Moves from `anchor` toward `reservation` as t -> 1."""
    t = min(max(t, 0.0), 1.0)
    frac = t ** (1.0 / beta) if beta > 0 else 1.0
    return reservation + (1 - frac) * (anchor - reservation)


def time_fraction(round_no: int, max_rounds: int | None, unlimited_tau: float = 6.0) -> float:
    """`round_no` is 1-indexed (round 1 = the first move, zero rounds
    actually elapsed yet), so the finite branch counts *elapsed* rounds
    (`round_no - 1`), not the current round index itself.

    CONFIRMED live (2026-08-22): using `round_no / max_rounds` directly made
    a one-shot ultimatum (`max_rounds=1`) hit t=1.0 -- "fully time-elapsed,
    concede everything" -- on the very first and only move, before any
    negotiation happened at all. As the seller (anchor=160, reservation/my
    own value=80), that collapsed our opening offer straight to our own
    floor (80, zero markup), which the buyer instantly accepted, handing
    them the entire surplus. Off-by-one, not a one-shot-only edge case: the
    same skew shifts every round's t up by `1/max_rounds`, just harmlessly
    small until max_rounds is tiny."""
    if max_rounds:
        return (round_no - 1) / max_rounds
    import math
    return 1 - math.exp(-round_no / unlimited_tau)


def deadline_pressure(rounds_remaining: int | None) -> float:
    """0 = no pressure (unlimited horizon), 1 = this is the last possible
    round (whoever proposes now can, in the pure no-discounting SPE, claim
    the whole surplus). A soft signal blended into the anchor, not applied
    literally."""
    if rounds_remaining is None:
        return 0.0
    if rounds_remaining <= 0:
        return 1.0
    return 1.0 / (1 + rounds_remaining)


class ValueBounds:
    """Tracks a [lo, hi] interval on the opponent's hidden reservation value,
    tightened only by hard revealed-preference evidence: a rejection at
    price p means a seller's true minimum is > p, or a buyer's true maximum
    is < p. Counteroffers are deliberately NOT used to tighten bounds —
    negotiators routinely anchor away from their true value, so a
    counteroffer is a biased/noisy signal, not a bound."""

    def __init__(self, lo: float, hi: float):
        self.lo = lo
        self.hi = hi

    def tighten_from_rejection(self, opponent_role: str, rejected_price: float) -> None:
        if opponent_role == "seller":
            self.lo = max(self.lo, rejected_price)
        else:
            self.hi = min(self.hi, rejected_price)

    def midpoint(self) -> float:
        return (self.lo + self.hi) / 2
