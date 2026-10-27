"""Kamenica-Gentzkow disclosure and buyer-trust estimation for persuasion.

The seller side computes the bluff rate that leaves the buyer indifferent and
adapts it to observed buy rates; the buyer side estimates P(high quality)
from its own purchase history.
"""

from __future__ import annotations


def optimal_bluff_prob(p: float, v: float, u: float, price: float) -> float:
    if price <= u:
        return 1.0
    if price >= v:
        return 0.0
    denom = (1 - p) * (price - u)
    if denom <= 0:
        return 0.0
    q = (p * (v - price)) / denom
    return min(max(q, 0.0), 1.0)


def pooled_value_given_signal(p: float, v: float, u: float, q: float) -> float:
    denom = p + (1 - p) * q
    if denom <= 0:
        return 0.0
    return (p * v + (1 - p) * q * u) / denom


class SellerDisclosureAdapter:

    def __init__(
        self, q0: float, step: float = 0.05, min_observations: int = 3,
        window: int = 5, consecutive_decline_threshold: int = 3,
    ):
        self.q = min(max(q0, 0.0), 1.0)
        self.step = step
        self.min_observations = min_observations
        self.window = window
        self.consecutive_decline_threshold = consecutive_decline_threshold
        self.recent: list[bool] = []
        self.consecutive_declines = 0

    def record(self, buyer_bought: bool) -> None:
        self.recent.append(buyer_bought)
        if len(self.recent) > self.window:
            self.recent.pop(0)
        self.consecutive_declines = 0 if buyer_bought else self.consecutive_declines + 1

    def adapted_q(self) -> float:
        # A string of refusals means the buyer has caught a bluff; honesty persists
        # across rounds, so stop bluffing entirely rather than nudging q down.
        if self.consecutive_declines >= self.consecutive_decline_threshold:
            self.q = 0.0
        elif len(self.recent) >= self.min_observations:
            accept_rate = sum(self.recent) / len(self.recent)
            if accept_rate > 0.8:
                self.q = min(self.q + self.step, 1.0)
            elif accept_rate < 0.4:
                self.q = max(self.q - self.step, 0.0)
        return self.q


class BuyerTrustModel:

    def __init__(self, prior_p: float, min_observations: int = 2, prior_pseudo_count: float = 20.0):
        self.prior_p = prior_p
        self.min_observations = min_observations
        self.prior_pseudo_count = prior_pseudo_count
        self.buys = 0
        self.highs = 0

    def record(self, quality_was_high: bool) -> None:
        self.buys += 1
        self.highs += int(quality_was_high)

    def posterior_high_given_signal(self) -> float:
        if self.buys < self.min_observations:
            return self.prior_p
        # A large pseudo-count keeps the estimate conservative on short horizons.
        k = self.prior_pseudo_count
        return (k * self.prior_p + self.highs) / (k + self.buys)
