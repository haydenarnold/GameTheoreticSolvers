"""Bayesian persuasion (Kamenica & Gentzkow, 2011) applied to the seller's
disclosure choice, plus simple online trust-modeling for both sides.

The seller always signals "buy" on high quality. On low quality, they
signal "buy" with probability q* chosen so the pooled posterior value,
conditional on a "buy" signal, sits exactly at the buyer's indifference
point (price). This is the textbook optimal-disclosure result: it
maximizes the sale rate subject to a rational buyer still finding "buy"
signals credible. If the unconditional prior alone already clears the price
(p*v + (1-p)*u >= price), q* comes out >= 1 after clipping — i.e. the docs'
own "always recommend" baseline is provably optimal, but only in that
regime; everywhere u < price < v it leaves money on the table (or gets
learned-around and ignored by a rational buyer) relative to a calibrated q*.
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
    """Nudges q toward whatever would make the observed buyer accept-rate
    (on rounds we signaled "buy") match a well-calibrated seller: if
    they're accepting more than that, they're under-skeptical — exploit
    further (raise q). If they're accepting notably less, they don't trust
    us — rebuild credibility (lower q). Matters most when we can't compute
    an exact q* (seller doesn't know v/u), where this is the only
    correction available.

    CONFIRMED live (2026-08-22), twice: a lifetime cumulative accept-rate
    reacts far too slowly once a buyer stops trusting "buy" signals. A
    seller that bluffed at q*=1.0 (textbook-optimal against a naive,
    one-shot Bayesian buyer) sold 5-7 times early, then the buyer's OWN
    trust model collapsed after a couple of low-quality purchases and
    declined every remaining round -- but the cumulative accept-rate
    (diluted by the early string of successes) took ~15-18 rounds to drift
    below the 0.4 correction threshold, by which point the game was nearly
    over. `window` makes the rate reflect RECENT behavior, not lifetime
    history, and `consecutive_decline_threshold` adds a fast path: once the
    buyer has declined that many signaled-buy rounds in a row, there is no
    cost to bluffing less (an already-distrustful buyer isn't buying
    regardless), so drop straight to full honesty rather than waiting on a
    slowly-drifting average."""

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
    """A running estimate of P(quality = high) conditional on rounds we
    chose to buy — quality is only observable to us on rounds we actually
    bought, so that's the only data available. Recalibrates our effective
    prior away from the game's stated `p` as evidence accumulates, instead
    of blindly trusting the static prior every round like the naive
    baseline does.

    CONFIRMED live (2026-08-22): a raw, unregularized empirical rate
    (`highs/buys`, no shrinkage) collapsed to 0.2 after just 5 purchases (1
    high, 4 low) against a TRUE, exactly-known prior of 0.333 -- ordinary
    small-sample noise, not evidence the seller's signal is untrustworthy,
    but with zero smoothing the buyer treated it as decisive and declined
    every one of the remaining 15 rounds, even though the raw prior alone
    (p*v+(1-p)*u) still comfortably cleared the price. `prior_pseudo_count`
    Beta-Binomial-smooths the estimate toward the known-true prior instead
    of a raw MLE: with few real observations it stays close to `prior_p`
    (correctly resisting small-sample noise), and only drifts meaningfully
    once genuine evidence accumulates -- e.g. a seller who bluffs "buy" on
    every low-quality round for 15+ rounds straight.

    REMOVED (2026-08-24): this used to also support periodic re-exploration
    -- after a lockout (posterior below the buy threshold), spend one more
    bounded test every `re_explore_cooldown` declined rounds, on the theory
    that quality is IID each round so an early unlucky sample shouldn't
    cause a permanent lockout. Measured directly against 102 real completed
    games (912 buy events replayed against the actual historical seller
    signals and quality outcomes): bootstrap buys landed a 57.4% high rate
    (+$98,129 avg), ordinary posterior-cleared buys landed 76.3%
    (+$318,073 avg), but periodic re-explore buys landed just 4.2% (3 of 72)
    for an average of -$320,874 -- effectively the mirror image of ordinary
    buying, not occasional bad luck. The flawed premise: quality IS IID
    per round, but a SELLER'S SIGNALING HONESTY isn't -- it's a fixed trait
    of that opponent for the whole game. Once real evidence shows a
    specific seller's "yes" doesn't predict quality, re-testing doesn't
    give them a fresh, independent chance; it just re-samples the same
    already-revealed-unreliable seller, and loses again almost every time.
    `note_declined_signal`/`rounds_since_purchase`/`re_explore_cooldown`
    removed along with it -- nothing else read them."""

    # CONFIRMED live (2026-08-25): `prior_pseudo_count=5.0` was still far too
    # weak. `p` in this game is GROUND TRUTH handed to us by the game engine
    # (`state["p"]`, always exact, always visible) -- not the seller's claim
    # to be verified. The only thing legitimately worth learning here is
    # whether the seller's MESSAGE carries information beyond that stated
    # rate. A real transcript (p=0.8, v=$200, u=$0, price=$100 -- unconditional
    # EV=$60/round, a comfortable 33% margin above the $120 breakeven-with-
    # margin bar) showed the seller sending the IDENTICAL message every one of
    # 20 rounds ("Good unit this round. Worth taking at the asking price.")
    # -- zero conditional information, since it never varies with the actual
    # quality. Two unlucky low draws on the very first two bootstrap
    # purchases (~4% likelihood at p=0.8, not evidence of anything) pulled
    # the posterior to (5*0.8+0)/(5+2)=0.571, just under the 0.60
    # margin-adjusted threshold -- and with no re-exploration once locked out
    # (see the REMOVED note above), that meant passing every remaining round
    # of a game with +$60/round unconditional EV, foregoing roughly $1,200 in
    # expectation for a $0 outcome. K=5 lets 2 samples of pure random noise
    # override a KNOWN, exact prior. K=20 requires meaningfully more real
    # evidence before the estimate can move enough to flip a comfortably
    # profitable base rate into a decline, while still reacting to a
    # genuinely bad, sustained pattern (many real observations, not 2) --
    # verified in `scratchpad/buyer_trust_sim.py` across the config space
    # actually seen this session (p in {0.333, 0.5, 0.8}, v/price ratios
    # 2-4): K=20 survives a worst-case 2-low bootstrap in every one of them,
    # where K=5 failed the p=0.8/v-over-price=2 case exactly as observed live.
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
        k = self.prior_pseudo_count
        return (k * self.prior_p + self.highs) / (k + self.buys)
