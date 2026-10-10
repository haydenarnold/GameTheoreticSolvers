from agent.persuasion.solver import (
    BuyerTrustModel,
    SellerDisclosureAdapter,
    optimal_bluff_prob,
    pooled_value_given_signal,
)


def test_naive_always_recommend_is_optimal_when_prior_alone_clears_price():
    p, v, u, price = 0.5, 100.0, 0.0, 40.0
    assert p * v + (1 - p) * u >= price
    q = optimal_bluff_prob(p, v, u, price)
    assert q == 1.0


def test_optimal_bluff_prob_calibrates_pooled_value_to_exact_indifference():
    p, v, u, price = 0.3, 100.0, 0.0, 40.0
    assert p * v + (1 - p) * u < price  # naive always-recommend would NOT clear the price here
    q = optimal_bluff_prob(p, v, u, price)
    assert 0.0 < q < 1.0
    pooled = pooled_value_given_signal(p, v, u, q)
    assert abs(pooled - price) < 1e-6


def test_optimal_bluff_prob_zero_when_price_at_or_above_high_value():
    assert optimal_bluff_prob(0.5, 100.0, 0.0, 150.0) == 0.0


def test_optimal_bluff_prob_one_when_price_at_or_below_low_value():
    assert optimal_bluff_prob(0.5, 100.0, 20.0, 10.0) == 1.0


def test_seller_disclosure_adapter_holds_until_enough_observations():
    adapter = SellerDisclosureAdapter(q0=0.5, min_observations=3)
    adapter.record(True)
    adapter.record(True)
    assert adapter.adapted_q() == 0.5  # only 2 observations so far, below the floor


def test_seller_disclosure_adapter_raises_q_when_buyer_is_under_skeptical():
    adapter = SellerDisclosureAdapter(q0=0.5, step=0.05, min_observations=3)
    for _ in range(5):
        adapter.record(True)  # buyer accepts every "buy" signal -> under-skeptical
    assert adapter.adapted_q() > 0.5


def test_seller_disclosure_adapter_lowers_q_when_buyer_is_over_skeptical():
    adapter = SellerDisclosureAdapter(q0=0.5, step=0.05, min_observations=3)
    for _ in range(5):
        adapter.record(False)  # buyer rejects every "buy" signal -> over-skeptical
    assert adapter.adapted_q() < 0.5


def test_seller_disclosure_adapter_reacts_within_a_few_rounds_not_fifteen():
    # Regression: a lifetime cumulative accept-rate, diluted by an early run
    # of successful sales, took ~15-18 rounds to drop below the correction
    # threshold live -- by which point two separate 20-round games were
    # nearly over. Several early wins followed by a real decline streak
    # must correct within a handful of rounds, not require the whole game.
    adapter = SellerDisclosureAdapter(q0=1.0, step=0.05, min_observations=3)
    for _ in range(6):
        adapter.record(True)  # a strong early run of sales (as observed live)
    for _ in range(3):
        adapter.record(False)  # buyer's trust has now visibly collapsed
    assert adapter.adapted_q() < 0.5


def test_seller_disclosure_adapter_consecutive_decline_fast_path_goes_fully_honest():
    adapter = SellerDisclosureAdapter(q0=1.0, consecutive_decline_threshold=3)
    for _ in range(3):
        adapter.record(False)
    assert adapter.adapted_q() == 0.0


def test_seller_disclosure_adapter_consecutive_decline_streak_resets_on_a_sale():
    adapter = SellerDisclosureAdapter(q0=0.7, consecutive_decline_threshold=3)
    adapter.record(False)
    adapter.record(False)
    adapter.record(True)  # resets the streak
    adapter.record(False)
    assert adapter.adapted_q() != 0.0


def test_buyer_trust_model_falls_back_to_prior_before_enough_observations():
    trust = BuyerTrustModel(prior_p=0.6, min_observations=2)
    trust.record(True)
    assert trust.posterior_high_given_signal() == 0.6


def test_buyer_trust_model_shrinks_toward_prior_with_few_observations():
    # Beta-Binomial shrinkage, not a raw MLE: 1 high / 2 buys should pull
    # only modestly away from the known prior (0.6), not jump straight to
    # the tiny-sample empirical rate (0.5) the way an unregularized
    # highs/buys estimate would.
    trust = BuyerTrustModel(prior_p=0.6, min_observations=2, prior_pseudo_count=5.0)
    trust.record(True)
    trust.record(False)
    posterior = trust.posterior_high_given_signal()
    assert abs(posterior - (5 * 0.6 + 1) / (5 + 2)) < 1e-9
    assert 0.5 < posterior < 0.6


def test_buyer_trust_model_drifts_further_from_prior_with_lots_of_contrary_evidence():
    # Enough real, consistently-bad data should still be able to move the
    # posterior well below the prior -- shrinkage resists small-sample
    # noise, it doesn't block genuine learning.
    trust = BuyerTrustModel(prior_p=0.6, min_observations=2, prior_pseudo_count=5.0)
    for _ in range(20):
        trust.record(False)
    assert trust.posterior_high_given_signal() < 0.2


def test_buyer_trust_model_default_pseudo_count_survives_an_unlucky_bootstrap():
    # CONFIRMED live (2026-08-25): a real p=0.8/v=$200/price=$100 game (60%
    # unconditional-EV margin) had its first 2 bootstrap buys both land low
    # (~4% likelihood, not evidence of anything) and the OLD default
    # (prior_pseudo_count=5.0) let that 2-sample draw crash the posterior
    # from 0.8 to 0.571 -- just under the 0.60 margin-adjusted threshold --
    # locking out the remaining 18 rounds of a game with +$60/round
    # unconditional EV. The new default must survive this exact case.
    trust = BuyerTrustModel(prior_p=0.8, min_observations=2)
    trust.record(False)
    trust.record(False)
    posterior = trust.posterior_high_given_signal()
    effective_value = posterior * 200.0
    required = 100.0 * 1.2  # BUYER_SAFETY_MARGIN=0.20, matching strategy.py
    assert effective_value >= required, (
        f"posterior {posterior} (effective value {effective_value}) must still clear "
        f"{required} after just 2 unlucky draws at a comfortably profitable p=0.8"
    )


def test_buyer_trust_model_default_pseudo_count_still_detects_sustained_bad_evidence():
    # The fix must not neuter genuine learning -- many real observations
    # consistently below the stated prior should still move the posterior
    # substantially, just not from 2 samples of ordinary noise.
    trust = BuyerTrustModel(prior_p=0.8, min_observations=2)
    for _ in range(30):
        trust.record(False)
    # (20*0.8+0)/(20+30) = 0.32 -- well below the 0.8 prior given real,
    # sustained contrary evidence, just not from 2 samples of noise.
    assert trust.posterior_high_given_signal() < 0.35


# Periodic re-exploration tests removed (2026-08-24) along with the
# mechanism itself -- see BuyerTrustModel's docstring in solver.py for why
# (measured net-negative against 102 real completed games).

# StyleBandit's own tests moved to test_rhetoric_bandit.py (2026-08-23) when
# the class moved to agent/rhetoric/bandit.py -- it's shared across all
# three families now, not persuasion-specific.
