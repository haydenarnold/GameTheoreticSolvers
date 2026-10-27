"""Tests for the disclosure bluff rate, the seller's adaptive bluff rate, and the
buyer trust model.
"""

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
    assert p * v + (1 - p) * u < price
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
    assert adapter.adapted_q() == 0.5


def test_seller_disclosure_adapter_raises_q_when_buyer_is_under_skeptical():
    adapter = SellerDisclosureAdapter(q0=0.5, step=0.05, min_observations=3)
    for _ in range(5):
        adapter.record(True)
    assert adapter.adapted_q() > 0.5


def test_seller_disclosure_adapter_lowers_q_when_buyer_is_over_skeptical():
    adapter = SellerDisclosureAdapter(q0=0.5, step=0.05, min_observations=3)
    for _ in range(5):
        adapter.record(False)
    assert adapter.adapted_q() < 0.5


def test_seller_disclosure_adapter_reacts_within_a_few_rounds_not_fifteen():
    adapter = SellerDisclosureAdapter(q0=1.0, step=0.05, min_observations=3)
    for _ in range(6):
        adapter.record(True)
    for _ in range(3):
        adapter.record(False)
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
    adapter.record(True)
    adapter.record(False)
    assert adapter.adapted_q() != 0.0


def test_buyer_trust_model_falls_back_to_prior_before_enough_observations():
    trust = BuyerTrustModel(prior_p=0.6, min_observations=2)
    trust.record(True)
    assert trust.posterior_high_given_signal() == 0.6


def test_buyer_trust_model_shrinks_toward_prior_with_few_observations():
    trust = BuyerTrustModel(prior_p=0.6, min_observations=2, prior_pseudo_count=5.0)
    trust.record(True)
    trust.record(False)
    posterior = trust.posterior_high_given_signal()
    assert abs(posterior - (5 * 0.6 + 1) / (5 + 2)) < 1e-9
    assert 0.5 < posterior < 0.6


def test_buyer_trust_model_drifts_further_from_prior_with_lots_of_contrary_evidence():
    trust = BuyerTrustModel(prior_p=0.6, min_observations=2, prior_pseudo_count=5.0)
    for _ in range(20):
        trust.record(False)
    assert trust.posterior_high_given_signal() < 0.2


def test_buyer_trust_model_default_pseudo_count_survives_an_unlucky_bootstrap():
    trust = BuyerTrustModel(prior_p=0.8, min_observations=2)
    trust.record(False)
    trust.record(False)
    posterior = trust.posterior_high_given_signal()
    effective_value = posterior * 200.0
    required = 100.0 * 1.2
    assert effective_value >= required, (
        f"posterior {posterior} (effective value {effective_value}) must still clear "
        f"{required} after just 2 unlucky draws at a comfortably profitable p=0.8"
    )


def test_buyer_trust_model_default_pseudo_count_still_detects_sustained_bad_evidence():
    trust = BuyerTrustModel(prior_p=0.8, min_observations=2)
    for _ in range(30):
        trust.record(False)
    assert trust.posterior_high_given_signal() < 0.35


