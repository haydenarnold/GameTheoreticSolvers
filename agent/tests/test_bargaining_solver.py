"""Tests for the Rubinstein solvers: infinite-horizon share, backward induction,
the damped blend, and the continuation value of rejecting.
"""

from agent.bargaining.solver import (
    damped_finite_horizon_take,
    finite_horizon_take,
    infinite_horizon_share,
    reject_continuation_value,
)


def test_infinite_horizon_symmetric_delta_matches_1_over_1_plus_delta():
    money = 100.0
    for delta in (0.5, 0.8, 0.95):
        share = infinite_horizon_share(money, delta, delta)
        assert abs(share - money / (1 + delta)) < 1e-9


def test_infinite_horizon_matches_closed_form_asymmetric():
    money, delta_me, delta_opp = 200.0, 0.9, 0.7
    share = infinite_horizon_share(money, delta_me, delta_opp)
    expected = money * (1 - delta_opp) / (1 - delta_me * delta_opp)
    assert abs(share - expected) < 1e-9


def test_finite_horizon_last_round_proposer_takes_all():
    take = finite_horizon_take(100.0, "alice", lambda p: 0.5, rounds_remaining=0)
    assert take == 100.0


def test_finite_horizon_two_round_symmetric_delta_splits_evenly():
    take = finite_horizon_take(100.0, "alice", lambda p: 0.5, rounds_remaining=1)
    assert abs(take - 50.0) < 1e-9


def test_finite_horizon_asymmetric_deltas_favor_impatient_opponent_loss():
    deltas = {"alice": 0.8, "bob": 0.5}
    take = finite_horizon_take(100.0, "alice", lambda p: deltas[p], rounds_remaining=2)
    assert abs(take - 90.0) < 1e-9


def test_reject_continuation_value_at_deadline_is_zero():
    value = reject_continuation_value(100.0, "alice", lambda p: 0.5, rounds_remaining=0)
    assert value == 0.0


def test_reject_continuation_value_matches_discounted_next_round_take():
    delta_of = lambda p: {"alice": 0.9, "bob": 0.6}[p]
    value = reject_continuation_value(100.0, "alice", delta_of, rounds_remaining=2)
    next_round_take = finite_horizon_take(100.0, "alice", delta_of, rounds_remaining=1)
    assert abs(value - delta_of("alice") * next_round_take) < 1e-9


def test_finite_horizon_oscillates_wildly_by_parity_even_with_real_discounting():
    take = finite_horizon_take(10_000.0, "alice", lambda p: 0.9, rounds_remaining=11)
    assert take < 4_000.0


def test_damped_take_converges_toward_infinite_horizon_for_symmetric_patient_players():
    money, delta = 10_000.0, 0.9
    damped = damped_finite_horizon_take(money, "alice", lambda p: delta, rounds_remaining=11)
    converged = infinite_horizon_share(money, delta, delta)
    assert abs(damped - converged) < 500.0


def test_damped_take_matches_exact_finite_value_at_the_true_deadline():
    damped = damped_finite_horizon_take(100.0, "alice", lambda p: 0.9, rounds_remaining=0)
    assert damped == 100.0


def test_damped_take_stays_above_half_for_a_more_patient_proposer_away_from_the_deadline():
    delta_of = lambda p: 0.97 if p == "alice" else 0.85
    for rr in range(5, 20):
        take = damped_finite_horizon_take(100.0, "alice", delta_of, rounds_remaining=rr)
        assert take >= 50.0, f"failed at rounds_remaining={rr}: {take}"
