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
    # Deadline gives the proposer everything; one round earlier, the
    # proposer only needs to offer the responder their (discounted)
    # walk-into-being-proposer-next-round value.
    take = finite_horizon_take(100.0, "alice", lambda p: 0.5, rounds_remaining=1)
    assert abs(take - 50.0) < 1e-9


def test_finite_horizon_asymmetric_deltas_favor_impatient_opponent_loss():
    deltas = {"alice": 0.8, "bob": 0.5}
    take = finite_horizon_take(100.0, "alice", lambda p: deltas[p], rounds_remaining=2)
    # Hand-computed backward induction: x[2]=100; x[1] (bob proposes,
    # alice responds) = 100 - delta_alice*x[2] = 100-0.8*100=20;
    # x[0] (alice proposes, bob responds) = 100 - delta_bob*x[1] = 100-0.5*20=90.
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
    # Documents the confirmed pathology damped_finite_horizon_take exists to
    # fix: symmetric delta=0.9, 11 rounds remaining gives the proposer only
    # 37.8% -- well under half -- purely because 11 is odd, not because of
    # anything strategically meaningful.
    take = finite_horizon_take(10_000.0, "alice", lambda p: 0.9, rounds_remaining=11)
    assert take < 4_000.0  # confirms the raw pathology this test documents


def test_damped_take_converges_toward_infinite_horizon_for_symmetric_patient_players():
    money, delta = 10_000.0, 0.9
    damped = damped_finite_horizon_take(money, "alice", lambda p: delta, rounds_remaining=11)
    converged = infinite_horizon_share(money, delta, delta)
    assert abs(damped - converged) < 500.0  # much closer to ~52.6% than the raw 37.8%


def test_damped_take_matches_exact_finite_value_at_the_true_deadline():
    # At rounds_remaining=0 there's no oscillation to damp -- the proposer
    # genuinely does take everything, and damping must not distort that.
    damped = damped_finite_horizon_take(100.0, "alice", lambda p: 0.9, rounds_remaining=0)
    assert damped == 100.0


def test_damped_take_stays_above_half_for_a_more_patient_proposer_away_from_the_deadline():
    # Near a real deadline (rr=0,1), a moderately patient responder who'd
    # grab everything by rejecting into the deadline can legitimately force
    # concessions even from a more-patient proposer -- that's real
    # near-deadline strategy, not parity noise, and damping correctly leaves
    # it alone (low rr means low alpha... low weight on the *infinite* value,
    # i.e. the exact finite dynamics still dominate close in). Away from the
    # deadline, though, damping should keep a more-patient proposer's share
    # above half regardless of parity.
    delta_of = lambda p: 0.97 if p == "alice" else 0.85
    for rr in range(5, 20):
        take = damped_finite_horizon_take(100.0, "alice", delta_of, rounds_remaining=rr)
        assert take >= 50.0, f"failed at rounds_remaining={rr}: {take}"
