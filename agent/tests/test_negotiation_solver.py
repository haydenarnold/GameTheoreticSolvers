from agent.negotiation.solver import ValueBounds, concession_target, deadline_pressure, time_fraction


def test_concession_target_linear_beta_one_is_the_simple_midpoint_rule():
    target = concession_target(anchor=100.0, reservation=50.0, t=0.5, beta=1.0)
    assert abs(target - 75.0) < 1e-9


def test_concession_target_boulware_beta_below_one_concedes_less_than_linear():
    linear = concession_target(anchor=100.0, reservation=50.0, t=0.5, beta=1.0)
    boulware = concession_target(anchor=100.0, reservation=50.0, t=0.5, beta=0.5)
    # Boulware holds closer to the anchor mid-negotiation than a linear concession would.
    assert boulware > linear


def test_concession_target_conceder_beta_above_one_concedes_more_than_linear():
    linear = concession_target(anchor=100.0, reservation=50.0, t=0.5, beta=1.0)
    conceder = concession_target(anchor=100.0, reservation=50.0, t=0.5, beta=2.0)
    assert conceder < linear


def test_concession_target_reaches_anchor_at_t_zero_and_reservation_at_t_one():
    assert abs(concession_target(100.0, 50.0, 0.0, 0.7) - 100.0) < 1e-9
    assert abs(concession_target(100.0, 50.0, 1.0, 0.7) - 50.0) < 1e-9


def test_concession_target_handles_buyer_direction_anchor_below_reservation():
    # Buyer: anchor (low opening bid) below reservation (max willing to pay);
    # target should rise from the anchor toward the reservation as t -> 1.
    early = concession_target(anchor=20.0, reservation=80.0, t=0.1, beta=1.0)
    late = concession_target(anchor=20.0, reservation=80.0, t=0.9, beta=1.0)
    assert early < late


def test_time_fraction_capped_horizon():
    # round_no is 1-indexed (round 1 = zero rounds actually elapsed), so
    # this measures elapsed rounds (round_no - 1), not the round index.
    assert time_fraction(5, 10) == 0.4


def test_time_fraction_is_zero_on_the_opening_move_of_a_one_shot_ultimatum():
    # Regression: a live max_rounds=1 game hit t=1.0 (full concession) on
    # the very first and only move, because round_no == max_rounds there
    # unconditionally. The opening offer of a one-shot game must anchor
    # fully aggressive (t=0), not fully conceded.
    assert time_fraction(1, 1) == 0.0


def test_time_fraction_unlimited_horizon_grows_toward_one():
    early = time_fraction(1, None)
    late = time_fraction(20, None)
    assert 0 < early < late < 1


def test_deadline_pressure_zero_when_unlimited():
    assert deadline_pressure(None) == 0.0


def test_deadline_pressure_one_at_the_last_round():
    assert deadline_pressure(0) == 1.0


def test_value_bounds_tighten_from_seller_rejection_raises_lower_bound():
    bounds = ValueBounds(lo=0.0, hi=1000.0)
    bounds.tighten_from_rejection("seller", rejected_price=60.0)
    assert bounds.lo == 60.0
    assert bounds.hi == 1000.0


def test_value_bounds_tighten_from_buyer_rejection_lowers_upper_bound():
    bounds = ValueBounds(lo=0.0, hi=1000.0)
    bounds.tighten_from_rejection("buyer", rejected_price=60.0)
    assert bounds.hi == 60.0
    assert bounds.lo == 0.0
