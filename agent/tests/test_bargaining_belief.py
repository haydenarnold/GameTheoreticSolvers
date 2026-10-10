from agent.bargaining.belief import DEFAULT_GRID, DeltaBelief


def test_fresh_belief_mean_is_the_grid_average_not_the_grid_sum():
    # Regression: unnormalized initial weights made mean() return the raw
    # sum of DEFAULT_GRID (8.19) instead of its average (0.819) before any
    # update() call — i.e. on the opening move of nearly every
    # incomplete-information bargaining game. A discount factor > 1 is
    # nonsensical and blew the backward-induction formula out to roughly
    # -$23.5M on a real $100 game, which then clamped to a $0 offer.
    belief = DeltaBelief()
    expected_average = sum(DEFAULT_GRID) / len(DEFAULT_GRID)
    assert abs(belief.mean() - expected_average) < 1e-9
    assert belief.mean() <= 1.0  # any valid discount factor must be in [0, 1]


def test_fresh_belief_weights_sum_to_one():
    belief = DeltaBelief()
    assert abs(sum(belief.weights) - 1.0) < 1e-9


def test_update_keeps_mean_in_valid_range():
    belief = DeltaBelief()
    belief.update(0.6, lambda d: d)  # opponent's take fraction roughly matches their own delta
    assert 0.0 <= belief.mean() <= 1.0


def test_single_observation_does_not_swing_the_mean_to_the_grid_edge():
    # Regression: a single offer observation swung the posterior mean from
    # the grid's uniform average (0.819) to 0.92, confirmed live on both the
    # offer and accept-threshold sides after just one round of evidence.
    # `predicted_fraction_fn` here says every hypothesis is fully consistent
    # with a 0.99-style take (an extreme, maximally-informative single
    # observation) -- even so, one data point alone must not fully collapse
    # the belief onto the grid's edge.
    belief = DeltaBelief()
    belief.update(0.99, lambda d: 0.99 if d >= 0.97 else 0.0)
    assert belief.mean() < 0.95


def test_consistent_repeated_observations_still_move_the_mean_substantially():
    # Shrinkage must resist noise from a lone data point, not block genuine
    # learning: several consistent observations should still pull the mean
    # well above the uniform prior.
    from agent.bargaining.belief import DEFAULT_GRID

    belief = DeltaBelief()
    for _ in range(6):
        belief.update(0.99, lambda d: 0.99 if d >= 0.97 else 0.0)
    uniform_mean = sum(DEFAULT_GRID) / len(DEFAULT_GRID)
    assert belief.mean() > uniform_mean + 0.1


def test_first_update_is_identical_to_the_old_uniform_shrink_target():
    # CONFIRMED (2026-08-25): the trust-region fix (shrink toward the
    # belief's own PREVIOUS weights, not a fixed uniform target) must be
    # mathematically identical to the old behavior on a belief's first-ever
    # update, since prev_weights == the initial uniform weights there --
    # this is exactly what preserves the 2026-08-22 incident protection.
    # Computed directly rather than a hardcoded constant, since the exact
    # value depends on DEFAULT_GRID/_KERNEL_SIGMA/SHRINKAGE_PSEUDO_COUNT.
    from agent.bargaining.belief import SHRINKAGE_PSEUDO_COUNT, _KERNEL_SIGMA
    import math

    grid = DEFAULT_GRID
    observed = 0.99
    predicted_fraction_fn = lambda d: 0.99 if d >= 0.97 else 0.0

    raw = []
    for d in grid:
        predicted = predicted_fraction_fn(d)
        err = predicted - observed
        raw.append(math.exp(-(err * err) / (2 * _KERNEL_SIGMA * _KERNEL_SIGMA)))
    total = sum(raw)
    raw_weights = [w / total for w in raw]
    mix = SHRINKAGE_PSEUDO_COUNT / (SHRINKAGE_PSEUDO_COUNT + 1)
    uniform = 1.0 / len(grid)
    expected_old_mean = sum(
        ((1 - mix) * w + mix * uniform) * d for w, d in zip(raw_weights, grid)
    )

    belief = DeltaBelief()
    belief.update(observed, predicted_fraction_fn)
    assert abs(belief.mean() - expected_old_mean) < 1e-9


def test_repeated_evidence_converges_closer_to_an_extreme_true_delta_than_before():
    # CONFIRMED (2026-08-25): the whole point of the fix -- perpetually
    # shrinking toward a fixed uniform target on every update (not just the
    # first) fights genuine convergence. Against a rational opponent whose
    # true delta sits at the grid's edge (0.99), repeated CONSISTENT
    # evidence should now close the gap much further than the old
    # every-update-toward-uniform behavior did.
    true_delta = 0.99
    belief = DeltaBelief()
    for _ in range(5):
        # opponent behaves exactly like their own true delta implies
        belief.update(true_delta, lambda d: d)
    # Old behavior (perpetual uniform shrink) left this well below 0.9 even
    # after several consistent observations; trust-region should land
    # meaningfully closer to the true 0.99.
    assert belief.mean() > 0.9


def test_single_observation_still_does_not_swing_to_the_edge_after_prior_updates():
    # The trust-region fix damps step size relative to the CURRENT belief,
    # not just the first call -- a single new, wildly-different observation
    # after several consistent ones should still be damped, not instantly
    # override accumulated evidence in one step.
    belief = DeltaBelief()
    for _ in range(3):
        belief.update(0.6, lambda d: d)  # several consistent low-delta observations
    mean_before = belief.mean()
    belief.update(0.99, lambda d: 0.99 if d >= 0.97 else 0.0)  # one wild outlier
    # One outlier must move the belief, but not slam it straight to the grid edge.
    assert belief.mean() < 0.95
    assert belief.mean() > mean_before  # it should still move in the outlier's direction
