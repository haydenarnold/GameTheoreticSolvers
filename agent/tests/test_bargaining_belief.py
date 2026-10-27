"""Tests for the discount-factor belief: grid prior, likelihood reweighting, and
the shrinkage that limits the pull of any single observation.
"""

from agent.bargaining.belief import DEFAULT_GRID, DeltaBelief


def test_fresh_belief_mean_is_the_grid_average_not_the_grid_sum():
    belief = DeltaBelief()
    expected_average = sum(DEFAULT_GRID) / len(DEFAULT_GRID)
    assert abs(belief.mean() - expected_average) < 1e-9
    assert belief.mean() <= 1.0


def test_fresh_belief_weights_sum_to_one():
    belief = DeltaBelief()
    assert abs(sum(belief.weights) - 1.0) < 1e-9


def test_update_keeps_mean_in_valid_range():
    belief = DeltaBelief()
    belief.update(0.6, lambda d: d)
    assert 0.0 <= belief.mean() <= 1.0


def test_single_observation_does_not_swing_the_mean_to_the_grid_edge():
    belief = DeltaBelief()
    belief.update(0.99, lambda d: 0.99 if d >= 0.97 else 0.0)
    assert belief.mean() < 0.95


def test_consistent_repeated_observations_still_move_the_mean_substantially():
    from agent.bargaining.belief import DEFAULT_GRID

    belief = DeltaBelief()
    for _ in range(6):
        belief.update(0.99, lambda d: 0.99 if d >= 0.97 else 0.0)
    uniform_mean = sum(DEFAULT_GRID) / len(DEFAULT_GRID)
    assert belief.mean() > uniform_mean + 0.1


def test_first_update_is_identical_to_the_old_uniform_shrink_target():
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
    true_delta = 0.99
    belief = DeltaBelief()
    for _ in range(5):
        belief.update(true_delta, lambda d: d)
    assert belief.mean() > 0.9


def test_single_observation_still_does_not_swing_to_the_edge_after_prior_updates():
    belief = DeltaBelief()
    for _ in range(3):
        belief.update(0.6, lambda d: d)
    mean_before = belief.mean()
    belief.update(0.99, lambda d: 0.99 if d >= 0.97 else 0.0)
    assert belief.mean() < 0.95
    assert belief.mean() > mean_before
