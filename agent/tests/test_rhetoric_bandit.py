import json
import os

import pytest

import agent.rhetoric.bandit as bandit_mod
from agent.rhetoric.bandit import StyleBandit, get_style_bandit, persist_style_bandit


def test_style_bandit_starts_with_all_five_foundations_uniform():
    bandit = StyleBandit()
    assert set(bandit.alpha) == set(bandit.beta) == set(StyleBandit.ARMS) == {
        "care", "fairness", "loyalty", "authority", "purity",
    }
    assert all(bandit.alpha[arm] == 1.0 and bandit.beta[arm] == 1.0 for arm in StyleBandit.ARMS)


def test_style_bandit_update_moves_only_the_credited_arm():
    bandit = StyleBandit()
    bandit.update("fairness", True)
    assert bandit.alpha["fairness"] == 2.0
    assert bandit.beta["fairness"] == 1.0
    for arm in StyleBandit.ARMS:
        if arm != "fairness":
            assert bandit.alpha[arm] == 1.0 and bandit.beta[arm] == 1.0

    bandit.update("authority", False)
    assert bandit.beta["authority"] == 2.0
    assert bandit.alpha["authority"] == 1.0


def test_style_bandit_selects_a_consistently_successful_arm_much_more_often():
    # Thompson Sampling over Beta posteriors: an arm with a long, clean
    # track record of successes should get selected far more often than one
    # with a long, clean track record of failures, over enough trials.
    bandit = StyleBandit()
    for _ in range(30):
        bandit.update("care", True)
    for _ in range(30):
        bandit.update("purity", False)

    trials = 500
    counts = {arm: 0 for arm in StyleBandit.ARMS}
    for _ in range(trials):
        counts[bandit.select_arm()] += 1

    assert counts["care"] > counts["purity"]
    assert counts["care"] > trials * 0.5  # dominates the other four arms combined


def test_style_bandit_select_arm_always_returns_a_known_arm():
    bandit = StyleBandit()
    for _ in range(50):
        assert bandit.select_arm() in StyleBandit.ARMS


def test_style_bandit_update_ignores_an_unknown_arm():
    bandit = StyleBandit()
    bandit.update("not-a-real-arm", True)
    assert all(bandit.alpha[arm] == 1.0 and bandit.beta[arm] == 1.0 for arm in StyleBandit.ARMS)


def test_style_bandit_to_dict_from_dict_round_trips():
    bandit = StyleBandit()
    bandit.update("care", True)
    bandit.update("care", True)
    bandit.update("purity", False)

    restored = StyleBandit.from_dict(bandit.to_dict())
    assert restored.alpha == bandit.alpha
    assert restored.beta == bandit.beta


def _isolate_store(tmp_path, monkeypatch):
    """Point the persistence layer at a throwaway file and a clean
    in-process cache, so tests never touch the real data/ directory or leak
    state into each other via the module-level cache."""
    monkeypatch.setattr(bandit_mod, "_STORE_PATH", str(tmp_path / "opponent_style_bandits.json"))
    monkeypatch.setattr(bandit_mod, "_CACHE", {})


def test_get_style_bandit_gives_a_fresh_bandit_for_hidden_identity(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    bandit = get_style_bandit("game-1", None)
    assert bandit.alpha == {arm: 1.0 for arm in StyleBandit.ARMS}


def test_get_style_bandit_returns_the_same_object_for_the_same_game(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    first = get_style_bandit("game-1", None)
    first.update("care", True)
    second = get_style_bandit("game-1", None)
    assert second is first
    assert second.alpha["care"] == 2.0


def test_hidden_identity_bandits_do_not_share_state_across_games(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    a = get_style_bandit("game-a", None)
    a.update("care", True)
    b = get_style_bandit("game-b", None)
    assert b.alpha["care"] == 1.0  # untouched -- a different game, no opponent identity to link them


def test_persist_style_bandit_is_a_noop_for_hidden_identity(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    bandit = get_style_bandit("game-1", None)
    bandit.update("care", True)
    persist_style_bandit("game-1", None)  # must not write anything
    assert not os.path.exists(bandit_mod._STORE_PATH)


def test_named_opponent_bandit_survives_a_simulated_process_restart(tmp_path, monkeypatch):
    # CONFIRMED live (2026-08-24): each campaign batch spawns a fresh
    # subprocess, so an in-memory-only cache would never actually persist
    # across games against the same named opponent -- this is the whole
    # point of the disk-backed store. Simulated here by clearing _CACHE
    # (the in-process half) while keeping the same on-disk file, exactly
    # what a fresh process would see.
    _isolate_store(tmp_path, monkeypatch)
    bandit = get_style_bandit("game-1", "Wordsworth")
    bandit.update("fairness", True)
    bandit.update("fairness", True)
    persist_style_bandit("game-1", "Wordsworth")

    monkeypatch.setattr(bandit_mod, "_CACHE", {})  # simulate a fresh process
    reloaded = get_style_bandit("game-2", "Wordsworth")
    # PERSISTENT_PRIOR_STRENGTH (4.0) + 2 real successes.
    assert reloaded.alpha["fairness"] == bandit_mod.PERSISTENT_PRIOR_STRENGTH + 2.0


def test_different_named_opponents_get_independent_persisted_bandits(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    a = get_style_bandit("game-1", "Wordsworth")
    a.update("care", True)
    persist_style_bandit("game-1", "Wordsworth")

    b = get_style_bandit("game-2", "Ra")
    assert b.alpha["care"] == bandit_mod.PERSISTENT_PRIOR_STRENGTH  # a different opponent, untouched by Wordsworth's update


def test_get_style_bandit_falls_back_to_fresh_on_corrupt_store_file(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    with open(bandit_mod._STORE_PATH, "w") as f:
        f.write("{not valid json")

    bandit = get_style_bandit("game-1", "Wordsworth")  # must not raise
    # Falls back to a FRESH PERSISTENT bandit (named opponent), not an
    # ordinary per-game one -- so the higher prior applies here too.
    assert bandit.alpha == {arm: bandit_mod.PERSISTENT_PRIOR_STRENGTH for arm in StyleBandit.ARMS}


def test_get_style_bandit_falls_back_to_fresh_on_corrupt_entry_for_this_opponent(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    with open(bandit_mod._STORE_PATH, "w") as f:
        json.dump({"Wordsworth": {"alpha": {"care": "not-a-number"}, "beta": {}}}, f)

    bandit = get_style_bandit("game-1", "Wordsworth")  # must not raise
    assert bandit.alpha == {arm: bandit_mod.PERSISTENT_PRIOR_STRENGTH for arm in StyleBandit.ARMS}


def test_persist_style_bandit_survives_a_missing_data_directory(tmp_path, monkeypatch):
    # The store path's parent directory doesn't exist yet -- persisting
    # must create it rather than fail.
    nested = tmp_path / "does" / "not" / "exist" / "opponent_style_bandits.json"
    monkeypatch.setattr(bandit_mod, "_STORE_PATH", str(nested))
    monkeypatch.setattr(bandit_mod, "_CACHE", {})

    bandit = get_style_bandit("game-1", "Wordsworth")
    bandit.update("care", True)
    persist_style_bandit("game-1", "Wordsworth")
    assert os.path.exists(nested)


def test_style_bandit_prior_strength_regularizes_a_fresh_bandit():
    # CONFIRMED live (2026-08-24): the real, live persistent store after
    # ~90 games showed several opponents with wild best/worst arm spreads
    # (0.89 vs 0.25) built from only 3-5 real trials per arm -- a Beta(1,1)
    # prior gets swamped by pure noise at that sample size. A higher
    # prior_strength requires substantially more real evidence to move a
    # posterior mean the same amount.
    weak = StyleBandit(prior_strength=1.0)
    strong = StyleBandit(prior_strength=bandit_mod.PERSISTENT_PRIOR_STRENGTH)
    for _ in range(3):
        weak.update("purity", True)
        strong.update("purity", True)
    weak_mean = weak.alpha["purity"] / (weak.alpha["purity"] + weak.beta["purity"])
    strong_mean = strong.alpha["purity"] / (strong.alpha["purity"] + strong.beta["purity"])
    assert weak_mean > strong_mean  # same 3 successes move the weak prior's mean further
    assert weak_mean > 0.75  # 4/5 -- an overconfident swing on thin evidence
    assert strong_mean < 0.65  # same evidence barely moves the stronger prior


def test_fresh_persistent_bandit_uses_the_stronger_prior_but_ordinary_per_game_bandit_does_not(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    named = get_style_bandit("game-1", "Wordsworth")
    hidden = get_style_bandit("game-2", None)
    assert named.alpha["care"] == bandit_mod.PERSISTENT_PRIOR_STRENGTH
    assert hidden.alpha["care"] == 1.0  # ordinary per-game bandit, unaffected


def test_style_bandit_prior_means_shifts_the_starting_point_not_the_concentration():
    means = {"care": 0.4, "fairness": 0.5, "loyalty": 0.65, "authority": 0.47, "purity": 0.58}
    bandit = StyleBandit(prior_strength=4.0, prior_means=means)
    for arm, mean in means.items():
        total = bandit.alpha[arm] + bandit.beta[arm]
        assert total == pytest.approx(8.0)  # same total concentration as 2*prior_strength
        assert bandit.alpha[arm] / total == pytest.approx(mean)


def test_style_bandit_prior_means_of_all_point_five_matches_uniform_default():
    uniform = {arm: 0.5 for arm in StyleBandit.ARMS}
    default = StyleBandit(prior_strength=3.0)
    informed = StyleBandit(prior_strength=3.0, prior_means=uniform)
    assert informed.alpha == default.alpha
    assert informed.beta == default.beta


def test_style_bandit_prior_means_is_overridden_by_real_evidence():
    means = {"care": 0.2, "fairness": 0.5, "loyalty": 0.9, "authority": 0.5, "purity": 0.5}
    bandit = StyleBandit(prior_strength=2.0, prior_means=means)
    for _ in range(40):
        bandit.update("care", True)
        bandit.update("loyalty", False)
    care_mean = bandit.alpha["care"] / (bandit.alpha["care"] + bandit.beta["care"])
    loyalty_mean = bandit.alpha["loyalty"] / (bandit.alpha["loyalty"] + bandit.beta["loyalty"])
    assert care_mean > loyalty_mean  # 40 real observations flip the population-prior ranking


def test_get_style_bandit_applies_prior_means_only_on_a_fresh_bandit(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    means = {"care": 0.9, "fairness": 0.5, "loyalty": 0.5, "authority": 0.5, "purity": 0.5}
    fresh = get_style_bandit("game-1", None, prior_means=means)
    assert fresh.alpha["care"] / (fresh.alpha["care"] + fresh.beta["care"]) == pytest.approx(0.9)

    # A second call for the SAME key (already cached) ignores a different
    # prior_means -- it's not a fresh bandit anymore.
    again = get_style_bandit("game-1", None, prior_means={arm: 0.1 for arm in StyleBandit.ARMS})
    assert again is fresh
    assert again.alpha["care"] / (again.alpha["care"] + again.beta["care"]) == pytest.approx(0.9)
