import json
import os

import pytest

import agent.rhetoric.opponent_profile as profile_mod
from agent.rhetoric.opponent_profile import (
    ConcessionProfile,
    TRUST_THRESHOLD_GAMES,
    get_concession_profile,
    record_concession_evidence,
)


def _isolate_store(tmp_path, monkeypatch):
    """Point the persistence layer at a throwaway file and a clean
    in-process cache, so tests never touch the real data/ directory or leak
    state into each other via the module-level cache."""
    monkeypatch.setattr(profile_mod, "_STORE_PATH", str(tmp_path / "opponent_concession_profile.json"))
    monkeypatch.setattr(profile_mod, "_CACHE", {})


def test_fresh_profile_starts_untrusted():
    profile = ConcessionProfile()
    assert profile.games_observed == 0
    assert profile.games_confirmed_hopeless == 0
    assert profile.is_trusted_hardliner() is False


def test_is_trusted_hardliner_requires_the_threshold_game_count():
    profile = ConcessionProfile(games_observed=TRUST_THRESHOLD_GAMES - 1, games_confirmed_hopeless=TRUST_THRESHOLD_GAMES - 1)
    assert profile.is_trusted_hardliner() is False  # one short of the bar

    profile = ConcessionProfile(games_observed=TRUST_THRESHOLD_GAMES, games_confirmed_hopeless=TRUST_THRESHOLD_GAMES)
    assert profile.is_trusted_hardliner() is True


def test_is_trusted_hardliner_requires_all_not_just_a_majority():
    # 5 observed, 4 confirmed hopeless (80%) -- still not trusted, since one
    # real game showed genuine concession.
    profile = ConcessionProfile(games_observed=5, games_confirmed_hopeless=4)
    assert profile.is_trusted_hardliner() is False


def test_to_dict_from_dict_round_trips():
    profile = ConcessionProfile(games_observed=7, games_confirmed_hopeless=7)
    restored = ConcessionProfile.from_dict(profile.to_dict())
    assert restored.games_observed == profile.games_observed
    assert restored.games_confirmed_hopeless == profile.games_confirmed_hopeless


def test_get_concession_profile_gives_a_fresh_profile_for_hidden_identity():
    profile = get_concession_profile(None, "bargaining")
    assert profile.games_observed == 0
    assert profile.is_trusted_hardliner() is False


def test_hidden_identity_profiles_do_not_persist(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    record_concession_evidence(None, "bargaining", True)  # must not write anything
    assert not os.path.exists(profile_mod._STORE_PATH)


def test_record_concession_evidence_accumulates_across_calls(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    record_concession_evidence("Wordsworth", "bargaining", True)
    record_concession_evidence("Wordsworth", "bargaining", True)
    profile = get_concession_profile("Wordsworth", "bargaining")
    assert profile.games_observed == 2
    assert profile.games_confirmed_hopeless == 2
    assert profile.is_trusted_hardliner() is True


def test_a_genuinely_conceding_game_lowers_the_ratio_without_erasing_history(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    record_concession_evidence("Wordsworth", "bargaining", True)
    record_concession_evidence("Wordsworth", "bargaining", True)
    record_concession_evidence("Wordsworth", "bargaining", False)  # genuine concession this time
    profile = get_concession_profile("Wordsworth", "bargaining")
    assert profile.games_observed == 3
    assert profile.games_confirmed_hopeless == 2
    assert profile.is_trusted_hardliner() is False  # no longer ALL games


def test_named_opponent_profile_survives_a_simulated_process_restart(tmp_path, monkeypatch):
    # Each campaign batch spawns a fresh subprocess -- an in-memory-only
    # cache would never actually persist across games against the same
    # named opponent. Simulated here by clearing _CACHE (the in-process
    # half) while keeping the same on-disk file, exactly what a fresh
    # process would see.
    _isolate_store(tmp_path, monkeypatch)
    record_concession_evidence("Wordsworth", "bargaining", True)
    record_concession_evidence("Wordsworth", "bargaining", True)

    monkeypatch.setattr(profile_mod, "_CACHE", {})  # simulate a fresh process
    reloaded = get_concession_profile("Wordsworth", "bargaining")
    assert reloaded.games_observed == 2
    assert reloaded.is_trusted_hardliner() is True


def test_different_named_opponents_get_independent_persisted_profiles(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    record_concession_evidence("Wordsworth", "bargaining", True)
    record_concession_evidence("Wordsworth", "bargaining", True)

    other = get_concession_profile("Ra", "bargaining")
    assert other.games_observed == 0  # untouched by Wordsworth's record


def test_same_name_is_scoped_independently_per_family(tmp_path, monkeypatch):
    # A name recurring across families must not share state -- a bargaining
    # hardliner isn't guaranteed to be one in negotiation or persuasion.
    _isolate_store(tmp_path, monkeypatch)
    record_concession_evidence("Wordsworth", "bargaining", True)
    record_concession_evidence("Wordsworth", "bargaining", True)

    negotiation_profile = get_concession_profile("Wordsworth", "negotiation")
    assert negotiation_profile.games_observed == 0
    assert negotiation_profile.is_trusted_hardliner() is False

    bargaining_profile = get_concession_profile("Wordsworth", "bargaining")
    assert bargaining_profile.games_observed == 2
    assert bargaining_profile.is_trusted_hardliner() is True


def test_get_concession_profile_falls_back_to_fresh_on_corrupt_store_file(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    with open(profile_mod._STORE_PATH, "w") as f:
        f.write("{not valid json")

    profile = get_concession_profile("Wordsworth", "bargaining")  # must not raise
    assert profile.games_observed == 0
    assert profile.is_trusted_hardliner() is False


def test_get_concession_profile_falls_back_to_fresh_on_corrupt_entry_for_this_opponent(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    with open(profile_mod._STORE_PATH, "w") as f:
        json.dump({"bargaining:Wordsworth": {"games_observed": "not-a-number"}}, f)

    profile = get_concession_profile("Wordsworth", "bargaining")  # must not raise
    assert profile.games_observed == 0


def test_record_concession_evidence_survives_a_missing_data_directory(tmp_path, monkeypatch):
    nested = tmp_path / "does" / "not" / "exist" / "opponent_concession_profile.json"
    monkeypatch.setattr(profile_mod, "_STORE_PATH", str(nested))
    monkeypatch.setattr(profile_mod, "_CACHE", {})

    record_concession_evidence("Wordsworth", "bargaining", True)
    assert os.path.exists(nested)


def test_record_concession_evidence_reloads_from_disk_if_not_yet_cached_in_process(tmp_path, monkeypatch):
    # Two independent processes/threads recording against the same name:
    # the second call must not clobber the first's persisted history just
    # because it wasn't in ITS in-process cache yet.
    _isolate_store(tmp_path, monkeypatch)
    record_concession_evidence("Wordsworth", "bargaining", True)

    monkeypatch.setattr(profile_mod, "_CACHE", {})  # simulate a second, fresh process
    record_concession_evidence("Wordsworth", "bargaining", True)

    monkeypatch.setattr(profile_mod, "_CACHE", {})
    profile = get_concession_profile("Wordsworth", "bargaining")
    assert profile.games_observed == 2
