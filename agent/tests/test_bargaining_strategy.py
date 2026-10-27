"""Tests for the bargaining strategy, including its guardrails: the 30-70% bounds,
the deadline safety zone, and handling of opponents who never concede.
"""

import pytest

import agent.rhetoric.opponent_profile as profile_mod
from agent.bargaining.strategy import HOPELESS_CONFIRMATION_ROUNDS, MIN_SELF_SHARE_FRAC, bargaining_strategy


def _isolate_concession_profile_store(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_mod, "_STORE_PATH", str(tmp_path / "opponent_concession_profile.json"))
    monkeypatch.setattr(profile_mod, "_CACHE", {})


def _offer_game(game_id, me, delta_me, delta_opp):
    delta_me_key = "delta_1" if me == "alice" else "delta_2"
    delta_opp_key = "delta_2" if me == "alice" else "delta_1"
    return {
        "game_id": game_id,
        "game_family": "bargaining",
        "your_player": me,
        "prompt": "Split the money.",
        "game_state": {
            "current_player": me,
            "proposer": me,
            "round": 1,
            "max_rounds": None,
            "horizon_known": False,
            "money_to_divide": 1_000_000.0,
            delta_me_key: delta_me,
            delta_opp_key: delta_opp,
            "complete_information": True,
            "messages_allowed": False,
            "last_offer": None,
        },
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
    }


def _decision_game(game_id, me, delta_me, delta_opp, my_gain, money=1_000_000.0):
    opp = "bob" if me == "alice" else "alice"
    delta_me_key = "delta_1" if me == "alice" else "delta_2"
    delta_opp_key = "delta_2" if me == "alice" else "delta_1"
    return {
        "game_id": game_id,
        "game_family": "bargaining",
        "your_player": me,
        "prompt": "Respond to the offer.",
        "game_state": {
            "current_player": me,
            "proposer": opp,
            "round": 1,
            "max_rounds": None,
            "horizon_known": False,
            "money_to_divide": money,
            delta_me_key: delta_me,
            delta_opp_key: delta_opp,
            "complete_information": True,
            "messages_allowed": False,
            "last_offer": {
                "alice_gain": my_gain if me == "alice" else money - my_gain,
                "bob_gain": my_gain if me == "bob" else money - my_gain,
                "proposer": opp,
                "round": 1,
            },
        },
        "valid_actions": {"type": "decision", "fields": {"decision": "'accept', 'reject', or 'walkaway'"}},
    }


def test_never_offers_away_the_entire_pot_to_an_infinitely_patient_opponent():
    game = _offer_game("barg-floor-offer", me="alice", delta_me=0.95, delta_opp=1.0)
    action = bargaining_strategy(game)
    assert action["alice_gain"] >= 0.30 * 1_000_000.0 - 1e-6


def test_never_accepts_next_to_nothing_from_an_infinitely_patient_opponent():
    game = _decision_game("barg-floor-decision", me="bob", delta_me=0.95, delta_opp=1.0, my_gain=50_000.0)
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_still_accepts_a_generous_offer_above_the_floor():
    game = _decision_game("barg-floor-decision-2", me="bob", delta_me=0.95, delta_opp=1.0, my_gain=700_000.0)
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_never_demands_the_entire_pot_when_we_are_the_infinitely_patient_side():
    game = _offer_game("barg-ceiling-offer", me="bob", delta_me=1.0, delta_opp=0.95)
    action = bargaining_strategy(game)
    assert action["bob_gain"] <= 0.70 * 1_000_000.0 + 1e-6


def test_never_requires_the_entire_pot_to_accept_when_we_are_the_infinitely_patient_side():
    game = _decision_game("barg-ceiling-decision", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=700_000.0)
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_panic_zone_accepts_a_net_positive_offer_even_below_the_normal_threshold():
    game = _decision_game("barg-panic-accept", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=612_531.573)
    game["game_state"]["round"] = 99
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_panic_zone_still_rejects_a_zero_or_negative_offer():
    game = _decision_game("barg-panic-reject-zero", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=0.0)
    game["game_state"]["round"] = 99
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_panic_zone_not_triggered_below_the_threshold_round():
    game = _decision_game("barg-not-yet-panic", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=400_000.0)
    game["game_state"]["round"] = 89
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_panic_zone_offer_mirrors_opponents_own_last_revealed_position():
    game = _offer_game("barg-panic-offer-mirror", me="bob", delta_me=1.0, delta_opp=0.95)
    game["game_state"]["round"] = 95
    game["game_state"]["last_offer"] = {
        "alice_gain": 387_468.427,
        "bob_gain": 612_531.573,
        "proposer": "alice",
        "round": 94,
    }
    action = bargaining_strategy(game)
    assert abs(action["alice_gain"] - 387_468.427) < 1e-6
    assert abs(action["bob_gain"] - 612_531.573) < 1e-6


def test_panic_zone_offer_splits_down_the_middle_with_no_revealed_opponent_position():
    game = _offer_game("barg-panic-offer-nohistory", me="bob", delta_me=1.0, delta_opp=0.95)
    game["game_state"]["round"] = 95
    action = bargaining_strategy(game)
    assert abs(action["bob_gain"] - 500_000.0) < 1e-6


def test_never_counter_offers_ourselves_less_than_the_standing_offer():
    game = _offer_game("barg-standing-offer-floor", me="bob", delta_me=0.9, delta_opp=0.95)
    game["game_state"]["last_offer"] = {
        "alice_gain": 350_000.0,
        "bob_gain": 650_000.0,
        "proposer": "alice",
        "round": 1,
    }
    action = bargaining_strategy(game)
    assert action["bob_gain"] >= 650_000.0 - 1e-6


def test_incomplete_information_finite_horizon_decision_does_not_crash():
    game = _decision_game("barg-crash-check", me="bob", delta_me=0.9, delta_opp=0.8, my_gain=400_000.0)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 3
    game["game_state"]["complete_information"] = False
    del game["game_state"]["delta_1"]
    action = bargaining_strategy(game)
    assert action["decision"] in ("accept", "reject")


def test_ceiling_bound_offer_softens_toward_midpoint_over_time():
    money = 1_000_000.0
    early = _offer_game("barg-ceiling-soften-early", me="bob", delta_me=1.0, delta_opp=0.95)
    early["game_state"]["round"] = 1
    late = _offer_game("barg-ceiling-soften-late", me="bob", delta_me=1.0, delta_opp=0.95)
    late["game_state"]["round"] = 60

    early_gain = bargaining_strategy(early)["bob_gain"]
    late_gain = bargaining_strategy(late)["bob_gain"]
    assert early_gain == pytest.approx(698_484.54, abs=1.0)
    assert late_gain < early_gain
    assert late_gain > 0.50 * money


def test_floor_bound_offer_softens_over_time_even_in_finite_horizon():
    money = 100.0
    game = _offer_game("barg-floor-finite-early", me="alice", delta_me=0.95, delta_opp=1.0)
    game["game_state"]["money_to_divide"] = money
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 1
    early_take = bargaining_strategy(game)["alice_gain"]

    late_game = _offer_game("barg-floor-finite-late", me="alice", delta_me=0.95, delta_opp=1.0)
    late_game["game_state"]["money_to_divide"] = money
    late_game["game_state"]["max_rounds"] = 12
    late_game["game_state"]["horizon_known"] = True
    late_game["game_state"]["round"] = 8
    late_take = bargaining_strategy(late_game)["alice_gain"]

    assert early_take > 0.65 * money
    assert late_take < early_take
    assert late_take >= MIN_SELF_SHARE_FRAC * money - 1e-6


def test_unlimited_horizon_offer_never_drops_below_the_min_floor_while_softening():
    money = 1_000_000.0
    game = _offer_game("barg-concession-floor-safety", me="bob", delta_me=0.5, delta_opp=0.5)
    game["game_state"]["round"] = 89
    take = bargaining_strategy(game)["bob_gain"]
    assert take >= MIN_SELF_SHARE_FRAC * money - 1e-6


def test_unlimited_horizon_accept_threshold_also_softens_over_time():
    money = 1_000_000.0
    offer_at_60_percent = 600_000.0

    early = _decision_game("barg-accept-soften-early", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=offer_at_60_percent)
    early["game_state"]["round"] = 1
    assert bargaining_strategy(early)["decision"] == "reject"

    late = _decision_game("barg-accept-soften-late", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=offer_at_60_percent)
    late["game_state"]["round"] = 60
    assert bargaining_strategy(late)["decision"] == "accept"


def test_belief_update_uses_damped_predictions_not_oscillating_raw_ones(monkeypatch):
    from agent.bargaining import belief as belief_mod
    from agent.bargaining import strategy as strategy_mod

    monkeypatch.setattr(strategy_mod, "DISABLE_DELTA_BELIEF", False)
    game = _decision_game(
        "barg-belief-damped", me="bob", delta_me=0.8, delta_opp=0.95, my_gain=2_500.0, money=10_000.0
    )
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 1
    game["game_state"]["complete_information"] = False
    del game["game_state"]["delta_1"]

    bargaining_strategy(game)

    mean = belief_mod.get_belief("barg-belief-damped").mean()
    assert mean < 0.94


def test_finite_horizon_panic_zone_offer_mirrors_opponent_near_the_real_deadline():
    game = _offer_game("barg-finite-panic-mirror", me="bob", delta_me=1.0, delta_opp=0.95)
    game["game_state"]["money_to_divide"] = 100.0
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 11
    game["game_state"]["last_offer"] = {
        "alice_gain": 60.0,
        "bob_gain": 40.0,
        "proposer": "alice",
        "round": 10,
    }
    action = bargaining_strategy(game)
    assert abs(action["bob_gain"] - 40.0) < 1e-6


def test_finite_horizon_panic_zone_decision_accepts_any_net_positive_offer():
    game = _decision_game("barg-finite-panic-accept", me="bob", delta_me=0.9, delta_opp=0.95, my_gain=5.0, money=100.0)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 11
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_degenerate_delta_me_does_not_panic_accept_early_in_finite_horizon():
    game = _decision_game(
        "barg-degenerate-no-early-panic", me="bob", delta_me=1.0, delta_opp=0.9, my_gain=39.0, money=100.0
    )
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 10
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_degenerate_delta_me_panic_accepts_at_rounds_remaining_one_when_that_is_the_true_last_chance():
    game = _decision_game(
        "barg-degenerate-panic-at-last-decision", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=43.0, money=100.0
    )
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 11
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_not_conceding_override_needs_more_confirmations_when_our_own_delta_is_degenerate():
    game = _decision_game(
        "barg-degenerate-not-conceding-holds-out", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=430000.0, money=1_000_000.0
    )
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 5
    game["game_state"]["last_offer"] = {"alice_gain": 570000.0, "bob_gain": 430000.0, "proposer": "alice", "round": 5}
    game["game_state"]["history"] = _stationary_offer_history("alice", opp_self_take=570000.0, rounds=[1, 3])
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_not_conceding_override_eventually_fires_with_enough_confirmations_when_degenerate():
    money = 1_000_000.0
    game = _decision_game(
        "barg-degenerate-not-conceding-eventually-fires", me="bob", delta_me=1.0, delta_opp=0.95,
        my_gain=500_000.0, money=money,
    )
    game["game_state"]["round"] = 14
    game["game_state"]["last_offer"] = {"alice_gain": 500_000.0, "bob_gain": 500_000.0, "proposer": "alice", "round": 14}
    game["game_state"]["history"] = _stationary_offer_history("alice", opp_self_take=500_000.0, rounds=[2, 4, 6, 8, 10, 12])
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_degenerate_delta_me_still_locks_in_a_stationary_offer_at_the_true_last_decision():
    game = _decision_game(
        "barg-degenerate-locks-in-at-last-decision", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=430000.0, money=1_000_000.0
    )
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 11
    game["game_state"]["last_offer"] = {"alice_gain": 570000.0, "bob_gain": 430000.0, "proposer": "alice", "round": 11}
    game["game_state"]["history"] = _stationary_offer_history("alice", opp_self_take=570000.0, rounds=[1, 3, 5, 7, 9])
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_degenerate_delta_me_still_panic_accepts_at_the_true_literal_deadline():
    game = _decision_game(
        "barg-degenerate-panic-at-deadline", me="bob", delta_me=1.0, delta_opp=0.9, my_gain=5.0, money=100.0
    )
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 12
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_finite_horizon_panic_zone_not_triggered_with_ample_rounds_left():
    game = _offer_game("barg-finite-not-yet-panic", me="bob", delta_me=1.0, delta_opp=0.95)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 3
    action = bargaining_strategy(game)
    assert action["bob_gain"] == pytest.approx(684_533.74, abs=1.0)


def _stationary_offer_history(opp: str, opp_self_take: float, rounds: list[int]) -> list[dict]:
    opp_gain_key = "alice_gain" if opp == "alice" else "bob_gain"
    me_gain_key = "bob_gain" if opp == "alice" else "alice_gain"
    return [
        {
            "speaker": "opponent", "round": r,
            "offer": {opp_gain_key: opp_self_take, me_gain_key: 100.0 - opp_self_take, "proposer": opp, "round": r},
        }
        for r in rounds
    ]


def test_accepts_immediately_when_opponent_offer_has_been_stationary_for_several_rounds():
    game = _decision_game("barg-stationary-accept", me="alice", delta_me=0.95, delta_opp=0.90, my_gain=53.0, money=100.0)
    game["game_state"]["round"] = 4
    game["game_state"]["last_offer"] = {"alice_gain": 53.0, "bob_gain": 47.0, "proposer": "bob", "round": 4}
    game["game_state"]["history"] = _stationary_offer_history("bob", opp_self_take=47.0, rounds=[1, 2, 3])
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_stationary_detection_fires_against_the_real_server_history_shape():
    game = _decision_game("barg-stationary-server-shape", me="alice", delta_me=0.95, delta_opp=0.90, my_gain=53.0, money=100.0)
    game["game_state"]["round"] = 4
    game["game_state"]["last_offer"] = {"alice_gain": 53.0, "bob_gain": 47.0, "proposer": "bob", "round": 4}
    game["game_state"]["history"] = [
        {
            "round": r, "proposer": "bob",
            "offer": {"alice_gain": 53.0, "bob_gain": 47.0, "proposer": "bob", "round": r},
            "decision": "reject", "response_time_ms": 1000,
        }
        for r in [1, 2, 3]
    ]
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_stationary_detection_includes_the_current_offer_not_just_prior_history():
    from agent.bargaining.strategy import _opponent_not_conceding

    game = _decision_game("barg-stationary-timing", me="alice", delta_me=0.95, delta_opp=0.90, my_gain=53.0, money=100.0)
    game["game_state"]["round"] = 4
    game["game_state"]["history"] = _stationary_offer_history("bob", opp_self_take=47.0, rounds=[2, 3])
    assert _opponent_not_conceding(game, "bob", 100.0, current_opp_gain=47.0)
    assert not _opponent_not_conceding(game, "bob", 100.0, current_opp_gain=30.0)


def test_stationary_detection_still_respects_the_minimum_share_floor():
    game = _decision_game("barg-stationary-lowball", me="alice", delta_me=0.95, delta_opp=0.90, my_gain=10.0, money=100.0)
    game["game_state"]["round"] = 4
    game["game_state"]["last_offer"] = {"alice_gain": 10.0, "bob_gain": 90.0, "proposer": "bob", "round": 4}
    game["game_state"]["history"] = _stationary_offer_history("bob", opp_self_take=90.0, rounds=[1, 2, 3])
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_stationary_detection_does_not_trigger_when_the_opponent_is_actually_moving():
    game = _decision_game("barg-not-stationary", me="alice", delta_me=0.95, delta_opp=0.90, my_gain=53.0, money=100.0)
    game["game_state"]["round"] = 4
    game["game_state"]["last_offer"] = {"alice_gain": 53.0, "bob_gain": 47.0, "proposer": "bob", "round": 4}
    game["game_state"]["history"] = [
        {"speaker": "opponent", "round": 1, "offer": {"alice_gain": 20.0, "bob_gain": 80.0, "proposer": "bob", "round": 1}},
        {"speaker": "opponent", "round": 2, "offer": {"alice_gain": 35.0, "bob_gain": 65.0, "proposer": "bob", "round": 2}},
        {"speaker": "opponent", "round": 3, "offer": {"alice_gain": 45.0, "bob_gain": 55.0, "proposer": "bob", "round": 3}},
    ]
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_not_conceding_fires_when_opponent_trends_worse_for_us_without_ever_going_flat():
    game = _decision_game("barg-trending-worse", me="bob", delta_me=0.95, delta_opp=0.90, my_gain=5366.261, money=10000.0)
    game["game_state"]["round"] = 17
    game["game_state"]["last_offer"] = {"alice_gain": 4667.668, "bob_gain": 5332.332, "proposer": "alice", "round": 17}
    game["game_state"]["history"] = [
        {
            "round": r, "proposer": "alice",
            "offer": {"alice_gain": alice_gain, "bob_gain": 10000.0 - alice_gain, "proposer": "alice", "round": r},
            "decision": "reject", "response_time_ms": 1000,
        }
        for r, alice_gain in [(13, 4633.739), (15, 4667.668)]
    ]
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_not_conceding_does_not_fire_on_only_two_data_points():
    from agent.bargaining.strategy import _opponent_not_conceding

    game = _decision_game("barg-not-enough-data", me="bob", delta_me=0.95, delta_opp=0.90, my_gain=53.0, money=100.0)
    game["game_state"]["history"] = _stationary_offer_history("alice", opp_self_take=47.0, rounds=[1])
    assert not _opponent_not_conceding(game, "alice", 100.0, current_opp_gain=47.0)


def test_finite_horizon_offer_is_not_capped_at_70_percent_when_non_degenerate():
    money = 1_000_000.0
    game = _offer_game("barg-uncapped-offer", me="alice", delta_me=0.7, delta_opp=0.5)
    game["game_state"]["money_to_divide"] = money
    game["game_state"]["max_rounds"] = 7
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 1
    action = bargaining_strategy(game)
    assert action["alice_gain"] > 0.72 * money
    assert action["alice_gain"] < money


def test_finite_horizon_accept_threshold_is_not_capped_at_70_percent_when_non_degenerate():
    money = 1_000_000.0
    game = _decision_game(
        "barg-uncapped-threshold", me="bob", delta_me=0.85, delta_opp=0.5,
        my_gain=0.72 * money, money=money,
    )
    game["game_state"]["max_rounds"] = 7
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 1
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"

    generous = _decision_game(
        "barg-uncapped-threshold-generous", me="bob", delta_me=0.85, delta_opp=0.5,
        my_gain=0.76 * money, money=money,
    )
    generous["game_state"]["max_rounds"] = 7
    generous["game_state"]["horizon_known"] = True
    generous["game_state"]["round"] = 1
    assert bargaining_strategy(generous)["decision"] == "accept"


def test_finite_horizon_accept_threshold_also_softens_toward_midpoint():
    money = 100.0
    offer_at_60_percent = 60.0

    early = _decision_game("barg-finite-accept-early", me="alice", delta_me=0.95, delta_opp=1.0, my_gain=offer_at_60_percent, money=money)
    early["game_state"]["max_rounds"] = 12
    early["game_state"]["horizon_known"] = True
    early["game_state"]["round"] = 1
    assert bargaining_strategy(early)["decision"] == "reject"

    late = _decision_game("barg-finite-accept-late", me="alice", delta_me=0.95, delta_opp=1.0, my_gain=offer_at_60_percent, money=money)
    late["game_state"]["max_rounds"] = 12
    late["game_state"]["horizon_known"] = True
    late["game_state"]["round"] = 8
    assert bargaining_strategy(late)["decision"] == "accept"


def test_style_bandit_arm_is_passed_to_compose_message_on_an_offer(monkeypatch):
    from agent.rhetoric.bandit import StyleBandit

    captured = {}

    def fake_compose_message(**kwargs):
        captured.update(kwargs)
        return "a message"

    monkeypatch.setattr("agent.bargaining.strategy.compose_message", fake_compose_message)
    game = _offer_game("barg-style-offer", me="alice", delta_me=0.9, delta_opp=0.9)
    game["game_state"]["messages_allowed"] = True
    action = bargaining_strategy(game)
    assert captured["style"] in StyleBandit.ARMS
    assert action["message"] == "a message"

    from agent.bargaining import strategy as strategy_mod
    assert strategy_mod._LAST_STYLE_ARM["barg-style-offer"] == captured["style"]


def test_no_style_selected_when_messages_not_allowed():
    game = _offer_game("barg-no-style-offer", me="alice", delta_me=0.9, delta_opp=0.9)
    game["game_state"]["messages_allowed"] = False
    action = bargaining_strategy(game)
    assert "message" not in action

    from agent.bargaining import strategy as strategy_mod
    assert "barg-no-style-offer" not in strategy_mod._LAST_STYLE_ARM


def _opponent_offer_history_entries(opp: str, opp_self_takes: list[float], me_gain_total: float = 100.0) -> list[dict]:
    return [
        {
            "round": r,
            "proposer": opp,
            "offer": {
                f"{opp}_gain": take,
                "bob_gain" if opp == "alice" else "alice_gain": me_gain_total - take,
                "proposer": opp,
                "round": r,
            },
            "decision": "reject",
        }
        for r, take in enumerate(opp_self_takes, start=1)
    ]


def test_style_bandit_credits_success_when_opponent_moves_toward_us():
    from agent.bargaining import strategy as strategy_mod

    game_id = "barg-style-credit-success"
    strategy_mod._LAST_STYLE_ARM[game_id] = "fairness"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    alpha_before = bandit.alpha["fairness"]

    game = _decision_game(game_id, me="bob", delta_me=0.9, delta_opp=0.9, my_gain=55.0, money=100.0)
    game["game_state"]["last_offer"] = {"alice_gain": 45.0, "bob_gain": 55.0, "proposer": "alice", "round": 3}
    game["game_state"]["history"] = _opponent_offer_history_entries("alice", [50.0])
    bargaining_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).alpha["fairness"] == alpha_before + 1.0
    assert game_id not in strategy_mod._LAST_STYLE_ARM


def test_style_bandit_credits_failure_when_opponent_does_not_move():
    from agent.bargaining import strategy as strategy_mod

    game_id = "barg-style-credit-failure"
    strategy_mod._LAST_STYLE_ARM[game_id] = "authority"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    beta_before = bandit.beta["authority"]

    game = _decision_game(game_id, me="bob", delta_me=0.9, delta_opp=0.9, my_gain=50.0, money=100.0)
    game["game_state"]["last_offer"] = {"alice_gain": 50.0, "bob_gain": 50.0, "proposer": "alice", "round": 3}
    game["game_state"]["history"] = _opponent_offer_history_entries("alice", [50.0])
    bargaining_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).beta["authority"] == beta_before + 1.0


def test_style_bandit_does_not_credit_success_for_noise_level_movement():
    from agent.bargaining import strategy as strategy_mod

    game_id = "barg-style-credit-noise-movement"
    strategy_mod._LAST_STYLE_ARM[game_id] = "purity"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    alpha_before, beta_before = bandit.alpha["purity"], bandit.beta["purity"]

    game = _decision_game(game_id, me="bob", delta_me=0.9, delta_opp=0.9, my_gain=50.10, money=100.0)
    game["game_state"]["last_offer"] = {"alice_gain": 49.90, "bob_gain": 50.10, "proposer": "alice", "round": 3}
    game["game_state"]["history"] = _opponent_offer_history_entries("alice", [50.0])
    bargaining_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).alpha["purity"] == alpha_before
    assert strategy_mod._style_bandit({"game_id": game_id}).beta["purity"] == beta_before + 1.0
    assert game_id not in strategy_mod._LAST_STYLE_ARM


def test_style_bandit_does_not_credit_with_no_prior_opponent_offer():
    from agent.bargaining import strategy as strategy_mod

    game_id = "barg-style-credit-insufficient-history"
    strategy_mod._LAST_STYLE_ARM[game_id] = "purity"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    alpha_before, beta_before = bandit.alpha["purity"], bandit.beta["purity"]

    game = _decision_game(game_id, me="bob", delta_me=0.9, delta_opp=0.9, my_gain=50.0, money=100.0)
    game["game_state"]["last_offer"] = {"alice_gain": 50.0, "bob_gain": 50.0, "proposer": "alice", "round": 1}
    game["game_state"]["history"] = []
    bargaining_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).alpha["purity"] == alpha_before
    assert strategy_mod._style_bandit({"game_id": game_id}).beta["purity"] == beta_before
    assert game_id not in strategy_mod._LAST_STYLE_ARM


def test_not_conceding_override_requires_a_fraction_of_the_real_threshold_not_just_the_bare_floor():
    game = _decision_game("barg-not-conceding-below-real-threshold", me="bob", delta_me=0.95, delta_opp=0.80, my_gain=4444.44, money=10000.0)
    game["game_state"]["last_offer"] = {"alice_gain": 5555.56, "bob_gain": 4444.44, "proposer": "alice", "round": 5}
    game["game_state"]["history"] = _stationary_offer_history("alice", opp_self_take=5555.56, rounds=[1, 3])
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_not_conceding_override_still_fires_when_offer_is_close_to_the_real_threshold():
    game = _decision_game("barg-not-conceding-close-to-real-threshold", me="bob", delta_me=0.95, delta_opp=0.95, my_gain=380469.0, money=1_000_000.0)
    game["game_state"]["last_offer"] = {"alice_gain": 619531.0, "bob_gain": 380469.0, "proposer": "alice", "round": 5}
    game["game_state"]["history"] = _stationary_offer_history("alice", opp_self_take=619999.0, rounds=[1]) + [
        {
            "round": 3, "proposer": "alice",
            "offer": {"alice_gain": 619939.0, "bob_gain": 380061.0, "proposer": "alice", "round": 3},
            "decision": "reject", "response_time_ms": 1000,
        },
    ]
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_degenerate_offer_no_longer_stays_pinned_across_a_real_live_sequence():
    early = _offer_game("barg-degenerate-live-early", me="alice", delta_me=1.0, delta_opp=0.9)
    early["game_state"]["max_rounds"] = 12
    early["game_state"]["horizon_known"] = False
    early["game_state"]["round"] = 1

    late = _offer_game("barg-degenerate-live-late", me="alice", delta_me=1.0, delta_opp=0.9)
    late["game_state"]["max_rounds"] = 12
    late["game_state"]["horizon_known"] = False
    late["game_state"]["round"] = 9

    early_gain = bargaining_strategy(early)["alice_gain"]
    late_gain = bargaining_strategy(late)["alice_gain"]
    assert early_gain == pytest.approx(0.70 * 1_000_000.0, abs=2_000.0)
    assert late_gain < early_gain


def test_finite_horizon_offer_no_longer_cliffs_at_the_degenerate_delta_boundary():
    money = 100.0
    prev_take = None
    max_jump = 0.0
    steps = 400
    for i in range(steps + 1):
        delta_me = 0.98 + (1.0 - 0.98) * i / steps
        game = _offer_game("barg-degenerate-continuity-sweep", me="alice", delta_me=delta_me, delta_opp=0.7)
        game["game_state"]["money_to_divide"] = money
        game["game_state"]["max_rounds"] = 12
        game["game_state"]["horizon_known"] = True
        game["game_state"]["round"] = 1
        take = bargaining_strategy(game)["alice_gain"]
        if prev_take is not None:
            max_jump = max(max_jump, abs(take - prev_take))
        prev_take = take
    assert max_jump < 0.5


def test_finite_horizon_offer_still_matches_old_behavior_well_below_the_threshold():
    game = _offer_game("barg-degenerate-blend-noop", me="alice", delta_me=0.9, delta_opp=0.5)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 1
    take = bargaining_strategy(game)["alice_gain"]
    assert take > 0.70 * 1_000_000.0


def test_finite_horizon_offer_still_hits_the_exact_degenerate_value_at_the_threshold():
    game = _offer_game("barg-degenerate-blend-full", me="alice", delta_me=1.0, delta_opp=0.9)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 1
    take = bargaining_strategy(game)["alice_gain"]
    assert take == pytest.approx(0.70 * 1_000_000.0, abs=2_000.0)


def test_confirmed_stationarity_speeds_up_incomplete_information_belief_convergence():
    from agent.bargaining.strategy import bargaining_strategy as strategy_fn

    money = 100.0
    hist = []
    alice_asks = {1: 58.0, 3: 58.0, 5: 54.0, 7: 54.0, 9: 54.0, 11: 54.0, 13: 54.0, 15: 54.0, 17: 54.0, 19: 54.0}
    accepted_round = None
    for r in sorted(alice_asks):
        alice_gain = alice_asks[r]
        bob_gain = money - alice_gain
        game = {
            "game_id": "barg-belief-speedup-live-replay",
            "game_family": "bargaining",
            "your_player": "bob",
            "prompt": "Respond to the offer.",
            "game_state": {
                "current_player": "bob", "proposer": "alice", "round": r, "max_rounds": None,
                "horizon_known": False, "money_to_divide": money, "delta_2": 0.95,
                "complete_information": False, "messages_allowed": True,
                "last_offer": {"alice_gain": alice_gain, "bob_gain": bob_gain, "proposer": "alice", "round": r},
                "history": list(hist),
            },
            "valid_actions": {"type": "decision", "fields": {"decision": "'accept' or 'reject'"}},
        }
        action = strategy_fn(game)
        if action["decision"] == "accept":
            accepted_round = r
            break
        hist.append({"round": r, "proposer": "alice", "offer": game["game_state"]["last_offer"], "decision": "reject", "response_time_ms": 1000})

    assert accepted_round is not None
    assert accepted_round < 19


def _offer_game_with_state(game_id, me, delta_me, delta_opp, round_no, max_rounds, last_offer, history):
    delta_me_key = "delta_1" if me == "alice" else "delta_2"
    delta_opp_key = "delta_2" if me == "alice" else "delta_1"
    return {
        "game_id": game_id,
        "game_family": "bargaining",
        "your_player": me,
        "prompt": "Split the money.",
        "game_state": {
            "current_player": me,
            "proposer": me,
            "round": round_no,
            "max_rounds": max_rounds,
            "horizon_known": max_rounds is not None,
            "money_to_divide": 100.0,
            delta_me_key: delta_me,
            delta_opp_key: delta_opp,
            "complete_information": True,
            "messages_allowed": False,
            "last_offer": last_offer,
            "history": history,
        },
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
    }


def test_offer_concession_is_dampened_against_a_non_reciprocating_opponent():
    game_id = "barg-damping-non-reciprocating"
    money = 100.0
    bob_asks = {2: 87.56, 4: 88.89, 6: 90.65, 8: 92.96}
    hist = []
    action = None
    for r in [1, 3, 5, 7, 9]:
        last_offer = None
        if r > 1:
            bob_ask = bob_asks[r - 1]
            last_offer = {"alice_gain": money - bob_ask, "bob_gain": bob_ask, "proposer": "bob", "round": r - 1}
        game = _offer_game_with_state(game_id, "alice", 0.80, 0.95, r, 12, last_offer, list(hist))
        action = bargaining_strategy(game)
        if last_offer:
            hist.append({"round": r - 1, "proposer": "bob", "offer": last_offer, "decision": "reject", "response_time_ms": 1000})
        hist.append({
            "round": r, "proposer": "alice",
            "offer": {"alice_gain": action["alice_gain"], "bob_gain": action["bob_gain"], "proposer": "alice", "round": r},
            "decision": "reject", "response_time_ms": 1000,
        })

    assert action["alice_gain"] > 47.587 + 5.0


def test_offer_concession_is_not_dampened_against_a_genuinely_conceding_opponent():
    game_id = "barg-damping-genuine-concession"
    money = 100.0
    bob_asks = {2: 73.0, 4: 66.0, 6: 59.0}
    hist = []
    undamped_action = None
    for r in [1, 3, 5, 7]:
        last_offer = None
        if r > 1:
            bob_ask = bob_asks[r - 1]
            last_offer = {"alice_gain": money - bob_ask, "bob_gain": bob_ask, "proposer": "bob", "round": r - 1}
        game = _offer_game_with_state(game_id, "alice", 1.0, 0.9, r, 12, last_offer, list(hist))
        action = bargaining_strategy(game)
        if last_offer:
            hist.append({"round": r - 1, "proposer": "bob", "offer": last_offer, "decision": "reject", "response_time_ms": 1000})
        hist.append({
            "round": r, "proposer": "alice",
            "offer": {"alice_gain": action["alice_gain"], "bob_gain": action["bob_gain"], "proposer": "alice", "round": r},
            "decision": "reject", "response_time_ms": 1000,
        })
        undamped_action = action

    from agent.negotiation.solver import concession_target, time_fraction
    t = time_fraction(7, 12, unlimited_tau=30.0)
    expected = concession_target(70.0, 50.0, t, 0.7)
    assert undamped_action["alice_gain"] == pytest.approx(expected, abs=0.5)


def test_offer_damping_does_not_apply_on_the_first_offer():
    game = _offer_game_with_state("barg-damping-first-offer", "alice", 0.80, 0.95, 1, 12, None, [])
    action = bargaining_strategy(game)
    assert action["alice_gain"] == pytest.approx(70.0, abs=0.5)


def test_offer_concession_is_dampened_via_history_when_last_offer_is_unresolvable():
    game_id = "barg-damping-last-offer-unresolvable"

    def make_game(round_no, hist):
        game = _offer_game_with_state(game_id, "alice", 1.0, 0.9, round_no, 12, None, list(hist))
        game["game_state"]["complete_information"] = False
        del game["game_state"]["delta_2"]
        return game

    hist = []
    action = bargaining_strategy(make_game(1, hist))
    hist.append({
        "round": 1, "proposer": "alice",
        "offer": {"alice_gain": action["alice_gain"], "bob_gain": action["bob_gain"], "proposer": "alice", "round": 1},
        "decision": "reject", "response_time_ms": 1000,
    })
    for bob_round, alice_round in [(2, 3), (4, 5), (6, 7)]:
        hist.append({
            "round": bob_round, "proposer": "bob",
            "offer": {"alice_gain": 50.0, "bob_gain": 50.0, "proposer": "bob", "round": bob_round},
            "decision": "reject", "response_time_ms": 1000,
        })
        action = bargaining_strategy(make_game(alice_round, hist))
        hist.append({
            "round": alice_round, "proposer": "alice",
            "offer": {"alice_gain": action["alice_gain"], "bob_gain": action["bob_gain"], "proposer": "alice", "round": alice_round},
            "decision": "reject", "response_time_ms": 1000,
        })

    assert action["alice_gain"] > 62.57002855431526 + 1.0


def test_resolve_opponent_last_ask_falls_back_to_history_when_last_offer_is_unusable():
    from agent.bargaining.strategy import _resolve_opponent_last_ask

    game = _offer_game_with_state(
        "barg-resolve-ask-fallback", "alice", 1.0, None, 5, 12, None,
        [{"round": 4, "proposer": "bob", "offer": {"alice_gain": 50.0, "bob_gain": 50.0, "proposer": "bob", "round": 4}}],
    )
    assert _resolve_opponent_last_ask(game, "bob") == 50.0


def test_resolve_opponent_last_ask_prefers_last_offer_when_it_is_usable():
    from agent.bargaining.strategy import _resolve_opponent_last_ask

    game = _offer_game_with_state(
        "barg-resolve-ask-last-offer", "alice", 1.0, None, 5, 12,
        {"alice_gain": 40.0, "bob_gain": 60.0, "proposer": "bob", "round": 4},
        [{"round": 4, "proposer": "bob", "offer": {"alice_gain": 50.0, "bob_gain": 50.0, "proposer": "bob", "round": 4}}],
    )
    assert _resolve_opponent_last_ask(game, "bob") == 60.0


def test_panic_zone_requires_a_meaningful_floor_before_the_true_last_chance():
    game = _decision_game("barg-panic-floor-reject-token-offer", me="bob", delta_me=0.95, delta_opp=1.0, my_gain=0.01, money=1_000_000.0)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 10
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_panic_zone_still_accepts_anything_positive_at_the_true_last_chance():
    game = _decision_game("barg-panic-floor-true-deadline-accepts", me="bob", delta_me=0.95, delta_opp=1.0, my_gain=0.01, money=1_000_000.0)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 12
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_panic_zone_floor_also_applies_mid_unlimited_horizon_panic_zone():
    game = _decision_game("barg-panic-floor-unlimited-mid", me="bob", delta_me=0.95, delta_opp=0.95, my_gain=1.0, money=100.0)
    game["game_state"]["round"] = 92
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_panic_zone_floor_relaxes_near_the_absolute_deadline_round():
    game = _decision_game("barg-panic-floor-unlimited-near-cap", me="bob", delta_me=0.95, delta_opp=0.95, my_gain=1.0, money=100.0)
    game["game_state"]["round"] = 98
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_panic_zone_offer_falls_back_to_history_when_last_offer_is_unusable():
    game = _offer_game("barg-panic-mirror-history-fallback", me="bob", delta_me=0.95, delta_opp=1.0)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 10
    game["game_state"]["last_offer"] = None
    game["game_state"]["history"] = [
        {"round": 9, "proposer": "alice", "offer": {"alice_gain": 75.0, "bob_gain": 25.0, "proposer": "alice", "round": 9}, "decision": "reject"},
    ]
    game["game_state"]["money_to_divide"] = 100.0
    action = bargaining_strategy(game)
    assert action["bob_gain"] == pytest.approx(25.0, abs=0.01)
    assert action["alice_gain"] == pytest.approx(75.0, abs=0.01)


def test_offer_side_panic_zone_also_respects_the_degenerate_delta_exclusion():
    game = _offer_game("barg-offer-panic-degenerate-exclusion", me="bob", delta_me=1.0, delta_opp=0.95)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 10
    action = bargaining_strategy(game)
    assert action["bob_gain"] == pytest.approx(567_399.0, abs=50.0)
    assert action["bob_gain"] != pytest.approx(500_000.0, abs=1.0)


def test_offer_side_panic_zone_still_engages_at_the_true_last_chance_even_when_degenerate():
    game = _offer_game("barg-offer-panic-degenerate-true-last-chance", me="bob", delta_me=1.0, delta_opp=0.95)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 11
    action = bargaining_strategy(game)
    assert action["bob_gain"] == pytest.approx(500_000.0, abs=1.0)


def test_soften_offer_floor_erode_over_time_does_not_change_early_round_behavior():
    from agent.bargaining.strategy import _soften_offer_floor

    result = _soften_offer_floor(
        69.286, 30.0, 70.0, 70.0, round_no=2, max_rounds=None, money=100.0,
        force_soften=False, erode_over_time=True,
    )
    assert result == pytest.approx(69.286, abs=0.01)


def test_soften_offer_floor_erode_over_time_lowers_a_frozen_offer_after_many_rounds():
    from agent.bargaining.strategy import _soften_offer_floor

    early = _soften_offer_floor(
        69.286, 30.0, 70.0, 70.0, round_no=2, max_rounds=None, money=100.0,
        force_soften=False, erode_over_time=True,
    )
    late = _soften_offer_floor(
        69.286, 30.0, 70.0, 70.0, round_no=88, max_rounds=None, money=100.0,
        force_soften=False, erode_over_time=True,
    )
    assert late < early - 10.0


def test_soften_offer_floor_erode_over_time_is_a_noop_when_false():
    from agent.bargaining.strategy import _soften_offer_floor

    result = _soften_offer_floor(
        69.286, 30.0, 70.0, 70.0, round_no=88, max_rounds=None, money=100.0,
        force_soften=False, erode_over_time=False,
    )
    assert result == pytest.approx(69.286, abs=0.001)


def test_soften_threshold_ceiling_erode_over_time_lowers_a_frozen_threshold_after_many_rounds():
    from agent.bargaining.strategy import _soften_threshold_ceiling

    early = _soften_threshold_ceiling(
        64.29, 30.0, 70.0, 100.0, round_no=2, max_rounds=None,
        force_soften=False, erode_over_time=True,
    )
    late = _soften_threshold_ceiling(
        64.29, 30.0, 70.0, 100.0, round_no=88, max_rounds=None,
        force_soften=False, erode_over_time=True,
    )
    assert early == pytest.approx(64.29, abs=0.01)
    assert late < early - 5.0


def test_unlimited_horizon_offer_no_longer_freezes_forever_against_a_stationary_opponent():
    game_id = "barg-90round-offer-no-longer-frozen"
    money = 100.0

    def make_offer_game(round_no, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Split the money.",
            "game_state": {
                "current_player": "bob", "proposer": "bob", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False, "last_offer": None, "history": hist,
            },
            "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
        }

    round2_action = bargaining_strategy(make_offer_game(2, []))
    hist = [
        {"round": r, "proposer": "alice", "offer": {"alice_gain": 55.56, "bob_gain": 44.44, "proposer": "alice", "round": r}, "decision": "reject"}
        for r in range(1, 88, 2)
    ]
    round88_action = bargaining_strategy(make_offer_game(88, hist))
    assert round2_action["bob_gain"] == pytest.approx(69.286, abs=0.01)
    assert round88_action["bob_gain"] < round2_action["bob_gain"] - 4.0


def test_unlimited_horizon_decision_accepts_a_stationary_offer_much_earlier_than_the_deadline_safety_net():
    game_id = "barg-90round-decision-accepts-earlier"
    money = 100.0

    def make_decision_game(round_no, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Respond.",
            "game_state": {
                "current_player": "bob", "proposer": "alice", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False,
                "last_offer": {"alice_gain": 55.56, "bob_gain": 44.44, "proposer": "alice", "round": round_no},
                "history": hist,
            },
            "valid_actions": {"type": "decision", "fields": {"decision": "'accept' or 'reject'"}},
        }

    hist = []
    accepted_round = None
    for r in range(1, 90, 2):
        action = bargaining_strategy(make_decision_game(r, list(hist)))
        if action["decision"] == "accept":
            accepted_round = r
            break
        hist.append({"round": r, "proposer": "alice", "offer": {"alice_gain": 55.56, "bob_gain": 44.44, "proposer": "alice", "round": r}, "decision": "reject"})
        hist.append({"round": r + 1, "proposer": "bob", "offer": {"alice_gain": 30.714, "bob_gain": 69.286, "proposer": "bob", "round": r + 1}, "decision": "reject"})

    assert accepted_round is not None
    assert accepted_round < 89


def test_offer_side_erosion_does_not_fire_against_a_genuinely_conceding_opponent():
    game_id = "barg-erosion-not-against-moving-opponent"
    money = 100.0

    def make_offer_game(round_no, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Split the money.",
            "game_state": {
                "current_player": "bob", "proposer": "bob", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False, "last_offer": None, "history": hist,
            },
            "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
        }

    hist = []
    action = None
    for rnd, opp_ask in [(2, 60.0), (20, 55.0), (40, 50.0), (60, 46.0), (80, 42.0)]:
        game = make_offer_game(rnd, list(hist))
        action = bargaining_strategy(game)
        prior_round = rnd - 1 if rnd > 2 else 1
        hist.append({
            "round": prior_round, "proposer": "alice",
            "offer": {"alice_gain": opp_ask, "bob_gain": 100.0 - opp_ask, "proposer": "alice", "round": prior_round},
            "decision": "reject",
        })

    assert action["bob_gain"] == pytest.approx(69.28571428571429, abs=0.01)


def test_offer_side_erosion_fires_once_stationarity_is_strongly_confirmed():
    game_id = "barg-erosion-fires-after-strong-confirmation"
    money = 100.0

    def make_offer_game(round_no, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Split the money.",
            "game_state": {
                "current_player": "bob", "proposer": "bob", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False, "last_offer": None, "history": hist,
            },
            "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
        }

    hist = [
        {"round": r, "proposer": "alice", "offer": {"alice_gain": 55.56, "bob_gain": 44.44, "proposer": "alice", "round": r}, "decision": "reject"}
        for r in [2, 4, 6, 8, 10, 12, 14, 16]
    ]
    action = bargaining_strategy(make_offer_game(17, hist))
    assert action["bob_gain"] < 69.28571428571429 - 1.0


def test_decision_side_erosion_does_not_fire_against_a_genuinely_conceding_opponent():
    game_id = "barg-decision-erosion-not-against-moving-opponent"
    money = 100.0

    def make_decision_game(round_no, offer_price, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Respond.",
            "game_state": {
                "current_player": "bob", "proposer": "alice", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False,
                "last_offer": {"alice_gain": 100.0 - offer_price, "bob_gain": offer_price, "proposer": "alice", "round": round_no},
                "history": hist,
            },
            "valid_actions": {"type": "decision", "fields": {"decision": "'accept' or 'reject'"}},
        }

    from agent.bargaining.strategy import _reject_value, _soften_threshold_ceiling
    hist = []
    action = None
    for rnd, my_gain_offered in [(2, 40.0), (20, 45.0), (40, 50.0), (60, 54.0), (80, 58.0)]:
        game = make_decision_game(rnd, my_gain_offered, hist)
        action = bargaining_strategy(game)
        prior_round = rnd - 1 if rnd > 2 else 1
        prior_gain = my_gain_offered - 1.0
        hist.append({
            "round": prior_round, "proposer": "alice",
            "offer": {"alice_gain": 100.0 - prior_gain, "bob_gain": prior_gain, "proposer": "alice", "round": prior_round},
            "decision": "reject",
        })

    raw_threshold = _reject_value(money, "bob", 0.9, 0.8, None)
    unfloored = _soften_threshold_ceiling(
        raw_threshold, 30.0, 100.0, money, 80, None, force_soften=False, erode_over_time=False,
    )
    assert action["decision"] == ("accept" if 58.0 >= unfloored else "reject")


def test_offer_side_erosion_uses_a_faster_tau_than_the_ordinary_concession_curve():
    game_id = "barg-erosion-tau-is-faster"
    money = 100.0

    def make_offer_game(round_no, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Split the money.",
            "game_state": {
                "current_player": "bob", "proposer": "bob", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False, "last_offer": None, "history": hist,
            },
            "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
        }

    hist = [
        {"round": r, "proposer": "alice", "offer": {"alice_gain": 55.56, "bob_gain": 44.44, "proposer": "alice", "round": r}, "decision": "reject"}
        for r in [2, 4, 6, 8, 10, 12, 14, 16]
    ]
    action = bargaining_strategy(make_offer_game(17, hist))

    from agent.bargaining.strategy import CONCESSION_BETA, CONCESSION_TAU, _soften_offer_floor
    from agent.negotiation.solver import time_fraction

    raw_take = 69.28571428571429 * (1 - 0.03)
    old_pace = _soften_offer_floor(
        raw_take, 30.0, 70.0, 70.0, 17, None, money=money,
        force_soften=False, erode_over_time=True, erosion_tau=CONCESSION_TAU,
    )
    assert action["bob_gain"] < old_pace


def test_offer_side_damping_does_not_re_slow_a_confirmed_erosion_step():
    game_id = "barg-damping-does-not-re-slow-erosion"
    money = 10000.0

    def make_offer_game(round_no, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Split the money.",
            "game_state": {
                "current_player": "bob", "proposer": "bob", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False, "last_offer": None, "history": hist,
            },
            "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
        }

    stationary_offer = {"alice_gain": 5555.56, "bob_gain": 4444.44}
    hist = []
    actions = []
    for r in range(1, 14, 2):
        hist.append({
            "round": r, "proposer": "alice",
            "offer": {**stationary_offer, "proposer": "alice", "round": r}, "decision": "reject",
        })
        action = bargaining_strategy(make_offer_game(r + 1, list(hist)))
        actions.append(action["bob_gain"])
        hist.append({
            "round": r + 1, "proposer": "bob",
            "offer": {"alice_gain": money - action["bob_gain"], "bob_gain": action["bob_gain"], "proposer": "bob", "round": r + 1},
            "decision": "reject",
        })

    biggest_single_round_drop = max(a - b for a, b in zip(actions, actions[1:]))
    assert biggest_single_round_drop > 200.0


def _hopeless_offer_game(game_id, round_no, hist, opponent_name=None, money=1_000_000.0):
    game = {
        "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
        "prompt": "Split the money.",
        "game_state": {
            "current_player": "bob", "proposer": "bob", "round": round_no,
            "max_rounds": None, "horizon_known": False, "money_to_divide": money,
            "delta_1": 1.0, "delta_2": 0.8, "complete_information": True,
            "messages_allowed": False, "last_offer": None, "history": hist,
        },
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
    }
    if opponent_name:
        game["opponent"] = {"name": opponent_name}
    return game


def _stationary_hopeless_history(rounds):
    return [
        {"round": r, "proposer": "alice", "offer": {"alice_gain": 920_000.0, "bob_gain": 80_000.0, "proposer": "alice", "round": r}, "decision": "reject"}
        for r in rounds
    ]


def test_trusted_hardliner_profile_halves_the_confirmation_requirement(tmp_path, monkeypatch):
    _isolate_concession_profile_store(tmp_path, monkeypatch)
    from agent.rhetoric.opponent_profile import get_concession_profile, record_concession_evidence

    name = "Known Stonewaller"
    record_concession_evidence(name, "bargaining", True)
    record_concession_evidence(name, "bargaining", True)
    assert get_concession_profile(name, "bargaining").is_trusted_hardliner()

    assert HOPELESS_CONFIRMATION_ROUNDS == 14
    mid_rounds = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    hist_mid = _stationary_hopeless_history(mid_rounds)
    action_untrusted = bargaining_strategy(
        _hopeless_offer_game("barg-untrusted-mid", mid_rounds[-1] + 2, hist_mid, opponent_name=None)
    )
    action_trusted = bargaining_strategy(
        _hopeless_offer_game("barg-trusted-mid", mid_rounds[-1] + 2, hist_mid, opponent_name=name)
    )
    assert action_trusted["bob_gain"] < action_untrusted["bob_gain"]
    assert action_trusted["bob_gain"] == pytest.approx(80_000.0, abs=1.0)


def test_hardliner_confirmation_still_required_even_when_trusted(tmp_path, monkeypatch):
    _isolate_concession_profile_store(tmp_path, monkeypatch)
    from agent.rhetoric.opponent_profile import record_concession_evidence

    name = "Known Stonewaller"
    record_concession_evidence(name, "bargaining", True)
    record_concession_evidence(name, "bargaining", True)

    hist = _stationary_hopeless_history([1])
    action = bargaining_strategy(_hopeless_offer_game("barg-trusted-but-no-evidence", 3, hist, opponent_name=name))
    assert action["bob_gain"] != pytest.approx(80_000.0, abs=1.0)


def test_accepting_records_concession_evidence_for_the_named_opponent(tmp_path, monkeypatch):
    _isolate_concession_profile_store(tmp_path, monkeypatch)
    from agent.rhetoric.opponent_profile import get_concession_profile

    name = "Fresh Opponent"
    money = 100.0
    game = {
        "game_id": "barg-records-evidence-on-accept", "game_family": "bargaining", "your_player": "bob",
        "prompt": "Respond.", "opponent": {"name": name},
        "game_state": {
            "current_player": "bob", "proposer": "alice", "round": 1,
            "max_rounds": None, "horizon_known": False, "money_to_divide": money,
            "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
            "messages_allowed": False,
            "last_offer": {"alice_gain": 25.0, "bob_gain": 75.0, "proposer": "alice", "round": 1},
            "history": [],
        },
        "valid_actions": {"type": "decision", "fields": {"decision": "'accept' or 'reject'"}},
    }
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"
    profile = get_concession_profile(name, "bargaining")
    assert profile.games_observed == 1
    assert profile.games_confirmed_hopeless == 0
