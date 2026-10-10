import time

import pytest

import agent.dispatch as dispatch_mod
from agent.dispatch import _fallback_action, strategy


def _bargaining_offer_game():
    return {
        "game_id": "g1", "game_family": "bargaining", "your_player": "alice",
        "game_state": {"money_to_divide": 1000.0},
        "valid_actions": {"type": "offer", "fields": {}},
    }


def _bargaining_decision_game():
    return {
        "game_id": "g2", "game_family": "bargaining", "your_player": "bob",
        "game_state": {"money_to_divide": 1000.0},
        "valid_actions": {"type": "decision", "fields": {}},
    }


def _negotiation_offer_game(role="seller"):
    return {
        "game_id": "g3", "game_family": "negotiation", "your_player": "player_1",
        "game_state": {"current_player": "player_1", "player_1_role": role, "player_1_value": 100.0},
        "valid_actions": {"type": "offer", "fields": {}},
    }


def _negotiation_decision_game():
    return {
        "game_id": "g4", "game_family": "negotiation", "your_player": "player_2",
        "game_state": {"current_player": "player_2", "player_2_role": "buyer", "player_2_value": 100.0},
        "valid_actions": {"type": "decision", "fields": {}},
    }


def _persuasion_game(action_type):
    return {
        "game_id": "g5", "game_family": "persuasion", "your_player": "player_1",
        "game_state": {},
        "valid_actions": {"type": action_type, "fields": {}},
    }


def test_strategy_returns_handler_result_when_fast(monkeypatch):
    monkeypatch.setattr(dispatch_mod, "STRATEGIES", {"bargaining": lambda g: {"decision": "accept"}})
    action = strategy(_bargaining_decision_game())
    assert action == {"decision": "accept"}


def test_strategy_falls_back_when_handler_exceeds_deadline(monkeypatch):
    monkeypatch.setattr(dispatch_mod, "TURN_DEADLINE_SECONDS", 0.05)

    def slow_handler(game):
        time.sleep(0.3)
        return {"decision": "should-never-be-seen"}

    monkeypatch.setattr(dispatch_mod, "STRATEGIES", {"bargaining": slow_handler})
    action = strategy(_bargaining_decision_game())
    assert action == {"decision": "accept"}  # the fallback, not the slow handler's real return


def test_unknown_family_still_raises():
    with pytest.raises(ValueError):
        strategy({"game_family": "unknown", "game_state": {}, "valid_actions": {"type": "offer"}})


def test_fallback_bargaining_offer_sums_to_exactly_the_pot():
    action = _fallback_action(_bargaining_offer_game())
    assert action["alice_gain"] + action["bob_gain"] == 1000.0
    assert action["alice_gain"] == action["bob_gain"] == 500.0


def test_fallback_bargaining_decision_is_lowercase_accept():
    assert _fallback_action(_bargaining_decision_game()) == {"decision": "accept"}


def test_fallback_negotiation_offer_seller_anchors_above_own_value():
    action = _fallback_action(_negotiation_offer_game(role="seller"))
    assert action["product_price"] == pytest.approx(110.0)


def test_fallback_negotiation_offer_buyer_anchors_below_own_value():
    game = _negotiation_offer_game(role="seller")
    game["game_state"]["player_1_role"] = "buyer"
    action = _fallback_action(game)
    assert action["product_price"] == pytest.approx(90.0)


def test_fallback_negotiation_decision_is_accept_offer_capitalized():
    assert _fallback_action(_negotiation_decision_game()) == {"decision": "AcceptOffer"}


def test_fallback_persuasion_seller_message_is_a_real_pitch():
    action = _fallback_action(_persuasion_game("seller_message"))
    assert isinstance(action["message"], str) and action["message"]


def test_fallback_persuasion_seller_recommendation_recommends():
    assert _fallback_action(_persuasion_game("seller_recommendation")) == {"decision": "yes"}


def test_fallback_persuasion_buyer_decision_declines_conservatively():
    # Declining (not buying) is the conservative default: a passed round is
    # always $0 for both sides, never a large realized loss, unlike a blind
    # "yes" on an unknown-quality round.
    assert _fallback_action(_persuasion_game("buyer_decision")) == {"decision": "no"}
