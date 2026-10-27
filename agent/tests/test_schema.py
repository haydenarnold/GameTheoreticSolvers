"""Tests for player-name normalization and bargaining offer construction.
"""

from agent.schema import build_bargaining_offer, read_gain


def test_build_offer_uses_literal_alice_bob_keys_even_when_told_player_2():
    action = build_bargaining_offer("player_2", my_gain=600.0, money=1000.0)
    assert set(action) == {"alice_gain", "bob_gain"}
    assert action["bob_gain"] == 600.0
    assert action["alice_gain"] == 400.0


def test_build_offer_uses_literal_alice_bob_keys_when_told_player_1():
    action = build_bargaining_offer("player_1", my_gain=600.0, money=1000.0)
    assert set(action) == {"alice_gain", "bob_gain"}
    assert action["alice_gain"] == 600.0
    assert action["bob_gain"] == 400.0


def test_build_offer_still_works_with_native_alice_bob_naming():
    action = build_bargaining_offer("alice", my_gain=600.0, money=1000.0)
    assert action == {"alice_gain": 600.0, "bob_gain": 400.0}


def test_build_offer_gains_always_sum_to_money_to_divide():
    for me in ("alice", "bob", "player_1", "player_2"):
        action = build_bargaining_offer(me, my_gain=333.33, money=1000.0)
        assert abs(sum(action[k] for k in ("alice_gain", "bob_gain")) - 1000.0) < 1e-9


def test_read_gain_tolerates_both_naming_conventions_when_reading():
    offer = {"alice_gain": 400.0, "bob_gain": 600.0}
    assert read_gain(offer, "bob") == 600.0
    assert read_gain(offer, "player_2") == 600.0
    assert read_gain(offer, "player_1") == 400.0
