"""Tests for recognizing opponent entries in both server and local history shapes.
"""

from agent.history import is_opponent_entry


def test_recognizes_server_shape_by_proposer_field():
    entry = {"round": 3, "proposer": "bob", "offer": {"bob_gain": 60.0}, "decision": "reject"}
    assert is_opponent_entry(entry, "bob") is True
    assert is_opponent_entry(entry, "alice") is False


def test_recognizes_local_fallback_shape_by_speaker_field():
    opponent_entry = {"speaker": "opponent", "round": 1, "offer": {"bob_gain": 60.0}}
    own_entry = {"speaker": "me", "round": 1, "action": {"decision": "accept"}}
    assert is_opponent_entry(opponent_entry, "bob") is True
    assert is_opponent_entry(own_entry, "bob") is False


def test_recognizes_negotiation_shape_by_nested_from_player_field():
    entry = {
        "round": 7, "offer": {"price": 118.8, "from_player": "player_1", "round": 7},
        "decision": "RejectOffer", "decided_by": "player_2",
    }
    assert is_opponent_entry(entry, "player_1") is True
    assert is_opponent_entry(entry, "player_2") is False
