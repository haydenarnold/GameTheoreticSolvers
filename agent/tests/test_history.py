from agent.history import is_opponent_entry


def test_recognizes_server_shape_by_proposer_field():
    # The real API's history entries carry no "speaker" key at all -- only
    # "proposer" (see history.py's module docstring for the live incident
    # this caused).
    entry = {"round": 3, "proposer": "bob", "offer": {"bob_gain": 60.0}, "decision": "reject"}
    assert is_opponent_entry(entry, "bob") is True
    assert is_opponent_entry(entry, "alice") is False


def test_recognizes_local_fallback_shape_by_speaker_field():
    opponent_entry = {"speaker": "opponent", "round": 1, "offer": {"bob_gain": 60.0}}
    own_entry = {"speaker": "me", "round": 1, "action": {"decision": "accept"}}
    assert is_opponent_entry(opponent_entry, "bob") is True
    assert is_opponent_entry(own_entry, "bob") is False


def test_recognizes_negotiation_shape_by_nested_from_player_field():
    # Negotiation's real history entries have NO top-level "proposer" at
    # all -- the proposer identity is nested one level down, inside "offer",
    # under a different key ("from_player"). Confirmed live by pulling a
    # real negotiation game_state directly from the API (see module
    # docstring) -- this shape is different from both bargaining's
    # top-level "proposer" and the local fallback's "speaker" tag.
    entry = {
        "round": 7, "offer": {"price": 118.8, "from_player": "player_1", "round": 7},
        "decision": "RejectOffer", "decided_by": "player_2",
    }
    assert is_opponent_entry(entry, "player_1") is True
    assert is_opponent_entry(entry, "player_2") is False
