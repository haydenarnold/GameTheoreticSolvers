"""Tests for parsing and classifying the seller's "discourage" signal, including
failure and low-confidence paths.
"""

from agent.persuasion.signal import _parse_response, classify_discourage_signal


def test_parse_response_valid_json():
    assert _parse_response('{"discourages": true, "confidence": 0.9}') == (True, 0.9)
    assert _parse_response('{"discourages": false, "confidence": 0.4}') == (False, 0.4)


def test_parse_response_strips_code_fences():
    raw = '```json\n{"discourages": true, "confidence": 0.8}\n```'
    assert _parse_response(raw) == (True, 0.8)


def test_parse_response_clamps_confidence_to_valid_range():
    assert _parse_response('{"discourages": true, "confidence": 1.5}') == (True, 1.0)
    assert _parse_response('{"discourages": true, "confidence": -0.5}') == (True, 0.0)


def test_parse_response_returns_none_for_malformed_json():
    assert _parse_response("not json at all") is None
    assert _parse_response("{not actually json") is None


def test_parse_response_returns_none_when_discourages_field_missing_or_wrong_type():
    assert _parse_response('{"confidence": 0.9}') is None
    assert _parse_response('{"discourages": "yes", "confidence": 0.9}') is None


def test_parse_response_returns_none_when_confidence_field_missing_or_wrong_type():
    assert _parse_response('{"discourages": true}') is None
    assert _parse_response('{"discourages": true, "confidence": "high"}') is None
    assert _parse_response('{"discourages": true, "confidence": true}') is None


def test_parse_response_returns_none_for_non_dict_json():
    assert _parse_response("[1, 2, 3]") is None


def test_classify_discourage_signal_returns_none_for_empty_message():
    assert classify_discourage_signal("") is None
    assert classify_discourage_signal("   ") is None


def test_classify_discourage_signal_returns_parsed_result_on_success(monkeypatch):
    class FakeMessage:
        content = '{"discourages": true, "confidence": 0.95}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    monkeypatch.setattr("agent.persuasion.signal.completion", lambda **kwargs: FakeResponse())
    assert classify_discourage_signal("This is below our usual quality.") == (True, 0.95)


def test_classify_discourage_signal_returns_none_when_the_call_raises(monkeypatch):
    def raise_error(**kwargs):
        raise TimeoutError("simulated timeout")

    monkeypatch.setattr("agent.persuasion.signal.completion", raise_error)
    assert classify_discourage_signal("Some message.") is None


def test_classify_discourage_signal_prompt_distinguishes_generic_filler_from_a_real_hedge():
    from agent.persuasion.signal import SYSTEM_PROMPT

    assert "no strong recommendation" in SYSTEM_PROMPT.lower()
    assert "discourages=false" in SYSTEM_PROMPT.lower()


def test_classify_discourage_signal_returns_none_for_unparseable_response(monkeypatch):
    class FakeMessage:
        content = "I refuse to answer in JSON."

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    monkeypatch.setattr("agent.persuasion.signal.completion", lambda **kwargs: FakeResponse())
    assert classify_discourage_signal("Some message.") is None
