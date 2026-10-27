"""Tests for the message layer's prompt contract and for parsing model output
into plain message text.
"""

from agent.rhetoric.compose import SYSTEM_PROMPT, TARGET_MESSAGE_CHARS, _extract_message_text


def test_system_prompt_asks_for_the_target_length_and_against_repetition():
    assert str(TARGET_MESSAGE_CHARS) in SYSTEM_PROMPT
    assert "concise" in SYSTEM_PROMPT.lower()
    assert "restate" in SYSTEM_PROMPT.lower() or "repeat" in SYSTEM_PROMPT.lower()


def test_extract_passes_through_plain_prose():
    assert _extract_message_text("That price doesn't hold up.") == "That price doesn't hold up."


def test_extract_unwraps_json_action_leak():
    raw = '{"decision": "RejectOffer", "product_price": 657220.13, "message": "Here is my counter."}'
    assert _extract_message_text(raw) == "Here is my counter."


def test_extract_strips_code_fences_before_parsing():
    raw = '```json\n{"message": "Fenced message."}\n```'
    assert _extract_message_text(raw) == "Fenced message."


def test_extract_returns_none_for_json_without_message_field():
    raw = '{"decision": "AcceptOffer"}'
    assert _extract_message_text(raw) is None


def test_extract_finds_message_nested_under_schema_shaped_wrapper():
    raw = '{"type": "decision", "fields": {"decision": "RejectOffer", "product_price": 534382.82, "message": "Nested counter."}}'
    assert _extract_message_text(raw) == "Nested counter."


def test_extract_returns_none_for_empty_string():
    assert _extract_message_text("   ") is None


def test_extract_treats_malformed_braces_as_literal_text():
    raw = "{not actually json"
    assert _extract_message_text(raw) == "{not actually json"
