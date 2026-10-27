"""Tests for the rhetorical profile: whole-word matching, negation handling, and
foundation detection.
"""

from agent.rhetoric.profile import classify


def test_substring_false_positives_are_not_counted():
    profile = classify(["I'm scared this deal falls through, it would be a real satisfaction to close."])
    assert profile["foundation_scores"]["care"] == 0
    assert profile["evidence_leaning"] <= 0


def test_whole_word_matches_still_count():
    profile = classify(["I care about a fair deal, and the data backs my number."])
    assert profile["foundation_scores"]["care"] == 1
    assert profile["foundation_scores"]["fairness"] == 1
    assert profile["evidence_leaning"] > 0


def test_no_messages_gives_neutral_profile():
    profile = classify([])
    assert profile["evidence_leaning"] == 0.0
    assert profile["dominant_foundation"] is None
    assert profile["n_messages"] == 0


def test_negated_mention_does_not_count_as_affirming():
    negated = classify(["This is not a fair deal for either of us."])
    affirmed = classify(["This is a genuinely fair deal for both of us."])
    assert negated["foundation_scores"]["fairness"] == 0
    assert affirmed["foundation_scores"]["fairness"] == 1


def test_negation_applies_to_evidence_leaning_too():
    negated = classify(["There is no evidence backing your number."])
    assert negated["evidence_leaning"] <= 0.0


def test_dominant_foundation_requires_minimum_evidence():
    profile = classify(["I care about closing this."])
    assert profile["foundation_scores"]["care"] == 1
    assert profile["dominant_foundation"] is None


def test_dominant_foundation_fires_once_minimum_evidence_is_met():
    profile = classify(["I care deeply about this, and protecting your safety matters to me."])
    assert profile["foundation_scores"]["care"] >= 2
    assert profile["dominant_foundation"] == "care"
