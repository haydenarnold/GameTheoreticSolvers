"""Infers an opponent's rhetorical profile from their past messages.

Scores evidence- vs. intuition-leaning language and picks the dominant moral
foundation by lexicon counts, which feed the message layer's framing.
"""

from __future__ import annotations

import re
from typing import Iterable

from agent.rhetoric.lexicons import EVIDENCE_WORDS, INTUITION_WORDS, MORAL_FOUNDATION_WORDS, NEGATION_WORDS

NEGATION_WINDOW_WORDS = 4

MIN_FOUNDATION_EVIDENCE = 2


def _compile(words: set) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b")


_EVIDENCE_RE = _compile(EVIDENCE_WORDS)
_INTUITION_RE = _compile(INTUITION_WORDS)
_FOUNDATION_RES = {name: _compile(words) for name, words in MORAL_FOUNDATION_WORDS.items()}
_NEGATION_RE = _compile(NEGATION_WORDS)


def _count(text: str, pattern: re.Pattern) -> int:
    return len(pattern.findall(text))


def _count_with_negation(text: str, pattern: re.Pattern) -> int:
    count = 0
    for m in pattern.finditer(text):
        # Skip hits shortly after a negation ("not fair") so they don't count as
        # support for that foundation or style.
        window_start = max(0, m.start() - 40)
        preceding = text[window_start:m.start()]
        preceding_words = preceding.split()[-NEGATION_WINDOW_WORDS:]
        if _NEGATION_RE.search(" ".join(preceding_words)):
            continue
        count += 1
    return count


def classify(messages: Iterable[str]) -> dict:
    texts = [m for m in messages if m]
    joined = " ".join(texts).lower()
    n_messages = len(texts)

    evidence = _count_with_negation(joined, _EVIDENCE_RE)
    intuition = _count_with_negation(joined, _INTUITION_RE)
    total = evidence + intuition
    evidence_leaning = (evidence - intuition) / total if total else 0.0

    foundation_scores = {name: _count_with_negation(joined, pattern) for name, pattern in _FOUNDATION_RES.items()}
    dominant_foundation = (
        max(foundation_scores, key=foundation_scores.get)
        if sum(foundation_scores.values()) >= MIN_FOUNDATION_EVIDENCE else None
    )

    return {
        "n_messages": n_messages,
        "evidence_leaning": evidence_leaning,
        "dominant_foundation": dominant_foundation,
        "foundation_scores": foundation_scores,
    }
