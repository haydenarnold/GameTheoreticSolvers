from __future__ import annotations

import re
from typing import Iterable

from agent.rhetoric.lexicons import EVIDENCE_WORDS, INTUITION_WORDS, MORAL_FOUNDATION_WORDS, NEGATION_WORDS

# How many words immediately before a lexicon match count as its negation
# scope -- "not a fair deal" negates at distance 2, "not exactly what I'd
# call a fair deal" would miss at a smaller window. 4 is a judgment call:
# wide enough to catch typical hedged negation, narrow enough that it
# shouldn't often reach back across an unrelated clause boundary (no clause
# boundary detection here, so a comma or "but" mid-window can still let a
# negation from a different clause bleed through incorrectly).
NEGATION_WINDOW_WORDS = 4

# A dominant_foundation call on 0 or 1 total matches is mostly just noting
# which single incidental word happened to appear -- not a real signal
# about which value the message is actually leaning on. Require at least
# this many total (post-negation) matches across all five foundations
# combined before naming a dominant one; below that, report None ("no
# signal yet") same as a message with zero matches at all.
MIN_FOUNDATION_EVIDENCE = 2


def _compile(words: set) -> re.Pattern:
    # Word-boundary alternation, not substring search: plain `text.count(w)`
    # false-positives on a lexicon word appearing inside an unrelated word
    # (e.g. "care" inside "scared", "fact" inside "satisfaction").
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b")


_EVIDENCE_RE = _compile(EVIDENCE_WORDS)
_INTUITION_RE = _compile(INTUITION_WORDS)
_FOUNDATION_RES = {name: _compile(words) for name, words in MORAL_FOUNDATION_WORDS.items()}
_NEGATION_RE = _compile(NEGATION_WORDS)


def _count(text: str, pattern: re.Pattern) -> int:
    return len(pattern.findall(text))


def _count_with_negation(text: str, pattern: re.Pattern) -> int:
    """Like `_count`, but drops any match preceded within
    `NEGATION_WINDOW_WORDS` words by a negation cue -- "not a fair deal"
    should not score as a fairness-affirming mention. DERIVED (2026-08-24),
    not yet live-tested: a simple preceding-window heuristic, not real
    dependency parsing, so it will occasionally over- or under-negate
    (see NEGATION_WINDOW_WORDS's own caveat about clause boundaries) -- a
    deliberate, documented limitation rather than an attempt at full
    negation scope resolution, which real NLP negation handling still
    struggles with in general."""
    count = 0
    for m in pattern.finditer(text):
        window_start = max(0, m.start() - 40)  # generous char slice, trimmed to words below
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
