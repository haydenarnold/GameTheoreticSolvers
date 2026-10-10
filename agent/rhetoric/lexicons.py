"""Keyword cues for two rhetorical-style axes used to profile an opponent
from their own message history:

- Evidence vs. intuition leaning: matching a persuasion channel to the
  target's leaning (data/logic vs. narrative/gut-feeling) outperforms a
  fixed style — see Nature Human Behaviour 10.1038/s41562-025-02136-2.
- Moral Foundations Theory emphasis (Haidt & Graham): which moral frame
  (care, fairness, loyalty, authority, purity) the opponent's own language
  already leans on, so we can mirror or deliberately contrast it.

Deliberately small, hand-picked lexicons rather than a full dictionary —
good enough to bias an LLM's phrasing, not meant to be a validated
psychometric instrument.
"""

EVIDENCE_WORDS = {
    "data", "evidence", "research", "study", "studies", "statistics", "proven", "proof",
    "shows", "because", "therefore", "analysis", "fact", "facts", "numbers", "report",
    "demonstrates", "objectively", "rational", "logic", "logical", "benchmark", "benchmarks",
    "metric", "metrics", "verified", "documented", "quantify", "quantified", "empirical",
    "measured", "measurable", "precedent", "standard", "standards", "calculation",
}

INTUITION_WORDS = {
    "feel", "feels", "feeling", "sense", "gut", "trust", "believe", "seems", "instinct",
    "imagine", "picture", "heart", "vibe", "honestly", "personally", "intuition", "instinctively",
    "hope", "hopeful", "worry", "worried", "comfortable", "uneasy", "impression", "sentiment",
}

# DERIVED (2026-08-24), not yet live-tested: expanded from the original
# 5-10-word-per-foundation sets to roughly 30-40 each. This is still a
# hand-constructed lexicon, not an import of a published, validated
# psychometric instrument (no network access to pull the actual academic
# MFD2.0/eMFD word lists) -- broader coverage, same "good enough to bias an
# LLM's phrasing" honesty caveat as before, not an upgrade in validation
# status. Each foundation now covers both its virtue framing (the value
# being upheld) and its vice framing (the value being violated), since a
# message can invoke a foundation either way ("this is fair" and "this is
# cheating" both invoke fairness) -- the original lists were skewed toward
# violation words for some foundations (fairness, purity) and virtue words
# for others (care, authority), an inconsistency that made cross-foundation
# comparison noisier than it needed to be.
MORAL_FOUNDATION_WORDS = {
    "care": {
        "harm", "harmful", "hurt", "suffer", "suffering", "care", "caring", "protect",
        "protecting", "protection", "safe", "safety", "kind", "kindness", "compassion",
        "compassionate", "vulnerable", "wellbeing", "welfare", "gentle", "shield", "shielding",
        "risk", "risky", "damage", "cruel", "cruelty", "pain", "painful", "comfort",
        "reassure", "reassurance", "secure", "security",
    },
    "fairness": {
        "fair", "fairness", "unfair", "unfairness", "equal", "equally", "equality",
        "balanced", "balance", "deserve", "deserved", "deserves", "cheat", "cheating",
        "cheated", "reciprocal", "reciprocity", "reciprocate", "justice", "just", "unjust",
        "impartial", "biased", "bias", "exploit", "exploiting", "exploitative", "onesided",
        "one-sided", "proportion", "proportional", "equitable", "equity", "level playing field",
    },
    "loyalty": {
        "loyal", "loyalty", "team", "teamwork", "together", "betray", "betrayal", "commitment",
        "committed", "partner", "partnership", "alliance", "united", "unity", "solidarity",
        "faithful", "faithfulness", "abandon", "abandoning", "abandoned", "collective",
        "shared", "mutual", "stand together", "back each other",
    },
    "authority": {
        "respect", "respected", "rules", "rule", "tradition", "traditional", "authority",
        "authoritative", "orderly", "proper", "propriety", "hierarchy", "protocol",
        "obey", "obedience", "defer", "deference", "expert", "expertise", "credential",
        "credentials", "legitimate", "legitimacy", "chain of command", "precedent",
        "established", "standard practice",
    },
    "purity": {
        "honest", "honesty", "integrity", "clean", "cleanliness", "pure", "purity",
        "corrupt", "corrupted", "corruption", "tainted", "taint", "disgust", "disgusting",
        "sacred", "defile", "defiled", "sanctity", "wholesome", "genuine", "authentic",
        "authenticity", "transparent", "transparency", "deceit", "deceitful", "shady",
        "sleazy", "dirty",
    },
}

# Negation cues checked in a short word-window immediately BEFORE a lexicon
# match (see `profile.py`'s `_count_with_negation`) -- "not fair" should not
# score as a fairness-affirming mention, and blindly counting bag-of-words
# hits without this treats affirmation and negation of the exact same value
# identically, which is backwards for a signal meant to infer WHICH frame
# the speaker is leaning on.
NEGATION_WORDS = {
    "not", "no", "never", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't",
    "didn't", "won't", "wouldn't", "can't", "cannot", "without", "lack", "lacks", "lacking",
    "hardly", "barely", "none",
}

# One-line gloss per foundation, for instructing an LLM to frame a pitch
# around a specific one (see rhetoric.compose's `style` param) -- kept next
# to the matching lexicon above since they're the same taxonomy.
MORAL_FOUNDATION_GLOSSES = {
    "care": "protecting the buyer from a bad outcome and looking out for their interests",
    "fairness": "equitable value and a fair price for what's genuinely being offered",
    "loyalty": "an ongoing relationship and standing behind this deal as part of it",
    "authority": "expertise, track record, and confident, credible endorsement",
    "purity": "honesty and integrity in what's being represented about the product",
}
