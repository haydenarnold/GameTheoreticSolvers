"""Word lists behind the rhetoric profile: evidence vs. intuition language,
the five moral-foundation lexicons, and negation cues.

The same lexicons classify our own messages offline in the rhetoric analysis.
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

NEGATION_WORDS = {
    "not", "no", "never", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't",
    "didn't", "won't", "wouldn't", "can't", "cannot", "without", "lack", "lacks", "lacking",
    "hardly", "barely", "none",
}

MORAL_FOUNDATION_GLOSSES = {
    "care": "protecting the buyer from a bad outcome and looking out for their interests",
    "fairness": "equitable value and a fair price for what's genuinely being offered",
    "loyalty": "an ongoing relationship and standing behind this deal as part of it",
    "authority": "expertise, track record, and confident, credible endorsement",
    "purity": "honesty and integrity in what's being represented about the product",
}
