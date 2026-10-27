"""Robustness checks for the per-foundation and register analyses (Appendix D.1).

Adds intervals, effect sizes and pairwise tests to the foundation success rates,
re-estimates them with standard errors clustered by game (messages within a game
are not independent), fits an adjusted persuasion model, tests whether what the
care label is actually capturing (honest "pass" messages vs. pitches), and
repeats the analysis for evidence- vs. intuition-leaning language.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from agent.persuasion.strategy import DISCOURAGE_PHRASES
from agent.rhetoric.lexicons import MORAL_FOUNDATION_WORDS
from agent.rhetoric.profile import (
    _EVIDENCE_RE,
    _FOUNDATION_RES,
    _INTUITION_RE,
    MIN_FOUNDATION_EVIDENCE,
    _compile,
    _count_with_negation,
)
from scripts.moral_foundation_analysis import FOUNDATIONS, dominant_foundation_for_message, load_latest_games

FAMILIES = ("persuasion", "bargaining", "negotiation")

# Care words that can also read as a warning to the buyer ("risk", "damage"),
# which would confound a care frame with a discouraging message.
AMBIGUOUS_CARE_WORDS = {
    "harm", "harmful", "hurt", "suffer", "suffering", "protect", "protecting", "protection",
    "safe", "safety", "risk", "risky", "damage", "cruel", "cruelty", "pain", "painful",
    "shield", "shielding", "secure", "security",
}


def register_leaning(text: str) -> float:
    lowered = text.lower()
    evidence = _count_with_negation(lowered, _EVIDENCE_RE)
    intuition = _count_with_negation(lowered, _INTUITION_RE)
    total = evidence + intuition
    return (evidence - intuition) / total if total else 0.0


# A "pass" message is an honest no-sale signal, so the buyer declining it is
# the right outcome rather than a persuasion failure; pitches are the messages
# where persuading is the point.
def is_discouraging(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in DISCOURAGE_PHRASES) or "pass on" in lowered


def _messages(d: dict, family: str):
    state = d["game_state"]
    me = d["your_player"]

    if family == "persuasion":
        if state.get("seller_message_type") != "text":
            return
        if (state.get(f"{me}_role") or state.get("your_role")) != "seller":
            return
        v, price = state.get("v"), state.get("product_price")
        ratio = v / price if v is not None and price else None
        for h in state.get("history", []):
            if h.get("seller_message") is None or h.get("bought") is None:
                continue
            yield h["seller_message"], bool(h["bought"]), {
                "quality": h.get("quality"), "round": h.get("round"), "ratio": ratio,
            }
        return

    sender_key, accept_value = ("from_player", "AcceptOffer") if family == "negotiation" else ("proposer", "accept")
    for h in state.get("history", []):
        offer = h.get("offer") or {}
        if offer.get(sender_key) != me or not offer.get("message") or h.get("decision") is None:
            continue
        yield offer["message"], h["decision"] == accept_value, {}


def collect_rows(games: dict, family: str, foundation_fn=dominant_foundation_for_message) -> list[dict]:
    rows = []
    for d in games.values():
        if d["family"] != family:
            continue
        for msg, success, extra in _messages(d, family):
            fnd = foundation_fn(msg)
            if fnd is None:
                continue
            rows.append({
                "game_id": d["game_id"], "fnd": fnd, "success": success, "msg": msg,
                "register": register_leaning(msg), "length": len(msg),
                "pitch": not is_discouraging(msg), **extra,
            })
    return rows


def _cluster_codes(rows: list[dict]) -> np.ndarray:
    codes: dict[str, int] = {}
    return np.array([codes.setdefault(r["game_id"], len(codes)) for r in rows])


def levels_in(rows: list[dict]) -> list[str]:
    return [f for f in FOUNDATIONS if any(r["fnd"] == f for r in rows)]


def fit_logit(rows: list[dict], controls: tuple[str, ...] = ()):
    # The first foundation present (care, when it is) is the reference level, so
    # odds ratios read against it.
    usable = [r for r in rows if all(r.get(c) is not None for c in controls)]
    terms = levels_in(usable)[1:]
    names = ["const"] + terms + list(controls)
    cols = [np.ones(len(usable))]
    cols += [np.array([r["fnd"] == f for r in usable], float) for f in terms]
    for c in controls:
        x = np.array([float(r[c] == "high") if c == "quality" else float(r[c]) for r in usable])
        if c in ("ratio", "length"):
            x = np.log(x)
        if c in ("round", "ratio", "length"):
            x = (x - x.mean()) / x.std()
        cols.append(x)
    X = np.column_stack(cols)
    y = np.array([float(r["success"]) for r in usable])
    result = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": _cluster_codes(usable)})
    return result, names, len(usable), terms


def joint_wald(result, names: list[str], terms: list[str]) -> tuple[float, float]:
    R = np.zeros((len(terms), len(names)))
    for i, t in enumerate(terms):
        R[i, names.index(t)] = 1.0
    w = result.wald_test(R, scalar=True)
    return float(w.statistic), float(w.pvalue)


def foundation_table(rows: list[dict], label: str) -> None:
    levels = levels_in(rows)
    counts = {f: [0, 0] for f in levels}
    for r in rows:
        counts[r["fnd"]][0 if r["success"] else 1] += 1
    contingency = [counts[f] for f in levels]
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    n = sum(map(sum, contingency))
    print(f"\n-- {label} (n={n}) --")
    print(f"  {'foundation':<10} {'n':>6} {'rate':>7}   95% Wilson CI")
    for f in sorted(levels, key=lambda f: -counts[f][0] / max(sum(counts[f]), 1)):
        s, k = counts[f][0], sum(counts[f])
        lo, hi = proportion_confint(s, k, method="wilson")
        print(f"  {f:<10} {k:>6} {s / k:>7.1%}   [{lo:.1%}, {hi:.1%}]")
    print(f"  naive chi2({dof})={chi2:.1f}, p={p:.3g}, Cramer's V={math.sqrt(chi2 / n):.3f}")

    result, names, _, terms = fit_logit(rows)
    stat, pc = joint_wald(result, names, terms)
    print(f"  game-clustered joint Wald chi2({len(terms)})={stat:.1f}, p={pc:.3g}")


def pairwise_table(rows: list[dict]) -> None:
    counts = {f: [0, 0] for f in FOUNDATIONS}
    for r in rows:
        counts[r["fnd"]][0 if r["success"] else 1] += 1
    pairs = list(combinations(FOUNDATIONS, 2))
    pvals = [stats.chi2_contingency([counts[a], counts[b]])[1] for a, b in pairs]
    reject, adjusted, _, _ = multipletests(pvals, method="holm")
    print("\n  pairwise differences (Holm-adjusted):")
    for (a, b), padj, sig in sorted(zip(pairs, adjusted, reject), key=lambda t: t[1]):
        ra = counts[a][0] / sum(counts[a])
        rb = counts[b][0] / sum(counts[b])
        print(f"    {a:<9} vs {b:<9} diff={ra - rb:+6.1%}  p_holm={padj:.3g}{'  *' if sig else ''}")


def adjusted_model(rows: list[dict], controls: tuple[str, ...], label: str) -> None:
    result, names, n, terms = fit_logit(rows, controls)
    ci = np.exp(result.conf_int())
    print(f"\n-- {label} (n={n}, {len(set(r['game_id'] for r in rows))} games, clustered SEs) --")
    print(f"  {'term':<10} {'OR':>7}   95% CI           p")
    for i, name in enumerate(names):
        if name == "const":
            continue
        print(f"  {name:<10} {math.exp(result.params[i]):>7.2f}   [{ci[i, 0]:.2f}, {ci[i, 1]:.2f}]   {result.pvalues[i]:.3g}")
    stat, p = joint_wald(result, names, terms)
    print(f"  joint foundation Wald chi2({len(terms)})={stat:.1f}, p={p:.3g}")


def register_table(rows: list[dict], label: str) -> None:
    bins = {"evidence-leaning": [], "intuition-leaning": [], "balanced/none": []}
    for r in rows:
        key = "evidence-leaning" if r["register"] > 0 else "intuition-leaning" if r["register"] < 0 else "balanced/none"
        bins[key].append(r)
    print(f"\n-- {label}: success by register (n={len(rows)}) --")
    print(f"  {'register':<18} {'n':>6} {'rate':>7}   95% Wilson CI")
    table = []
    for name, rs in bins.items():
        s = sum(r["success"] for r in rs)
        k = len(rs)
        table.append([s, k - s])
        if k:
            lo, hi = proportion_confint(s, k, method="wilson")
            print(f"  {name:<18} {k:>6} {s / k:>7.1%}   [{lo:.1%}, {hi:.1%}]")
    nonempty = [t for t in table if sum(t)]
    chi2, p, dof, _ = stats.chi2_contingency(nonempty)
    print(f"  naive chi2({dof})={chi2:.1f}, p={p:.3g}, Cramer's V={math.sqrt(chi2 / len(rows)):.3f}")


def register_model(rows: list[dict], controls: tuple[str, ...], label: str) -> None:
    result, names, n, _ = fit_logit(rows, ("register",) + controls)
    i = names.index("register")
    lo, hi = np.exp(result.conf_int()[i])
    print(f"  {label}: OR per +1.0 evidence-leaning (n={n}) = {math.exp(result.params[i]):.2f} "
          f"[{lo:.2f}, {hi:.2f}], p={result.pvalues[i]:.3g}")


def register_by_foundation(rows: list[dict]) -> None:
    print("\n  share of evidence-leaning / intuition-leaning messages by foundation (persuasion):")
    for f in FOUNDATIONS:
        rs = [r for r in rows if r["fnd"] == f]
        ev = sum(r["register"] > 0 for r in rs) / len(rs)
        it = sum(r["register"] < 0 for r in rs) / len(rs)
        print(f"    {f:<10} n={len(rs):>5} evidence={ev:.1%} intuition={it:.1%}")


def reduced_care_foundation_fn():
    patterns = dict(_FOUNDATION_RES)
    patterns["care"] = _compile(MORAL_FOUNDATION_WORDS["care"] - AMBIGUOUS_CARE_WORDS)

    def dominant(text: str | None) -> str | None:
        if not text:
            return None
        lowered = text.lower()
        scores = {name: _count_with_negation(lowered, p) for name, p in patterns.items()}
        if sum(scores.values()) < MIN_FOUNDATION_EVIDENCE:
            return None
        return max(scores, key=scores.get)

    return dominant


def care_diagnostics(care_rows: list[dict]) -> None:
    words = Counter(w for r in care_rows for w in _FOUNDATION_RES["care"].findall(r["msg"].lower()))
    print("\n  most frequent care-lexicon hits:", ", ".join(f"{w} ({c})" for w, c in words.most_common(6)))
    reduced = reduced_care_foundation_fn()
    relabeled = Counter(reduced(r["msg"]) or "none" for r in care_rows)
    still = relabeled.get("care", 0)
    print(f"  without warning-flavored words, {still} of {len(care_rows)} care messages "
          f"({still / len(care_rows):.1%}) are still labeled care; the rest become: "
          + ", ".join(f"{k} ({v})" for k, v in relabeled.most_common() if k != "care"))
    discouraging = sum(not r["pitch"] for r in care_rows)
    print(f"  {discouraging / len(care_rows):.1%} of care messages are discouraging (\"pass\"/low-quality) messages")


def signal_split(rows: list[dict]) -> None:
    print("\n  share of each foundation's messages that are pitches (vs. honest \"pass\" messages):")
    for f in FOUNDATIONS:
        rs = [r for r in rows if r["fnd"] == f]
        print(f"    {f:<10} n={len(rs):>5} pitch={sum(r['pitch'] for r in rs) / len(rs):.1%}")
    foundation_table([r for r in rows if r["pitch"]], "pitch messages only")
    foundation_table([r for r in rows if not r["pitch"]], "discouraging messages only")


def main() -> None:
    games = load_latest_games()
    rows = {family: collect_rows(games, family) for family in FAMILIES}
    pers = rows["persuasion"]
    pitches = [r for r in pers if r["pitch"]]

    print("=== 1. Foundation success rates: intervals, effect sizes, clustered tests ===")
    for family in FAMILIES:
        foundation_table(rows[family], family)
    pairwise_table(pers)

    print("\n=== 2. Persuasion: adjusted models (all messages) ===")
    adjusted_model(pers, (), "foundation only")
    adjusted_model(pers, ("quality", "round"), "+ quality, round")
    adjusted_model(pers, ("quality", "round", "ratio"), "+ quality, round, value/price (seller knew v only)")

    print("\n=== 3. What the persuasion care label captures ===")
    care_diagnostics([r for r in pers if r["fnd"] == "care"])
    signal_split(pers)
    controls = ("quality", "round")
    adjusted_model(pitches, (), "pitch messages, foundation only")
    adjusted_model(pitches, controls, "pitch messages, + quality, round")
    adjusted_model(pitches, controls + ("ratio",), "pitch messages, + quality, round, value/price")
    bluffs = [r for r in pitches if r["quality"] == "low"]
    foundation_table(bluffs, "true bluffs: pitch messages on low-quality rounds")
    
    print("\n=== 4. Evidence vs. intuition register ===")
    for family in FAMILIES:
        register_table(rows[family], family)
    print("\n  logistic models (clustered by game, foundation dummies included):")
    register_model(pers, (), "persuasion, all messages")
    register_model(pers, ("quality", "round", "length"), "persuasion, + quality, round, length")
    register_model(pitches, (), "persuasion, pitches only")
    register_model(pitches, ("quality", "round", "length"), "persuasion, pitches, + quality, round, length")
    register_model(bluffs, ("round", "length"), "persuasion, true bluffs, + round, length")
    register_model(rows["bargaining"], ("length",), "bargaining, + length")
    register_model(rows["negotiation"], ("length",), "negotiation, + length")
    register_by_foundation(pers)
    register_table(pitches, "persuasion pitches only")
    register_table(bluffs, "persuasion true bluffs")


if __name__ == "__main__":
    main()
