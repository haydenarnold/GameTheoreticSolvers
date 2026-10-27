"""Ablation of the opponent-modeling layer over the Monte Carlo opponents.

Runs each family with all three estimators enabled, then again with them
replaced by fixed priors on identical seeds, and reports the score difference.
"""

from __future__ import annotations

import logging
import os
import statistics
import sys

logging.disable(logging.WARNING)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import montecarlo_opponents as mc

from agent.bargaining import belief as belief_mod
from agent.bargaining import strategy as barg_strategy_mod
from agent.negotiation import solver as neg_solver_mod
from agent.negotiation import strategy as neg_strategy_mod
from agent.persuasion import solver as pers_solver_mod
from agent.persuasion import strategy as pers_strategy_mod


def force_mechanisms_on():
    barg_strategy_mod.DISABLE_DELTA_BELIEF = False
    neg_strategy_mod.DISABLE_VALUE_BOUNDS = False
    pers_strategy_mod.DISABLE_BUYER_TRUST_MODEL = False


def apply_ablation():
    belief_mod.DeltaBelief.update = lambda self, *a, **k: None
    neg_solver_mod.ValueBounds.tighten_from_rejection = lambda self, *a, **k: None
    pers_solver_mod.BuyerTrustModel.posterior_high_given_signal = lambda self: self.prior_p


# Strategy modules keep per-game state in module globals; clear it between runs
# so the baseline cannot leak into the ablated pass.
def reset_per_game_state():
    belief_mod._BELIEFS.clear()
    neg_strategy_mod._BOUNDS.clear()
    neg_strategy_mod._LAST_SENT_PRICE.clear()
    neg_strategy_mod._LAST_STYLE_ARM.clear()
    barg_strategy_mod._LAST_STYLE_ARM.clear()
    barg_strategy_mod._LAST_SENT_TAKE.clear()
    pers_strategy_mod._SELLER_ADAPTERS.clear()
    pers_strategy_mod._LAST_SELLER_SIGNAL.clear()
    pers_strategy_mod._LAST_SELLER_PAYOFF.clear()
    pers_strategy_mod._LAST_STYLE_ARM.clear()
    pers_strategy_mod._BUYER_TRUST.clear()
    pers_strategy_mod._LAST_BUYER_BOUGHT.clear()
    pers_strategy_mod._LAST_BUYER_PAYOFF.clear()
    pers_strategy_mod._BUYER_NO_TRUST.clear()
    pers_strategy_mod._LAST_BUYER_TESTED_NO.clear()


def run(families, apply_ablation_fn):
    reset_per_game_state()
    if apply_ablation_fn:
        apply_ablation_fn()
    trials = mc.TRIALS_PER_CELL
    results_by_family = {}
    for family in families:
        results, crashes = mc.run_family(family, trials)
        results_by_family[family] = (results, crashes)
    return results_by_family


def report(label, family, results, crashes):
    print(f"\n=== {label} {family} (n={mc.TRIALS_PER_CELL}/archetype) ===")
    ranked = sorted(results.items(), key=lambda kv: statistics.mean(kv[1]) if kv[1] else -999)
    for archetype, scores in ranked:
        if not scores:
            print(f"  {archetype:20s} ALL CRASHED ({crashes[archetype]})")
            continue
        mean = statistics.mean(scores)
        median = statistics.median(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        crash_note = f" ({crashes[archetype]} crashed)" if crashes.get(archetype) else ""
        print(f"  {archetype:20s} n={len(scores):3d} mean={mean:+.4f} median={median:+.4f} stdev={stdev:.4f}{crash_note}")


def main():
    families = sys.argv[1:] or ["bargaining", "negotiation", "persuasion"]
    force_mechanisms_on()

    baseline = run(families, apply_ablation_fn=None)
    for family in families:
        results, crashes = baseline[family]
        report("BASELINE (mechanisms ON)", family, results, crashes)

    ablated = run(families, apply_ablation_fn=apply_ablation)
    for family in families:
        results, crashes = ablated[family]
        report("ABLATED (mechanisms OFF)", family, results, crashes)

    print("\n=== aggregate mean, baseline vs ablated ===")
    for family in families:
        base_scores = [s for scores in baseline[family][0].values() for s in scores]
        abl_scores = [s for scores in ablated[family][0].values() for s in scores]
        base_mean = statistics.mean(base_scores) if base_scores else float("nan")
        abl_mean = statistics.mean(abl_scores) if abl_scores else float("nan")
        print(f"  {family:12s} baseline={base_mean:+.4f}  ablated={abl_mean:+.4f}  delta(ablated-baseline)={abl_mean - base_mean:+.4f}")


if __name__ == "__main__":
    main()
