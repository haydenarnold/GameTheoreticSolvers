"""Opponent-profiling ablation: same 11-archetype Monte Carlo suite as
montecarlo_opponents.py, but with the three opponent-profiling mechanisms
monkeypatched to no-ops in-process (never touching the on-disk strategy
files) to isolate their marginal contribution.

Ablated (no-op'd for the duration of this process only):
  - agent.bargaining.belief.DeltaBelief.update -> frozen at the prior mean
    (agent never revises its estimate of the opponent's discount factor)
  - agent.negotiation.solver.ValueBounds.tighten_from_rejection -> no-op
    (anchor/reservation never tighten from an observed rejection)
  - agent.persuasion.solver.BuyerTrustModel.posterior_high_given_signal ->
    always returns the known prior p (buyer never recalibrates on evidence)

Rhetoric profiling is NOT ablated here: this harness never generates or
scores messages (see montecarlo_opponents.py's own docstring), so it has no
mechanism through which a rhetoric-bandit ablation could show up in these
numbers one way or the other.

FIXED (2026-08-29): this is a corrected copy of the original scratchpad
version, which imported a stale scratchpad copy of montecarlo_opponents.py
still carrying the fresh-game_id-per-round bug (see that file's own
run_bargaining_trial/run_negotiation_trial docstring comments) -- under
that bug, DeltaBelief and ValueBounds could never accumulate state across
rounds in the FIRST PLACE, so both the original bargaining ablation ("zero
measured effect") and the negotiation ablation ("mechanism never fires")
were measuring a harness artifact, not the real mechanism. This copy
imports the corrected scripts/montecarlo_opponents.py (one stable game_id
per trial) so the numbers below are the first trustworthy measurement of
either mechanism's real marginal effect.

Usage: .venv/bin/python scripts/ablation_montecarlo.py [family ...]
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
    # The live strategy files currently ship these three DISABLE_* flags set
    # to True (a separate, real production decision -- see each flag's own
    # comment). This ablation experiment needs the mechanisms genuinely
    # callable in BOTH the baseline and ablated runs below, so the class
    # -method monkeypatch in `apply_ablation` below is the only thing that
    # differs between them -- otherwise "baseline" here would silently
    # inherit whatever the live flags happen to say today, not measure the
    # mechanism's own effect.
    barg_strategy_mod.DISABLE_DELTA_BELIEF = False
    neg_strategy_mod.DISABLE_VALUE_BOUNDS = False
    pers_strategy_mod.DISABLE_BUYER_TRUST_MODEL = False


def apply_ablation():
    belief_mod.DeltaBelief.update = lambda self, *a, **k: None
    neg_solver_mod.ValueBounds.tighten_from_rejection = lambda self, *a, **k: None
    pers_solver_mod.BuyerTrustModel.posterior_high_given_signal = lambda self: self.prior_p


def reset_per_game_state():
    # FIXED (2026-08-29): run_family() reseeds its own rng_master to the same
    # fixed seed (12345) on every call -- by design, so baseline and ablated
    # passes see identical trial configs. But that also means every trial
    # gets the exact same `game_id` string in both passes, and every
    # mechanism under test here is cached in a MODULE-LEVEL dict keyed by
    # game_id (belief_mod._BELIEFS, neg_strategy_mod._BOUNDS/
    # _LAST_SENT_PRICE, pers_strategy_mod._BUYER_TRUST/_BUYER_NO_TRUST,
    # barg_strategy_mod._LAST_SENT_TAKE, plus each family's _LAST_STYLE_ARM).
    # Without clearing these between passes, the "ablated" pass's lookups
    # for a given game_id found the SAME object the baseline pass already
    # populated with real accumulated evidence -- e.g. a bargaining
    # DeltaBelief the baseline pass had already updated toward the
    # opponent's true delta -- instead of a fresh, uniform-prior one.
    # Confirmed via direct comparison: this script's own "ablated" bargaining
    # numbers differed from a genuinely fresh, single-pass run of
    # montecarlo_opponents.py under the live DISABLE_DELTA_BELIEF=True flag
    # (hardliner mean 0.0161 here vs 0.0104 fresh) even though both are
    # supposed to represent the identical "mechanism off" condition.
    # Persuasion's numbers were unaffected by the same collision only
    # because its ablation patches `posterior_high_given_signal` itself
    # (the read), not the `.buys`/`.highs` counters that accumulate the
    # contamination -- so a stale, contaminated trust object still produces
    # the correct ablated decision once that method is overridden. Clearing
    # every per-game_id cache before each pass removes the collision
    # entirely without changing the paired-seed methodology (game_id
    # STRINGS still repeat across passes, deliberately, so configs stay
    # identical -- only the leftover STATE attached to them is wiped).
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
    # Clear every per-game_id cache before EACH pass (not just once at
    # startup) -- see reset_per_game_state's own docstring for why a stale
    # entry from a prior pass, found via a repeated game_id, would silently
    # leak that prior pass's accumulated state into this one.
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
