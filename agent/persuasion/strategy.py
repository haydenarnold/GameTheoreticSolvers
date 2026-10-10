"""Persuasion strategy: wires the disclosure solver + trust models into
seller/buyer actions. Payoff deltas (not message parsing) are used to infer
last round's outcome — `seller_total_payoff`/`buyer_total_payoff` are
documented cumulative fields, so this works in both text and binary mode
without needing to parse free text.

CONFIRMED live (2026-08-22): the buyer's trust model is a strict bootstrap
deadlock without the exploration carve-out below — it can only learn from
realized purchases, purchases only happen when the current (prior-only,
before any data) expected value already clears the price, and when it
doesn't, the buyer never buys, so it never gets data, so it never learns.
A real 20-round game hit this exactly: unconditional EV landed precisely on
price, the buyer declined every single round despite the seller signaling
"buy" 6 times (a plausibly honest ~30% rate matching the prior), and ended
0-for-20. See `buyer_decision` below for the fix.

CONFIRMED live (2026-08-22), a second and third finding along the same
theme -- a strategy shouldn't lock itself out for the rest of a 20-round
game over a small, early, possibly-unlucky sample:

(1) Buyer side: even with the bootstrap fix above, once the post-bootstrap
estimate first dipped below price, the buyer stayed locked out
PERMANENTLY -- one real game went 15+ rounds with zero further purchases
despite the raw prior remaining exactly as favorable the whole time (true
quality is IID each round; nothing about the game state actually changed).
`BuyerTrustModel` used to add a periodic re-exploration mechanism for this
(one more bounded test every `re_explore_cooldown` declined-signal rounds).

REMOVED (2026-08-24): periodic re-exploration was itself confirmed live to
be badly net-negative -- replayed against 102 real completed games (912 buy
events), it landed only a 4.2% high-quality rate (avg -$320,874/event)
vs. 76.3% (avg +$318,073/event) for ordinary posterior-cleared buys. The
premise's flaw: quality IS IID per round, but a SELLER'S SIGNALING HONESTY
isn't -- it's a fixed trait of that opponent for the whole game, so
re-testing after real evidence of unreliability doesn't get a fresh,
independent draw, it just re-samples the same already-revealed-unreliable
seller. See `BuyerTrustModel`'s docstring in `solver.py` for the full
numbers. The bootstrap fix above (a single bounded test, not a repeating
one) is unaffected and stays -- it fires once, before there's any evidence
either way, which is a fundamentally different situation.

(2) Seller side, the more damaging direction: with a textbook-optimal q*
computed to bluff on essentially every low-quality round (the common
breakeven-prior case, `p*v ~= price`), a SINGLE early bluff that happens to
land on bad luck can poison an adaptive buyer's trust for the ENTIRE REST
of the game. Confirmed live: a seller (us) bluffed on round 1 of a 20-round
game, the buyer bought and got burned, and never engaged again -- not even
once, across roughly ten more rounds where we were completely honest about
genuine high-quality items. The textbook q* formula treats every round as
an isolated one-shot signal with no reputation cost; a real 20-round
relationship has much more value in a track record than in one early
gamble. `SELLER_TRUST_BUILDING_ROUNDS`/`_MAX_Q` cap how aggressively we
bluff before any track record exists, regardless of what q* computes to.
"""

from __future__ import annotations

import random

from agent import history
from agent.persuasion.signal import classify_discourage_signal
from agent.persuasion.solver import (
    BuyerTrustModel,
    SellerDisclosureAdapter,
    optimal_bluff_prob,
)
from agent.rhetoric.bandit import StyleBandit, get_style_bandit, persist_style_bandit
from agent.rhetoric.compose import compose_message

FALLBACK_LOW_QUALITY_BLUFF_PROB = 0.35

# DERIVED (2026-08-23), not yet live-tested: every threshold below used to
# require only bare EV non-negativity (>= breakeven) before buying or
# spending a purchase on exploration. That's the right bar if the score
# were raw payoff -- it is NOT the right bar for what's actually scored:
# percentile rank against a reference pool on this exact configuration,
# which is not itself full of EV-argmax optimizers. In an exact- or
# near-breakeven spot, resolving every tie toward "buy" adds variance that
# helps our percentile about as often as it hurts it -- and several real
# games (an exact p*v==price tie repeatedly re-exploring into bad luck; a
# decline-test firing at exactly the shape-ratio boundary) came back
# confirming that pattern costs real rating points, not just occasional
# noise. BUYER_SAFETY_MARGIN requires a genuine cushion above breakeven --
# not just non-negative -- before taking on that variance at all. Applied
# uniformly to every buy/explore decision below rather than tuned
# per-mechanism, to keep the reasoning in one place; 0.20 is a judgment
# call (not derived from anything more precise), chosen to clearly exclude
# the exact-tie and boundary cases actually observed live while still
# leaving real room for genuinely favorable configs (the original
# motivating case for exploration at all, ratio ~2-3x, stays far above it).
BUYER_SAFETY_MARGIN = 0.20
# Ablation study (Table 2) showed trust model costs ~0.040/game vs
# prior-only. This only gates USING the LEARNED posterior
# (`trust.posterior_high_given_signal()` / `no_trust.posterior_high_given_
# signal()`) once a bootstrap test has run -- it must NOT gate the
# one-shot bounded bootstrap test itself (BOOTSTRAP_EV_RATIO_THRESHOLD /
# DECLINE_TEST_UPSIDE_RATIO_THRESHOLD below), which is a separate,
# independently-validated mechanism. It briefly did gate the bootstrap too
# (until 2026-08-30 -- see that fix's comment at both bootstrap call
# sites), which silently zeroed out bootstrap testing live despite the
# regression tests for it passing (they monkeypatched this flag off).
DISABLE_BUYER_TRUST_MODEL = True

# DERIVED (2026-08-30), from real live play: a seller who sends the exact
# same message every single round -- whether a literal repeated "yes"/"no"
# in binary mode, or byte-identical text in text mode -- is by definition
# not conditioning that message on true quality, so it carries zero
# information regardless of what DISABLE_BUYER_TRUST_MODEL says about
# trusting a LEARNED posterior. The one place this still mattered even with
# the trust model already disabled: on a discourage ("no") signal, the
# existing code declines unconditionally with no fallback to the raw prior
# at all (the `not DISABLE_BUYER_TRUST_MODEL and worth_testing` gate below
# is never true while the trust model is off) -- so a seller who ALWAYS
# discourages, regardless of true quality, currently locks us out of the
# whole game even when the raw, message-independent expected value clearly
# favors buying. Requiring at least 3 consecutive identical messages
# (including the current one) before concluding "no intel" mirrors the
# STATIONARY_OPPONENT_ROUNDS convention used for the same purpose in
# bargaining/negotiation -- one or two matching rounds could still be
# coincidence; three is a real pattern.
MIN_ROUNDS_FOR_NO_INTEL_DETECTION = 3

# Gates whether the unconditional prior is favorable enough to be worth
# testing a discourage ("no") signal at all -- see `unconditional_ev_ok`
# below. (Used to also gate the buyer-side periodic re-explore mechanism,
# removed 2026-08-24 -- see the module docstring.)
EXPLORATION_EV_RATIO_THRESHOLD = 1.0 + BUYER_SAFETY_MARGIN

# CONFIRMED live (2026-08-23): a real 20-round game had p=0.8, v=$125, u=$0,
# price=$100 -- unconditional EV landed exactly on price (ratio 1.0), just
# below EXPLORATION_EV_RATIO_THRESHOLD (1.2). The seller signaled "buy" 19 of
# 20 rounds with a specific, consistent claim ("I checked this unit myself
# and it's high quality"). We declined all 20 -- 0-for-20, reproducing the
# original bootstrap-deadlock bug this module already fixed once, via a
# different path: while `trust.buys < trust.min_observations`,
# `posterior_high_given_signal()` returns the raw `prior_p`, so gating the
# bootstrap branch on EXPLORATION_EV_RATIO_THRESHOLD made it compute the
# EXACT SAME quantity, against the EXACT SAME threshold, as the final
# else-branch decision below it -- the bootstrap branch could only ever fire
# "yes" in cases where the plain decision would already say "yes" anyway,
# making it dead code for its entire purpose (learning below that bar). A
# single bounded bootstrap test (`min_observations` purchases, default 2) is
# cheap information-gathering, not the repeated-re-exploration pattern the
# margin above was built to prevent -- so it only needs to clear bare
# breakeven, not the full safety margin.
BOOTSTRAP_EV_RATIO_THRESHOLD = 0.95
# DERIVED (2026-08-25), empirically re-checked against real data before
# lowering: a 400-game live pull found the ONLY sub-1.0 config that actually
# recurs is p=0.8, v=1.2x price, u=0 (ratio=0.96) -- it appeared in 21/200
# (10.5%) of real buyer-role games, every one of them a guaranteed $0 under
# the old threshold, since the bootstrap gate never even tried it. `quality`
# is genuinely unobservable on rounds we don't buy (confirmed: 0 of 2,501
# non-bought history entries carry it), so this can't be measured directly
# for the exact blocked population -- the best available proxy is the
# empirical high-quality rate on real p=0.8 "yes"-signaled purchases
# elsewhere in the same 400-game pull: 88.0% (n=50, bootstrap-window buys
# specifically) and 85.7% (n=321, all p=0.8 yes-signaled buys), both above
# this config's 83.3% breakeven (price=1.0/v=1.2 -> (price-u)/(v-u)). The
# 95% binomial CI on the n=50 estimate is roughly [79%, 97%] -- straddles
# breakeven, so this isn't airtight, but the risk is hard-bounded to
# min_observations (2) test purchases per affected game, same bounded
# mechanism the original 1.0 threshold already relies on (which itself was
# set from a single live anecdote -- this is a stronger evidence base, not
# a weaker one). 0.95 clears exactly this validated config with a small
# margin, without extrapolating to unvalidated lower ratios.

# DERIVED (2026-08-23), not yet live-tested: a discourage signal used to be
# treated as fully decisive, unconditionally -- no mechanism ever tested
# whether THIS seller's "no" was actually reliable, no matter how favorable
# the raw configuration was. Found via a real game (p=0.5, v=$4,000,000,
# u=$0, price=$1,000,000): the unconditional EV already clears price 2x
# over, and the breakeven reliability for a "no" signal to be worth obeying
# there is only 75% (`r* = (v-price)/(v-u)`) -- so even a seller whose "no"
# is right 70% of the time would still make testing it +EV, and we had zero
# way to find out where a given seller actually falls on that scale.
#
# DECLINE_TEST_UPSIDE_RATIO_THRESHOLD gates whether it's even worth
# bootstrapping a test at all: (v-price)/(price-u) is the upside-if-wrong
# vs. downside-if-right ratio, and 1.0 would be the breakeven point under a
# maximally uninformative 50/50 guess about the signal's reliability.
# CONFIRMED live (2026-08-23): a real game (p=0.33, v=$2,000,000, u=$0,
# price=$1,000,000) had this ratio land at EXACTLY 1.0 -- clearing the old
# bare-breakeven gate -- while the unconditional prior (0.66) sat below
# EXPLORATION_EV_RATIO_THRESHOLD, so "no" got bootstrap-tested twice (both
# genuinely low quality, -$1M each) while every real "yes" endorsement in
# the same game was declined. Requiring the same margin here as the yes
# side keeps both exploration directions consistent and excludes exactly
# this boundary case, not just clearly-unfavorable ones.
DECLINE_TEST_UPSIDE_RATIO_THRESHOLD = 1.0 + BUYER_SAFETY_MARGIN

# Starting belief that a "no" signal will turn out to be high quality
# anyway (i.e. 1 minus our assumed default reliability). A seller has a
# real game-theoretic reason to be honest on "no" -- bluffing "yes" is
# always weakly better for their own immediate payoff, so a genuine "no"
# usually reflects either real quality concern or reputation-building (the
# same logic our own SELLER_TRUST_BUILDING_ROUNDS relies on) -- so this
# starts skeptical of testing (0.15, i.e. 85% assumed reliability), and
# only shifts based on real evidence from bounded testing, same as
# BuyerTrustModel already does for the "yes" side.
DECLINE_TRUST_PRIOR_P_HIGH = 0.15

# CONFIRMED live (2026-08-25): a population-level measurement pass across
# 1,017 classified real seller messages (300-game sample) found buy rates
# varying substantially and consistently by which moral-foundation StyleBandit
# arm the message led with: loyalty 64.6% (n=223), purity 57.8% (n=218),
# fairness 53.3% (n=214), authority 46.9% (n=49), care 40.6% (n=313) --
# chi2=45.0 across the 5 arms, p<<0.001. Checked against three plausible
# confounds before trusting this: (1) true quality -- gap survives when
# split by quality, and is STARKEST on bluff/low-quality rounds specifically
# (care-framed bluffs converted 24.9% vs loyalty-framed bluffs at 73.3%,
# ruling out "care just gets used more on genuinely weak items"); (2) round
# number -- the same ranking (loyalty highest, care lowest) holds
# independently within early/mid/late round buckets, ruling out "care
# correlates with less-trust-built-up early rounds"; (3) config -- mean
# v/price ratio is 2.09-2.28 across all five arms with no meaningful skew,
# confirming Thompson Sampling's arm choice is config-independent as
# expected by construction. StyleBandit's per-opponent posterior (Beta(1,1)
# or Beta(PERSISTENT_PRIOR_STRENGTH,...) uniform prior) has no way to
# encode this population-level information -- every fresh bandit starts
# genuinely agnostic between all 5 arms and can only learn it's wrong about
# `care` from real per-opponent evidence, one opponent at a time, which the
# separate 2026-08-25 measurement pass found hasn't happened yet for any of
# 48 tracked opponents (all 90% credible intervals still overlap, even for
# the opponent with the most accumulated data). SELLER_FOUNDATION_PRIOR_MEANS
# seeds StyleBandit's prior at these population rates instead of a flat 0.5
# per arm (see StyleBandit.__init__'s `prior_means` param) -- real
# per-opponent evidence still fully overrides this as it accumulates, same
# as it already overrides the existing uniform prior; this only changes
# where a bandit STARTS, not whether it can still learn a specific opponent
# doesn't fit this population pattern. Scoped to persuasion's seller-side
# `_style_bandit` only -- bargaining/negotiation's own StyleBandit usage is
# untouched (their own measurement pass found a different, weaker-evidence
# pattern on a different pair of foundations, not re-derived here).
SELLER_FOUNDATION_PRIOR_MEANS = {
    "loyalty": 0.646,
    "purity": 0.578,
    "fairness": 0.533,
    "authority": 0.469,
    "care": 0.406,
}

# A single early bluff that lands on bad luck can poison an adaptive
# buyer's trust for the entire rest of a repeated 20-round game -- far more
# costly than the small edge q* buys by bluffing on round 1 specifically,
# before we have any track record at all. Cap effective q well below
# textbook-optimal for the first few rounds regardless of what q* computes.
#
# CONFIRMED live (2026-08-23): 0.3 still wasn't low enough -- a real game
# had a round-1 bluff land (a 30% draw) and the buyer bought exactly once
# (that bluff) and declined 8 subsequent GENUINELY honest high-quality
# offers for the rest of the game: +$10K from the bluff, -$80K in foregone
# honest sales. This is the third real game this session showing the same
# shape (one early/isolated bluff -> outsized, long-lasting trust
# collapse), and it happened WITHIN this already-conservative window, where
# SELLER_BLUFF_DAMPENING doesn't even apply (0.3 was already below half of
# every q* seen so far). Lowered further given how consistently the
# downside has outweighed the upside.
SELLER_TRUST_BUILDING_ROUNDS = 3
# CONFIRMED live (2026-08-25): 0.1 still wasn't low enough -- a real game
# (p=0.5, v=$3M, u=$0, price=$1M, where the prior alone already clears
# breakeven so q* clips to 1.0 -- textbook "always recommend," valid only
# for a memoryless one-shot buyer) had a round-1 bluff land inside that 10%
# window. The buyer bought once (the bluff, +$1M) then declined all 9
# subsequent GENUINELY honest high-quality "yes" signals for the rest of
# the game -- $9M in foregone honest sales against the $1M gained. This is
# at least the FOURTH real game this session (across two prior tightenings,
# 0.3->0.1) showing the identical shape: the round with the least possible
# trust to lose (no track record has even started yet) is exactly where a
# caught bluff does maximum, game-long damage, and the cost has
# consistently outweighed the gain by 3-9x every time it's been measured.
# Rather than find another smaller-but-still-nonzero value that keeps
# failing the same way, eliminate bluffing entirely during this window --
# q*=1.0 in particular reflects zero-information "always say yes," which
# has no signaling value against a learning buyer to begin with, only
# downside risk once caught.
SELLER_TRUST_BUILDING_MAX_Q = 0.0
# RE-EXAMINED (2026-08-26), no change: the same 500-game real pull and
# counterfactual simulation used to re-derive SELLER_BLUFF_DAMPENING below
# also swept this constant (0.0/0.05/0.10/0.15). The effect size was tiny
# and inconsistent across three independent sweeps (roughly -0.6% to +0.6%,
# no clean monotonic direction, well within simulation noise) -- nowhere
# near the clear, monotonic signal that justified moving
# SELLER_BLUFF_DAMPENING. Given four real, documented live incidents all
# showing the same catastrophic, asymmetric shape (3-9x cost/benefit) when
# bluffing happens in this exact window, and no real evidence here to
# offset that, 0.0 stays as-is -- this is a case where the data genuinely
# doesn't support a change, not a value nobody rechecked.

# CONFIRMED live (2026-08-23): q* maximizes one-shot expected value assuming
# a fast, cleanly Bayesian opponent -- it doesn't price in that a caught or
# even successful bluff can trigger an extended, slow-recovering trust
# collapse in a real opponent. Two real games (same session) both showed
# this: a single successful bluff past the SELLER_TRUST_BUILDING_ROUNDS
# window (round 8 in both cases) was followed by 3-5 subsequent GENUINELY
# high-quality, honestly-signaled rounds all getting declined anyway --
# $3-5M in foregone honest sales against $1-2M gained from the bluffs
# themselves. Net negative both times, not just occasional noise.
# SELLER_BLUFF_DAMPENING scales q* down for the WHOLE game, not just the
# first few rounds -- trading some theoretical one-shot EV for materially
# lower risk of triggering a collapse that's now shown to cost several
# times what it gains. SellerDisclosureAdapter still adapts on top of this
# (can still raise q for a genuinely under-skeptical buyer), this just sets
# the untested starting point.
#
# DERIVED (2026-08-26): re-examined empirically rather than left as an
# unrevisited judgment call, now that a large real sample exists (500-game
# pull, 243 seller-role games, 4,860 rounds across 39 distinct configs --
# see scratchpad/persuasion_bluff_analysis.py) and the buyer-side fix
# earlier this session (BuyerTrustModel prior_pseudo_count 5->20) changes
# the dynamics this constant was tuned against. A counterfactual simulation
# through the REAL current persuasion_strategy() code end-to-end (both
# seller and buyer sides; scratchpad/persuasion_seller_sim.py), sweeping
# dampening across the 39 real configs at ~15-20 trials each, found a
# consistent, monotonic seller-payoff gain moving from 0.5 toward 1.0 (three
# independent sweeps all agreed: 0.5->0.7 alone already captures roughly
# half of the total 0.5->1.0 gain, ~+1-2% mean payoff). Checked the tail
# specifically, not just the mean, since the original 0.5 value exists to
# avoid a catastrophic-collapse tail, not to maximize average payoff:
# p10/p25/worst-5 of the per-game payoff distribution were IDENTICAL across
# every dampening level tested (scratchpad/persuasion_seller_sim_tail.py) --
# raising dampening improved the middle/upper of the distribution without
# creating or worsening any bad tail, at least against a well-calibrated
# buyer.
#
# That caveat is the reason this moves to 0.7, not all the way to the 1.0
# the simulation's mean actually favors: the simulated "opponent" is our
# OWN buyer-side logic standing in for a real one, and it is now
# specifically MORE FORGIVING of bad luck than it used to be (that's the
# whole point of the K=20 fix) -- likely more forgiving than the unknown
# mix of real opponents (humans included) whose brittler reactions produced
# the original live incidents this constant was built to prevent. 0.7
# banks about half the measured, real, tail-safe gain while remaining a
# meaningfully smaller step than the full theoretical optimum, consistent
# with how SELLER_TRUST_BUILDING_MAX_Q itself was walked down incrementally
# (0.3->0.1->0.0) as evidence accumulated, rather than jumped in one move.
SELLER_BLUFF_DAMPENING = 0.7

# Asymmetric-trust cue for text mode: a profit-seeking seller has zero cost
# to send a positive pitch every round (cheap talk, weak evidence on its
# own — this is exactly what q*-calibrated bluffing exploits), but a
# seller has every incentive to NEVER volunteer a negative statement unless
# the round genuinely warrants it, so an explicit discouraging phrase is
# unusually credible and treated as decisive. CONFIRMED live: a seller
# wrote "This product is low quality — you should pass on it" and we
# bought anyway, losing the full price, because text mode had no signal
# detection at all before this fix.
#
# CONFIRMED live (2026-08-22), the lexicon was still too narrow: a seller
# used relative/comparative discouragement -- "this one is ordinary. Wait
# for the next -- I'll tell you when it's worth it." -- with no absolute
# negative phrase in it at all ("worth it" even appears in a positive
# framing). None of the existing phrases matched, so it defaulted to an
# implicit buy signal and we bought twice on this exact wording, both times
# genuinely low quality. Comparative/deferral language ("this one is
# X, wait for a better one") is just as credible a discourage signal as an
# absolute one -- a profit-seeking seller still has zero incentive to
# volunteer it unless the round genuinely doesn't warrant a sale.
DISCOURAGE_PHRASES = (
    "low quality", "low-quality", "wouldn't recommend", "would not recommend",
    "don't recommend", "do not recommend", "not recommend", "pass on it",
    "should pass", "hold off", "skip this", "steer clear", "not worth it",
    "poor quality", "subpar", "avoid this", "wouldn't buy", "would not buy",
    "ordinary", "mediocre", "middling", "nothing special", "run of the mill",
    "run-of-the-mill", "wait for the next", "wait for a better", "not this one",
    "not the one", "sit this one out", "sit this round out",
    # CONFIRMED live (2026-08-22): "you may want to pass today" slipped past
    # every existing phrase (closest was "should pass"/"pass on it") -- cost
    # two real purchases (-100 each) on genuinely low-quality rounds in the
    # same game. "isn't my best" is the same seller's other unmatched
    # hedge ("this one isn't my best unit") from the same message.
    "may want to pass", "want to pass today", "isn't my best",
    # CONFIRMED live (2026-08-23): "This unit is below our usual quality
    # this round" -- a relative-comparison hedge, not caught by any existing
    # phrase -- was used consistently and honestly by a seller across 6
    # rounds of a 20-round game (all 6 genuinely low quality), and we bought
    # all 6 anyway (a hugely favorable prior, EV=$240 vs $100 price, meant
    # the buyer defaulted straight to "yes" on the unmatched message). Cost
    # $600 out of an achievable $2800 -- over a quarter of the game's profit
    # -- in a game we otherwise played perfectly (bought all 14 genuine
    # high-quality rounds too). Third distinct discourage-phrasing miss this
    # session; see the module's live incident log for the other two.
    "below our usual quality", "below our usual",
)

# DERIVED (2026-08-23), not yet live-tested: three separate discourage-
# phrasing misses in one session (see above) made clear a fixed keyword list
# is structurally always one phrasing behind -- it can only ever catch
# wording that's already been seen live once. `agent.persuasion.signal`
# classifies the message with an LLM instead (a real, general "does this
# discourage buying" judgment), and DISCOURAGE_PHRASES above is now only a
# FALLBACK: used when the LLM call fails outright, or reports confidence
# below this threshold. 0.7 is a judgment call, not derived from anything --
# LLM self-reported confidence isn't well-calibrated in general, so this
# picks a moderately conservative bar that still lets clear cases (which is
# most of them; the phrase-list misses above were all fairly unambiguous
# discouragement, just novel wording) resolve via the LLM, while deferring
# genuinely ambiguous ones to the phrase list's proven-conservative default.
DISCOURAGE_CONFIDENCE_THRESHOLD = 0.7

_SELLER_ADAPTERS: dict[str, SellerDisclosureAdapter] = {}
_LAST_SELLER_SIGNAL: dict[str, bool] = {}
_LAST_SELLER_PAYOFF: dict[str, float] = {}

# Per-game (never pooled across STRANGERS -- see StyleBandit's docstring)
# Thompson Sampling over which moral-foundation framing to lead a "buy"
# pitch with, adapted to this game's buyer's own revealed behavior.
# `get_style_bandit`/`persist_style_bandit` (agent.rhetoric.bandit) resolve
# to a disk-persisted, cross-game bandit keyed by the opponent's revealed
# name when known, or an ordinary fresh per-game bandit when identity is
# hidden -- see that module's docstring for the full rationale.
_LAST_STYLE_ARM: dict[str, str] = {}

_BUYER_TRUST: dict[str, BuyerTrustModel] = {}
_LAST_BUYER_BOUGHT: dict[str, bool] = {}
_LAST_BUYER_PAYOFF: dict[str, float] = {}

# Mirrors _BUYER_TRUST but for testing "no" signals instead of "yes" ones --
# same BuyerTrustModel class, just a second independent instance per game,
# seeded with DECLINE_TRUST_PRIOR_P_HIGH. _LAST_BUYER_TESTED_NO records
# which model a pending purchase's quality result should be credited to,
# since a "yes" purchase and a "test the no" purchase must update different
# models even though both show up as `bought=True` on the next round.
_BUYER_NO_TRUST: dict[str, BuyerTrustModel] = {}
_LAST_BUYER_TESTED_NO: dict[str, bool] = {}


def _seller_adapter(game_id: str, q0: float) -> SellerDisclosureAdapter:
    if game_id not in _SELLER_ADAPTERS:
        _SELLER_ADAPTERS[game_id] = SellerDisclosureAdapter(q0)
    return _SELLER_ADAPTERS[game_id]


def _opponent_display_name(game: dict) -> str | None:
    return (game.get("opponent") or {}).get("name")


def _style_bandit(game: dict) -> StyleBandit:
    return get_style_bandit(game["game_id"], _opponent_display_name(game), prior_means=SELLER_FOUNDATION_PRIOR_MEANS)


def _update_seller_adapter_from_last_round(game: dict, state: dict, price: float) -> None:
    game_id = game["game_id"]
    if game_id not in _LAST_SELLER_SIGNAL:
        return
    signaled = _LAST_SELLER_SIGNAL.pop(game_id)
    prev_payoff = _LAST_SELLER_PAYOFF.get(game_id, 0.0)
    current_payoff = state.get("seller_total_payoff", 0.0)
    bought = (current_payoff - prev_payoff) > price / 2
    if signaled and game_id in _SELLER_ADAPTERS:
        _SELLER_ADAPTERS[game_id].record(bought)
    arm = _LAST_STYLE_ARM.pop(game_id, None)
    if arm is not None:
        _style_bandit(game).update(arm, bought)
        persist_style_bandit(game_id, _opponent_display_name(game))


def _buyer_trust(game_id: str, prior_p: float) -> BuyerTrustModel:
    if game_id not in _BUYER_TRUST:
        _BUYER_TRUST[game_id] = BuyerTrustModel(prior_p)
    return _BUYER_TRUST[game_id]


def _buyer_no_trust(game_id: str) -> BuyerTrustModel:
    if game_id not in _BUYER_NO_TRUST:
        _BUYER_NO_TRUST[game_id] = BuyerTrustModel(DECLINE_TRUST_PRIOR_P_HIGH, min_observations=1)
    return _BUYER_NO_TRUST[game_id]


def _seller_signaled_buy(state: dict) -> bool:
    """Whether THIS round's seller message endorses buying. Binary mode
    carries a clean yes/no. Text mode has no reliable positive signal to
    parse (a generic pitch is cheap and uninformative on its own), but an
    explicit discouraging message is a different matter — a seller has no
    incentive to volunteer one unless it's genuinely warranted, so it's
    treated as decisive rather than ignored.

    Text mode tries the LLM classifier first (a general judgment, not
    limited to previously-seen phrasings); only falls back to the fixed
    DISCOURAGE_PHRASES list when that call fails outright or comes back
    below DISCOURAGE_CONFIDENCE_THRESHOLD -- see the constant's own comment
    for why a fixed list alone kept missing new phrasings."""
    raw = state.get("seller_message")
    if state.get("seller_message_type") == "binary":
        if isinstance(raw, str):
            return raw.strip().lower() in ("yes", "true", "buy", "recommend")
        return bool(raw)
    if isinstance(raw, str) and raw.strip():
        llm_result = classify_discourage_signal(raw)
        if llm_result is not None:
            discourages, confidence = llm_result
            if confidence >= DISCOURAGE_CONFIDENCE_THRESHOLD:
                return not discourages
        lowered = raw.lower()
        if any(phrase in lowered for phrase in DISCOURAGE_PHRASES):
            return False
    return True


def _seller_message_repeated_every_round(state: dict) -> bool:
    """Whether the seller's message -- the literal string, so this covers
    binary mode's "yes"/"no" and text mode's actual prose the same way --
    has been byte-identical for every round of the game so far, including
    the current one. Requires at least MIN_ROUNDS_FOR_NO_INTEL_DETECTION
    matching rounds before concluding this isn't coincidence -- see that
    constant's derivation."""
    current = state.get("seller_message")
    if current is None:
        return False
    prior = [h.get("seller_message") for h in state.get("history") or []]
    seen = prior + [current]
    if len(seen) < MIN_ROUNDS_FOR_NO_INTEL_DETECTION:
        return False
    return all(m == current for m in seen)


def _update_buyer_trust_from_last_round(game_id: str, state: dict, price: float, v: float, u: float) -> None:
    if game_id not in _LAST_BUYER_BOUGHT:
        return
    bought = _LAST_BUYER_BOUGHT.pop(game_id)
    tested_no = _LAST_BUYER_TESTED_NO.pop(game_id, False)
    if not bought:
        return
    prev_payoff = _LAST_BUYER_PAYOFF.get(game_id, 0.0)
    current_payoff = state.get("buyer_total_payoff", 0.0)
    delta = current_payoff - prev_payoff
    quality_was_high = abs(delta - (v - price)) < abs(delta - (u - price))
    # A purchase that tested a discourage signal must credit the decline
    # model, not the (unrelated) "yes"-signal trust model, even though both
    # look identical as `bought=True` from here.
    if tested_no:
        if game_id in _BUYER_NO_TRUST:
            _BUYER_NO_TRUST[game_id].record(quality_was_high)
    elif game_id in _BUYER_TRUST:
        _BUYER_TRUST[game_id].record(quality_was_high)


def persuasion_strategy(game: dict) -> dict:
    state = game["game_state"]
    valid_actions = game["valid_actions"]
    game_id = game["game_id"]
    price = state["product_price"]
    p = state["p"]

    if valid_actions["type"] in ("seller_message", "seller_recommendation"):
        _update_seller_adapter_from_last_round(game, state, price)

        quality_high = state["current_quality"] == "high"
        know_cv = state.get("is_seller_know_cv", False) and "v" in state and "u" in state
        q_base = (
            optimal_bluff_prob(p, state["v"], state["u"], price)
            if know_cv
            else FALLBACK_LOW_QUALITY_BLUFF_PROB
        )
        q_base *= SELLER_BLUFF_DAMPENING
        q = _seller_adapter(game_id, q_base).adapted_q()
        if int(state.get("round", 1)) <= SELLER_TRUST_BUILDING_ROUNDS:
            q = min(q, SELLER_TRUST_BUILDING_MAX_Q)
        signal_buy = quality_high or (random.random() < q)

        if valid_actions["type"] == "seller_recommendation":
            action = {"decision": "yes" if signal_buy else "no"}
        else:
            # Style adaptation only applies to a genuine "buy" pitch -- it's
            # about maximizing conversion, not about making a discourage
            # message more credible (a different, out-of-scope goal).
            style = _style_bandit(game).select_arm() if signal_buy else None
            message = compose_message(
                family="persuasion",
                situation=game["prompt"],
                stance={
                    "role": "seller",
                    "signaling_buy": signal_buy,
                    "fallback_message": (
                        "This is a strong unit — I'd recommend it."
                        if signal_buy
                        else "I'd hold off on this one."
                    ),
                },
                profile=None,
                style=style,
            )
            action = {"message": message}
            if style is not None:
                _LAST_STYLE_ARM[game_id] = style

        _LAST_SELLER_SIGNAL[game_id] = signal_buy
        _LAST_SELLER_PAYOFF[game_id] = state.get("seller_total_payoff", 0.0)
        history.record_own_action(game, action)
        return action

    if valid_actions["type"] == "buyer_decision":
        v, u = state["v"], state["u"]
        _update_buyer_trust_from_last_round(game_id, state, price, v, u)
        tested_no = False

        if _seller_message_repeated_every_round(state):
            # See MIN_ROUNDS_FOR_NO_INTEL_DETECTION's derivation -- a
            # message that never varies carries no information regardless
            # of what it says THIS round, so decide from the raw,
            # message-independent prior alone (identical to the
            # DISABLE_BUYER_TRUST_MODEL "yes" branch below) instead of
            # branching on `_seller_signaled_buy`. This is the one case
            # that branch doesn't already cover: an always-discouraging
            # seller currently locks us out unconditionally (the
            # `not DISABLE_BUYER_TRUST_MODEL and worth_testing` gate is
            # never true while the trust model is off), even when the raw
            # prior clearly favors buying.
            effective_p = p
            expected_value = effective_p * v + (1 - effective_p) * u
            decision = "yes" if expected_value >= price * (1 + BUYER_SAFETY_MARGIN) else "no"
        elif _seller_signaled_buy(state):
            trust = _buyer_trust(game_id, p)
            prior_ev = p * v + (1 - p) * u
            bootstrap_ev_ratio_ok = price > 0 and prior_ev / price >= BOOTSTRAP_EV_RATIO_THRESHOLD
            if trust.buys < trust.min_observations and bootstrap_ev_ratio_ok:
                # CONFIRMED live: a real 20-round game had unconditional EV
                # land exactly on price (p*v+(1-p)*u == price to the cent).
                # trust.posterior_high_given_signal() can only move once it
                # has real purchase data, and that data can only come from
                # buying -- so a buyer that starts out unable to justify a
                # purchase under the raw prior is stuck there permanently,
                # 0-for-N even against a seller signaling "buy" 30% of the
                # time (a plausibly honest rate). Spend a small, bounded
                # number of purchases on "buy" signals specifically to find
                # out whether this seller is worth trusting, rather than
                # staying blind for the whole game -- but only when the raw
                # prior is already close enough to profitable that learning
                # the truth could plausibly change the verdict. Uses the
                # lower BOOTSTRAP_EV_RATIO_THRESHOLD, not
                # EXPLORATION_EV_RATIO_THRESHOLD -- see that constant's
                # comment for why using the same bar as the final decision
                # made this branch dead code.
                #
                # CONFIRMED live (2026-08-30): this check must run BEFORE the
                # DISABLE_BUYER_TRUST_MODEL branch below, not after it. It
                # used to be the `elif`, so with the flag on (its live
                # default) this bounded, separately-validated bootstrap test
                # was unreachable -- every recurring p=0.8/v=1.2x/u=0 config
                # (ratio 0.96, empirically 85-88% real high-quality rate,
                # see BOOTSTRAP_EV_RATIO_THRESHOLD's derivation) still went
                # 0-for-20 live, the exact regression this threshold was
                # added to fix. A 198-game live pull the same day found 11
                # such games, all 0-for-20, $0 total. The ablation cost
                # cited below is for the LEARNED posterior
                # (`trust.posterior_high_given_signal()`, used only once
                # this bootstrap is exhausted), which is a distinct
                # mechanism from this one-shot bounded test and was never
                # itself measured as costly.
                decision = "yes"
            elif DISABLE_BUYER_TRUST_MODEL:
                effective_p = p
                expected_value = effective_p * v + (1 - effective_p) * u
                decision = "yes" if expected_value >= price * (1 + BUYER_SAFETY_MARGIN) else "no"
            else:
                effective_p = trust.posterior_high_given_signal()
                expected_value = effective_p * v + (1 - effective_p) * u
                # Require a genuine margin, not just >= breakeven -- see
                # BUYER_SAFETY_MARGIN above.
                decision = "yes" if expected_value >= price * (1 + BUYER_SAFETY_MARGIN) else "no"
        else:
            # A rational seller USUALLY has no incentive to discourage a
            # sale unless the round is genuinely bad -- but "usually" isn't
            # "always", and whether that's worth acting on depends on how
            # favorable the raw configuration is (see
            # DECLINE_TEST_UPSIDE_RATIO_THRESHOLD's derivation above: even a
            # "no" that's right 70% of the time can still be worth
            # overriding if the upside is big enough). Test it, bounded,
            # only when doing so clears that bar; otherwise fall back to
            # the old unconditional trust -- most configurations won't
            # clear it, so this stays the default behavior in practice.
            downside = price - u
            upside = v - price
            # CONFIRMED live (2026-08-23): the upside/downside shape ratio
            # alone isn't enough -- a real game (p=0.33, v=$2,000,000,
            # u=$0, price=$1,000,000; unconditional EV ratio 0.66, BELOW
            # the yes-side's own EXPLORATION_EV_RATIO_THRESHOLD of 0.75)
            # had upside/downside=(v-price)/(price-u)=1.0, clearing this
            # gate on its own -- so the buyer bootstrap-tested TWO
            # discourage signals (bought, both genuinely low quality, -$1M
            # each) while every genuine "yes" endorsement in the same game
            # got declined, because the yes-side bootstrap never even had a
            # chance to fire under that same unfavorable unconditional
            # prior. Net result: -$2,000,000, worse than doing nothing.
            # Backwards: testing "no" MORE aggressively than "yes" in a
            # config where "yes" isn't even trusted enough to test is never
            # right. Now requires the SAME unconditional-EV bar the yes
            # side already uses, in addition to the shape ratio -- testing
            # either direction should only happen when the raw prior is
            # already reasonably favorable to begin with.
            unconditional_ev_ok = price > 0 and (p * v + (1 - p) * u) / price >= EXPLORATION_EV_RATIO_THRESHOLD
            worth_testing = (
                downside > 0 and upside > 0
                and upside / downside >= DECLINE_TEST_UPSIDE_RATIO_THRESHOLD
                and unconditional_ev_ok
            )
            if worth_testing:
                no_trust = _buyer_no_trust(game_id)
                if no_trust.buys < no_trust.min_observations:
                    # Same fix as the yes-side bootstrap above (2026-08-30):
                    # this bounded test must fire regardless of
                    # DISABLE_BUYER_TRUST_MODEL -- only the LEARNED posterior
                    # below stays behind that flag.
                    decision, tested_no = "yes", True  # bounded bootstrap test
                elif DISABLE_BUYER_TRUST_MODEL:
                    decision = "no"
                else:
                    effective_p_high = no_trust.posterior_high_given_signal()
                    expected_value = effective_p_high * v + (1 - effective_p_high) * u
                    # Same margin as the yes-side threshold above -- not
                    # just > breakeven.
                    if expected_value > price * (1 + BUYER_SAFETY_MARGIN):
                        decision, tested_no = "yes", True
                    else:
                        decision = "no"
            else:
                decision = "no"

        _LAST_BUYER_TESTED_NO[game_id] = tested_no
        _LAST_BUYER_BOUGHT[game_id] = decision == "yes"
        _LAST_BUYER_PAYOFF[game_id] = state.get("buyer_total_payoff", 0.0)
        action = {"decision": decision}
        history.record_own_action(game, action)
        return action

    return {}
