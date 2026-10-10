"""Bargaining strategy: wires the solver + belief tracker into actions.

Walkaway is provably weakly dominated by reject in this game: rejecting
never forecloses anything walkaway doesn't already foreclose (both end in
$0 if pushed to the limit), but rejecting preserves the option value of a
future counteroffer. So we never choose it.

CONFIRMED live (2026-08-22): an opponent reported at delta=1.0 (0%
inflation, unlimited horizon) makes the textbook infinite-horizon formula
`money*(1-delta_opp)/(1-delta_me*delta_opp)` evaluate to exactly zero — a
real degenerate limit of Rubinstein's model (a perfectly patient opponent
captures the whole pie in the idealized theory), not a bug in the formula.
But blindly executing that is bad: we already know "unlimited horizon"
isn't really infinite (a hidden round cap kicks in around round ~99-100
regardless of `horizon_known`), so an opponent reported as delta=1.0 is not
actually infinitely patient in practice, and handing over the entire pot on
a knife-edge theoretical extrapolation is an unforced, catastrophic loss.
MIN_SELF_SHARE_FRAC is a hard floor against that: we never propose keeping,
or require accepting, less than this fraction of the pie, no matter what
the raw formula says — applied only where the formula has no natural
deadline pulling it toward zero for a good reason (i.e. not the finite
near-deadline case, where accepting a small amount really is optimal).

CONFIRMED live (2026-08-22), the mirror-image failure: a game where WE were
reported at delta=1.0 made the same formula degenerate the other way — our
offer pinned at exactly 97% of the pot every single round (zero variation,
because delta_me=1.0 makes the formula return `money` regardless of
delta_opp), AND our own accept threshold came out to the full pot, meaning
we would only ever accept getting literally everything. That's not just an
aggressive opening stance — it makes agreement mathematically impossible
against any opponent unwilling to hand over 100%, which is exactly what
produced a 99-round deadlock ending in $0 for both. MAX_SELF_SHARE_FRAC
mirrors the floor: a hard ceiling on both what we offer to keep and what we
require to accept, so the same knife-edge can't lock us out of every
achievable deal in either direction.

CONFIRMED live (2026-08-22): even with both bounds in place, an unlimited-
horizon game reached round 99 with the opponent offering us $612,531 of a
$1,000,000 pot (61%) — a large, unambiguous net-positive amount — and we
rejected it, because our formula-driven threshold (capped at
MAX_SELF_SHARE_FRAC) still demanded 70%. The game then ended at the
confirmed hidden round cap with $0 for both. No theoretical threshold is
worth risking the proven $0 alternative for once we're deep enough into a
game that we know a hard cutoff is coming. DEADLINE_PANIC_ROUND is a
hardcoded override for exactly that situation: past this round, in an
unlimited-horizon game, we accept anything net-positive on the decision
side, and mirror the opponent's own last stated position on the offer side
(the single highest-probability-of-acceptance move available, since it's
literally what they already told us they want) rather than continue
proposing a split they've shown no sign of taking.

Even with the panic override, the unlimited-horizon path still spent two
full improvement cycles as the top-priority gap: between round 1 and
DEADLINE_PANIC_ROUND, `_my_take`/`_reject_value` return a value that only
depends on delta_me/delta_opp, never on elapsed rounds, so the ask (and
accept threshold) sits flat at whatever the MIN/MAX band allows for the
entire game, then hits a hard cliff at the panic round. Negotiation already
solved exactly this shape of problem with a continuous time-dependent
concession curve (Faratin, Sierra & Jennings 1998), reused here (imported
straight from `negotiation.solver`, not reimplemented).

CONFIRMED live (2026-08-22): that fix wasn't enough on its own. A
COMPLETE-information, FINITE-horizon game (max_rounds=12) had the opponent's
delta reported as exactly 1.0 -- and it turns out that's degenerate at
every finite horizon too, not just the infinite-horizon formula: with
delta_opp=1.0, the opponent loses literally nothing by rejecting and
becoming proposer at any future round, including the deadline itself, so
backward induction collapses to zero for us at every node, regardless of
rounds_remaining or parity. Our round-1 offer jumped straight to the 30%
floor with zero negotiation runway -- and the opponent accepted instantly,
the same "instant accept" signal that showed the original 15%/85% band was
too conservative. The unlimited-horizon softening never even ran, because
it was gated on `rounds_remaining is None` and this game had a real
max_rounds.

CONFIRMED live (2026-08-22), cycle 3's live 3-games-per-family check: a
COMPLETE-information, FINITE-horizon game (delta_1=0.8, delta_2=0.9,
max_rounds=12) held the ceiling-bound ask flat at 70% the entire game (the
deliberate, then-untested design choice below), and the opponent did not
concede until round 12 -- the literal, certain deadline. It worked out
this time, but there was no safety margin at all: one more round of
opponent stubbornness would have produced a guaranteed $0/$0, and unlike
the unlimited-horizon case, finite-horizon games had NO panic-zone
protection whatsoever -- `in_panic_zone` was gated on `rounds_remaining is
None`, so it could never fire when a real `max_rounds` was set, even
though a finite deadline is *more* certain and dangerous than the
unlimited-horizon soft ~99-round cap that `DEADLINE_PANIC_ROUND` guards
against. Fixed by widening `in_panic_zone` to also trigger a few rounds
before a real, known deadline -- reusing the same already-proven
mirror-their-last-ask / accept-anything-positive logic, not a new curve.

CONFIRMED live (2026-08-22), cycle 5: the finite-horizon panic-zone fix
above introduced its own regression for the mirror case. A game with
delta_me=1.0 (us truly, exactly infinitely patient -- zero cost to
waiting) held firm at $700k/$300k for five straight proposer rounds
against an opponent rigidly repeating $390k/$610k, our own messages
explicitly declaring "I will let the clock run out to Round 12... I
won't [blink]" -- and then accepted the opponent's unmoved $390k at round
10, TWO rounds before the real deadline. Verified directly: at round 10
the ordinary (still-softened, force_soften) threshold was $567,399 --
Bob's $390k should have been rejected -- but `FINITE_PANIC_ROUNDS_REMAINING`
fired anyway and blanket-accepted anything net-positive, because it does
not distinguish WHY a game is in the finite panic zone. It was built for a
player who genuinely loses real decaying value by continuing to wait (the
original bug); for a genuinely patient delta_me, waiting costs nothing, so
folding two rounds early throws away value (and contradicts what our own
message had just said) for no reason. The blanket accept-anything override
must be reserved for the literal last round when delta_me is degenerate --
not applied `FINITE_PANIC_ROUNDS_REMAINING` rounds early -- while a
non-degenerate delta_me keeps the original, wider margin (real decay makes
that margin actually necessary there).

CONFIRMED live (2026-08-22), a second cycle-4 finding on the same theme: an
incomplete-information, finite-horizon game (delta_me=0.8, opponent hidden)
had us counter-offer a confident 68.85% for ourselves at round 2, then
collapse to accepting a mere 26.19% by round 3 -- with 9 of 12 rounds still
left. Two more games with the same setup showed the identical pattern
(25.07%, 28.00%). Root cause: the belief update's likelihood function
compared the opponent's observed offer against `finite_horizon_take`
(undamped, round-PARITY-oscillating -- the exact defect
`damped_finite_horizon_take` exists to fix for our OWN decisions) instead
of the damped version. At the odd rounds_remaining=11 in the flagged game,
the undamped formula predicts an opponent at delta=0.99 takes only 72.4% --
so an observed 75% offer looked achievable ONLY under an almost-maximally-
patient hypothesis, collapsing the posterior mean to 0.958 from a SINGLE
data point (the damped formula predicts 94.3% for that same delta=0.99 at
the same rounds_remaining -- a 22-point gap purely from the parity quirk,
not from anything informative about the opponent). Fixed by using
`damped_finite_horizon_take` in the belief update too, so the likelihood
model relies on the same already-validated, non-oscillating predictions our
own decisions do.

Two separate, DIFFERENTLY-DIRECTED fixes below, not one symmetric
mechanism (an earlier version of this fix tried to unify them and got the
threshold's direction backwards -- worth recording why they differ):
`_soften_offer_floor` widens to cover finite horizon too, still only ever
touching a floor-bound offer, and still converges toward OUR OWN floor
(the most we'd ever concede) -- there's no live evidence a ceiling-bound
offer (asking for our own maximum immediately) causes any problem, so that
case is untouched. `_soften_threshold_ceiling` is cycle 2's original,
already-validated fix (confirmed to turn the $612,531-rejected-at-round-99
case into an accept by round 89), now also widened to finite horizon --
it converges toward a neutral 50/50, NOT toward the floor, because an
accept threshold becoming more lenient over time is the goal (matching
DEADLINE_PANIC_ROUND's own logic), whereas becoming more lenient is
exactly backwards for what an offer should do in the mirror situation.

DISCOVERED while verifying the above (2026-08-22, not yet seen live but
caught before shipping): the "widen to finite horizon" fix alone was still
broken, because it patched the two softening helpers without addressing
what feeds them. `damped_finite_horizon_take` does NOT stabilize when a
delta is exactly at the degenerate value 1.0, because the infinite-horizon
"anchor" it damps toward is ALSO exactly 0 in that case (or exactly `money`
for the delta_me=1.0 mirror) -- blending an oscillating finite component
toward an anchor that carries no real information doesn't smooth anything.
Diagnosed directly: for delta_me=0.95, delta_opp=1.0, max_rounds=12, the
raw (damped) take swung 0 -> 85.7 -> 0 -> 73.5 -> 0 -> 63.0 ... across
consecutive rounds_remaining values, and the raw reject-continuation
threshold swung 44.0 -> 0 -> 95.0 the same way -- pure round-parity noise,
not a converging concession curve. Root-caused, not patched again: a delta
this close to 1.0 makes the *opponent* (or us, in the mirror case) lose
essentially nothing by ever rejecting, so finite backward induction
collapses to "whoever is proposer at the literal deadline takes
everything" and swings entirely on parity. `DEGENERATE_DELTA_THRESHOLD`
detects this upstream and routes BOTH `_my_take` and `_reject_value`
through the stable `infinite_horizon_share` formula whenever either delta
is degenerate, regardless of whether the horizon is finite -- the same
formula already used for genuinely-unlimited-horizon games, feeding into
the same already-validated softening curves via the real `max_rounds`.
This also exposed that `_soften_threshold_ceiling` had only ever been
exercised on the degenerate-HIGH side (delta_me=1.0, raw threshold pinned
at `money`) -- the mirror degenerate-LOW side (delta_opp=1.0, raw
threshold pinned at 0) fell through to the floor with no time-softening at
all, meaning we'd have been willing to accept a merely-mediocre offer
starting on round 1. `force_soften` makes both degenerate directions
converge the same way: start near our own ceiling, soften toward a neutral
50/50 over time, exactly like the already-validated degenerate-HIGH case.

CONFIRMED live (2026-08-22), three separate games: an opponent that
repeats the IDENTICAL offer round after round -- not moving a single
dollar for 20, or even 39, consecutive proposal rounds -- while our own
side kept applying the ordinary generic time-based softening curve,
taking far longer than necessary to finally accept a split the opponent
had already revealed, many rounds earlier, was their fixed final position.
One game held out to round 40 against an opponent stationary since round
1, compounding 5%/round decay down to ~13.5% of the nominal payoff by the
time we finally accepted. Another held out to round 8 against an opponent
stationary since round 2. In both cases the eventually-accepted split was
already comfortably within our own acceptable band the entire time -- the
only thing gained by continued holdout was additional decay, since the
opponent had already proven, empirically, that no further movement was
coming. `_opponent_not_conceding` detects this directly from revealed
behavior (K consecutive near-identical opponent offers) rather than
waiting for a generic, opponent-behavior-blind curve to eventually grind
down to a target already reached.

CONFIRMED live (2026-08-23), the pure-flatness bar above was still too
narrow: two real games lost heavily to holdout against an opponent that
was never exactly stationary, just never conceding.
(1) delta_me=0.95/delta_opp=0.90, unlimited horizon, messages allowed: the
opponent's own self-ask INCREASED every round (4385.82 -> 4667.67 across 8
of her turns) -- i.e. what she offered US strictly DECREASED round over
round, the whole game. That's a clearly non-improving trend from round 1
onward (every extra round of holdout was doubly negative: nominal terms
got worse AND the 5%/round decay kept compounding), but the strict
within-0.5%-band check requires near-total flatness, so it didn't fire
until round 17 (her last two steps finally shrank below the band) --
accepting a WORSE nominal amount, discounted by 16 rounds of 5% decay
(~44% of face value), than what was on the table at round 1. (2) A second
game (delta_me=0.90, finite 12-round horizon, incomplete information) had
a genuinely, perfectly flat opponent ($5000/$5000 every single turn from
round 1) -- correctly detected, but only at round 5, the earliest the old
K=3-consecutive-exact-matches bar could possibly fire, by which point
10%/round decay had already erased ~34% of the value. In both cases the
opponent's revealed trend already made the verdict clear before the old
flatness bar could confirm it. `_opponent_not_conceding` (renamed from
`_opponent_offer_is_stationary`) now fires whenever the recent window is
flat OR trending against us (each step's opponent self-ask doesn't DROP by
more than the existing tolerance) -- a strict superset of the old
within-band check, so every previously-passing case (including a real
opponent conceding meaningfully in our favor, which must NOT trigger
this) still behaves identically; see the function's own docstring.

DERIVED, not yet live-tested (2026-08-22): a rating-strategy review noticed
MAX_SELF_SHARE_FRAC's ceiling was a blanket cap on every game, not just the
degenerate/unlimited-horizon cases it was built for. Checked against the
real solver: 31% of realistic (delta_me, delta_opp, rounds_remaining)
combinations legitimately compute an equilibrium share above 70% -- e.g.
delta_me=0.6 vs delta_opp=0.5 alone already computes to 71.3%, nowhere near
degenerate -- and the blanket cap was giving away an average 15 points of
the pot whenever it fired. `_soften_offer_floor` and
`_soften_threshold_ceiling` now only apply the MAX_SELF_SHARE_FRAC clip
when `is_degenerate or rounds_remaining is None`; a genuinely finite, known
max_rounds, non-degenerate game trusts the exact backward-induction number
directly (it's provably correct for that real deadline, unlike unlimited
horizon's hidden-cap mismatch risk). Every existing degenerate/unlimited
protection -- the floor, the panic zones, stationary-opponent detection --
is untouched, since none of them key off this ceiling. Needs a live smoke
test before trusting at volume: the open question is how often real
opponents (especially humans, who may reject on fairness grounds beyond
what a rational-agent model predicts) actually concede to these higher,
now-uncapped asks before the deadline panic zone has to bail us out.
"""

from __future__ import annotations

from agent import history
from agent.bargaining import belief as belief_mod
from agent.bargaining.solver import (
    damped_finite_horizon_take,
    infinite_horizon_share,
)
from agent.negotiation.solver import concession_target, time_fraction
from agent.rhetoric.bandit import StyleBandit, get_style_bandit, persist_style_bandit
from agent.rhetoric.opponent_profile import get_concession_profile, record_concession_evidence
from agent.rhetoric.compose import compose_message
from agent.rhetoric.profile import classify
from agent.schema import build_bargaining_offer, opponent_name, read_gain

ROBUSTNESS_BUFFER_FRAC = 0.03
ACCEPT_TOLERANCE_FRAC = 0.01
# DERIVED (2026-08-30), from real human play (not synthetic testing):
# under incomplete information and unlimited horizon, the formula-driven
# opening ask -- computed from the frozen belief prior (DEFAULT_GRID's
# uniform mean, 0.819; see DISABLE_DELTA_BELIEF's own derivation for why
# it never moves from that) -- asks for as much as ~67-70% of the pot at
# delta_me=0.80-0.90 (10-20% inflation). Real opening-round outcomes
# playing AS a human against real opponents found a flat 50/50 open
# performed meaningfully better than that at this patience level; at
# delta_me=0.95-1.00 (0-5% inflation) the reverse held -- an aggressive,
# formula-driven open (already >50%, unaffected by this) outperformed
# 50/50, consistent with existing behavior there. Only delta_me<=0.90
# actually needs correcting: at delta_me=0.80 the existing formula already
# lands close to 50/50 (50.9%) on its own. Scoped strictly to what was
# observed -- round 1 (the literal opening move), incomplete information,
# unlimited horizon -- not extended to finite-horizon games or later-round
# offers, which have their own separate, already-validated concession
# logic untouched by this.
OPENING_OFFER_50_50_DELTA_ME_MAX = 0.90
MIN_SELF_SHARE_FRAC = 0.30
# CONFIRMED (2026-08-29): this ablation was originally measured under a
# scripts/montecarlo_opponents.py bug -- a fresh random game_id every
# round instead of once per trial meant DeltaBelief could never actually
# accumulate evidence across a trial's rounds either way, so the original
# "zero measured effect" claim risked being a harness artifact rather than
# a real finding. A SECOND bug in the ablation script itself (baseline and
# ablated passes reused the identical game_id per trial, so the "ablated"
# pass found baseline's already-updated belief instead of a fresh one --
# see scripts/ablation_montecarlo.py's reset_per_game_state docstring)
# inflated the first corrected measurement in the OTHER direction. With
# both fixed: baseline=+0.238, ablated=+0.233, delta=-0.005 -- still
# negligible, well inside this suite's per-cell noise floor (stdev
# 0.02-0.18 at n=400/archetype). The original conclusion survives a real
# measurement; disabling eliminates noise from mis-estimated delta at no
# material cost.
DISABLE_DELTA_BELIEF = True
MAX_SELF_SHARE_FRAC = 0.70
DEADLINE_PANIC_ROUND = 90  # conservative margin below the confirmed ~99-100 hidden round cap
FINITE_PANIC_ROUNDS_REMAINING = 2  # same panic-zone safety net, triggered this close to a real,
                                    # known (and therefore harder/more certain) finite deadline

# CONFIRMED live (2026-08-24): the panic zone's "accept if my_gain > 0" had
# no floor beyond literally positive. A real game ($1,000,000 pot, finite
# 12-round horizon, non-degenerate delta_me=0.95) had the opponent hold an
# identical $0.01-for-us / $999,999.99-for-them offer across 4 straight
# rounds; we accepted it at round 10 (rounds_remaining=2). Rejecting there
# risked NOTHING -- we still had a real decision left (our own round-11
# counter, then the true round-12 deadline) -- so caving to an amount this
# extreme bought no safety at all, just gave away the entire pot to any
# opponent (deliberate or not) holding an extortionate anchor through this
# window. PANIC_ZONE_MIN_FRAC requires a genuinely meaningful floor while
# any further decision is still possible; only at the TRUE last chance
# (see ABSOLUTE_DEADLINE_ROUND below, and the identical parity reasoning
# already applied to `decision_panic_zone`'s finite case) does "anything
# positive" apply again, since the alternative there really is $0 for
# certain. 0.10 is a judgment call, not derived from anything more precise.
PANIC_ZONE_MIN_FRAC = 0.10

# CONFIRMED live (2026-08-24): DEADLINE_PANIC_ROUND (90) is a conservative
# buffer BEFORE the confirmed ~99-100 hidden round cap, not itself a literal
# deadline -- there's real room between round 90 and the actual cap. Unlike
# the finite-horizon case (where rounds_remaining <= 1 is an exact,
# provable "no further decision" point), the unlimited-horizon cap isn't
# known precisely, so this is a second, closer heuristic buffer: PANIC_ZONE_
# MIN_FRAC's floor still applies through most of the panic zone (rounds
# 90+), and only relaxes to "accept anything positive" once genuinely close
# to the confirmed cap, where the risk of holding out for a better floor
# starts to outweigh the benefit.
ABSOLUTE_DEADLINE_ROUND = 97
CONCESSION_TAU = 30.0  # rounds for the unlimited-horizon softening curve to meaningfully engage
CONCESSION_BETA = 0.7  # mildly Boulware, matching negotiation's own default: hold firm early
DEGENERATE_DELTA_THRESHOLD = 0.999  # a delta this close to 1.0 makes finite backward
                                     # induction oscillate wildly by round parity (see
                                     # module docstring) -- treat it like unlimited horizon

# DERIVED (2026-08-25), offline stress-tested (see _degeneracy_weight):
# comfortably below DEGENERATE_DELTA_THRESHOLD -- below this, `_degeneracy_
# weight` returns exactly 0.0 and every downstream computation is byte-
# identical to before this fix. Only deltas in the last ~1% of patience
# space blend at all.
DEGENERATE_BLEND_START = 0.99
STATIONARY_OPPONENT_ROUNDS = 3  # this many consecutive near-identical opponent offers is
                                  # treated as proof further holdout won't extract more
STATIONARY_OPPONENT_TOLERANCE_FRAC = 0.005  # offers within 0.5% of the pot count as "the same"

# DERIVED (2026-08-23), not yet live-tested: how close a stationary offer
# must be to the properly-reasoned, softened threshold before the
# not-conceding override accepts it, instead of just clearing
# MIN_SELF_SHARE_FRAC's bare survival floor -- see the override's own
# comment for the two real games (neither degenerate) that motivated this:
# accepting a stationary offer 24% and 10.7% of the pot below what the
# formula actually said we were owed. 0.75 is a judgment call, not derived
# from anything more precise -- high enough to require the offer be
# genuinely close to our real position, not just survivable, while still
# meaningfully below 1.0 so the override still fires well before the
# threshold's own natural softening would otherwise reach the same level
# (its whole reason for existing).
NOT_CONCEDING_ACCEPT_FRAC = 0.75

# CONFIRMED live (2026-08-24): the not-conceding override used to be
# disabled outright whenever delta_me is degenerate (see its own comment --
# the fix for a real game where it accepted a stationary $430,000 at bare
# MIN_SELF_SHARE_FRAC when the real threshold wanted $658,368). That fix
# conflated two different things: "don't cave CHEAP when patience is free"
# (correct) with "never trust proven stationarity when patience is free"
# (not correct, and costly on its own). A real game (delta_me=1.0, $1M pot,
# unlimited horizon) had the opponent offer EXACTLY $500,000 identically
# for all 45 of their turns across 90 rounds, with an explicit commitment
# message every time ("I will not chase a worse split"). With the override
# fully disabled, we held out to the literal DEADLINE_PANIC_ROUND=90
# backstop before accepting -- the SAME $500,000 that had been on the
# table, unchanged, since round 1. That's not just wasted rounds: 90 is
# only 10 rounds short of the CONFIRMED hidden round cap (~99-100) that
# has already zeroed out a real game this session -- grinding a
# degenerate-delta game this close to that cliff for zero additional value
# is a real, avoidable risk of the same catastrophic outcome. Since
# patience is genuinely free here, there's no reason to cave at the first
# sign of stationarity the way a decaying player should -- but there's
# also no reason to refuse ANY amount of proof. DEGENERATE_NOT_CONCEDING_ROUNDS
# requires substantially more confirmations (more rounds of costless
# patience spent verifying) before trusting stationarity when delta_me is
# degenerate, rather than disabling the override entirely. Checked against
# this exact game's numbers: NOT_CONCEDING_ACCEPT_FRAC's bar (0.75x the
# softening threshold) already crosses $500,000 by ~round 10 -- the
# binding constraint becomes this confirmation count, not the acceptance
# bar. At 7 confirmations (Bob repeats every 2 rounds), this now accepts
# around round 14 instead of round 90: still 2x+ the ordinary 3-round bar
# (reflecting that patience really is free, so extra caution costs
# nothing), but nowhere near the hidden-cap danger zone for a genuinely,
# thoroughly proven-immovable opponent.
DEGENERATE_NOT_CONCEDING_ROUNDS = 7

# DERIVED (2026-08-24), not yet live-tested: everything above governs
# whether to ACCEPT an opponent's offer -- nothing governed how much WE
# concede when making our OWN offer, which follows `_my_take`/
# `_soften_offer_floor`'s time-based curve unconditionally, regardless of
# whether the opponent has shown any reciprocal movement at all. A batch of
# 8 real heavy losses (all -7.0 to -7.5) shared one shape: in several, the
# opponent's own ask kept RISING every round (e.g. 87.56% -> 96% across a
# whole game) while our own offer kept falling on schedule (70% -> 47.6%)
# regardless -- we conceded steadily INTO an opponent moving the opposite
# direction the entire time. An unconditional, purely time-based concession
# curve is exploitable for free by any opponent (or opposing strategy) that
# simply never reciprocates, since caving costs them nothing and gains
# everything once they can predict we'll soften on schedule no matter what
# they do. OFFER_CONCESSION_DAMPING_FRAC slows (not freezes -- a fully
# reciprocity-gated curve risks the mutual-holdout deadlock this session's
# own panic-zone fixes exist to prevent) how much of each round's
# formula-driven concession we actually take when the opponent hasn't
# reciprocated, so a non-conceding opponent no longer gets our full
# concession for free. 0.3 is a judgment call: still lets real pressure
# (the deadline, the panic zone) eventually resolve a truly immovable
# opponent, while cutting the giveaway to a non-reciprocating one to
# roughly a third of what the raw curve would otherwise hand over each
# round.
OFFER_CONCESSION_DAMPING_FRAC = 0.3

# CONFIRMED live (2026-08-24): `erode_over_time` (see `_soften_offer_floor`/
# `_soften_threshold_ceiling`) was originally gated purely on unlimited
# horizon -- erosion toward a neutral 50/50 began on ROUND COUNT ALONE,
# with no check that the opponent had actually proven immovable. Tested
# directly against a real game's numbers (delta_me=0.9, delta_opp=0.8,
# unlimited horizon): by round 40 our own offer had already eroded from the
# true 69.29% target down to 61.12%, and by round 88 down to 54.13% --
# giving away real leverage on a schedule, regardless of whether the
# opponent had shown even a single sign of being unmovable. That's the same
# mistake OFFER_CONCESSION_DAMPING_FRAC above already fixed for the
# ordinary concession curve, just reintroduced in the erosion mechanism
# built to fix a DIFFERENT problem. Erosion should only bind once there's
# real evidence extra patience won't pay off, not just because rounds
# elapsed. EROSION_CONFIRMATION_ROUNDS requires substantially more
# confirmed-non-conceding rounds than the ordinary
# STATIONARY_OPPONENT_ROUNDS=3 bar before erosion is allowed to engage at
# all -- erosion trades away real backward-induction-justified value, a
# much bigger commitment than the ordinary damping/accept mechanisms make,
# so it deliberately demands much stronger proof first. Matches
# DEGENERATE_NOT_CONCEDING_ROUNDS's same reasoning (extra confirmation
# before giving up a real advantage), reusing its exact value.
EROSION_CONFIRMATION_ROUNDS = DEGENERATE_NOT_CONCEDING_ROUNDS

# CONFIRMED live (2026-08-25): even after EROSION_CONFIRMATION_ROUNDS gates
# WHEN erosion is allowed to start, the erosion curve itself reused
# CONCESSION_TAU=30 -- the same pace built for cautious, still-uncertain
# early-game softening. But erosion only ever engages after 7 rounds of
# much stronger proof than an ordinary concession decision requires, so the
# caution CONCESSION_TAU encodes has already been paid for by the
# confirmation requirement itself; reusing the same slow pace on top of
# that is redundant, not extra safety. Reproduced offline against a
# scripted fully-stationary opponent matching a real game's exact numbers
# (delta_me=0.9, delta_opp=0.8, unlimited horizon): with CONCESSION_TAU,
# convergence took 33 rounds and ended at 1.5% of the pot after decay. A
# live sample of 9/100 recent games hit a genuinely stationary opponent and
# still ran 10-91 rounds, several ending at near-total value destruction
# (shares of 0.0, 1.9e-05, 0.07, 0.10). EROSION_TAU is a third of
# CONCESSION_TAU -- still a real curve, not an instant jump (an instant
# jump to the eroded target risks conceding more than a still-real chance
# the opponent moves warrants), but reaching the same eventual erosion
# target in roughly a third of the rounds once the stronger bar has already
# been cleared.
EROSION_TAU = CONCESSION_TAU / 3.0

# DERIVED (2026-08-25), from a controlled Monte Carlo simulation (400 trials
# vs. a scripted, provably-unmovable "hardliner" opponent: identical ask
# every round, never conceding): even with EROSION_TAU's faster convergence,
# erosion only ever moves us TOWARD our own MIN_SELF_SHARE_FRAC floor, never
# below it -- correctly, since that floor exists to prevent excessive
# capitulation. But against a hardliner whose position genuinely sits below
# that floor (this simulation: demanding 92% for itself, leaving us <=15%),
# NO floor-respecting deal exists at all -- every single trial (400/400)
# resolved only via the calendar-based DEADLINE_PANIC_ROUND=90 safety net,
# by which point per-round decay had erased ~99% of nominal value (mean
# score 0.011 vs. money/2's ~0.5 naive expectation). The panic zone's own
# mirror-their-position logic (see `panic_mode_engaged` below) already
# handles exactly this case correctly once triggered -- the only problem is
# WHEN it triggers: purely by round count, with no regard for how much
# evidence already exists that further waiting is pointless. Real games
# show the identical shape (e.g. a 90-round game where the opponent's own
# offer was IDENTICAL from round 1 -- 45 consecutive confirmations before
# panic finally engaged, all of it pure decay for zero new information).
# HOPELESS_CONFIRMATION_ROUNDS gates a NEW, earlier trigger for exactly this
# case (see `confirmed_hopeless` below): a much stronger bar than
# EROSION_CONFIRMATION_ROUNDS (7), since crossing below our own protective
# floor is a bigger commitment than eroding down to it -- roughly double,
# reflecting real confidence the opponent's position is unmovable, not the
# generic non-conceding signal that's enough to justify erosion alone.
HOPELESS_CONFIRMATION_ROUNDS = 2 * EROSION_CONFIRMATION_ROUNDS

# Per-game (never pooled across games -- see StyleBandit's own docstring)
# Thompson Sampling over which moral-foundation framing to lead an offer's
# message with, adapted to how THIS opponent's offers actually move in
# response. Ported from persuasion's seller-side StyleBandit (2026-08-23):
# unlike persuasion, there's no clean "did they take the specific action we
# were pitching" signal here, since accepting our offer ends the game
# immediately and we never get called again to observe it. The only
# feedback we CAN observe is the reject-and-counter case: whether the
# opponent's next self-ask moved toward us relative to their own previous
# offer. That conflates the natural concession curve with any actual
# message effect (no counterfactual is observable within one game), so this
# is a noisier proxy than persuasion's bought/didn't-buy signal -- but it's
# the only one available, and it's still directly informative about whether
# holding this style is correlated with the opponent continuing to move.
# CONFIRMED live (2026-08-24): `get_style_bandit`/`persist_style_bandit`
# (agent.rhetoric.bandit) resolve to a disk-persisted, cross-game bandit
# keyed by the opponent's revealed name when one is known, or an ordinary
# fresh per-game bandit when identity is hidden -- see that module's
# docstring for the full rationale. `_style_bandit` here is now a thin
# wrapper resolving the opponent's display name from `game`, not its own
# storage.
_LAST_STYLE_ARM: dict[str, str] = {}

# Tracks our own most recently sent offer per game, so the next offer can
# tell whether the raw formula wants to concede further and, if so, whether
# that concession should be dampened -- see OFFER_CONCESSION_DAMPING_FRAC.
_LAST_SENT_TAKE: dict[str, float] = {}


def _opponent_display_name(game: dict) -> str | None:
    return (game.get("opponent") or {}).get("name")


def _style_bandit(game: dict) -> StyleBandit:
    return get_style_bandit(game["game_id"], _opponent_display_name(game))


def _credit_style_bandit_from_opponent_response(
    game: dict, opp: str, game_id: str, current_opp_gain: float, money: float,
) -> None:
    """Pops any style credited to our last offer and scores it against
    whether the opponent's current self-ask moved toward us relative to
    their own previous offer. `history.get_history` only ever contains
    PRIOR, already-decided rounds -- same timing point as
    `_opponent_not_conceding` -- so the round currently under decision
    arrives separately as `current_opp_gain`, not folded into history yet.
    Requires at least one prior opponent offer to compare against; if this
    is their first offer in the game, there's nothing to compare yet, so
    this is a no-op rather than a guessed failure.

    DERIVED (2026-08-24), not yet live-tested: the raw moved/didn't-move
    binary treated a $0.01 shift and a $50,000 shift identically, crediting
    tiny noise-level jitter toward us as full "success" -- not real evidence
    the message's framing changed anything. An EARLIER version of this fix
    tried filtering out all small movements (including exact zero) as
    "ambiguous," but that's wrong: an opponent who repeats their exact prior
    offer is giving the clearest possible failure signal there is, not an
    ambiguous one -- it broke an existing, correct test for exactly that
    reason. The right asymmetry is on the SUCCESS side only: success now
    requires movement to clear STATIONARY_OPPONENT_TOLERANCE_FRAC * money
    (the same noise band `_opponent_not_conceding` already uses elsewhere in
    this file), while zero, negative, or noise-level-small movement all
    still correctly count as failure -- nothing is discarded, only
    small-positive is prevented from masquerading as clean success."""
    arm = _LAST_STYLE_ARM.pop(game_id, None)
    if arm is None:
        return
    prior_opponent_offers = [
        read_gain(h["offer"], opp) for h in history.get_history(game)
        if history.is_opponent_entry(h, opp) and h.get("offer")
    ]
    if not prior_opponent_offers:
        return
    moved_by = prior_opponent_offers[-1] - current_opp_gain  # positive = moved toward us
    success = moved_by > STATIONARY_OPPONENT_TOLERANCE_FRAC * money
    _style_bandit(game).update(arm, success)
    persist_style_bandit(game_id, _opponent_display_name(game))


def _is_degenerate(delta_me: float, delta_opp: float) -> bool:
    return delta_me >= DEGENERATE_DELTA_THRESHOLD or delta_opp >= DEGENERATE_DELTA_THRESHOLD


def _degeneracy_weight(delta_me: float, delta_opp: float) -> float:
    """0.0 well below DEGENERATE_DELTA_THRESHOLD, ramping linearly to 1.0
    AT the threshold (deltas at or above it are unchanged from before this
    fix -- still fully routed through the degenerate-case formula and the
    capped/softened ceiling).

    DERIVED (2026-08-25), offline stress-tested: `_my_take`/`_reject_value`'s
    formula choice and the downstream ceiling/`force_soften` application
    were both gated on the single boolean `_is_degenerate`, producing a
    sharp cliff right at the threshold -- confirmed directly via the offline
    monotonicity sweep (stress_test_invariants.py): delta_me=0.95 computed a
    smooth, un-capped ~97% share (100-money game, delta_opp=0.01, finite
    12-round horizon), while delta_me=0.999 -- comfortably more patient, not
    a knife-edge -- fell back to a completely different formula AND got
    unconditionally capped down to a curve starting at 70%, a 27-point drop
    for increased patience. 52 monotonicity violations found by the sweep;
    all but the below-floor-opening-anchor ones (a separate, pre-existing,
    intentional mechanic -- see `_soften_offer_floor`'s `raw_take <
    floor_val` branch, unrelated to this fix) clustered at exactly this
    boundary. A hard binary switch can't be smoothed by retuning constants;
    it needs the switch itself to become a continuous ramp. Callers blend
    between the non-degenerate and degenerate computations using this
    weight (call both, `(1-w)*a + w*b`) rather than this function trying to
    smooth the ceiling/formula internals directly -- lower risk, since both
    endpoints stay byte-identical to the already-live-validated code paths.
    `max(delta_me, delta_opp)` (not just delta_me) because either side's
    degeneracy triggers the same formula collapse (see the module
    docstring's mirror-case incidents)."""
    d = max(delta_me, delta_opp)
    if d <= DEGENERATE_BLEND_START:
        return 0.0
    if d >= DEGENERATE_DELTA_THRESHOLD:
        return 1.0
    return (d - DEGENERATE_BLEND_START) / (DEGENERATE_DELTA_THRESHOLD - DEGENERATE_BLEND_START)


def _resolve_opponent_last_ask(game: dict, opp: str) -> float | None:
    """The opponent's own gain in their most recently made offer, resolved
    via `state["last_offer"]` when it correctly attributes to them, falling
    back to `history.get_history` otherwise.

    CONFIRMED live (2026-08-24): `state["last_offer"]` is only reliably
    populated during DECISION-type turns (the API guarantees it there --
    see `valid_actions["type"] == "decision"` below, which reads it
    directly with no `.get()` fallback). During OFFER-type turns it's
    unreliable -- first confirmed for the panic-zone offer branch (pulled
    real completed games via the API: 4 of 5 real panic-zone offer turns had
    `state["last_offer"]` fail to resolve to the opponent's actual last
    offer), and this fallback was originally written ONLY for that branch.
    Replaying a real transcript showed the identical gap in the ORDINARY,
    non-panic offer branch: with `last_offer` unresolved there, BOTH the
    "never counter-propose worse than what's on the table" floor AND
    `OFFER_CONCESSION_DAMPING_FRAC`'s reciprocity gate silently no-op on
    every non-panic offer turn, since both require `last_offer.get
    ("proposer") == opp` to fire at all. A real $100 game reproduced this
    exactly: the opponent camped at an identical $50 offer for 5 straight
    rounds while our own counter kept sliding on an undamped Boulware
    curve (70 -> 68.45 -> 65.84 -> 62.57 -> 58.79) all the way down to
    literally proposing their own $50/$50 split ourselves, which they then
    accepted -- the full 20-point premium given away for nothing, because
    the damping gate never once saw a resolvable `last_offer` to fire on.
    Shared by both offer-branch code paths now, not panic-zone-only."""
    last_offer = game["game_state"].get("last_offer")
    if last_offer and last_offer.get("proposer") == opp:
        return read_gain(last_offer, opp)
    opponent_offers = [
        read_gain(h["offer"], opp) for h in history.get_history(game)
        if history.is_opponent_entry(h, opp) and h.get("offer")
    ]
    return opponent_offers[-1] if opponent_offers else None


def _opponent_not_conceding(
    game: dict, opp: str, money: float, current_opp_gain: float,
    required_rounds: int = STATIONARY_OPPONENT_ROUNDS,
) -> bool:
    """Whether the opponent's last `required_rounds` self-asks,
    INCLUDING the one currently on the table, show no real sign of
    conceding to us -- flat, OR trending further in the opponent's own
    favor (never mind exactly flat).

    DERIVED (2026-08-23), not yet live-tested: originally checked only
    `history.get_history` -- which, like negotiation's server history, only
    ever contains prior, already-decided rounds; the round under active
    decision arrives separately as `last_offer` and isn't folded into
    `history` until after we act on it. That meant three-in-a-row could only
    be confirmed by a FOURTH matching offer, one round later than the third
    repeat itself. Found while porting this same pattern to negotiation
    (agent/negotiation/strategy.py) and tracing a real incident where that
    one-round lag mattered: fixed there by combining the current offer with
    prior history instead of reading history alone. Applying the identical
    fix here too -- bargaining has per-round decay, so firing one round
    later than necessary has a direct, real cost (see the module docstring's
    39-round holdout incident, where each extra round cost 5% of the pot).

    CONFIRMED live (2026-08-23), a second finding: requiring near-total
    flatness (this function used to be `_opponent_offer_is_stationary`, an
    exact-match-band check) missed the far more common and equally
    decisive case of an opponent who is trending WORSE for us -- their own
    self-ask rising round over round -- without ever going flat. A real
    game held out 17 rounds against exactly that pattern before the old
    band check finally caught up, paying 16 rounds of 5%/round decay (and
    an ever-worsening nominal offer) for nothing. The condition below is a
    strict superset of the old one: it fires whenever no step in the recent
    window DROPS (in the opponent's own self-ask) by more than the existing
    tolerance -- flat sequences still pass trivially (as before), rising
    ones now pass too, and a genuinely, meaningfully conceding opponent
    (self-ask dropping well beyond tolerance) still correctly does not.

    CONFIRMED live (2026-08-30): the per-STEP tolerance check above has a
    real, apparently deliberately-exploited gap -- a real opponent (a
    2100-rated human-observed player) conceded at a slow, steady ~0.2% of
    the pot per round, comfortably under STATIONARY_OPPONENT_TOLERANCE_
    FRAC's 0.5%-per-round bar every single round, so it was classified as
    "not conceding" starting at `required_rounds` and never reconsidered
    afterward even though the opponent kept giving real ground the whole
    time. Simulated directly against this exact tactic (unlimited horizon,
    incomplete information, delta_me=1.0): our own bot didn't accept until
    round 89, taking only 37.6% of the pot, saved from worse only by the
    unrelated DEADLINE_PANIC_ROUND=90 backstop. A per-step check can never
    catch a concession rate tuned to sit just under it, no matter how the
    single-step tolerance is set -- what it misses is CUMULATIVE movement
    across the whole confirmation window. Comparing the window's two
    endpoints directly (oldest vs. newest, scaled to the window's actual
    length rather than a fixed per-step amount) still passes a genuinely
    flat or worsening opponent exactly as before, but now correctly detects
    a real, if slow, concession trend that a fixed per-step bar alone
    cannot -- the two checks are complementary, not a replacement of each
    other, since a *single* large jump-and-hold should still count as
    "not conceding" going forward even if the cumulative window looks
    favorable at the moment it happens.
    """
    if money <= 0:
        return False
    prior_opponent_offers = [
        read_gain(h["offer"], opp) for h in history.get_history(game)
        if history.is_opponent_entry(h, opp) and h.get("offer")
    ]
    recent = prior_opponent_offers[-(required_rounds - 1):] + [current_opp_gain]
    if len(recent) < required_rounds:
        return False
    tolerance = STATIONARY_OPPONENT_TOLERANCE_FRAC * money
    per_step_not_conceding = all(recent[i] >= recent[i - 1] - tolerance for i in range(1, len(recent)))
    # Deliberately NOT scaled by window length: the tolerance represents
    # "how much single-round wobble counts as noise, not a real move" --
    # scaling it up by (required_rounds - 1) would make the cumulative
    # check MORE lenient the longer the confirmation window is, exactly
    # backwards from the point of adding it (a slow, steady drip that
    # clears one round's worth of tolerance summed over many rounds is
    # real, sustained movement, not noise, regardless of how many rounds
    # it took to get there).
    cumulative_not_conceding = recent[-1] >= recent[0] - tolerance
    return per_step_not_conceding and cumulative_not_conceding


def _soften_offer_floor(
    raw_take: float, floor_val: float, opening_anchor: float, hard_cap: float,
    round_no: int, max_rounds: int | None, money: float = 0.0, force_soften: bool = False,
    erode_over_time: bool = False, erosion_tau: float = CONCESSION_TAU,
    force_soften_tau: float = CONCESSION_TAU, below_floor_tau: float = CONCESSION_TAU,
) -> float:
    """Offer side. A degenerate-low raw take starts at `opening_anchor`
    (the most assertive opening consistent with our own band) and concedes
    down to the floor as rounds accumulate, instead of jumping straight to
    the floor on move one. A ceiling-bound raw take softens from `hard_cap`
    toward a neutral 50/50 over time (mirroring the threshold side below)
    instead of staying pinned at the ceiling for the whole game. An in-band
    raw take is returned as-is.

    DERIVED, not yet live-tested (2026-08-22): `hard_cap` used to always
    equal MAX_SELF_SHARE_FRAC, applied to every game regardless of whether
    the degenerate-formula risk that ceiling exists for was actually
    present. Checked against the real solver across a realistic delta grid:
    31% of (delta_me, delta_opp, rounds_remaining) combinations produce a
    LEGITIMATE, non-degenerate equilibrium share above 70% -- e.g.
    delta_me=0.6/delta_opp=0.5 alone (only mildly more patient) already
    computes to 71.3% -- and the blanket cap was giving away an average 15
    points of the pot in exactly those cases. For a genuinely finite, known
    max_rounds, non-degenerate game the backward-induction number is exact
    for the real deadline (no hidden-cap mismatch the way unlimited horizon
    has), so the caller now passes `hard_cap=money` (i.e. trust the raw
    number) there, reserving the MAX_SELF_SHARE_FRAC clip for the
    degenerate/unlimited-horizon cases it was actually built to guard
    against. The opening_anchor for a below-floor raw take is untouched --
    that's a separate, already-validated aggressive-opening tactic, not part
    of this fix. The deadline panic-zone override is unaffected either way,
    so the $0 protection this cap exists for stays intact.

    CONFIRMED live (2026-08-23): a real game (delta_me=1.0, degenerate for
    us -- zero decay cost) had our own OFFER pinned at exactly 70% of the
    pot for 5 straight rounds (1, 3, 5, 7, 9), completely unchanged, because
    a degenerate raw_take (always == money) just got clamped to `hard_cap`
    here with no time-dependence at all -- unlike `_soften_threshold_
    ceiling` below, which DOES soften an equally-degenerate threshold down
    toward money/2 over time via `force_soften`. The opponent's own offer
    to us worsened over that same stretch (27 -> 34 -> 27 -> 19 -> 15),
    plausibly in response to our total rigidity. This mirrors the threshold
    side's fix exactly: a ceiling-bound raw take now softens toward money/2
    over time too, whether the cap was hit because of degeneracy
    (force_soften) or simply because a legitimate, non-degenerate unlimited
    -horizon share exceeded the cap on its own (raw_take > hard_cap) --
    same two triggers, same mirrored logic as the threshold side already
    uses.

    CONFIRMED live (2026-08-24): the two triggers above still left a real
    gap -- an unlimited-horizon, NON-degenerate raw_take that sits BELOW
    hard_cap gets neither trigger, so it's returned completely unchanged,
    with NO time-dependence, for the entire game. A real 90-round game
    (delta_me=0.9, delta_opp=0.8) had our own offer pinned at exactly
    $69.286 (the raw infinite_horizon_share) for 44 straight rounds while
    the opponent proved permanently immovable at $44.44 -- paying real
    per-round decay the whole time for zero benefit, only resolving at the
    DEADLINE_PANIC_ROUND=90 safety net. `erode_over_time` adds a THIRD,
    softer effect for exactly this case: not a hard override like the two
    triggers above (which would make an already-reasonable raw_take WORSE
    early on by replacing it with a curve that starts at hard_cap, above
    most legitimate raw values -- confirmed live to break existing
    not-conceding-override tests at moderate round counts), but a `min()`
    against the same erosion curve. Early in a long game the curve sits
    near hard_cap, well above a real non-degenerate raw_take, so `min()`
    picks the raw_take unchanged (no behavior change for ordinary games).
    Only once the curve has decayed, many rounds in, below the raw_take
    does it start to bind -- giving the exact "eventually accept reality
    over the untouchable theoretical target" behavior the degenerate case
    already had, without ever making an early-game offer worse than the
    honest formula value."""
    if raw_take < floor_val:
        # CONFIRMED live (2026-08-25): this branch fires whenever our own
        # theoretical entitlement (raw_take) sits below our own survival
        # floor -- e.g. delta_me=0.8 vs delta_opp=0.95 computes only ~20.8%
        # for us, below MIN_SELF_SHARE_FRAC's 30% -- and, like the
        # `force_soften` branch above before its own fix, used
        # CONCESSION_TAU unconditionally with NO opponent-awareness at all.
        # Two real games reproduced the exact same catastrophic shape the
        # erosion/force_soften_tau fixes were built to solve, just via this
        # THIRD, previously-unexamined branch: one had the opponent
        # propose an EXACT, byte-identical split across all 45 of their
        # proposal rounds (spread=0.0) while our own offer crept down this
        # branch's unconditional curve for 90 rounds before the
        # DEADLINE_PANIC_ROUND=90 safety net finally resolved it (predicted
        # values from this exact formula matched the real observed offers
        # to the cent: $692,031.73 @ round 2, $618,036.45 @ round 12,
        # $330,060.65 @ round 88); a second (incomplete-information) game
        # showed the identical signature over 98 rounds. `below_floor_tau`
        # applies the same, by-now-proven-safe pattern as `force_soften_tau`:
        # the caller passes CONCESSION_TAU (unchanged, round-1-safe) until
        # `confirmed_not_conceding` has independently fired via
        # EROSION_CONFIRMATION_ROUNDS, then EROSION_TAU.
        t = time_fraction(round_no, max_rounds, unlimited_tau=below_floor_tau)
        return concession_target(opening_anchor, floor_val, t, CONCESSION_BETA)
    if force_soften or raw_take > hard_cap:
        # DERIVED (2026-08-25), tried and reverted, then fixed properly:
        # first attempted reusing `erosion_tau` unconditionally here too --
        # but unlike erosion (only ever engaged after
        # EROSION_CONFIRMATION_ROUNDS of real confirmed evidence),
        # `force_soften` fires unconditionally from round 1 whenever
        # is_degenerate, with no confirmation gate at all. Speeding up its
        # tau unconditionally broke two deliberately live-validated tests
        # asserting round-1 stays close to the ceiling
        # (test_ceiling_bound_offer_softens_toward_midpoint_over_time,
        # test_degenerate_offer_no_longer_stays_pinned_across_a_real_live_
        # sequence) -- reverted.
        #
        # CONFIRMED live (2026-08-25): but the opposite gap is real too. A
        # game (delta_me=0.9, delta_opp=1.0 degenerate, unlimited horizon)
        # had the opponent propose the IDENTICAL 95%/5% split across all 47
        # of their proposal rounds over a 94-round game -- perfectly
        # stationary, the strongest possible confirmation signal -- while
        # our own offer, stuck on this branch's CONCESSION_TAU pace the
        # whole time, took 88 rounds to converge down to their proven
        # position. By the time we finally offered to match them, the
        # opponent walked away, ending the game at $0/$0 after we'd held a
        # strong 69% opening position. `force_soften_tau` fixes this the
        # right way this time: the caller passes CONCESSION_TAU (unchanged
        # behavior, matching the two tests above) until
        # `confirmed_not_conceding` has independently fired via the same
        # EROSION_CONFIRMATION_ROUNDS bar erosion already uses, then
        # EROSION_TAU -- so round-1 behavior is untouched, but a genuinely,
        # repeatedly proven-stationary opponent (like this one) no longer
        # gets 88 rounds of free decay before we even start moving toward
        # reality.
        t = time_fraction(round_no, max_rounds, unlimited_tau=force_soften_tau)
        return concession_target(hard_cap, money / 2, t, CONCESSION_BETA)
    if erode_over_time:
        t = time_fraction(round_no, max_rounds, unlimited_tau=erosion_tau)
        eroded = concession_target(hard_cap, money / 2, t, CONCESSION_BETA)
        return min(raw_take, eroded)
    return raw_take


def _soften_threshold_ceiling(
    raw_threshold: float, floor_val: float, ceiling_val: float, money: float,
    round_no: int, max_rounds: int | None, force_soften: bool = False,
    erode_over_time: bool = False, erosion_tau: float = CONCESSION_TAU,
    force_soften_tau: float = CONCESSION_TAU,
) -> float:
    """Threshold side. Floor applies immediately if relevant (unlimited
    horizon, or a degenerate delta -- see the call site). Either a
    degenerate-high raw threshold (pinned at `money`) or a degenerate-low one
    (pinned at 0, `force_soften=True`) starts at the ceiling and softens
    toward a neutral 50/50 -- NOT toward the floor -- as rounds accumulate,
    so we become more willing to accept a good-but-not-maximal offer over
    time rather than staying stuck demanding our own maximum, or folding
    immediately, for the whole game.

    DERIVED, not yet live-tested (2026-08-22): mirrors the offer-side fix
    above. The caller now passes `ceiling_val=money` for a genuinely finite,
    non-degenerate game -- `threshold > ceiling_val` can then never fire
    (raw_threshold is always <= money by construction) and `force_soften` is
    already False there, so the exact backward-induction threshold is used
    directly instead of being softened toward a neutral 50/50 whenever it
    legitimately exceeds 70%.

    CONFIRMED live (2026-08-24): see `_soften_offer_floor`'s identical
    `erode_over_time` addition -- an unlimited-horizon, non-degenerate
    threshold below `ceiling_val` was frozen with no time-dependence,
    so the not-conceding accept bar (NOT_CONCEDING_ACCEPT_FRAC * threshold)
    never adapted no matter how many rounds confirmed the opponent's offer
    was genuinely, permanently below it (a real 90-round game needed
    my_gain >= $48.22 forever against an opponent proven immovable at
    $44.44). `erode_over_time` takes `min()` against the same erosion
    curve rather than replacing the threshold outright -- early on the
    curve sits near `ceiling_val`, above most legitimate raw thresholds,
    so it doesn't bind and existing moderate-round-count behavior is
    unchanged; only after many rounds, once the curve decays below the
    raw threshold, does it start lowering the accept bar."""
    threshold = max(raw_threshold, floor_val)
    if force_soften or threshold > ceiling_val:
        # See `_soften_offer_floor`'s identical fix -- same rationale,
        # mirrored for the accept-threshold side: CONCESSION_TAU until
        # `confirmed_not_conceding_for_erosion` has independently fired,
        # then EROSION_TAU, via the caller-computed `force_soften_tau`.
        t = time_fraction(round_no, max_rounds, unlimited_tau=force_soften_tau)
        return concession_target(ceiling_val, money / 2, t, CONCESSION_BETA)
    if erode_over_time:
        t = time_fraction(round_no, max_rounds, unlimited_tau=erosion_tau)
        eroded = concession_target(ceiling_val, money / 2, t, CONCESSION_BETA)
        return min(threshold, eroded)
    return threshold


def _delta_index(player: str) -> str:
    return "1" if player in ("alice", "player_1") else "2"


def _rounds_remaining(state: dict) -> int | None:
    if not state.get("horizon_known", state.get("max_rounds") is not None):
        return None
    max_rounds = state.get("max_rounds")
    if max_rounds is None:
        return None
    return max(int(max_rounds) - int(state["round"]), 0)


def _my_take(money, me, opp, delta_me, delta_opp, rounds_remaining) -> float:
    # DERIVED (2026-08-24), not yet live-tested: offline stress testing
    # found a sharp discontinuity right at DEGENERATE_DELTA_THRESHOLD --
    # delta_me=0.9989 (finite horizon) computed a sensible, bounded 96.999%
    # share via damped_finite_horizon_take, while delta_me=0.999 -- one
    # part in a thousand MORE patient -- fell back to this plain closed
    # form and got capped down to a flat 70% by the ceiling downstream.
    # Tried simply removing the degenerate bypass here so finite horizons
    # always use damped_finite_horizon_take: that made the discontinuity
    # WORSE, not better, for the single most common real degenerate case
    # (delta_me EXACTLY 1.0, i.e. literal "0% inflation") -- confirmed
    # directly: at delta_me=1.0, damped_finite_horizon_take's OWN finite
    # and infinite components BOTH independently collapse to "take the
    # entire pot" (a truly infinitely-patient player facing any decaying
    # opponent has a real game-theoretic claim to everything), so removing
    # the ceiling there just let a live-validated, already-tuned test's
    # expected $567,399 balloon to $970,000 -- the exact kind of unrealistic
    # extreme-claim risk the ceiling exists to guard against, just moved to
    # a more common trigger point instead of eliminated. Left as the
    # original is_degenerate-gated bypass; narrowing the discontinuity
    # properly needs the ceiling's own on/off switch to become continuous
    # (blend toward the cap as delta_me approaches 1.0, not a hard
    # threshold), which is a more involved change than this function alone
    # can fix -- not attempted here without further discussion.
    if rounds_remaining is None or _is_degenerate(delta_me, delta_opp):
        return infinite_horizon_share(money, delta_me, delta_opp)
    delta_of = lambda p: delta_me if p == me else delta_opp
    return damped_finite_horizon_take(money, me, delta_of, rounds_remaining)


def _reject_value(money, me, delta_me, delta_opp, rounds_remaining) -> float:
    # See `_my_take`'s docstring -- reverted for the same reason.
    if rounds_remaining is None or _is_degenerate(delta_me, delta_opp):
        return delta_me * infinite_horizon_share(money, delta_me, delta_opp)
    if rounds_remaining <= 0:
        return 0.0  # this decision is at (or past) the deadline
    delta_of = lambda p: delta_me if p == me else delta_opp
    next_round_take = damped_finite_horizon_take(money, me, delta_of, rounds_remaining - 1)
    return delta_me * next_round_take


def bargaining_strategy(game: dict) -> dict:
    state = game["game_state"]
    valid_actions = game["valid_actions"]
    me = game["your_player"]
    opp = opponent_name(me)
    money = state["money_to_divide"]

    delta_me = state[f"delta_{_delta_index(me)}"]
    opp_delta_key = f"delta_{_delta_index(opp)}"
    complete_information = state.get("complete_information", opp_delta_key in state)

    game_belief = belief_mod.get_belief(game["game_id"])
    if opp_delta_key in state:
        delta_opp = state[opp_delta_key]
    else:
        delta_opp = game_belief.mean()

    rounds_remaining = _rounds_remaining(state)
    is_degenerate = _is_degenerate(delta_me, delta_opp)
    messages_allowed = state.get("messages_allowed", False)
    opponent_profile = classify(
        m["offer"].get("message", "")
        for m in history.get_history(game)
        if history.is_opponent_entry(m, opp) and m.get("offer", {}).get("message")
    ) if messages_allowed else None

    # DERIVED (2026-08-25): see HOPELESS_CONFIRMATION_ROUNDS's derivation --
    # `_resolve_opponent_last_ask` already handles both turn types correctly
    # (checks `state["last_offer"]` first, which the API guarantees resolves
    # to the opponent's own ask on decision-type turns; falls back to
    # history on offer-type turns where it doesn't), so this single early
    # call is valid for both branches below, not just one.
    opp_last_ask_early = _resolve_opponent_last_ask(game, opp)
    # Only for a genuinely non-degenerate delta_me: a degenerate delta_me
    # means waiting costs US nothing, so there's no reason to force an
    # early below-floor resolution when we could just keep waiting for
    # free, hoping the opponent eventually cracks -- same reasoning as the
    # existing finite-horizon degenerate exclusion below. "Below floor"
    # means their proven position would give us less than MIN_SELF_SHARE_
    # FRAC even if we accepted/mirrored it outright -- if it's ABOVE our
    # floor, the ordinary erosion/below-floor-tau mechanisms already
    # converge us toward it without needing this stronger, more drastic
    # trigger.
    #
    # CONFIRMED live (2026-08-25): this used to also exclude on `is_degenerate`
    # (delta_me OR delta_opp >= DEGENERATE_DELTA_THRESHOLD), not just
    # delta_me -- but "patience is free" is only true when OUR OWN delta is
    # degenerate. A real game (delta_me=0.8 non-degenerate, delta_opp=1.0
    # degenerate: the opponent's OWN patience is free, ours isn't) had us
    # paying real per-round decay against an opponent proven immovable at a
    # catastrophic share for 40+ rounds, but the blanket `is_degenerate`
    # exclusion (true here purely because of delta_opp) blocked
    # confirmed_hopeless from ever engaging -- the game ran to the literal
    # DEADLINE_PANIC_ROUND=90 backstop (83-90 rounds observed) instead of
    # resolving once genuinely proven hopeless, uncomfortably close to the
    # confirmed ~99-100 hidden round cap for zero benefit. Every other term
    # here (opp_last_ask_early, the floor comparison, _opponent_not_conceding)
    # is already degeneracy-independent -- only this exclusion needed
    # narrowing to the side that actually justifies waiting for free.
    # DERIVED (2026-08-25): `get_concession_profile` (agent.rhetoric.
    # opponent_profile) tracks, per opponent NAME and per FAMILY, whether
    # every prior confirmed-relevant game against this exact name showed
    # the same proven-immovable pattern (strict all-of, not majority --
    # see that module's docstring: a single real concession disqualifies
    # the name until confirmed-hopeless games catch back up to equal every
    # observed one again). `is_trusted_hardliner()` only ever HALVES the
    # in-game confirmation bar, never skips it -- real evidence from THIS
    # game is still required either way, this just lets a name with a
    # strong prior track record clear that bar faster.
    concession_profile = get_concession_profile(_opponent_display_name(game), "bargaining")
    effective_hopeless_rounds = (
        HOPELESS_CONFIRMATION_ROUNDS // 2 if concession_profile.is_trusted_hardliner()
        else HOPELESS_CONFIRMATION_ROUNDS
    )
    confirmed_hopeless = (
        delta_me < DEGENERATE_DELTA_THRESHOLD
        and opp_last_ask_early is not None
        and (money - opp_last_ask_early) < MIN_SELF_SHARE_FRAC * money
        and _opponent_not_conceding(
            game, opp, money, opp_last_ask_early, required_rounds=effective_hopeless_rounds,
        )
    )

    finite_panic_zone = rounds_remaining is not None and rounds_remaining <= FINITE_PANIC_ROUNDS_REMAINING
    unlimited_panic_zone = rounds_remaining is None and int(state["round"]) >= DEADLINE_PANIC_ROUND
    in_panic_zone = finite_panic_zone or unlimited_panic_zone or confirmed_hopeless
    # Decision side only: a degenerate delta_me means waiting costs us
    # nothing, so the finite-horizon override (built for the opposite case
    # -- real decay pressure) should only blanket-accept anything positive
    # at the true literal last chance to lock in a deal, not
    # FINITE_PANIC_ROUNDS_REMAINING early.
    #
    # CONFIRMED live (2026-08-23): "the true literal last chance" is NOT
    # always rounds_remaining == 0 -- that assumes this player will
    # eventually get a decision at rounds_remaining == 0, which depends on
    # round parity and only holds for the player who proposes SECOND. A
    # responder's own next decision, if they reject now, is always 2 rounds
    # later (one round for their own counteroffer, one for the opponent's
    # reply) -- so once rounds_remaining <= 1, there IS no further decision
    # coming for this player no matter what: rejecting now only leads to
    # making their own final offer and hoping the opponent accepts it, with
    # no chance to reconsider if it's refused. A real game (Bob, 12-round
    # horizon, Alice proposing first) never gives Bob a decision at
    # rounds_remaining == 0 at all -- his decisions land at
    # rounds_remaining = 11, 9, 7, 5, 3, 1, bottoming out at 1, never 0. The
    # old `rounds_remaining > 0` exclusion never released for him, so a
    # degenerate-delta_me Bob would hold out through his literal last
    # decision (round 11) and then be forced into a real $0-risk gamble on
    # his own final round-12 offer -- exactly the outcome this whole
    # override exists to avoid. `rounds_remaining > 1` is the correct,
    # parity-independent bound: rounds_remaining <= 1 always means no
    # further decision is coming, regardless of who proposed first.
    # CONFIRMED live (2026-08-24): this exclusion used to gate the decision
    # (accept) side only. A degenerate delta_me offer has the identical
    # reasoning for not rushing into panic-mode behavior early -- patience
    # is free, so there's no need to abandon the still-correctly-aggressive
    # formula-driven ask just because rounds_remaining <= 2. A real game
    # (delta_me=1.0, $1,000,000 pot, round 10 of 12) had the ordinary
    # formula wanting $567,399 for us at that exact round, but the
    # (un-gated) offer-side panic zone collapsed it straight to a plain
    # $500,000 -- giving up real value two full rounds before there was any
    # need to. Shared by both branches now, not decision-only.
    panic_mode_engaged = in_panic_zone and not (
        finite_panic_zone and delta_me >= DEGENERATE_DELTA_THRESHOLD and rounds_remaining > 1
    )
    # See PANIC_ZONE_MIN_FRAC/ABSOLUTE_DEADLINE_ROUND above: "no further
    # decision is possible" is the only point where accepting literally
    # anything positive is actually forced -- everywhere else in the panic
    # zone, rejecting a token offer risks nothing.
    true_last_chance = (rounds_remaining is not None and rounds_remaining <= 1) or (
        rounds_remaining is None and int(state["round"]) >= ABSOLUTE_DEADLINE_ROUND
    )

    if valid_actions["type"] == "offer":
        if panic_mode_engaged:
            # CONFIRMED live (2026-08-24): see `_resolve_opponent_last_ask`'s
            # docstring -- `state["last_offer"]` is unreliable specifically
            # during OFFER-type turns, so this falls back to `history`
            # instead of trusting it alone.
            opp_self_ask = _resolve_opponent_last_ask(game, opp)
            # Match exactly what they last asked for themselves (the highest-
            # probability-of-acceptance move, since it's their own revealed
            # position) or split down the middle if we've never seen an offer
            # from them to anchor to.
            my_take = (money - opp_self_ask) if opp_self_ask is not None else money / 2
            my_take = min(max(my_take, 0.0), money)
        elif (
            not complete_information and rounds_remaining is None
            and int(state["round"]) == 1 and delta_me <= OPENING_OFFER_50_50_DELTA_ME_MAX
        ):
            # See OPENING_OFFER_50_50_DELTA_ME_MAX's derivation -- a flat
            # 50/50 open outperforms the formula-driven ~67-70% ask real
            # opponents actually faced at this patience level. Recorded
            # into _LAST_SENT_TAKE like the ordinary branch below, so a
            # later round's OFFER_CONCESSION_DAMPING_FRAC check has an
            # accurate reference point for "did we just concede further
            # than our own last offer" starting from this opening move.
            my_take = money / 2.0
            _LAST_SENT_TAKE[game["game_id"]] = my_take
        else:
            raw_take = _my_take(money, me, opp, delta_me, delta_opp, rounds_remaining)
            raw_take *= (1 - ROBUSTNESS_BUFFER_FRAC)
            max_rounds = state.get("max_rounds") if rounds_remaining is not None else None
            # Ceiling protection only applies where the degenerate-formula or
            # hidden-cap risk it was built for actually exists (see
            # _soften_offer_floor's docstring); a genuinely finite,
            # non-degenerate game trusts the exact backward-induction share.
            # (Reverted an attempt to also drop this for finite+degenerate --
            # see `_my_take`'s docstring for why that made things worse, not
            # better, for the delta_me==1.0 case specifically.)
            #
            # DERIVED (2026-08-25): this used to be the boolean
            # `is_degenerate` (fed into `force_soften` below) -- see
            # `_degeneracy_weight`'s docstring for the cliff that produced
            # (a 27-point drop for a one-part-in-a-thousand increase in
            # patience). Now a continuous weight; see the blended
            # `_soften_offer_floor` calls below. The CEILING VALUE (not this
            # weight) is what stays unconditional for unlimited horizon --
            # `nondegenerate_hard_cap` below still pins it to
            # MAX_SELF_SHARE_FRAC*money there regardless of weight, matching
            # the pre-fix, independently-already-continuous unlimited-
            # horizon behavior exactly at weight=0.
            degeneracy_weight = _degeneracy_weight(delta_me, delta_opp)

            # CONFIRMED live (2026-08-24): see `_resolve_opponent_last_ask`'s
            # docstring -- `state["last_offer"]` is unreliable on OFFER-type
            # turns. Resolved via the same history fallback used elsewhere
            # in this file, not `state.get("last_offer")` alone. Computed
            # here (before softening, not after) because EROSION_CONFIRMATION_
            # ROUNDS below now needs it too, not just the "never propose
            # worse" floor.
            opp_last_ask = _resolve_opponent_last_ask(game, opp)
            # CONFIRMED live (2026-08-24): see EROSION_CONFIRMATION_ROUNDS's
            # derivation -- `erode_over_time` must not fire on round count
            # alone; it needs real, strong evidence the opponent won't
            # reciprocate, using the same detector as the damping check
            # below but requiring far more confirmations first, since
            # erosion gives up real backward-induction-justified value
            # rather than just slowing a concession already in progress.
            confirmed_not_conceding = opp_last_ask is not None and _opponent_not_conceding(
                game, opp, money, opp_last_ask, required_rounds=EROSION_CONFIRMATION_ROUNDS,
            )
            # CONFIRMED live (2026-08-24): see `_soften_offer_floor`'s
            # `erode_over_time` docstring -- an unlimited-horizon,
            # non-degenerate raw take under the 70% ceiling used to be
            # returned with NO time-dependence at all, staying frozen for
            # the entire game against a proven-immovable opponent (a real
            # 90-round game paid decay for 44 straight rounds pinned at
            # $69.286 before resolving only at the DEADLINE_PANIC_ROUND=90
            # safety net). `erode_over_time` adds gradual, min()-bounded
            # erosion for an unlimited-horizon game, but ONLY once
            # `confirmed_not_conceding` -- not `force_soften`, which would
            # make ordinary early-game offers worse -- see that parameter's
            # own docstring.
            # DERIVED (2026-08-25): blend between the "trust the exact
            # number" computation (weight 0) and the "capped/softened"
            # computation (weight 1) using `degeneracy_weight`, rather than
            # a hard `force_soften=is_degenerate` switch -- see
            # `_degeneracy_weight`'s docstring. Both endpoints call the
            # exact same, already-live-validated `_soften_offer_floor`
            # unchanged (force_soften=False/hard_cap=money, and
            # force_soften=True/hard_cap=MAX_SELF_SHARE_FRAC*money), so
            # weight 0.0 and weight 1.0 are byte-identical to the pre-fix
            # behavior -- only deltas strictly between DEGENERATE_BLEND_
            # START and DEGENERATE_DELTA_THRESHOLD get a new, interpolated
            # value. Skips the second call entirely when weight is exactly
            # 0.0 (the overwhelming majority of real games), so this adds no
            # extra work outside the narrow blend window.
            common_kwargs = dict(
                floor_val=MIN_SELF_SHARE_FRAC * money, opening_anchor=MAX_SELF_SHARE_FRAC * money,
                round_no=int(state["round"]), max_rounds=max_rounds, money=money,
                erode_over_time=rounds_remaining is None and confirmed_not_conceding,
                erosion_tau=EROSION_TAU,
                force_soften_tau=EROSION_TAU if confirmed_not_conceding else CONCESSION_TAU,
                below_floor_tau=EROSION_TAU if confirmed_not_conceding else CONCESSION_TAU,
            )
            # DERIVED (2026-08-27), not yet live-tested: this used to keep
            # the ceiling unconditional for ALL unlimited-horizon games
            # (`nondegenerate_hard_cap = MAX_SELF_SHARE_FRAC * money if
            # rounds_remaining is None else money`), reasoning that
            # unlimited horizon's unknown ~99-100-round hidden cap makes the
            # equilibrium share less trustworthy than a known, finite
            # deadline's. But that reasoning only bears on how long we
            # should be willing to hold out for a given ask -- a question
            # already handled separately by the erosion/panic-zone/
            # confirmed-hopeless machinery -- not on what the CORRECT
            # opening ask is when delta_opp is exact, not estimated. A real
            # complete-information, unlimited-horizon game (delta_me=0.95,
            # delta_opp=0.80, comfortably non-degenerate) computed a true
            # equilibrium share of 83.3%, got capped at 70% here, and the
            # opponent accepted our capped $6,984.85/$10,000 offer instantly
            # at round 1 with no pushback -- the same "instant accept means
            # we under-asked" signal already flagged elsewhere in this file
            # -- leaving roughly $1,345 (13.4% of the pot) on the table in a
            # single round, before any of the hidden-cap risk this ceiling
            # exists for could even become relevant. The cap now only
            # applies when delta_opp is genuinely uncertain (incomplete
            # information, i.e. a belief estimate rather than a known
            # value); finite horizon's existing exception is unchanged, and
            # the degenerate-delta blending above (`degeneracy_weight`) is
            # untouched either way, so the original knife-edge protection
            # this ceiling was built for still fully applies.
            nondegenerate_hard_cap = (
                money if (rounds_remaining is not None or complete_information)
                else MAX_SELF_SHARE_FRAC * money
            )
            nondegenerate_take = _soften_offer_floor(
                raw_take, hard_cap=nondegenerate_hard_cap, force_soften=False, **common_kwargs,
            )
            if degeneracy_weight <= 0.0:
                my_take = nondegenerate_take
            else:
                degenerate_take = _soften_offer_floor(
                    raw_take, hard_cap=MAX_SELF_SHARE_FRAC * money, force_soften=True, **common_kwargs,
                )
                my_take = (
                    (1 - degeneracy_weight) * nondegenerate_take + degeneracy_weight * degenerate_take
                )
            my_take = min(max(my_take, 0.0), money)
            # Never counter-propose worse for ourselves than what's already
            # on the table. CONFIRMED live: we rejected an opponent offer
            # giving us $622,331, then counter-offered ourselves only
            # $610,909 -- the formula-driven ask was computed independently
            # of their revealed offer and happened to be lower, which makes
            # no sense regardless of theory: if we're countering, it should
            # be because we want MORE, never less.
            if opp_last_ask is not None:
                my_take = max(my_take, money - opp_last_ask)
                my_take = min(my_take, money)

            # CONFIRMED live (2026-08-24): see OFFER_CONCESSION_DAMPING_FRAC's
            # derivation. If the opponent hasn't reciprocated (flat or
            # trending worse for us) and the formula wants to concede
            # further than our own last offer, only take a fraction of that
            # concession rather than the full amount -- a non-reciprocating
            # opponent shouldn't get our full scheduled concession for free.
            #
            # CONFIRMED live (2026-08-25): this doesn't distinguish an
            # ORDINARY schedule-driven concession (still genuinely uncertain
            # whether the opponent will reciprocate -- damping is exactly
            # right here) from erosion's OWN deliberate decrease, which only
            # ever runs after EROSION_CONFIRMATION_ROUNDS (7) rounds of much
            # stronger proof than damping's own 3-round bar. Applying the
            # same anti-exploitation brake to a decrease that's already
            # gated behind stronger evidence double-slows convergence for no
            # added protection: reproduced offline against a scripted fully
            # stationary opponent matching a real game's exact numbers
            # (delta_me=0.9, delta_opp=0.8, unlimited horizon, opponent
            # fixed at 55.6%/44.4% every round) -- 33 rounds to converge,
            # ending at 1.5% of the pot after decay despite being the
            # theoretically-favored, more-patient side. Live sample: 9/100
            # recent games hit a fully stationary opponent and still ran
            # 10-91 rounds, several ending at near-total value destruction
            # (shares of 0.0, 1.9e-05, 0.07, 0.10). Skip the brake once
            # erosion's own stronger confirmation has already fired --
            # see EROSION_TAU below for the other half of this fix (the
            # erosion curve itself was also just slower than it needed to
            # be once that same strong confirmation is in hand).
            prev_take = _LAST_SENT_TAKE.get(game["game_id"])
            if (
                prev_take is not None and my_take < prev_take and opp_last_ask is not None
                and not confirmed_not_conceding
            ):
                if _opponent_not_conceding(game, opp, money, opp_last_ask):
                    my_take = prev_take - OFFER_CONCESSION_DAMPING_FRAC * (prev_take - my_take)
            _LAST_SENT_TAKE[game["game_id"]] = my_take

        message = None
        if messages_allowed:
            style = _style_bandit(game).select_arm()
            message = compose_message(
                family="bargaining",
                situation=game["prompt"],
                stance={
                    "role": "proposer",
                    "my_gain": my_take,
                    "opponent_gain": money - my_take,
                    "money_to_divide": money,
                    "fallback_message": "This split reflects how patient each of us can afford to be.",
                },
                profile=opponent_profile,
                style=style,
            )
            _LAST_STYLE_ARM[game["game_id"]] = style
        action = build_bargaining_offer(me, my_take, money, message)
        history.record_own_action(game, action)
        return action

    if valid_actions["type"] == "decision":
        last_offer = state["last_offer"]
        history.record_opponent_offer(game, last_offer)
        _credit_style_bandit_from_opponent_response(game, opp, game["game_id"], read_gain(last_offer, opp), money)
        my_gain = read_gain(last_offer, me)

        if not complete_information and rounds_remaining is not None:
            opp_take_fraction = (money - my_gain) / money if money else 0.0
            predicted_fraction_fn = lambda d: damped_finite_horizon_take(
                money, opp, lambda p, d=d: delta_me if p == me else d, rounds_remaining
            ) / money if money else 0.0
            if not DISABLE_DELTA_BELIEF:
                game_belief.update(opp_take_fraction, predicted_fraction_fn)
        elif not complete_information and rounds_remaining is None:
            opp_take_fraction = (money - my_gain) / money if money else 0.0
            predicted_fraction_fn = lambda d: infinite_horizon_share(money, d, delta_me) / money if money else 0.0
            if not DISABLE_DELTA_BELIEF:
                game_belief.update(opp_take_fraction, predicted_fraction_fn)
        else:
            predicted_fraction_fn = None

        opponent_not_conceding = _opponent_not_conceding(game, opp, money, read_gain(last_offer, opp))
        if not DISABLE_DELTA_BELIEF and not complete_information and predicted_fraction_fn is not None and opponent_not_conceding:
            # CONFIRMED live (2026-08-24): two real incomplete-information
            # games had a genuinely, repeatedly stationary opponent (offer
            # unchanged for 8-10 of their own turns) that the not-conceding
            # override still took another 8-10 rounds to accept, because
            # DeltaBelief's SHRINKAGE_PSEUDO_COUNT deliberately damps each
            # individual update toward the uniform prior (built to stop a
            # single noisy observation from dominating -- see belief.py's
            # own incident log) -- but that damping schedule is keyed to
            # `n_observations` (how many rounds have been SEEN), not to how
            # CONSISTENT those observations have been. Once
            # `_opponent_not_conceding` has already independently confirmed
            # repetition, the same evidence repeating yet again isn't a new,
            # noisy data point that needs the same caution -- it's
            # confirmation of what's already been established. Feeding the
            # SAME observation into the belief a second time (reusing the
            # exact update already computed above, no new formula) lets
            # `n_observations` -- and therefore the shrinkage -- catch up to
            # match the real confidence the repetition itself already
            # earned, instead of crawling there one round at a time while
            # 5-10%/round decay keeps compounding against a target that
            # hasn't moved in many rounds. Real cost observed: 34-40% of the
            # pot's value lost to this exact gap in two live games.
            game_belief.update(opp_take_fraction, predicted_fraction_fn)

        if panic_mode_engaged:
            if true_last_chance:
                # Deep into an unlimited-horizon game (past
                # ABSOLUTE_DEADLINE_ROUND) or at the true literal deadline of
                # a finite one: no formula-driven threshold -- or floor -- is
                # worth risking the proven $0 no-deal alternative for. Take
                # anything net-positive over the certainty of nothing.
                decision = "accept" if my_gain > 0 else "reject"
            else:
                # CONFIRMED live: still inside the panic zone but a further
                # decision genuinely remains possible -- see
                # PANIC_ZONE_MIN_FRAC's derivation. Rejecting a token,
                # extortionate offer here costs nothing, so demand a
                # meaningful floor rather than literally anything positive.
                decision = "accept" if my_gain >= PANIC_ZONE_MIN_FRAC * money else "reject"
        else:
            raw_threshold = _reject_value(money, me, delta_me, delta_opp, rounds_remaining)
            max_rounds = state.get("max_rounds") if rounds_remaining is not None else None
            # DERIVED (2026-08-25): the floor and ceiling used to be gated
            # by the same hard `is_degenerate` boolean as the offer side --
            # see `_degeneracy_weight`'s docstring for the cliff this
            # produced. For unlimited horizon both stay unconditional
            # (untouched, unrelated to this fix). For finite horizon, blend
            # continuously below via two calls to the same, unchanged
            # `_soften_threshold_ceiling` (floor=0/ceiling=money at weight 0,
            # floor=MIN_SELF_SHARE_FRAC*money/ceiling=MAX_SELF_SHARE_FRAC*
            # money at weight 1) -- blending the RESULTS handles the floor
            # smoothly too, without needing a separately-blended floor_val.
            degeneracy_weight = _degeneracy_weight(delta_me, delta_opp)
            # CONFIRMED live (2026-08-24): see EROSION_CONFIRMATION_ROUNDS's
            # derivation -- `erode_over_time` must not fire on round count
            # alone, only once there's real evidence (many more confirmed
            # rounds than the ordinary STATIONARY_OPPONENT_ROUNDS=3 bar) the
            # opponent genuinely won't reciprocate further. `opponent_not_
            # conceding` above already checked the standard bar for other
            # purposes; erosion demands a much stronger, separate check
            # before it's allowed to give away real backward-induction-
            # justified value.
            confirmed_not_conceding_for_erosion = _opponent_not_conceding(
                game, opp, money, read_gain(last_offer, opp), required_rounds=EROSION_CONFIRMATION_ROUNDS,
            )
            # CONFIRMED live (2026-08-24): see `_soften_threshold_ceiling`'s
            # `erode_over_time` docstring -- an unlimited-horizon,
            # non-degenerate threshold under the 70% ceiling used to be
            # completely frozen, so the not-conceding accept bar
            # (NOT_CONCEDING_ACCEPT_FRAC * threshold) never adapted no
            # matter how many rounds confirmed the opponent's offer was
            # genuinely, permanently below it. The same real 90-round game
            # never accepted the opponent's proven-immovable $44.44 (needed
            # >= 0.75*64.29=$48.22 forever) until the round-90 safety net --
            # 89 rounds of real per-round decay paid for a target that was
            # never going to be met. `erode_over_time` (not `force_soften`,
            # which would raise the bar early -- see that parameter's own
            # docstring) fixes this the same way as the offer side, but only
            # once `confirmed_not_conceding_for_erosion` -- not on round
            # count alone (see EROSION_CONFIRMATION_ROUNDS).
            common_kwargs = dict(
                money=money, round_no=int(state["round"]), max_rounds=max_rounds,
                erode_over_time=rounds_remaining is None and confirmed_not_conceding_for_erosion,
                erosion_tau=EROSION_TAU,
                force_soften_tau=EROSION_TAU if confirmed_not_conceding_for_erosion else CONCESSION_TAU,
            )
            # Unlimited horizon keeps the floor and ceiling unconditional
            # (matches pre-fix behavior exactly at weight=0 -- see the
            # comment above `degeneracy_weight`); only finite horizon's
            # non-degenerate endpoint drops both to "trust the raw number."
            nondegenerate_floor = MIN_SELF_SHARE_FRAC * money if rounds_remaining is None else 0.0
            nondegenerate_ceiling = MAX_SELF_SHARE_FRAC * money if rounds_remaining is None else money
            nondegenerate_threshold = _soften_threshold_ceiling(
                raw_threshold, floor_val=nondegenerate_floor, ceiling_val=nondegenerate_ceiling,
                force_soften=False, **common_kwargs,
            )
            if degeneracy_weight <= 0.0:
                threshold = nondegenerate_threshold
            else:
                degenerate_threshold = _soften_threshold_ceiling(
                    raw_threshold, floor_val=MIN_SELF_SHARE_FRAC * money,
                    ceiling_val=MAX_SELF_SHARE_FRAC * money, force_soften=True, **common_kwargs,
                )
                threshold = (
                    (1 - degeneracy_weight) * nondegenerate_threshold
                    + degeneracy_weight * degenerate_threshold
                )

            not_conceding_required_rounds = (
                STATIONARY_OPPONENT_ROUNDS if delta_me < DEGENERATE_DELTA_THRESHOLD
                else DEGENERATE_NOT_CONCEDING_ROUNDS
            )
            opponent_not_conceding_for_override = (
                opponent_not_conceding if not_conceding_required_rounds == STATIONARY_OPPONENT_ROUNDS
                else _opponent_not_conceding(
                    game, opp, money, read_gain(last_offer, opp), required_rounds=not_conceding_required_rounds,
                )
            )
            if (
                my_gain >= max(MIN_SELF_SHARE_FRAC * money, NOT_CONCEDING_ACCEPT_FRAC * threshold)
                and opponent_not_conceding_for_override
            ):
                # CONFIRMED live: an opponent repeating the identical offer,
                # OR never conceding any real ground to us, for many
                # consecutive rounds has already revealed, empirically, that
                # further holdout won't extract more -- continuing to grind
                # through the generic time-based curve toward a target
                # we've already reached (or will never reach) only pays
                # additional decay for nothing.
                #
                # CONFIRMED live (2026-08-23): that reasoning assumes WE
                # also pay for continued holdout -- it's weaker when delta_me
                # is degenerate (0% inflation, no decay cost to us at all). A
                # real game (delta_me=1.0, delta_opp=0.95, $1,000,000 pot)
                # had the opponent repeat an identical $570,000/$430,000
                # offer 3 times; this override used to accept the $430,000
                # (barely above MIN_SELF_SHARE_FRAC's 30% floor) immediately,
                # when the ordinary formula-driven threshold -- which
                # correctly accounts for the fact that waiting is FREE for us
                # while the opponent keeps decaying 5%/round -- wanted at
                # least $658,368 at that same round. This used to be fixed by
                # disabling the override outright whenever delta_me is
                # degenerate, but that overcorrected -- see
                # DEGENERATE_NOT_CONCEDING_ROUNDS's derivation for the real
                # 90-round holdout that cost (a game held out to the literal
                # unlimited-horizon panic-zone deadline, dangerously close to
                # the confirmed hidden round cap, for an offer that had been
                # on the table unchanged since round 1). The fix now scales
                # the CONFIRMATION COUNT with how free our patience is
                # (`not_conceding_required_rounds` above), rather than fully
                # disabling the override -- still refuses to cave cheap when
                # waiting costs nothing, but no longer refuses to ever trust
                # overwhelming, repeated proof either.
                #
                # CONFIRMED live (2026-08-23), a second finding, non-
                # degenerate this time: the MIN_SELF_SHARE_FRAC floor alone
                # (30%) is disconnected from what we're actually owed. Two
                # real games (neither delta anywhere near degenerate: 0.95
                # vs 0.80, and 0.95 vs 0.95, both unlimited horizon) had this
                # override accept a stationary offer far below the properly
                # -reasoned threshold at that same round -- $4,444 accepted
                # vs a $6,862 threshold (24% of the pot given up), and
                # $380,469 accepted vs a $487,179 threshold (10.7% given
                # up). Bare survival was never the right bar for "good
                # enough to stop holding out" -- it should scale with what
                # we're actually owed. NOT_CONCEDING_ACCEPT_FRAC requires
                # the offer to clear a meaningful fraction of the real
                # threshold, not just MIN_SELF_SHARE_FRAC's floor, while
                # still letting the override fire well before the threshold
                # fully softens down to the opponent's level (its entire
                # reason for existing) -- and the deadline panic-zone above
                # remains a hard backstop against pure infinite holdout
                # either way.
                decision = "accept"
            else:
                decision = "accept" if my_gain >= threshold - ACCEPT_TOLERANCE_FRAC * money else "reject"
        action = {"decision": decision}
        history.record_own_action(game, action)
        if decision == "accept":
            # DERIVED (2026-08-25): the SDK never calls this function again
            # once we submit an accepting decision -- the game is over from
            # our side. This is the one guaranteed end-of-game signal
            # available for OUR OWN acceptances; a game that instead ends
            # via the OPPONENT accepting one of OUR offers has no
            # equivalent hook (we're simply never called again), so this
            # records evidence for a real subset of games, not all of them
            # -- an accepted, documented partial-coverage gap, not a bug.
            # Partial data is still valid signal for `is_trusted_hardliner`
            # (which only ever halves, never skips, in-game confirmation).
            record_concession_evidence(_opponent_display_name(game), "bargaining", confirmed_hopeless)
        return action

    return {}
