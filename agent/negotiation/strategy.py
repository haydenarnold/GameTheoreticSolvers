"""Negotiation strategy: role-aware concession tactics with revealed-bound
opponent-value inference. See negotiation/solver.py for the game-theoretic
rationale (no discounting exists in this family; only the round cap and
revealed preferences drive the strategy).

CONFIRMED live (2026-08-22): two gaps found via real transcripts.

(1) `time_fraction`'s unlimited-horizon branch was called without an
explicit `unlimited_tau`, so it used the function's own default (6.0) --
much faster than bargaining's deliberately-chosen 30.0 for the equivalent
unlimited-horizon softening. A real 32-round buyer negotiation fully
conceded to its own $12,000 reservation value (t effectively at 1.0) well
before round 30, then held flat there while the seller also flatlined,
ending in agreement at EXACTLY $12,000 -- zero surplus captured, despite
the seller having shown willingness to go as low as ~$12,000-12,121,
suggesting real room existed below our ceiling. Explicitly setting the
same tau bargaining uses keeps the concession curve from collapsing our
entire bargaining position within the first ~30 rounds.

(2) A genuine one-shot ultimatum (`max_rounds == 1`) as PROPOSER used
`t=0` (full aggression, per an earlier fix to stop one-shot offers from
collapsing to zero markup) -- but a true one-shot has no future round to
lean on, so anchoring at the full speculative ceiling
(`my_value * DEFAULT_PRIOR_SPREAD`, a 2x guess with no real basis) risks
pricing out of any deal at all. Confirmed live: a $80-value seller priced
a one-shot at $160 and got rejected outright. `ONE_SHOT_ULTIMATUM_T`
splits the difference between full aggression and a fully safe reservation
price, instead of blindly reusing the same logic multi-round openers use.

CONFIRMED live (2026-08-23): a real unlimited-horizon game had the seller
offer an identical $118.80 for three straight rounds (7, 9, 11) -- after an
earlier identical $148.50 for rounds 1, 3, 5 -- while our own Boulware curve
kept crawling up slowly ($66.47 -> $72.16 -> $79.11 -> $86.68). $118.80 was
already profitable for us (my_value=$120). Rather than recognizing the
seller had proven immovable and accepting, we kept countering -- and at
round 12 the seller simply WALKED AWAY, turning a locked-in +$1.20 into a
guaranteed $0 for both sides. Negotiation has no discounting, so grinding
extra rounds against a genuinely stationary opponent doesn't cost decayed
value the way it does in bargaining (the case bargaining's own stationary
check was built for) -- but it carries a DIFFERENT, equally real risk here:
the opponent can just leave. `_opponent_offer_is_stationary` below ports
bargaining's exact pattern (same threshold, same tolerance) so a profitable,
proven-immovable offer gets taken immediately instead of risking a walkaway
by continuing to push past a point already shown, empirically, not to move.

CONFIRMED live (2026-08-24): the stationary-accept fix above had an
unguarded edge -- it only checked `profitable` (a bare `>=`/`<=`), so an
opponent camping EXACTLY at our reservation value, never moving, counted as
"profitable" and got accepted immediately at $0 surplus. Five real
transcripts in one batch shared this shape (a stationary opponent within
0-12% of our reservation while our own time-based counter kept sliding
toward them every round, with no reciprocity gating on the offer side at
all); two were exact ties, handing over 100% of a known $2,000-$3,000 ZOPA
for literally nothing. Two fixes, both gated correctly against the ORIGINAL
$118.80/$120 incident above (itself a thin, ~3%-of-guessed-range margin
under incomplete information) so neither reproduces that walkaway:
`STATIONARY_ACCEPT_MIN_MARGIN_FRAC` requires a stationary offer clear a
real fraction of the ZOPA before treating it as good enough to take
immediately -- but only under `complete_information`, where `bounds` is a
known ZOPA rather than a guess (see `_stationary_offer_clears_min_margin`'s
docstring for why gating matters here). `OFFER_CONCESSION_DAMPING_FRAC`
(ported from bargaining's identical fix) separately slows our own counter's
approach toward a proven-stationary opponent's number, applying regardless
of complete/incomplete information since it doesn't require knowing the
true ZOPA -- it only throttles how fast we move toward a number already
shown fixed.
"""

from __future__ import annotations

from agent import history
from agent.negotiation.solver import ValueBounds, concession_target, deadline_pressure, time_fraction
from agent.rhetoric.bandit import StyleBandit, get_style_bandit, persist_style_bandit
from agent.rhetoric.compose import compose_message
from agent.rhetoric.profile import classify
from agent.schema import check_fields, opponent_name

# DERIVED (2026-08-23), not yet live-tested: a batch-level review of 103
# recent negotiation games found only 49.5% ended in agreement -- "no deal"
# (hit the round cap) averaged -0.82 rating/game and "walked away" averaged
# -1.20/game, versus +0.89/game for actual agreements. Unlike bargaining,
# this family has no per-round decay to eventually force convergence -- the
# only thing that ends a stuck negotiation is one side leaving, so a
# stubborn early posture carries real, demonstrated walk-away risk with no
# compensating pressure on the opponent. The old DEFAULT_PRIOR_SPREAD=2.0
# anchor (2x reservation for a seller, half for a buyer under incomplete
# information) was a large, somewhat arbitrary opening position -- several
# real transcripts had opponents push back on it as bad-faith ("completely
# disconnected from market reality"). CONFIRMED live: a seller (my_value=80)
# opened at $160 (the old 2.0x anchor); the buyer's very first counter was
# already a genuinely profitable $116.10, but at round 3 of 10 (t=0.2) the
# old anchor+beta combination still wanted $151.97 -- nowhere close -- and
# the buyer walked away the very next round. With the new anchor (120,
# i.e. 1.5x) alone, the SAME round-3 target becomes $115.99 -- almost
# exactly the buyer's real offer, and would have closed the deal on the
# spot. CONCESSION_BETA also moves from 0.7 toward 1.0 (linear) -- NOT
# lower, despite an earlier draft of this fix saying so: per this
# function's own docstring, beta<1 is Boulware (SLOWER early concession)
# and beta>1 is Conceder (FASTER); the direction needed here is UP, not
# down, to concede faster early rather than slower. 0.85 is a modest,
# deliberately incomplete move toward linear (not all the way to a full
# Conceder posture, which would give away too much of the patient-opponent
# upside this curve still exists to capture) -- both changes trade some
# theoretical max-extraction for a meaningfully lower chance of the
# opponent leaving before any deal is reached at all.
CONCESSION_BETA = 0.85
DEFAULT_PRIOR_SPREAD = 1.5
# CORRECTED (2026-08-29): this used to cite a Monte Carlo ablation that
# actually never exercised this mechanism at all -- scripts/
# montecarlo_opponents.py's negotiation trial generated a fresh random
# game_id every ROUND instead of once per trial, so `_LAST_SENT_PRICE.get
# (game_id)` (needed for `tighten_from_rejection` to fire) was always None.
# That bug is now fixed (one stable game_id per trial, matching
# persuasion's already-correct pattern -- see that file's own fix
# comment). A SECOND bug in the ablation script itself (baseline and
# ablated passes reused the identical game_id per trial, so the "ablated"
# pass found baseline's already-updated ValueBounds instead of a fresh
# one -- see scripts/ablation_montecarlo.py's reset_per_game_state
# docstring) inflated the first corrected measurement; with both fixed,
# the mechanism genuinely engages (per-archetype scores now differ,
# baseline vs ablated -- e.g. tit_for_tat -0.273 vs -0.322) and the
# aggregate effect is small but real: baseline=-0.251, ablated=-0.262,
# delta=-0.011, still well inside this suite's per-cell noise floor
# (stdev 0.54-0.64 at n=400/archetype). Disabling remains a validated
# "no material cost" simplification, not an untested guess.
DISABLE_VALUE_BOUNDS = True
UNLIMITED_TAU = 30.0  # matches bargaining's own CONCESSION_TAU -- don't let the concession
                       # curve collapse to our reservation value within the first ~30 rounds
# CONFIRMED live (2026-08-25): the 0.5 "split the difference" value above
# still priced most one-shot ultimatums out of a deal. Pulled every
# max_rounds==1 SELLER-proposer game from a 150-game sample (this game
# variant always makes the seller the sole proposer when max_rounds==1 --
# 58/58 confirmed): under incomplete information, T=0.5 with
# DEFAULT_PRIOR_SPREAD=1.5 deterministically produces an ask of exactly
# 1.2788x our own reservation value (verified via this function's own
# formula), and that ask was accepted only 6 of 22 times (27%) -- the other
# 16 got a guaranteed $0, since a one-shot rejection has no recovery. By
# contrast, the 5 games in the same sample where the ask was grounded in a
# REAL known buyer value (complete_information=True) instead of the blind
# 1.5x guess landed at a 1.11-1.14x ratio and were accepted 5/5. There's no
# direct blind-condition data at intermediate ratios to fit an optimal curve
# from, but the direction and magnitude of the current failure (73% total
# loss) is unambiguous, and a one-shot's payoff asymmetry (reject = guaranteed
# $0, no future round to recover in) means underpricing costs far less than
# overpricing here. Moving T from 0.5 to 0.75 brings the blind-condition ask
# down to 1.1435x -- close to the 1.14x ratio that cleared 100% in the
# complete-information sample -- a deliberately moderate, evidence-anchored
# correction, not a claimed-optimal one; the recurring transcript checks
# should keep tracking this specific game shape (max_rounds==1, seller,
# incomplete information) to see whether the new ratio's acceptance rate
# holds up, and refine further if it doesn't.
ONE_SHOT_ULTIMATUM_T = 0.75

# DERIVED (2026-08-27), not yet live-tested: the T=0.75 derivation above
# never actually distinguished complete from incomplete information -- it
# just so happened that the 5 complete-information games in that sample had
# narrow true ZOPAs, so the same formula's absolute dollar shortfall looked
# like near-full extraction (1.11-1.14x reservation) purely because there
# wasn't much room to leave on the table either way. A real live game broke
# that coincidence: seller (my_value=$10,000), complete information, true
# buyer value known exactly at $12,000 (a full $2,000 ZOPA -- 20% of
# reservation, much wider than the validating sample). T=0.75 anchored on
# the REAL known ceiling still only asked $10,574.25 -- 28.7% of the $2,000
# surplus -- accepted instantly by the buyer with zero pushback, the same
# "instant accept means we under-asked" signal already flagged elsewhere in
# this codebase. T=0.75 exists to hedge against NOT knowing the buyer's true
# ceiling (the whole point of ONE_SHOT_ULTIMATUM_T's derivation); once that
# ceiling is exact, not guessed, there's nothing left to hedge against -- a
# rational responder accepts anything at or below their true value, and a
# genuine one-shot has no future round to recover an underpriced ask in
# either direction. ONE_SHOT_ULTIMATUM_COMPLETE_INFO_T anchors close to the
# known ceiling instead (0.03, mirroring bargaining's ROBUSTNESS_BUFFER_FRAC
# in both magnitude and purpose): the same $10,000/$12,000 case becomes an
# $11,967.60 ask, capturing ~98% of the surplus instead of 29%, while still
# leaving the buyer $32.40 rather than pricing at the exact knife-edge of
# their reservation.
ONE_SHOT_ULTIMATUM_COMPLETE_INFO_T = 0.03
STATIONARY_OPPONENT_ROUNDS = 3  # matches bargaining's threshold -- this many consecutive
                                  # near-identical opponent offers counts as proof they've stopped moving
STATIONARY_OPPONENT_TOLERANCE_FRAC = 0.005  # offers within 0.5% of my_value count as "the same"

# CONFIRMED live (2026-08-23): a DIFFERENT walk-away risk than the stationary
# case above -- that one needs three repeated offers to detect, but an
# opponent can walk away after just ONE round if our own concession looks
# too stingy, before any repetition ever happens. A real game: seller
# (my_value=$80, i.e. cost) opened at $160 (the standard 2x anchor); the
# buyer's very first counter was $116.10 -- already a comfortable 45% margin
# over cost, genuinely profitable -- but CONCESSION_BETA's deliberately
# Boulware curve (hold firm early, concede near the deadline) only pulled
# our counter down to $157.02, a 1.9% move. The buyer walked at the very
# next round, turning a solid, already-available profit into $0. Boulware is
# a real, working tradeoff (extract more from a patient opponent) but it's
# only worth the walk-away risk it carries when the offer on the table isn't
# already clearly good -- once it clears a large enough margin over our own
# reservation value, holding out for theoretically more isn't worth risking
# the whole deal. This is a strict ADDITION to the existing accept paths
# above (stationary, target-met) -- it only ever turns a reject into an
# accept, never the reverse, and only for offers already far more profitable
# than anything in the existing test suite, so it can't affect any case
# already relying on patient Boulware extraction of a thin-margin offer.
# 0.30 is a judgment call (not derived from anything more precise), picked
# comfortably below the 45% margin actually observed walking away, so it
# would have caught that real case with real room to spare.
EARLY_ACCEPT_MARGIN_FRAC = 0.30

# CONFIRMED live (2026-08-24): the stationary-accept path above gates only on
# `profitable`, a bare `>=`/`<=` -- an opponent who plants their offer
# EXACTLY at our reservation value and never moves counts as "profitable"
# and triggers immediate acceptance at literally $0 surplus. Five real
# transcripts in one batch shared this shape: a stationary opponent camped
# within 0-12% of our own reservation value while our own Boulware counter
# kept sliding toward them every round (no reciprocity gating existed on the
# OFFER side at all -- see OFFER_CONCESSION_DAMPING_FRAC below), and the
# instant our converging counter happened to clear the bare `profitable`
# line, we accepted immediately. Two of the five were EXACT ties (seller
# my_value=$8,000/$12,000, buyer offer identically $8,000/$12,000 for 3
# straight rounds) -- handing over 100% of a known $2,000-$3,000 ZOPA for
# nothing, when holding out had zero downside (a walkaway would have cost
# the same $0 already on the table). STATIONARY_ACCEPT_MIN_MARGIN_FRAC
# requires the stationary offer clear a real fraction of the ZOPA before
# treating "proven immovable" as reason enough to take it immediately.
#
# Deliberately gated to `complete_information` only (see
# `_stationary_offer_clears_min_margin`'s docstring): under incomplete
# information, `bounds` is a guessed range (DEFAULT_PRIOR_SPREAD), not a
# real ZOPA measurement, and the ORIGINAL motivating case for the
# stationary-accept path itself (a real walkaway after we kept pushing past
# a proven-immovable $118.80 offer, my_value=$120 -- see
# `_opponent_offer_is_stationary`'s docstring) was itself an incomplete-
# information game with a margin just as thin ($1.20, 3% of the guessed
# $40 range). Applying this bar there would have reproduced the exact
# walkaway this mechanism exists to prevent. Under complete information the
# five real cases above all had EXACT, known ZOPA widths, so "10% of a
# guess" becomes "10% of the real thing" -- a meaningfully different,
# actually-trustworthy signal. 0.10 is a judgment call, chosen comfortably
# above the five real thin-margin cases (0%, 0%, 2.3%, 3%, 12.2% of ZOPA)
# so it catches the first four while still letting the fifth (a genuinely
# decent 12.2% margin) through untouched.
STATIONARY_ACCEPT_MIN_MARGIN_FRAC = 0.10

# CONFIRMED live (2026-08-24): see STATIONARY_ACCEPT_MIN_MARGIN_FRAC's
# derivation above -- the margin fix above only tightens when we're willing
# to ACCEPT a stationary offer outright. It does nothing about the deeper
# cause: our own OFFER-side counter, driven purely by `concession_target`'s
# time-based curve, slides toward the opponent's fixed number every single
# round regardless of whether they've shown any reciprocity at all. Ported
# from bargaining's identical fix (OFFER_CONCESSION_DAMPING_FRAC there):
# once the opponent is proven stationary and the formula wants to concede
# further than our own last offer, only take a fraction of that concession
# rather than the full scheduled amount. Applies regardless of
# complete/incomplete information -- unlike the margin fix, this doesn't
# require knowing the true ZOPA, it only slows how fast we move toward a
# number that has already been proven fixed.
OFFER_CONCESSION_DAMPING_FRAC = 0.3

_BOUNDS: dict[str, ValueBounds] = {}
_LAST_SENT_PRICE: dict[str, float] = {}


def _get_bounds(game_id: str, role: str, my_value: float, opponent_true_value: float | None) -> ValueBounds:
    if game_id not in _BOUNDS:
        if opponent_true_value is not None:
            # Complete information: the exact zone of agreement is known, not
            # guessed — anchor at the opponent's real reservation instead of a
            # generic multiplier, so we're not leaving free surplus on the
            # table or anchoring somewhere unrealistic.
            seller_value = my_value if role == "seller" else opponent_true_value
            buyer_value = my_value if role == "buyer" else opponent_true_value
            _BOUNDS[game_id] = ValueBounds(lo=seller_value, hi=buyer_value)
        elif role == "seller":
            _BOUNDS[game_id] = ValueBounds(lo=my_value, hi=my_value * DEFAULT_PRIOR_SPREAD)
        else:
            _BOUNDS[game_id] = ValueBounds(lo=my_value / DEFAULT_PRIOR_SPREAD, hi=my_value)
    return _BOUNDS[game_id]


def _rounds_remaining(state: dict) -> int | None:
    max_rounds = state.get("max_rounds")
    if not state.get("horizon_known", max_rounds is not None) or max_rounds is None:
        return None
    return max(int(max_rounds) - int(state["round"]), 0)


def _opponent_offer_is_stationary(game: dict, opp: str, my_value: float, current_price: float) -> bool:
    """Whether the opponent's last STATIONARY_OPPONENT_ROUNDS offers,
    INCLUDING the one currently on the table, are all within tolerance of
    each other.

    Deliberately includes `current_price`, not just `history.get_history`
    (which only ever contains prior, already-decided rounds -- the round
    under active decision is passed separately as `last_offer` and hasn't
    been folded into `history` yet). Checking history alone would need a
    FOURTH matching offer to confirm three-in-a-row, one round later than
    the third repeat itself -- in the real incident this was built from
    (a seller offering an identical $118.80 for rounds 7, 9, 11), that extra
    round of confirmation would only have fired while deciding on a
    round-13 offer that never came, because the seller walked away at round
    12. Including the current offer lets this fire in time, at round 11.
    """
    if my_value <= 0:
        return False
    prior_opponent_prices = [
        h["offer"]["price"] for h in history.get_history(game)
        if history.is_opponent_entry(h, opp) and h.get("offer", {}).get("price") is not None
    ]
    recent = prior_opponent_prices[-(STATIONARY_OPPONENT_ROUNDS - 1):] + [current_price]
    if len(recent) < STATIONARY_OPPONENT_ROUNDS:
        return False
    return max(recent) - min(recent) <= STATIONARY_OPPONENT_TOLERANCE_FRAC * my_value


def _stationary_offer_clears_min_margin(
    role: str, my_value: float, offer_price: float, bounds: ValueBounds, complete_information: bool,
) -> bool:
    """Whether a stationary offer clears STATIONARY_ACCEPT_MIN_MARGIN_FRAC of
    the ZOPA before the stationary-accept shortcut treats it as good enough
    to take immediately, instead of just bare non-negative profit.

    Only applies under `complete_information` -- `bounds` is the REAL,
    known ZOPA there (see `_get_bounds`), so "10% of the ZOPA" is a
    trustworthy signal that real room is being left on the table. Under
    incomplete information `bounds` is a guessed range
    (`DEFAULT_PRIOR_SPREAD`), not a real measurement, and gating on it risks
    reproducing the exact real walkaway (`_opponent_offer_is_stationary`'s
    own motivating incident) this margin check was built to avoid causing
    -- so incomplete-information games always clear this check (fall back
    to the original bare-profitable bar)."""
    if not complete_information:
        return True
    zopa_width = bounds.hi - bounds.lo
    if zopa_width <= 0:
        return True
    margin = (offer_price - my_value) if role == "seller" else (my_value - offer_price)
    return margin >= STATIONARY_ACCEPT_MIN_MARGIN_FRAC * zopa_width


def _offer_clears_safety_margin(role: str, my_value: float, offer_price: float) -> bool:
    """Whether `offer_price` is already so far past bare profitability that
    holding out for the Boulware target isn't worth the walk-away risk --
    see EARLY_ACCEPT_MARGIN_FRAC's derivation. Does NOT check `profitable`
    itself (the caller already does); this only measures how comfortably
    past that bare line the offer sits."""
    if my_value <= 0:
        return False
    if role == "seller":
        return (offer_price - my_value) / my_value >= EARLY_ACCEPT_MARGIN_FRAC
    return (my_value - offer_price) / my_value >= EARLY_ACCEPT_MARGIN_FRAC


# Per-game (never pooled across games -- see StyleBandit's own docstring)
# Thompson Sampling over which moral-foundation framing to lead an offer's
# message with. Ported from bargaining's identical port from persuasion's
# seller-side StyleBandit (2026-08-23): the only observable feedback is the
# reject-and-counter case (accepting or walking away both end the game
# immediately, with no further call to us to observe it), scored by whether
# the opponent's next price moved toward us relative to their own previous
# price. See bargaining.strategy's equivalent for the full rationale.
# CONFIRMED live (2026-08-24): `get_style_bandit`/`persist_style_bandit`
# (agent.rhetoric.bandit) resolve to a disk-persisted, cross-game bandit
# keyed by the opponent's revealed name when known, or an ordinary fresh
# per-game bandit when identity is hidden -- see that module's docstring.
_LAST_STYLE_ARM: dict[str, str] = {}


def _opponent_display_name(game: dict) -> str | None:
    return (game.get("opponent") or {}).get("name")


def _style_bandit(game: dict) -> StyleBandit:
    return get_style_bandit(game["game_id"], _opponent_display_name(game))


def _credit_style_bandit_from_opponent_response(
    game: dict, opp: str, game_id: str, role: str, current_price: float, my_value: float,
) -> None:
    """Pops any style credited to our last offer and scores it against
    whether the opponent's current price moved toward us (lower for a
    seller judging a buyer's price is backwards -- a SELLER wants the
    opponent's price to go UP, a BUYER wants it to go DOWN) relative to
    their own previous price. `history.get_history` only contains prior,
    already-decided rounds -- same timing point as
    `_opponent_offer_is_stationary` -- so the round under active decision
    arrives separately as `current_price`. Requires at least one prior
    opponent price to compare against; their first offer in the game has
    nothing to compare yet, so this is a no-op rather than a guessed
    failure.

    DERIVED (2026-08-24), not yet live-tested: see bargaining's identical
    fix for the full derivation (including a corrected earlier attempt that
    wrongly filtered out exact-zero movement as "ambiguous" -- it isn't,
    it's the clearest possible failure signal there is). Success now
    requires movement to clear STATIONARY_OPPONENT_TOLERANCE_FRAC *
    my_value (the same noise band `_opponent_offer_is_stationary` already
    uses); zero, negative, or noise-level-small movement all still
    correctly count as failure."""
    arm = _LAST_STYLE_ARM.pop(game_id, None)
    if arm is None:
        return
    prior_opponent_prices = [
        h["offer"]["price"] for h in history.get_history(game)
        if history.is_opponent_entry(h, opp) and h.get("offer", {}).get("price") is not None
    ]
    if not prior_opponent_prices:
        return
    moved_by = (
        current_price - prior_opponent_prices[-1] if role == "seller"
        else prior_opponent_prices[-1] - current_price
    )  # positive = moved toward us
    success = moved_by > STATIONARY_OPPONENT_TOLERANCE_FRAC * my_value
    _style_bandit(game).update(arm, success)
    persist_style_bandit(game_id, _opponent_display_name(game))


def negotiation_strategy(game: dict) -> dict:
    state = game["game_state"]
    valid_actions = game["valid_actions"]
    me = game["your_player"]
    opp = opponent_name(me)
    role = state[f"{me}_role"]
    opponent_role = "buyer" if role == "seller" else "seller"
    my_value = state[f"{me}_value"]
    game_id = game["game_id"]

    opp_value_key = "player_1_value" if f"{me}_value" == "player_2_value" else "player_2_value"
    complete_information = state.get("complete_information", opp_value_key in state)

    rounds_remaining = _rounds_remaining(state)
    messages_allowed = state.get("messages_allowed", False)
    opponent_profile = classify(
        m["offer"].get("message", "")
        for m in history.get_history(game)
        if history.is_opponent_entry(m, opp) and m.get("offer", {}).get("message")
    ) if messages_allowed else None

    opponent_true_value = state.get(opp_value_key) if complete_information else None
    bounds = _get_bounds(game_id, role, my_value, opponent_true_value)

    if valid_actions["type"] == "decision":
        last_offer = state["last_offer"]
        history.record_opponent_offer(game, last_offer)
        offer_price = last_offer["price"]
        _credit_style_bandit_from_opponent_response(game, opp, game_id, role, offer_price, my_value)

        prior_price = _LAST_SENT_PRICE.get(game_id)
        if not DISABLE_VALUE_BOUNDS and prior_price is not None and not complete_information:
            bounds.tighten_from_rejection(opponent_role, prior_price)

        profitable = offer_price <= my_value if role == "buyer" else offer_price >= my_value

        if rounds_remaining == 0:
            action = {"decision": "AcceptOffer"} if profitable else {"decision": "WalkAway"}
            check_fields(valid_actions, action)
            history.record_own_action(game, action)
            return action

        if (
            profitable
            and _opponent_offer_is_stationary(game, opp, my_value, offer_price)
            and _stationary_offer_clears_min_margin(role, my_value, offer_price, bounds, complete_information)
        ):
            # CONFIRMED live: an opponent repeating the identical price for
            # several rounds has already revealed, empirically, that further
            # pushing won't extract more -- and unlike bargaining, continuing
            # to push here risks the opponent simply walking away (no
            # discounting exists in this family to make holding out costly
            # on its own). Take the proven, profitable offer now.
            action = {"decision": "AcceptOffer"}
            check_fields(valid_actions, action)
            history.record_own_action(game, action)
            return action

        if profitable and _offer_clears_safety_margin(role, my_value, offer_price):
            # CONFIRMED live: see EARLY_ACCEPT_MARGIN_FRAC's derivation --
            # an offer this far past bare profitability is worth taking
            # immediately rather than risking a walk-away by holding out for
            # the (still-early, still-firm) Boulware target.
            action = {"decision": "AcceptOffer"}
            check_fields(valid_actions, action)
            history.record_own_action(game, action)
            return action

        t = time_fraction(state["round"], state.get("max_rounds"), unlimited_tau=UNLIMITED_TAU)
        anchor = bounds.hi if role == "seller" else bounds.lo
        target = concession_target(anchor, my_value, t, CONCESSION_BETA)

        meets_target = offer_price >= target if role == "seller" else offer_price <= target
        if meets_target and profitable:
            action = {"decision": "AcceptOffer"}
            check_fields(valid_actions, action)
            history.record_own_action(game, action)
            return action

        counter = target
        counter = max(counter, my_value) if role == "seller" else min(counter, my_value)

        # CONFIRMED live (2026-08-24): see OFFER_CONCESSION_DAMPING_FRAC's
        # derivation. If the opponent has proven stationary and the formula
        # wants to concede further than our own last offer, only take a
        # fraction of that concession rather than sliding the full scheduled
        # amount toward a number already shown not to move.
        if prior_price is not None:
            conceded_further = counter < prior_price if role == "seller" else counter > prior_price
            if conceded_further and _opponent_offer_is_stationary(game, opp, my_value, offer_price):
                counter = (
                    prior_price - OFFER_CONCESSION_DAMPING_FRAC * (prior_price - counter) if role == "seller"
                    else prior_price + OFFER_CONCESSION_DAMPING_FRAC * (counter - prior_price)
                )

        message = None
        if messages_allowed:
            style = _style_bandit(game).select_arm()
            message = compose_message(
                family="negotiation",
                situation=game["prompt"],
                stance={
                    "role": role,
                    "my_reservation_value": my_value,
                    "their_last_price": offer_price,
                    "my_counter_price": counter,
                    "deadline_pressure": deadline_pressure(rounds_remaining),
                    "fallback_message": "That price doesn't reflect the real value here — here's my counter.",
                },
                profile=opponent_profile,
                style=style,
            )
            _LAST_STYLE_ARM[game_id] = style
        action = {"decision": "RejectOffer", "product_price": counter}
        if message:
            action["message"] = message
        _LAST_SENT_PRICE[game_id] = counter
        check_fields(valid_actions, action)
        history.record_own_action(game, action)
        return action

    if valid_actions["type"] == "offer":
        if state.get("max_rounds") == 1:
            # See ONE_SHOT_ULTIMATUM_COMPLETE_INFO_T's derivation: the blind
            # T=0.75 hedge only makes sense when the opponent's true
            # reservation value is a guess, not a known fact.
            t = ONE_SHOT_ULTIMATUM_COMPLETE_INFO_T if complete_information else ONE_SHOT_ULTIMATUM_T
        else:
            t = time_fraction(state["round"], state.get("max_rounds"), unlimited_tau=UNLIMITED_TAU)
        anchor = bounds.hi if role == "seller" else bounds.lo
        price = concession_target(anchor, my_value, t, CONCESSION_BETA)
        price = max(price, my_value) if role == "seller" else min(price, my_value)

        message = None
        if messages_allowed:
            style = _style_bandit(game).select_arm()
            message = compose_message(
                family="negotiation",
                situation=game["prompt"],
                stance={
                    "role": role,
                    "my_reservation_value": my_value,
                    "opening_price": price,
                    "fallback_message": "Here's my opening price — grounded in the real value of this deal.",
                },
                profile=opponent_profile,
                style=style,
            )
            _LAST_STYLE_ARM[game_id] = style
        action = {"product_price": price}
        if message:
            action["message"] = message
        _LAST_SENT_PRICE[game_id] = price
        check_fields(valid_actions, action)
        history.record_own_action(game, action)
        return action

    return {}
