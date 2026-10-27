"""Bargaining strategy: Rubinstein shares wrapped in guardrails for live opponents.

Proposals and accept/reject thresholds come from the (damped) Rubinstein
solution, clamped to a 30-70% band when the model degenerates, softened over
time against opponents who never concede, and overridden near the deadline
(Section 3.1 and Appendix A). A language model then argues for the chosen split.
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
OPENING_OFFER_50_50_DELTA_ME_MAX = 0.90
# Hard bounds on our share in degenerate/unlimited-horizon cases, where the raw
# formula can demand the whole pot or collapse to zero.
MIN_SELF_SHARE_FRAC = 0.30
# Skips updating the discount-factor belief, so delta_opp stays at the prior mean.
DISABLE_DELTA_BELIEF = True
MAX_SELF_SHARE_FRAC = 0.70
DEADLINE_PANIC_ROUND = 90
FINITE_PANIC_ROUNDS_REMAINING = 2

PANIC_ZONE_MIN_FRAC = 0.10

# Unlimited-horizon games appear to end around round 99-100; start treating
# offers as final a little before that.
ABSOLUTE_DEADLINE_ROUND = 97
CONCESSION_TAU = 30.0
CONCESSION_BETA = 0.7
DEGENERATE_DELTA_THRESHOLD = 0.999

DEGENERATE_BLEND_START = 0.99
STATIONARY_OPPONENT_ROUNDS = 3
STATIONARY_OPPONENT_TOLERANCE_FRAC = 0.005

NOT_CONCEDING_ACCEPT_FRAC = 0.75

DEGENERATE_NOT_CONCEDING_ROUNDS = 7

OFFER_CONCESSION_DAMPING_FRAC = 0.3

EROSION_CONFIRMATION_ROUNDS = DEGENERATE_NOT_CONCEDING_ROUNDS

EROSION_TAU = CONCESSION_TAU / 3.0

HOPELESS_CONFIRMATION_ROUNDS = 2 * EROSION_CONFIRMATION_ROUNDS

_LAST_STYLE_ARM: dict[str, str] = {}

_LAST_SENT_TAKE: dict[str, float] = {}


def _opponent_display_name(game: dict) -> str | None:
    return (game.get("opponent") or {}).get("name")


def _style_bandit(game: dict) -> StyleBandit:
    return get_style_bandit(game["game_id"], _opponent_display_name(game))


def _credit_style_bandit_from_opponent_response(
    game: dict, opp: str, game_id: str, current_opp_gain: float, money: float,
) -> None:
    arm = _LAST_STYLE_ARM.pop(game_id, None)
    if arm is None:
        return
    prior_opponent_offers = [
        read_gain(h["offer"], opp) for h in history.get_history(game)
        if history.is_opponent_entry(h, opp) and h.get("offer")
    ]
    if not prior_opponent_offers:
        return
    moved_by = prior_opponent_offers[-1] - current_opp_gain
    success = moved_by > STATIONARY_OPPONENT_TOLERANCE_FRAC * money
    _style_bandit(game).update(arm, success)
    persist_style_bandit(game_id, _opponent_display_name(game))


def _is_degenerate(delta_me: float, delta_opp: float) -> bool:
    return delta_me >= DEGENERATE_DELTA_THRESHOLD or delta_opp >= DEGENERATE_DELTA_THRESHOLD


# Ramps from 0 to 1 as a factor approaches 1, so the strategy blends smoothly
# into the degenerate-case handling instead of switching abruptly.
def _degeneracy_weight(delta_me: float, delta_opp: float) -> float:
    d = max(delta_me, delta_opp)
    if d <= DEGENERATE_BLEND_START:
        return 0.0
    if d >= DEGENERATE_DELTA_THRESHOLD:
        return 1.0
    return (d - DEGENERATE_BLEND_START) / (DEGENERATE_DELTA_THRESHOLD - DEGENERATE_BLEND_START)


def _resolve_opponent_last_ask(game: dict, opp: str) -> float | None:
    last_offer = game["game_state"].get("last_offer")
    if last_offer and last_offer.get("proposer") == opp:
        return read_gain(last_offer, opp)
    opponent_offers = [
        read_gain(h["offer"], opp) for h in history.get_history(game)
        if history.is_opponent_entry(h, opp) and h.get("offer")
    ]
    return opponent_offers[-1] if opponent_offers else None


# True when the opponent's own take has not dropped over the last few offers.
def _opponent_not_conceding(
    game: dict, opp: str, money: float, current_opp_gain: float,
    required_rounds: int = STATIONARY_OPPONENT_ROUNDS,
) -> bool:
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
    cumulative_not_conceding = recent[-1] >= recent[0] - tolerance
    return per_step_not_conceding and cumulative_not_conceding


def _soften_offer_floor(
    raw_take: float, floor_val: float, opening_anchor: float, hard_cap: float,
    round_no: int, max_rounds: int | None, money: float = 0.0, force_soften: bool = False,
    erode_over_time: bool = False, erosion_tau: float = CONCESSION_TAU,
    force_soften_tau: float = CONCESSION_TAU, below_floor_tau: float = CONCESSION_TAU,
) -> float:
    if raw_take < floor_val:
        t = time_fraction(round_no, max_rounds, unlimited_tau=below_floor_tau)
        return concession_target(opening_anchor, floor_val, t, CONCESSION_BETA)
    if force_soften or raw_take > hard_cap:
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
    threshold = max(raw_threshold, floor_val)
    if force_soften or threshold > ceiling_val:
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
    if rounds_remaining is None or _is_degenerate(delta_me, delta_opp):
        return infinite_horizon_share(money, delta_me, delta_opp)
    delta_of = lambda p: delta_me if p == me else delta_opp
    return damped_finite_horizon_take(money, me, delta_of, rounds_remaining)


def _reject_value(money, me, delta_me, delta_opp, rounds_remaining) -> float:
    if rounds_remaining is None or _is_degenerate(delta_me, delta_opp):
        return delta_me * infinite_horizon_share(money, delta_me, delta_opp)
    if rounds_remaining <= 0:
        return 0.0
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

    opp_last_ask_early = _resolve_opponent_last_ask(game, opp)
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

    # Deadline safety zone: near a real or suspected deadline, mirror the
    # opponent's last position rather than hold out into a zero-payoff ending.
    finite_panic_zone = rounds_remaining is not None and rounds_remaining <= FINITE_PANIC_ROUNDS_REMAINING
    unlimited_panic_zone = rounds_remaining is None and int(state["round"]) >= DEADLINE_PANIC_ROUND
    in_panic_zone = finite_panic_zone or unlimited_panic_zone or confirmed_hopeless
    panic_mode_engaged = in_panic_zone and not (
        finite_panic_zone and delta_me >= DEGENERATE_DELTA_THRESHOLD and rounds_remaining > 1
    )
    true_last_chance = (rounds_remaining is not None and rounds_remaining <= 1) or (
        rounds_remaining is None and int(state["round"]) >= ABSOLUTE_DEADLINE_ROUND
    )

    if valid_actions["type"] == "offer":
        if panic_mode_engaged:
            opp_self_ask = _resolve_opponent_last_ask(game, opp)
            my_take = (money - opp_self_ask) if opp_self_ask is not None else money / 2
            my_take = min(max(my_take, 0.0), money)
        elif (
            not complete_information and rounds_remaining is None
            and int(state["round"]) == 1 and delta_me <= OPENING_OFFER_50_50_DELTA_ME_MAX
        ):
            my_take = money / 2.0
            _LAST_SENT_TAKE[game["game_id"]] = my_take
        else:
            raw_take = _my_take(money, me, opp, delta_me, delta_opp, rounds_remaining)
            # Small haircut so a slightly misestimated discount factor doesn't
            # push the ask past what the opponent will accept.
            raw_take *= (1 - ROBUSTNESS_BUFFER_FRAC)
            max_rounds = state.get("max_rounds") if rounds_remaining is not None else None
            degeneracy_weight = _degeneracy_weight(delta_me, delta_opp)

            opp_last_ask = _resolve_opponent_last_ask(game, opp)
            confirmed_not_conceding = opp_last_ask is not None and _opponent_not_conceding(
                game, opp, money, opp_last_ask, required_rounds=EROSION_CONFIRMATION_ROUNDS,
            )
            common_kwargs = dict(
                floor_val=MIN_SELF_SHARE_FRAC * money, opening_anchor=MAX_SELF_SHARE_FRAC * money,
                round_no=int(state["round"]), max_rounds=max_rounds, money=money,
                erode_over_time=rounds_remaining is None and confirmed_not_conceding,
                erosion_tau=EROSION_TAU,
                force_soften_tau=EROSION_TAU if confirmed_not_conceding else CONCESSION_TAU,
                below_floor_tau=EROSION_TAU if confirmed_not_conceding else CONCESSION_TAU,
            )
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
            if opp_last_ask is not None:
                my_take = max(my_take, money - opp_last_ask)
                my_take = min(my_take, money)

            # Against a holdout, move our ask down only part of the way each
            # round so a drop in the raw target doesn't become a large concession.
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
            game_belief.update(opp_take_fraction, predicted_fraction_fn)

        if panic_mode_engaged:
            if true_last_chance:
                decision = "accept" if my_gain > 0 else "reject"
            else:
                decision = "accept" if my_gain >= PANIC_ZONE_MIN_FRAC * money else "reject"
        else:
            raw_threshold = _reject_value(money, me, delta_me, delta_opp, rounds_remaining)
            max_rounds = state.get("max_rounds") if rounds_remaining is not None else None
            degeneracy_weight = _degeneracy_weight(delta_me, delta_opp)
            confirmed_not_conceding_for_erosion = _opponent_not_conceding(
                game, opp, money, read_gain(last_offer, opp), required_rounds=EROSION_CONFIRMATION_ROUNDS,
            )
            common_kwargs = dict(
                money=money, round_no=int(state["round"]), max_rounds=max_rounds,
                erode_over_time=rounds_remaining is None and confirmed_not_conceding_for_erosion,
                erosion_tau=EROSION_TAU,
                force_soften_tau=EROSION_TAU if confirmed_not_conceding_for_erosion else CONCESSION_TAU,
            )
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

            # Accept a stalled, still-reasonable offer instead of waiting for a
            # better one that a non-conceding opponent will never make.
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
                decision = "accept"
            else:
                decision = "accept" if my_gain >= threshold - ACCEPT_TOLERANCE_FRAC * money else "reject"
        action = {"decision": decision}
        history.record_own_action(game, action)
        if decision == "accept":
            record_concession_evidence(_opponent_display_name(game), "bargaining", confirmed_hopeless)
        return action

    return {}
