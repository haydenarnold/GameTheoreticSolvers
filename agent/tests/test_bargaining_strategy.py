import pytest

import agent.rhetoric.opponent_profile as profile_mod
from agent.bargaining.strategy import HOPELESS_CONFIRMATION_ROUNDS, MIN_SELF_SHARE_FRAC, bargaining_strategy


def _isolate_concession_profile_store(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_mod, "_STORE_PATH", str(tmp_path / "opponent_concession_profile.json"))
    monkeypatch.setattr(profile_mod, "_CACHE", {})


def _offer_game(game_id, me, delta_me, delta_opp):
    delta_me_key = "delta_1" if me == "alice" else "delta_2"
    delta_opp_key = "delta_2" if me == "alice" else "delta_1"
    return {
        "game_id": game_id,
        "game_family": "bargaining",
        "your_player": me,
        "prompt": "Split the money.",
        "game_state": {
            "current_player": me,
            "proposer": me,
            "round": 1,
            "max_rounds": None,
            "horizon_known": False,
            "money_to_divide": 1_000_000.0,
            delta_me_key: delta_me,
            delta_opp_key: delta_opp,
            "complete_information": True,
            "messages_allowed": False,
            "last_offer": None,
        },
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
    }


def _decision_game(game_id, me, delta_me, delta_opp, my_gain, money=1_000_000.0):
    opp = "bob" if me == "alice" else "alice"
    delta_me_key = "delta_1" if me == "alice" else "delta_2"
    delta_opp_key = "delta_2" if me == "alice" else "delta_1"
    return {
        "game_id": game_id,
        "game_family": "bargaining",
        "your_player": me,
        "prompt": "Respond to the offer.",
        "game_state": {
            "current_player": me,
            "proposer": opp,
            "round": 1,
            "max_rounds": None,
            "horizon_known": False,
            "money_to_divide": money,
            delta_me_key: delta_me,
            delta_opp_key: delta_opp,
            "complete_information": True,
            "messages_allowed": False,
            "last_offer": {
                "alice_gain": my_gain if me == "alice" else money - my_gain,
                "bob_gain": my_gain if me == "bob" else money - my_gain,
                "proposer": opp,
                "round": 1,
            },
        },
        "valid_actions": {"type": "decision", "fields": {"decision": "'accept', 'reject', or 'walkaway'"}},
    }


def test_never_offers_away_the_entire_pot_to_an_infinitely_patient_opponent():
    # Regression: opponent delta=1.0 (0% inflation), unlimited horizon made
    # the raw Rubinstein formula evaluate to exactly $0 for us, and we
    # offered the opponent the full $1,000,000 in a real live game.
    game = _offer_game("barg-floor-offer", me="alice", delta_me=0.95, delta_opp=1.0)
    action = bargaining_strategy(game)
    assert action["alice_gain"] >= 0.30 * 1_000_000.0 - 1e-6


def test_never_accepts_next_to_nothing_from_an_infinitely_patient_opponent():
    # Same degeneracy on the responder side: the reject-continuation
    # threshold also collapsed to ~$0, meaning we'd accept literally any
    # positive offer. A lowball offer below the floor must now be rejected.
    game = _decision_game("barg-floor-decision", me="bob", delta_me=0.95, delta_opp=1.0, my_gain=50_000.0)
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_still_accepts_a_generous_offer_above_the_floor():
    # Genuinely generous (at our own ceiling ask) must clear even a round-1
    # threshold, which now correctly holds out near the ceiling early rather
    # than flatlining at the 30% floor for the whole game (see the
    # force_soften fix) -- 500,000 (50%) no longer qualifies as "generous"
    # once round-1 holds out near 70%, so use an offer at the ceiling itself.
    game = _decision_game("barg-floor-decision-2", me="bob", delta_me=0.95, delta_opp=1.0, my_gain=700_000.0)
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_never_demands_the_entire_pot_when_we_are_the_infinitely_patient_side():
    # Regression: WE were reported at delta=1.0 (0% inflation), unlimited
    # horizon, against an opponent at 0.95. The formula pinned our offer at
    # exactly 97% of the pot every single round (zero variation, since
    # delta_me=1.0 makes it return `money` regardless of delta_opp) — a real
    # $10,000 game deadlocked for 99 rounds ending in $0/$0 as a result.
    game = _offer_game("barg-ceiling-offer", me="bob", delta_me=1.0, delta_opp=0.95)
    action = bargaining_strategy(game)
    assert action["bob_gain"] <= 0.70 * 1_000_000.0 + 1e-6


def test_never_requires_the_entire_pot_to_accept_when_we_are_the_infinitely_patient_side():
    # Same degeneracy on the accept side: our own threshold came out to the
    # full pot, meaning agreement was mathematically impossible against any
    # opponent unwilling to hand over 100%. An offer at the ceiling itself
    # must now be accepted rather than rejected outright.
    game = _decision_game("barg-ceiling-decision", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=700_000.0)
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_panic_zone_accepts_a_net_positive_offer_even_below_the_normal_threshold():
    # Regression: at round 99 of an unlimited-horizon game, the opponent
    # offered us $612,531 of a $1,000,000 pot (61%) -- a large, unambiguous
    # net-positive amount -- and we rejected it because our formula-driven
    # threshold (even capped at 70%) still demanded more. The game then hit
    # the confirmed hidden round cap and ended $0/$0 for both.
    game = _decision_game("barg-panic-accept", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=612_531.573)
    game["game_state"]["round"] = 99
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_panic_zone_still_rejects_a_zero_or_negative_offer():
    game = _decision_game("barg-panic-reject-zero", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=0.0)
    game["game_state"]["round"] = 99
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_panic_zone_not_triggered_below_the_threshold_round():
    # One round short of the panic threshold, normal (formula + bounds +
    # time-softened) logic still applies, not the "accept anything positive"
    # override -- an offer well below even the round-89 softened threshold
    # (~51% at this delta pairing) is still rejected.
    game = _decision_game("barg-not-yet-panic", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=400_000.0)
    game["game_state"]["round"] = 89
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_panic_zone_offer_mirrors_opponents_own_last_revealed_position():
    # The highest-probability-of-acceptance move at the deadline: give them
    # exactly what they last asked for themselves rather than continuing to
    # propose a split they've shown no willingness to accept.
    game = _offer_game("barg-panic-offer-mirror", me="bob", delta_me=1.0, delta_opp=0.95)
    game["game_state"]["round"] = 95
    game["game_state"]["last_offer"] = {
        "alice_gain": 387_468.427,
        "bob_gain": 612_531.573,
        "proposer": "alice",
        "round": 94,
    }
    action = bargaining_strategy(game)
    assert abs(action["alice_gain"] - 387_468.427) < 1e-6
    assert abs(action["bob_gain"] - 612_531.573) < 1e-6


def test_panic_zone_offer_splits_down_the_middle_with_no_revealed_opponent_position():
    game = _offer_game("barg-panic-offer-nohistory", me="bob", delta_me=1.0, delta_opp=0.95)
    game["game_state"]["round"] = 95
    action = bargaining_strategy(game)
    assert abs(action["bob_gain"] - 500_000.0) < 1e-6


def test_never_counter_offers_ourselves_less_than_the_standing_offer():
    # Regression: the opponent offered us $622,331, we rejected it, then our
    # formula-driven counter-offer (computed independently of their revealed
    # offer) asked for only $610,909 for ourselves -- literally less than
    # what was already on the table. If we're countering, it should be
    # because we want MORE, never less, regardless of what the theory says.
    game = _offer_game("barg-standing-offer-floor", me="bob", delta_me=0.9, delta_opp=0.95)
    game["game_state"]["last_offer"] = {
        "alice_gain": 350_000.0,
        "bob_gain": 650_000.0,
        "proposer": "alice",
        "round": 1,
    }
    action = bargaining_strategy(game)
    assert action["bob_gain"] >= 650_000.0 - 1e-6


def test_incomplete_information_finite_horizon_decision_does_not_crash():
    # Regression: the damping-fix refactor dropped finite_horizon_take from
    # strategy.py's imports while a belief-update call sites still
    # referenced it directly -- a live NameError crash risk for any
    # incomplete-information, finite-horizon (max_rounds set) decision that
    # needed to update the opponent-delta belief. No test exercised this
    # exact combination at the time, so it shipped silently.
    game = _decision_game("barg-crash-check", me="bob", delta_me=0.9, delta_opp=0.8, my_gain=400_000.0)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 3
    game["game_state"]["complete_information"] = False
    del game["game_state"]["delta_1"]  # opponent's delta must be genuinely hidden
    action = bargaining_strategy(game)  # must not raise
    assert action["decision"] in ("accept", "reject")


def test_ceiling_bound_offer_softens_toward_midpoint_over_time():
    # Reversed on purpose (2026-08-23): this used to assert the ceiling-
    # bound offer stays flat at 70% forever with zero time-dependence.
    # CONFIRMED live: a real game (delta_me=1.0, degenerate) had our own
    # offer pinned at EXACTLY 70%/30% for 5 straight rounds (1, 3, 5, 7, 9)
    # with zero movement, while the opponent's own offer to us steadily
    # worsened over that same stretch -- plausibly in response to our total
    # rigidity. `_soften_offer_floor`'s ceiling branch now mirrors
    # `_soften_threshold_ceiling`'s already-existing behavior: it softens
    # from the ceiling toward a neutral 50/50 as rounds accumulate, instead
    # of staying pinned at the ceiling for the whole game.
    money = 1_000_000.0
    early = _offer_game("barg-ceiling-soften-early", me="bob", delta_me=1.0, delta_opp=0.95)
    early["game_state"]["round"] = 1
    late = _offer_game("barg-ceiling-soften-late", me="bob", delta_me=1.0, delta_opp=0.95)
    late["game_state"]["round"] = 60

    early_gain = bargaining_strategy(early)["bob_gain"]
    late_gain = bargaining_strategy(late)["bob_gain"]
    assert early_gain == pytest.approx(698_484.54, abs=1.0)  # round 1: only barely softened off the ceiling
    assert late_gain < early_gain  # round 60: meaningfully softened down
    assert late_gain > 0.50 * money  # but still hasn't collapsed all the way to a neutral split


def test_floor_bound_offer_softens_over_time_even_in_finite_horizon():
    # The actual fix: a COMPLETE-information, FINITE-horizon game with the
    # opponent's delta at exactly 1.0 degenerates to zero at every node of
    # the backward induction (not just the infinite-horizon formula), so
    # this used to jump straight to the 30% floor on round 1 with zero
    # negotiation runway -- confirmed live, the opponent accepted instantly.
    # It should now start near the ceiling and concede toward the floor as
    # rounds accumulate, using the real max_rounds (not the unlimited-
    # horizon curve).
    money = 100.0
    game = _offer_game("barg-floor-finite-early", me="alice", delta_me=0.95, delta_opp=1.0)
    game["game_state"]["money_to_divide"] = money
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 1
    early_take = bargaining_strategy(game)["alice_gain"]

    late_game = _offer_game("barg-floor-finite-late", me="alice", delta_me=0.95, delta_opp=1.0)
    late_game["game_state"]["money_to_divide"] = money
    late_game["game_state"]["max_rounds"] = 12
    late_game["game_state"]["horizon_known"] = True
    late_game["game_state"]["round"] = 8
    late_take = bargaining_strategy(late_game)["alice_gain"]

    assert early_take > 0.65 * money  # opens assertively, not at the floor
    assert late_take < early_take  # genuinely concedes as rounds pass
    assert late_take >= MIN_SELF_SHARE_FRAC * money - 1e-6  # never below the floor


def test_unlimited_horizon_offer_never_drops_below_the_min_floor_while_softening():
    money = 1_000_000.0
    game = _offer_game("barg-concession-floor-safety", me="bob", delta_me=0.5, delta_opp=0.5)
    game["game_state"]["round"] = 89  # right at the edge of the panic threshold
    take = bargaining_strategy(game)["bob_gain"]
    assert take >= MIN_SELF_SHARE_FRAC * money - 1e-6


def test_unlimited_horizon_accept_threshold_also_softens_over_time():
    # Cycle 2's original fix, preserved: a firm-but-reasonable offer that
    # falls short of the ceiling early should have a real chance of clearing
    # our (now lower) bar later in the game.
    money = 1_000_000.0
    offer_at_60_percent = 600_000.0

    early = _decision_game("barg-accept-soften-early", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=offer_at_60_percent)
    early["game_state"]["round"] = 1
    assert bargaining_strategy(early)["decision"] == "reject"

    late = _decision_game("barg-accept-soften-late", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=offer_at_60_percent)
    late["game_state"]["round"] = 60
    assert bargaining_strategy(late)["decision"] == "accept"


def test_belief_update_uses_damped_predictions_not_oscillating_raw_ones(monkeypatch):
    # Regression: a live game had the belief update compare an opponent's
    # offer against `finite_horizon_take` (undamped, round-parity-
    # oscillating -- the exact defect `damped_finite_horizon_take` exists
    # to fix for our own decisions). At the odd rounds_remaining=11 in this
    # setup, even a delta_opp=0.99 hypothesis only predicts a 72.4% self-ask
    # under the undamped formula, so an observed 75% offer collapsed the
    # posterior mean to ~0.96 from a SINGLE observation. Using the damped
    # formula lands a materially less overconfident mean (~0.92) for the
    # same evidence -- still high (a 75% self-ask genuinely doesn't fit
    # moderate-delta hypotheses well even damped), but not a parity artifact.
    from agent.bargaining import belief as belief_mod
    from agent.bargaining import strategy as strategy_mod

    monkeypatch.setattr(strategy_mod, "DISABLE_DELTA_BELIEF", False)
    game = _decision_game(
        "barg-belief-damped", me="bob", delta_me=0.8, delta_opp=0.95, my_gain=2_500.0, money=10_000.0
    )
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 1
    game["game_state"]["complete_information"] = False
    del game["game_state"]["delta_1"]  # opponent's true delta must be genuinely hidden

    bargaining_strategy(game)

    mean = belief_mod.get_belief("barg-belief-damped").mean()
    assert mean < 0.94


def test_finite_horizon_panic_zone_offer_mirrors_opponent_near_the_real_deadline():
    # Regression: a live max_rounds=12 game held the ceiling flat (0%
    # softening, by design) the entire game, and the opponent didn't
    # concede until round 12 itself -- the certain deadline, with zero
    # safety margin. Finite horizons never had any panic-zone protection at
    # all (it was gated on rounds_remaining is None), unlike the unlimited-
    # horizon soft ~99-round cap. This close to a real, known deadline, we
    # must mirror the opponent's own last revealed position instead of
    # continuing to hold the ceiling.
    game = _offer_game("barg-finite-panic-mirror", me="bob", delta_me=1.0, delta_opp=0.95)
    game["game_state"]["money_to_divide"] = 100.0
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 11  # rounds_remaining = 1, inside the finite panic margin
    game["game_state"]["last_offer"] = {
        "alice_gain": 60.0,
        "bob_gain": 40.0,
        "proposer": "alice",
        "round": 10,
    }
    action = bargaining_strategy(game)
    assert abs(action["bob_gain"] - 40.0) < 1e-6


def test_finite_horizon_panic_zone_decision_accepts_any_net_positive_offer():
    # Non-degenerate delta_me: waiting genuinely costs real decaying value,
    # so the wider FINITE_PANIC_ROUNDS_REMAINING margin (not just the
    # literal last round) correctly applies here.
    game = _decision_game("barg-finite-panic-accept", me="bob", delta_me=0.9, delta_opp=0.95, my_gain=5.0, money=100.0)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 11  # rounds_remaining = 1
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_degenerate_delta_me_does_not_panic_accept_early_in_finite_horizon():
    # Regression: a live game had delta_me=1.0 (us, truly infinitely
    # patient -- zero cost to waiting) accept a lopsided $390k/$610k offer
    # at round 10 of 12 (rounds_remaining=2), TWO rounds before the real
    # deadline, despite the ordinary softened threshold there ($567k) being
    # far above the accepted amount. The panic override was built for a
    # player who DOES lose real value by waiting -- for a genuinely patient
    # delta_me, folding early throws away value for no reason. This must
    # still reject at rounds_remaining=2 and hold to the ordinary threshold.
    game = _decision_game(
        "barg-degenerate-no-early-panic", me="bob", delta_me=1.0, delta_opp=0.9, my_gain=39.0, money=100.0
    )
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 10  # rounds_remaining = 2
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_degenerate_delta_me_panic_accepts_at_rounds_remaining_one_when_that_is_the_true_last_chance():
    # CONFIRMED live (2026-08-23): the test above (rounds_remaining=2)
    # correctly stays patient, but the old `rounds_remaining > 0` exclusion
    # assumed a decision at rounds_remaining==0 would eventually arrive --
    # which never happens for a responder whose own decisions always land
    # on odd rounds_remaining in this exact shape (Bob, 12-round horizon,
    # Alice proposing first: Bob's decisions occur at rounds_remaining =
    # 11,9,7,5,3,1 -- never 0). A degenerate-delta_me Bob held out through
    # his literal last decision (round 11, rounds_remaining=1) and was then
    # forced to gamble on his own final round-12 offer with no way to
    # reconsider if it was refused -- exactly the $0 risk this override
    # exists to avoid. rounds_remaining <= 1 always means no further
    # decision is coming (the next one, if any, is always 2 rounds later),
    # so this must panic-accept here too, not just at rounds_remaining==0.
    game = _decision_game(
        "barg-degenerate-panic-at-last-decision", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=43.0, money=100.0
    )
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 11  # rounds_remaining = 1, Bob's true last decision round
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_not_conceding_override_needs_more_confirmations_when_our_own_delta_is_degenerate():
    # CONFIRMED live (2026-08-23): a real game (delta_me=1.0 -- 0%
    # inflation, zero cost to us from waiting -- delta_opp=0.95, $1,000,000
    # pot) had the opponent repeat an identical $570,000/$430,000 offer 3
    # times. The not-conceding override accepted the $430,000 (barely above
    # MIN_SELF_SHARE_FRAC's 30% floor) at round 5, when the properly-reasoned,
    # softened threshold at that same round wanted at least $658,368 -- and
    # further holdout was genuinely free, since delta_me=1.0 means WE don't
    # decay at all. Only 3 total confirmations exist here (rounds 1, 3, 5) --
    # below DEGENERATE_NOT_CONCEDING_ROUNDS (7) -- so this must still reject,
    # now because more proof is required before trusting it when patience is
    # free, not because the override is disabled outright (see
    # test_not_conceding_override_eventually_fires_with_enough_confirmations_when_degenerate
    # for the case where enough proof exists).
    game = _decision_game(
        "barg-degenerate-not-conceding-holds-out", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=430000.0, money=1_000_000.0
    )
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 5  # rounds_remaining = 7, nowhere near the true last decision
    game["game_state"]["last_offer"] = {"alice_gain": 570000.0, "bob_gain": 430000.0, "proposer": "alice", "round": 5}
    game["game_state"]["history"] = _stationary_offer_history("alice", opp_self_take=570000.0, rounds=[1, 3])
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_not_conceding_override_eventually_fires_with_enough_confirmations_when_degenerate():
    # CONFIRMED live (2026-08-24): a real game (delta_me=1.0, $1,000,000
    # pot, unlimited horizon) had the opponent offer EXACTLY $500,000
    # identically for all 45 of their turns across 90 rounds. With the
    # override fully disabled for degenerate delta_me, we held out to the
    # literal DEADLINE_PANIC_ROUND=90 backstop -- 10 rounds short of the
    # confirmed hidden round cap -- for the same $500,000 available since
    # round 1. With DEGENERATE_NOT_CONCEDING_ROUNDS (7) confirmations of a
    # stationary $500,000 (well past the 0.75x-threshold bar, which this
    # exact config crosses around round 10), this must now accept -- around
    # round 14 rather than round 90.
    money = 1_000_000.0
    game = _decision_game(
        "barg-degenerate-not-conceding-eventually-fires", me="bob", delta_me=1.0, delta_opp=0.95,
        my_gain=500_000.0, money=money,
    )
    game["game_state"]["round"] = 14
    game["game_state"]["last_offer"] = {"alice_gain": 500_000.0, "bob_gain": 500_000.0, "proposer": "alice", "round": 14}
    game["game_state"]["history"] = _stationary_offer_history("alice", opp_self_take=500_000.0, rounds=[2, 4, 6, 8, 10, 12])
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_degenerate_delta_me_still_locks_in_a_stationary_offer_at_the_true_last_decision():
    # End-to-end replay of the actual live game above, continued out to
    # Bob's true last decision round (11) with the opponent still
    # perfectly stationary at $570,000/$430,000 the whole way: the
    # not-conceding override stays disabled (delta_me degenerate), so the
    # ordinary threshold governs -- and by round 11 the panic-zone fix
    # above correctly locks in the positive offer rather than risking a
    # gamble on Bob's own final round-12 counter.
    game = _decision_game(
        "barg-degenerate-locks-in-at-last-decision", me="bob", delta_me=1.0, delta_opp=0.95, my_gain=430000.0, money=1_000_000.0
    )
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 11  # rounds_remaining = 1, the true last decision
    game["game_state"]["last_offer"] = {"alice_gain": 570000.0, "bob_gain": 430000.0, "proposer": "alice", "round": 11}
    game["game_state"]["history"] = _stationary_offer_history("alice", opp_self_take=570000.0, rounds=[1, 3, 5, 7, 9])
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_degenerate_delta_me_still_panic_accepts_at_the_true_literal_deadline():
    # The safety net must still exist at the actual last round -- a
    # degenerate delta_me shouldn't risk a certain $0/$0 either.
    game = _decision_game(
        "barg-degenerate-panic-at-deadline", me="bob", delta_me=1.0, delta_opp=0.9, my_gain=5.0, money=100.0
    )
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 12  # rounds_remaining = 0, the true deadline
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_finite_horizon_panic_zone_not_triggered_with_ample_rounds_left():
    # Same delta pairing that produced the round-12 nail-biter live, but
    # with plenty of rounds_remaining -- normal ceiling-bound logic still
    # applies (now softening toward midpoint over time, per the offer-side
    # fix above), not the panic override. $684,533.74 is the exact
    # concession-curve value at round 3 of 12 -- meaningfully below the
    # 70% ceiling (confirming softening is engaged) but nowhere near a
    # panic-zone "accept/offer anything" value either.
    game = _offer_game("barg-finite-not-yet-panic", me="bob", delta_me=1.0, delta_opp=0.95)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 3  # rounds_remaining = 9, well outside the panic margin
    action = bargaining_strategy(game)
    assert action["bob_gain"] == pytest.approx(684_533.74, abs=1.0)


def _stationary_offer_history(opp: str, opp_self_take: float, rounds: list[int]) -> list[dict]:
    opp_gain_key = "alice_gain" if opp == "alice" else "bob_gain"
    me_gain_key = "bob_gain" if opp == "alice" else "alice_gain"
    return [
        {
            "speaker": "opponent", "round": r,
            "offer": {opp_gain_key: opp_self_take, me_gain_key: 100.0 - opp_self_take, "proposer": opp, "round": r},
        }
        for r in rounds
    ]


def test_accepts_immediately_when_opponent_offer_has_been_stationary_for_several_rounds():
    # Regression: a live game had the opponent repeat the identical offer
    # for 39 straight rounds while we kept applying the ordinary
    # time-based softening curve, not accepting until round 40 -- by then,
    # 5%/round decay had eaten the real payoff down to ~13.5% of nominal.
    # A non-degenerate pairing this early in an unlimited-horizon game
    # would ordinarily still be holding out near the ceiling; three
    # identical prior offers should override that and accept now.
    game = _decision_game("barg-stationary-accept", me="alice", delta_me=0.95, delta_opp=0.90, my_gain=53.0, money=100.0)
    game["game_state"]["round"] = 4
    game["game_state"]["last_offer"] = {"alice_gain": 53.0, "bob_gain": 47.0, "proposer": "bob", "round": 4}
    game["game_state"]["history"] = _stationary_offer_history("bob", opp_self_take=47.0, rounds=[1, 2, 3])
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_stationary_detection_fires_against_the_real_server_history_shape():
    # Regression: the server's actual history entries have NO "speaker" key
    # at all -- confirmed live by pulling a real game_state() directly from
    # the API, which returns entries shaped {"round", "proposer", "offer",
    # "decision", "response_time_ms"}. The old `h.get("speaker") ==
    # "opponent"` filter silently matched zero entries against this real
    # shape (only the local fallback log ever used "speaker"), so
    # stationary-opponent detection never actually fired in production. One
    # real game ground out to round 91 against an opponent who proposed an
    # identical 60/40 split for 88+ consecutive rounds, and our own delta
    # (0.8) compounded over that holdout crushed the realized payoff to
    # ~$0. Same parameters as the local-shape test above (proven to reject
    # under the ordinary, non-stationary threshold), but with genuine
    # server-shape history -- must now also accept.
    game = _decision_game("barg-stationary-server-shape", me="alice", delta_me=0.95, delta_opp=0.90, my_gain=53.0, money=100.0)
    game["game_state"]["round"] = 4
    game["game_state"]["last_offer"] = {"alice_gain": 53.0, "bob_gain": 47.0, "proposer": "bob", "round": 4}
    game["game_state"]["history"] = [
        {
            "round": r, "proposer": "bob",
            "offer": {"alice_gain": 53.0, "bob_gain": 47.0, "proposer": "bob", "round": r},
            "decision": "reject", "response_time_ms": 1000,
        }
        for r in [1, 2, 3]
    ]
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_stationary_detection_includes_the_current_offer_not_just_prior_history():
    # The timing fix's actual point: only 2 PRIOR entries exist in history
    # (round 4's own offer isn't part of history yet, since it's the one
    # currently under decision) -- combined with the current offer, that's
    # 3 total and must fire now, not require a 4th prior confirmation later.
    from agent.bargaining.strategy import _opponent_not_conceding

    game = _decision_game("barg-stationary-timing", me="alice", delta_me=0.95, delta_opp=0.90, my_gain=53.0, money=100.0)
    game["game_state"]["round"] = 4
    game["game_state"]["history"] = _stationary_offer_history("bob", opp_self_take=47.0, rounds=[2, 3])
    assert _opponent_not_conceding(game, "bob", 100.0, current_opp_gain=47.0)
    # A current offer that concedes real ground (opponent's self-ask drops
    # well past tolerance) must NOT be treated as "not conceding".
    assert not _opponent_not_conceding(game, "bob", 100.0, current_opp_gain=30.0)


def test_stationary_detection_still_respects_the_minimum_share_floor():
    # A stationary but genuinely bad (below-floor) offer must not be
    # rewarded just because it hasn't moved -- the floor protection still
    # applies.
    game = _decision_game("barg-stationary-lowball", me="alice", delta_me=0.95, delta_opp=0.90, my_gain=10.0, money=100.0)
    game["game_state"]["round"] = 4
    game["game_state"]["last_offer"] = {"alice_gain": 10.0, "bob_gain": 90.0, "proposer": "bob", "round": 4}
    game["game_state"]["history"] = _stationary_offer_history("bob", opp_self_take=90.0, rounds=[1, 2, 3])
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_stationary_detection_does_not_trigger_when_the_opponent_is_actually_moving():
    game = _decision_game("barg-not-stationary", me="alice", delta_me=0.95, delta_opp=0.90, my_gain=53.0, money=100.0)
    game["game_state"]["round"] = 4
    game["game_state"]["last_offer"] = {"alice_gain": 53.0, "bob_gain": 47.0, "proposer": "bob", "round": 4}
    game["game_state"]["history"] = [
        {"speaker": "opponent", "round": 1, "offer": {"alice_gain": 20.0, "bob_gain": 80.0, "proposer": "bob", "round": 1}},
        {"speaker": "opponent", "round": 2, "offer": {"alice_gain": 35.0, "bob_gain": 65.0, "proposer": "bob", "round": 2}},
        {"speaker": "opponent", "round": 3, "offer": {"alice_gain": 45.0, "bob_gain": 55.0, "proposer": "bob", "round": 3}},
    ]
    # Should fall through to the ordinary (non-stationary) threshold logic,
    # which at round 4 of an unlimited-horizon, non-degenerate pairing is
    # still holding out well above 53%.
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_not_conceding_fires_when_opponent_trends_worse_for_us_without_ever_going_flat():
    # CONFIRMED live (2026-08-23): delta_me=0.95/delta_opp=0.90, unlimited
    # horizon. The opponent's own self-ask rose every round (never flat) --
    # what she offered us strictly worsened round over round -- but the old
    # within-0.5%-band stationary check requires near-total flatness, so it
    # didn't fire until round 17, 16 rounds of 5%/round decay too late. Real
    # numbers from that game (opponent's own self-ask, i.e. her ask rising =
    # our offer shrinking): round13=4633.739, round15=4667.668,
    # round17=4667.668 (money=10000). This must now be treated as "not
    # conceding" as soon as 3 data points exist, not only once the last two
    # steps happen to also go flat.
    game = _decision_game("barg-trending-worse", me="bob", delta_me=0.95, delta_opp=0.90, my_gain=5366.261, money=10000.0)
    game["game_state"]["round"] = 17
    game["game_state"]["last_offer"] = {"alice_gain": 4667.668, "bob_gain": 5332.332, "proposer": "alice", "round": 17}
    game["game_state"]["history"] = [
        {
            "round": r, "proposer": "alice",
            "offer": {"alice_gain": alice_gain, "bob_gain": 10000.0 - alice_gain, "proposer": "alice", "round": r},
            "decision": "reject", "response_time_ms": 1000,
        }
        for r, alice_gain in [(13, 4633.739), (15, 4667.668)]
    ]
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_not_conceding_does_not_fire_on_only_two_data_points():
    # Guard against a false-early trigger: the window needs
    # STATIONARY_OPPONENT_ROUNDS total data points (matching the existing
    # timing-fix contract) before concluding anything about the trend, even
    # if the two points available so far look flat-or-worse.
    from agent.bargaining.strategy import _opponent_not_conceding

    game = _decision_game("barg-not-enough-data", me="bob", delta_me=0.95, delta_opp=0.90, my_gain=53.0, money=100.0)
    game["game_state"]["history"] = _stationary_offer_history("alice", opp_self_take=47.0, rounds=[1])
    assert not _opponent_not_conceding(game, "alice", 100.0, current_opp_gain=47.0)


def test_finite_horizon_offer_is_not_capped_at_70_percent_when_non_degenerate():
    # The fix: MAX_SELF_SHARE_FRAC's ceiling used to clip every offer to 70%
    # of the pot, even in an ordinary, non-degenerate, genuinely finite game
    # where the exact backward-induction share legitimately exceeds that --
    # delta_me=0.7/delta_opp=0.5 at rounds_remaining=6 computes to ~76.9%
    # (~74.6% after the standard 3% robustness haircut). That's real,
    # provably-correct equilibrium value, not a degenerate-formula artifact,
    # so it should no longer be clipped down to 700,000.
    money = 1_000_000.0
    game = _offer_game("barg-uncapped-offer", me="alice", delta_me=0.7, delta_opp=0.5)
    game["game_state"]["money_to_divide"] = money
    game["game_state"]["max_rounds"] = 7
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 1  # rounds_remaining = 6
    action = bargaining_strategy(game)
    assert action["alice_gain"] > 0.72 * money  # clearly above the old 70% cap
    assert action["alice_gain"] < money  # still sane, not the whole pot


def test_finite_horizon_accept_threshold_is_not_capped_at_70_percent_when_non_degenerate():
    # Mirror of the offer-side fix on the decision side: delta_me=0.85
    # (bob)/delta_opp=0.5 (alice) at rounds_remaining=6 computes a genuine
    # ~73.8% reject-continuation threshold. Under the old blanket cap this
    # would have softened to exactly 70% at round 1 (t=0), so a 72% offer
    # would have been accepted; now the real, higher threshold applies and
    # the same offer must be rejected instead.
    money = 1_000_000.0
    game = _decision_game(
        "barg-uncapped-threshold", me="bob", delta_me=0.85, delta_opp=0.5,
        my_gain=0.72 * money, money=money,
    )
    game["game_state"]["max_rounds"] = 7
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 1  # rounds_remaining = 6
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"

    generous = _decision_game(
        "barg-uncapped-threshold-generous", me="bob", delta_me=0.85, delta_opp=0.5,
        my_gain=0.76 * money, money=money,
    )
    generous["game_state"]["max_rounds"] = 7
    generous["game_state"]["horizon_known"] = True
    generous["game_state"]["round"] = 1
    assert bargaining_strategy(generous)["decision"] == "accept"


def test_finite_horizon_accept_threshold_also_softens_toward_midpoint():
    # The same threshold-softening now also applies to finite horizon, not
    # just unlimited -- a ceiling-bound threshold (delta_opp=1.0, complete
    # info, real max_rounds) should relax enough by round 8 of 12 to accept
    # an offer that a round-1 threshold would still reject.
    money = 100.0
    offer_at_60_percent = 60.0

    early = _decision_game("barg-finite-accept-early", me="alice", delta_me=0.95, delta_opp=1.0, my_gain=offer_at_60_percent, money=money)
    early["game_state"]["max_rounds"] = 12
    early["game_state"]["horizon_known"] = True
    early["game_state"]["round"] = 1
    assert bargaining_strategy(early)["decision"] == "reject"

    late = _decision_game("barg-finite-accept-late", me="alice", delta_me=0.95, delta_opp=1.0, my_gain=offer_at_60_percent, money=money)
    late["game_state"]["max_rounds"] = 12
    late["game_state"]["horizon_known"] = True
    late["game_state"]["round"] = 8
    assert bargaining_strategy(late)["decision"] == "accept"


def test_style_bandit_arm_is_passed_to_compose_message_on_an_offer(monkeypatch):
    from agent.rhetoric.bandit import StyleBandit

    captured = {}

    def fake_compose_message(**kwargs):
        captured.update(kwargs)
        return "a message"

    monkeypatch.setattr("agent.bargaining.strategy.compose_message", fake_compose_message)
    game = _offer_game("barg-style-offer", me="alice", delta_me=0.9, delta_opp=0.9)
    game["game_state"]["messages_allowed"] = True
    action = bargaining_strategy(game)
    assert captured["style"] in StyleBandit.ARMS
    assert action["message"] == "a message"

    from agent.bargaining import strategy as strategy_mod
    assert strategy_mod._LAST_STYLE_ARM["barg-style-offer"] == captured["style"]


def test_no_style_selected_when_messages_not_allowed():
    game = _offer_game("barg-no-style-offer", me="alice", delta_me=0.9, delta_opp=0.9)
    game["game_state"]["messages_allowed"] = False
    action = bargaining_strategy(game)
    assert "message" not in action

    from agent.bargaining import strategy as strategy_mod
    assert "barg-no-style-offer" not in strategy_mod._LAST_STYLE_ARM


def _opponent_offer_history_entries(opp: str, opp_self_takes: list[float], me_gain_total: float = 100.0) -> list[dict]:
    return [
        {
            "round": r,
            "proposer": opp,
            "offer": {
                f"{opp}_gain": take,
                "bob_gain" if opp == "alice" else "alice_gain": me_gain_total - take,
                "proposer": opp,
                "round": r,
            },
            "decision": "reject",
        }
        for r, take in enumerate(opp_self_takes, start=1)
    ]


def test_style_bandit_credits_success_when_opponent_moves_toward_us():
    from agent.bargaining import strategy as strategy_mod

    game_id = "barg-style-credit-success"
    strategy_mod._LAST_STYLE_ARM[game_id] = "fairness"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    alpha_before = bandit.alpha["fairness"]

    game = _decision_game(game_id, me="bob", delta_me=0.9, delta_opp=0.9, my_gain=55.0, money=100.0)
    game["game_state"]["last_offer"] = {"alice_gain": 45.0, "bob_gain": 55.0, "proposer": "alice", "round": 3}
    # Opponent's own take dropped 50 -> 45 across their last two offers -- moved toward us.
    game["game_state"]["history"] = _opponent_offer_history_entries("alice", [50.0])
    bargaining_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).alpha["fairness"] == alpha_before + 1.0
    assert game_id not in strategy_mod._LAST_STYLE_ARM  # popped regardless of outcome


def test_style_bandit_credits_failure_when_opponent_does_not_move():
    from agent.bargaining import strategy as strategy_mod

    game_id = "barg-style-credit-failure"
    strategy_mod._LAST_STYLE_ARM[game_id] = "authority"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    beta_before = bandit.beta["authority"]

    game = _decision_game(game_id, me="bob", delta_me=0.9, delta_opp=0.9, my_gain=50.0, money=100.0)
    game["game_state"]["last_offer"] = {"alice_gain": 50.0, "bob_gain": 50.0, "proposer": "alice", "round": 3}
    # Opponent's own take held flat 50 -> 50 -- did not move toward us.
    game["game_state"]["history"] = _opponent_offer_history_entries("alice", [50.0])
    bargaining_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).beta["authority"] == beta_before + 1.0


def test_style_bandit_does_not_credit_success_for_noise_level_movement():
    # DERIVED (2026-08-24): a $0.10 shift on a $100 pot is well below
    # STATIONARY_OPPONENT_TOLERANCE_FRAC's noise band (0.5% = $0.50) --
    # this should NOT count as clean success (that was the actual bug this
    # fix targets), but must still be recorded as a failure, not skipped.
    from agent.bargaining import strategy as strategy_mod

    game_id = "barg-style-credit-noise-movement"
    strategy_mod._LAST_STYLE_ARM[game_id] = "purity"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    alpha_before, beta_before = bandit.alpha["purity"], bandit.beta["purity"]

    game = _decision_game(game_id, me="bob", delta_me=0.9, delta_opp=0.9, my_gain=50.10, money=100.0)
    game["game_state"]["last_offer"] = {"alice_gain": 49.90, "bob_gain": 50.10, "proposer": "alice", "round": 3}
    game["game_state"]["history"] = _opponent_offer_history_entries("alice", [50.0])
    bargaining_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).alpha["purity"] == alpha_before  # not credited as success
    assert strategy_mod._style_bandit({"game_id": game_id}).beta["purity"] == beta_before + 1.0  # still counted as failure
    assert game_id not in strategy_mod._LAST_STYLE_ARM


def test_style_bandit_does_not_credit_with_no_prior_opponent_offer():
    from agent.bargaining import strategy as strategy_mod

    game_id = "barg-style-credit-insufficient-history"
    strategy_mod._LAST_STYLE_ARM[game_id] = "purity"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    alpha_before, beta_before = bandit.alpha["purity"], bandit.beta["purity"]

    game = _decision_game(game_id, me="bob", delta_me=0.9, delta_opp=0.9, my_gain=50.0, money=100.0)
    game["game_state"]["last_offer"] = {"alice_gain": 50.0, "bob_gain": 50.0, "proposer": "alice", "round": 1}
    game["game_state"]["history"] = []  # this is the opponent's first-ever offer -- nothing to compare
    bargaining_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).alpha["purity"] == alpha_before
    assert strategy_mod._style_bandit({"game_id": game_id}).beta["purity"] == beta_before
    assert game_id not in strategy_mod._LAST_STYLE_ARM  # still popped, just not scored


def test_not_conceding_override_requires_a_fraction_of_the_real_threshold_not_just_the_bare_floor():
    # CONFIRMED live (2026-08-23): delta_me=0.95, delta_opp=0.80 (neither
    # degenerate), unlimited horizon. Alice repeated an identical
    # $5,555.56/$4,444.44 offer 3 times. The old override accepted the
    # $4,444.44 immediately since it cleared MIN_SELF_SHARE_FRAC's bare 30%
    # floor ($3,000) -- but the properly-reasoned, softened threshold at
    # that round was $6,862.47. Giving up 24% of the pot to an offer that
    # only cleared bare survival, not anything close to what the formula
    # actually said we were owed. Must now reject and keep holding out.
    game = _decision_game("barg-not-conceding-below-real-threshold", me="bob", delta_me=0.95, delta_opp=0.80, my_gain=4444.44, money=10000.0)
    game["game_state"]["last_offer"] = {"alice_gain": 5555.56, "bob_gain": 4444.44, "proposer": "alice", "round": 5}
    game["game_state"]["history"] = _stationary_offer_history("alice", opp_self_take=5555.56, rounds=[1, 3])
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_not_conceding_override_still_fires_when_offer_is_close_to_the_real_threshold():
    # CONFIRMED live (2026-08-23): delta_me=0.95, delta_opp=0.95 (symmetric,
    # neither degenerate), unlimited horizon. Alice's near-stationary offer
    # ($380,469 for us) was only 10.7% below the real threshold ($487,179)
    # -- comfortably clears 75% of it ($365,384) -- so this must still
    # accept, same as before the fix. Confirms the fraction-based bar
    # doesn't just make the override strictly harder to satisfy across the
    # board -- offers genuinely close to the real threshold still resolve
    # the same way they did previously.
    game = _decision_game("barg-not-conceding-close-to-real-threshold", me="bob", delta_me=0.95, delta_opp=0.95, my_gain=380469.0, money=1_000_000.0)
    game["game_state"]["last_offer"] = {"alice_gain": 619531.0, "bob_gain": 380469.0, "proposer": "alice", "round": 5}
    game["game_state"]["history"] = _stationary_offer_history("alice", opp_self_take=619999.0, rounds=[1]) + [
        {
            "round": 3, "proposer": "alice",
            "offer": {"alice_gain": 619939.0, "bob_gain": 380061.0, "proposer": "alice", "round": 3},
            "decision": "reject", "response_time_ms": 1000,
        },
    ]
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_degenerate_offer_no_longer_stays_pinned_across_a_real_live_sequence():
    # End-to-end replay of the actual live incident's shape: Alice,
    # delta_me=1.0 (degenerate), incomplete info, max_rounds=12, round 1 vs
    # round 9 (the real game held flat at exactly 70/30 across rounds
    # 1, 3, 5, 7, 9 -- a perfectly rigid offer the whole way). Confirms the
    # two offers now genuinely differ instead of both landing on the
    # ceiling. (_offer_game fixes the pot at $1,000,000; the real game's
    # pot was $100, but the underlying fraction-of-pot behavior is
    # identical regardless of scale.)
    early = _offer_game("barg-degenerate-live-early", me="alice", delta_me=1.0, delta_opp=0.9)
    early["game_state"]["max_rounds"] = 12
    early["game_state"]["horizon_known"] = False  # matches the live game's incomplete-info framing
    early["game_state"]["round"] = 1

    late = _offer_game("barg-degenerate-live-late", me="alice", delta_me=1.0, delta_opp=0.9)
    late["game_state"]["max_rounds"] = 12
    late["game_state"]["horizon_known"] = False
    late["game_state"]["round"] = 9

    early_gain = bargaining_strategy(early)["alice_gain"]
    late_gain = bargaining_strategy(late)["alice_gain"]
    assert early_gain == pytest.approx(0.70 * 1_000_000.0, abs=2_000.0)  # round 1: still close to the ceiling
    assert late_gain < early_gain  # round 9: meaningfully softened, not flat


def test_finite_horizon_offer_no_longer_cliffs_at_the_degenerate_delta_boundary():
    # DERIVED (2026-08-25): offline stress testing (stress_test_invariants.py's
    # monotonicity sweep) found a sharp discontinuity right at
    # DEGENERATE_DELTA_THRESHOLD=0.999 for a genuinely finite horizon --
    # delta_me=0.95 computed a smooth, uncapped ~97% share ($100 pot,
    # delta_opp=0.01, max_rounds=12, round 1), while delta_me=0.999 fell back
    # to the degenerate formula AND got unconditionally capped down to a flat
    # 70% -- a 27-point cliff for increased patience, which the game's own
    # theory says should never make our position WORSE. Fine sweep across
    # [0.98, 1.0] (400 points) confirms the fix: no consecutive-point jump
    # exceeds a small tolerance anywhere in the blend window, vs. a ~27-point
    # jump before this fix (see `_degeneracy_weight`'s docstring for the
    # full diagnosis and the blend design). 0.5 (half a percent of the pot)
    # is a generous tolerance for a 400-point sweep -- the actual measured
    # max jump is roughly 0.15 on a $100 pot.
    money = 100.0
    prev_take = None
    max_jump = 0.0
    steps = 400
    for i in range(steps + 1):
        delta_me = 0.98 + (1.0 - 0.98) * i / steps
        game = _offer_game("barg-degenerate-continuity-sweep", me="alice", delta_me=delta_me, delta_opp=0.7)
        game["game_state"]["money_to_divide"] = money
        game["game_state"]["max_rounds"] = 12
        game["game_state"]["horizon_known"] = True
        game["game_state"]["round"] = 1
        take = bargaining_strategy(game)["alice_gain"]
        if prev_take is not None:
            max_jump = max(max_jump, abs(take - prev_take))
        prev_take = take
    assert max_jump < 0.5  # was ~27 (of a $100 pot) before this fix


def test_finite_horizon_offer_still_matches_old_behavior_well_below_the_threshold():
    # The blend must be a no-op far from the boundary -- DEGENERATE_BLEND_
    # START keeps ordinary, comfortably-non-degenerate games byte-identical
    # to before this fix.
    game = _offer_game("barg-degenerate-blend-noop", me="alice", delta_me=0.9, delta_opp=0.5)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 1
    take = bargaining_strategy(game)["alice_gain"]
    # Uncapped, exact backward-induction share for this comfortably non-
    # degenerate case -- not clamped toward MAX_SELF_SHARE_FRAC.
    assert take > 0.70 * 1_000_000.0


def test_finite_horizon_offer_still_hits_the_exact_degenerate_value_at_the_threshold():
    # Weight must reach exactly 1.0 AT the threshold, not just approach it --
    # delta_me=1.0 (the single most common real degenerate case) must be
    # completely unchanged from the pre-fix, already-live-validated behavior.
    game = _offer_game("barg-degenerate-blend-full", me="alice", delta_me=1.0, delta_opp=0.9)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 1
    take = bargaining_strategy(game)["alice_gain"]
    assert take == pytest.approx(0.70 * 1_000_000.0, abs=2_000.0)


def test_confirmed_stationarity_speeds_up_incomplete_information_belief_convergence():
    # CONFIRMED live (2026-08-24): two real incomplete-information games had
    # a genuinely, repeatedly stationary opponent (unchanged for 8-10 of
    # their own turns), but the not-conceding override still took another
    # 8-10 rounds beyond when stationarity first confirmed to actually
    # accept -- DeltaBelief's shrinkage is keyed to round count, not
    # evidence consistency, so it kept crawling toward the same answer the
    # repetition had already established. This replays the real numbers
    # from one of those games (money=$100, delta_me=0.95, Alice flat at
    # $54/$46 from round 5 onward) and confirms the fix moves acceptance
    # meaningfully earlier than before (round 19) without claiming it's
    # instant -- see the function's own docstring for why the boost is
    # deliberately bounded rather than aggressive.
    from agent.bargaining.strategy import bargaining_strategy as strategy_fn

    money = 100.0
    hist = []
    alice_asks = {1: 58.0, 3: 58.0, 5: 54.0, 7: 54.0, 9: 54.0, 11: 54.0, 13: 54.0, 15: 54.0, 17: 54.0, 19: 54.0}
    accepted_round = None
    for r in sorted(alice_asks):
        alice_gain = alice_asks[r]
        bob_gain = money - alice_gain
        game = {
            "game_id": "barg-belief-speedup-live-replay",
            "game_family": "bargaining",
            "your_player": "bob",
            "prompt": "Respond to the offer.",
            "game_state": {
                "current_player": "bob", "proposer": "alice", "round": r, "max_rounds": None,
                "horizon_known": False, "money_to_divide": money, "delta_2": 0.95,
                "complete_information": False, "messages_allowed": True,
                "last_offer": {"alice_gain": alice_gain, "bob_gain": bob_gain, "proposer": "alice", "round": r},
                "history": list(hist),
            },
            "valid_actions": {"type": "decision", "fields": {"decision": "'accept' or 'reject'"}},
        }
        action = strategy_fn(game)
        if action["decision"] == "accept":
            accepted_round = r
            break
        hist.append({"round": r, "proposer": "alice", "offer": game["game_state"]["last_offer"], "decision": "reject", "response_time_ms": 1000})

    assert accepted_round is not None
    assert accepted_round < 19  # strictly earlier than the pre-fix live outcome


def _offer_game_with_state(game_id, me, delta_me, delta_opp, round_no, max_rounds, last_offer, history):
    delta_me_key = "delta_1" if me == "alice" else "delta_2"
    delta_opp_key = "delta_2" if me == "alice" else "delta_1"
    return {
        "game_id": game_id,
        "game_family": "bargaining",
        "your_player": me,
        "prompt": "Split the money.",
        "game_state": {
            "current_player": me,
            "proposer": me,
            "round": round_no,
            "max_rounds": max_rounds,
            "horizon_known": max_rounds is not None,
            "money_to_divide": 100.0,
            delta_me_key: delta_me,
            delta_opp_key: delta_opp,
            "complete_information": True,
            "messages_allowed": False,
            "last_offer": last_offer,
            "history": history,
        },
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
    }


def test_offer_concession_is_dampened_against_a_non_reciprocating_opponent():
    # CONFIRMED live (2026-08-24): a real game (delta_me=0.80, delta_opp=0.95)
    # had Bob's own self-ask RISE every round (87.56% -> 96%, never
    # reciprocating) while our own offer fell on schedule regardless
    # (70% -> 47.6% over the same 5 offers). Replays that exact shape:
    # round 9's dampened offer must land meaningfully above the old,
    # undampened live value (47.587).
    game_id = "barg-damping-non-reciprocating"
    money = 100.0
    bob_asks = {2: 87.56, 4: 88.89, 6: 90.65, 8: 92.96}
    hist = []
    action = None
    for r in [1, 3, 5, 7, 9]:
        last_offer = None
        if r > 1:
            bob_ask = bob_asks[r - 1]
            last_offer = {"alice_gain": money - bob_ask, "bob_gain": bob_ask, "proposer": "bob", "round": r - 1}
        game = _offer_game_with_state(game_id, "alice", 0.80, 0.95, r, 12, last_offer, list(hist))
        action = bargaining_strategy(game)
        if last_offer:
            hist.append({"round": r - 1, "proposer": "bob", "offer": last_offer, "decision": "reject", "response_time_ms": 1000})
        hist.append({
            "round": r, "proposer": "alice",
            "offer": {"alice_gain": action["alice_gain"], "bob_gain": action["bob_gain"], "proposer": "alice", "round": r},
            "decision": "reject", "response_time_ms": 1000,
        })

    assert action["alice_gain"] > 47.587 + 5.0  # meaningfully above the old, undampened live outcome


def test_offer_concession_is_not_dampened_against_a_genuinely_conceding_opponent():
    # Mirror-image check: when the opponent IS reciprocating (their own
    # ask to us rising each round), _opponent_not_conceding is False, so
    # the ordinary, undampened concession curve must still apply exactly
    # as before -- this fix must not slow down concession universally.
    game_id = "barg-damping-genuine-concession"
    money = 100.0
    bob_asks = {2: 73.0, 4: 66.0, 6: 59.0}  # Bob's own ask falling -- genuine concession
    hist = []
    undamped_action = None
    for r in [1, 3, 5, 7]:
        last_offer = None
        if r > 1:
            bob_ask = bob_asks[r - 1]
            last_offer = {"alice_gain": money - bob_ask, "bob_gain": bob_ask, "proposer": "bob", "round": r - 1}
        game = _offer_game_with_state(game_id, "alice", 1.0, 0.9, r, 12, last_offer, list(hist))
        action = bargaining_strategy(game)
        if last_offer:
            hist.append({"round": r - 1, "proposer": "bob", "offer": last_offer, "decision": "reject", "response_time_ms": 1000})
        hist.append({
            "round": r, "proposer": "alice",
            "offer": {"alice_gain": action["alice_gain"], "bob_gain": action["bob_gain"], "proposer": "alice", "round": r},
            "decision": "reject", "response_time_ms": 1000,
        })
        undamped_action = action

    # Round 7's offer should match the plain, undampened ceiling-softening
    # curve at round 7 (money=100, ceiling=70, midpoint=50, t at round 7 of 12).
    from agent.negotiation.solver import concession_target, time_fraction
    t = time_fraction(7, 12, unlimited_tau=30.0)
    expected = concession_target(70.0, 50.0, t, 0.7)
    assert undamped_action["alice_gain"] == pytest.approx(expected, abs=0.5)


def test_offer_damping_does_not_apply_on_the_first_offer():
    # No prior own offer exists yet, so there's nothing to dampen against --
    # the very first offer of the game must equal the plain formula value.
    game = _offer_game_with_state("barg-damping-first-offer", "alice", 0.80, 0.95, 1, 12, None, [])
    action = bargaining_strategy(game)
    assert action["alice_gain"] == pytest.approx(70.0, abs=0.5)  # standard round-1 ceiling anchor, undamped


def test_offer_concession_is_dampened_via_history_when_last_offer_is_unresolvable():
    # CONFIRMED live (2026-08-24): `state["last_offer"]` is unreliable on
    # ordinary (non-panic) OFFER-type turns too, not just panic-zone ones --
    # unlike the two tests above, this reproduces the REAL live gap by
    # passing `last_offer=None` on every round (matching what the actual
    # API sends) while `history` carries the real opponent offers, exactly
    # as a real transcript did: Bob camped at an identical $50/$100 offer
    # for 5 straight rounds while our own counter kept sliding on the
    # undampened curve (70 -> 68.45 -> 65.84 -> 62.57 -> 58.79) all the way
    # down to proposing his own $50/$50 split ourselves. With the fallback,
    # round 5 (the first round with 3 confirmed stationary Bob offers)
    # must land meaningfully above the old undamped value of 65.8367911.
    game_id = "barg-damping-last-offer-unresolvable"

    def make_game(round_no, hist):
        game = _offer_game_with_state(game_id, "alice", 1.0, 0.9, round_no, 12, None, list(hist))
        game["game_state"]["complete_information"] = False
        del game["game_state"]["delta_2"]  # incomplete info -- use the belief estimate, matching the real game
        return game

    hist = []
    action = bargaining_strategy(make_game(1, hist))
    hist.append({
        "round": 1, "proposer": "alice",
        "offer": {"alice_gain": action["alice_gain"], "bob_gain": action["bob_gain"], "proposer": "alice", "round": 1},
        "decision": "reject", "response_time_ms": 1000,
    })
    for bob_round, alice_round in [(2, 3), (4, 5), (6, 7)]:
        hist.append({
            "round": bob_round, "proposer": "bob",
            "offer": {"alice_gain": 50.0, "bob_gain": 50.0, "proposer": "bob", "round": bob_round},
            "decision": "reject", "response_time_ms": 1000,
        })
        action = bargaining_strategy(make_game(alice_round, hist))
        hist.append({
            "round": alice_round, "proposer": "alice",
            "offer": {"alice_gain": action["alice_gain"], "bob_gain": action["bob_gain"], "proposer": "alice", "round": alice_round},
            "decision": "reject", "response_time_ms": 1000,
        })

    # Round 7's damped offer must land meaningfully above what round 7 gave
    # BEFORE this fix, undamped (62.57002855431526).
    assert action["alice_gain"] > 62.57002855431526 + 1.0


def test_resolve_opponent_last_ask_falls_back_to_history_when_last_offer_is_unusable():
    from agent.bargaining.strategy import _resolve_opponent_last_ask

    game = _offer_game_with_state(
        "barg-resolve-ask-fallback", "alice", 1.0, None, 5, 12, None,
        [{"round": 4, "proposer": "bob", "offer": {"alice_gain": 50.0, "bob_gain": 50.0, "proposer": "bob", "round": 4}}],
    )
    assert _resolve_opponent_last_ask(game, "bob") == 50.0


def test_resolve_opponent_last_ask_prefers_last_offer_when_it_is_usable():
    from agent.bargaining.strategy import _resolve_opponent_last_ask

    game = _offer_game_with_state(
        "barg-resolve-ask-last-offer", "alice", 1.0, None, 5, 12,
        {"alice_gain": 40.0, "bob_gain": 60.0, "proposer": "bob", "round": 4},
        [{"round": 4, "proposer": "bob", "offer": {"alice_gain": 50.0, "bob_gain": 50.0, "proposer": "bob", "round": 4}}],
    )
    # Uses the resolvable last_offer's 60.0, not history's stale 50.0.
    assert _resolve_opponent_last_ask(game, "bob") == 60.0


def test_panic_zone_requires_a_meaningful_floor_before_the_true_last_chance():
    # CONFIRMED live (2026-08-24): a real game ($1,000,000 pot, finite
    # 12-round horizon, non-degenerate delta_me=0.95) had the opponent hold
    # an identical $0.01-for-us offer across 4 straight rounds; we accepted
    # it at round 10 (rounds_remaining=2) because the old panic zone only
    # checked my_gain > 0. Rejecting there risked nothing -- a real
    # decision (round 11 counter, then the true round-12 deadline) still
    # existed. Must now reject a token offer this far from the true deadline.
    game = _decision_game("barg-panic-floor-reject-token-offer", me="bob", delta_me=0.95, delta_opp=1.0, my_gain=0.01, money=1_000_000.0)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 10  # rounds_remaining = 2, not the true last chance
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_panic_zone_still_accepts_anything_positive_at_the_true_last_chance():
    # The floor must not apply at the true literal deadline -- the whole
    # reason the panic zone exists is to avoid risking $0 there.
    game = _decision_game("barg-panic-floor-true-deadline-accepts", me="bob", delta_me=0.95, delta_opp=1.0, my_gain=0.01, money=1_000_000.0)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 12  # rounds_remaining = 0, the true last chance
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_panic_zone_floor_also_applies_mid_unlimited_horizon_panic_zone():
    # DEADLINE_PANIC_ROUND (90) is a buffer before the confirmed hidden cap,
    # not itself a literal deadline -- the floor should still apply well
    # into the unlimited-horizon panic zone, only relaxing near
    # ABSOLUTE_DEADLINE_ROUND.
    game = _decision_game("barg-panic-floor-unlimited-mid", me="bob", delta_me=0.95, delta_opp=0.95, my_gain=1.0, money=100.0)
    game["game_state"]["round"] = 92  # in the panic zone, but well short of ABSOLUTE_DEADLINE_ROUND (97)
    action = bargaining_strategy(game)
    assert action["decision"] == "reject"


def test_panic_zone_floor_relaxes_near_the_absolute_deadline_round():
    game = _decision_game("barg-panic-floor-unlimited-near-cap", me="bob", delta_me=0.95, delta_opp=0.95, my_gain=1.0, money=100.0)
    game["game_state"]["round"] = 98  # past ABSOLUTE_DEADLINE_ROUND (97)
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"


def test_panic_zone_offer_falls_back_to_history_when_last_offer_is_unusable():
    # CONFIRMED live (2026-08-24): pulled real completed games via the API
    # and found state["last_offer"] did not reliably resolve to the
    # opponent's actual last offer on OFFER-type panic-zone turns (4 of 5
    # real sampled cases fell back to a plain 50/50 split despite a clear
    # recent opponent position sitting in history) -- e.g. Alice had just
    # asked for $75 of a $100 pot, and we offered her $50 instead of
    # mirroring her real $75 ask. Replays that exact shape with
    # last_offer=None (the observed failure condition) and a populated
    # history -- must now mirror Alice's real position via history, not
    # fall back to a plain 50/50 split.
    game = _offer_game("barg-panic-mirror-history-fallback", me="bob", delta_me=0.95, delta_opp=1.0)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 10
    game["game_state"]["last_offer"] = None
    game["game_state"]["history"] = [
        {"round": 9, "proposer": "alice", "offer": {"alice_gain": 75.0, "bob_gain": 25.0, "proposer": "alice", "round": 9}, "decision": "reject"},
    ]
    game["game_state"]["money_to_divide"] = 100.0
    action = bargaining_strategy(game)
    assert action["bob_gain"] == pytest.approx(25.0, abs=0.01)
    assert action["alice_gain"] == pytest.approx(75.0, abs=0.01)


def test_offer_side_panic_zone_also_respects_the_degenerate_delta_exclusion():
    # CONFIRMED live (2026-08-24): a real game (delta_me=1.0 -- degenerate,
    # 0% inflation, patience genuinely free -- $1,000,000 pot, finite
    # 12-round horizon) had our round-10 offer (rounds_remaining=2)
    # collapse to a plain $500,000 via the (previously un-gated) offer-side
    # panic zone, when the ordinary, still-correctly-aggressive
    # formula-driven offer at that exact round wants $567,399 -- giving up
    # real value two full rounds before there was any need to rush. The
    # offer side must now share the identical degenerate-delta exclusion
    # the decision side already had, deferring panic-zone offer behavior
    # (mirror/50-50) until the true last chance too.
    game = _offer_game("barg-offer-panic-degenerate-exclusion", me="bob", delta_me=1.0, delta_opp=0.95)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 10  # rounds_remaining = 2, not the true last chance
    action = bargaining_strategy(game)
    assert action["bob_gain"] == pytest.approx(567_399.0, abs=50.0)
    assert action["bob_gain"] != pytest.approx(500_000.0, abs=1.0)  # must not collapse to the old panic-zone split


def test_offer_side_panic_zone_still_engages_at_the_true_last_chance_even_when_degenerate():
    # Once rounds_remaining <= 1 (the true last chance -- no further
    # decision is coming), the offer side must still switch to the
    # highest-probability-of-acceptance panic-zone move, same as before.
    game = _offer_game("barg-offer-panic-degenerate-true-last-chance", me="bob", delta_me=1.0, delta_opp=0.95)
    game["game_state"]["max_rounds"] = 12
    game["game_state"]["horizon_known"] = True
    game["game_state"]["round"] = 11  # rounds_remaining = 1, the true last chance
    action = bargaining_strategy(game)
    assert action["bob_gain"] == pytest.approx(500_000.0, abs=1.0)  # no revealed opponent position -> split down the middle


def test_soften_offer_floor_erode_over_time_does_not_change_early_round_behavior():
    from agent.bargaining.strategy import _soften_offer_floor

    # money=100, hard_cap=70, raw_take=69.286 (below the cap -- neither of
    # the two hard triggers fires). At round 2 of an unlimited-horizon game,
    # the erosion curve sits near hard_cap (70), well above 69.286, so
    # min() must pick raw_take unchanged -- no behavior change this early.
    result = _soften_offer_floor(
        69.286, 30.0, 70.0, 70.0, round_no=2, max_rounds=None, money=100.0,
        force_soften=False, erode_over_time=True,
    )
    assert result == pytest.approx(69.286, abs=0.01)


def test_soften_offer_floor_erode_over_time_lowers_a_frozen_offer_after_many_rounds():
    from agent.bargaining.strategy import _soften_offer_floor

    # CONFIRMED live (2026-08-24): the same 69.286 raw_take, many rounds
    # into an unlimited-horizon game, must now have eroded meaningfully
    # below its round-2 value instead of staying pinned there forever.
    early = _soften_offer_floor(
        69.286, 30.0, 70.0, 70.0, round_no=2, max_rounds=None, money=100.0,
        force_soften=False, erode_over_time=True,
    )
    late = _soften_offer_floor(
        69.286, 30.0, 70.0, 70.0, round_no=88, max_rounds=None, money=100.0,
        force_soften=False, erode_over_time=True,
    )
    assert late < early - 10.0


def test_soften_offer_floor_erode_over_time_is_a_noop_when_false():
    from agent.bargaining.strategy import _soften_offer_floor

    # Without the flag, a raw_take under the cap is returned completely
    # unchanged regardless of round number -- this is the exact behavior
    # that produced the real 44-round frozen offer before this fix.
    result = _soften_offer_floor(
        69.286, 30.0, 70.0, 70.0, round_no=88, max_rounds=None, money=100.0,
        force_soften=False, erode_over_time=False,
    )
    assert result == pytest.approx(69.286, abs=0.001)


def test_soften_threshold_ceiling_erode_over_time_lowers_a_frozen_threshold_after_many_rounds():
    from agent.bargaining.strategy import _soften_threshold_ceiling

    early = _soften_threshold_ceiling(
        64.29, 30.0, 70.0, 100.0, round_no=2, max_rounds=None,
        force_soften=False, erode_over_time=True,
    )
    late = _soften_threshold_ceiling(
        64.29, 30.0, 70.0, 100.0, round_no=88, max_rounds=None,
        force_soften=False, erode_over_time=True,
    )
    assert early == pytest.approx(64.29, abs=0.01)  # unchanged early on
    assert late < early - 5.0  # meaningfully eroded many rounds in


def test_unlimited_horizon_offer_no_longer_freezes_forever_against_a_stationary_opponent():
    # CONFIRMED live (2026-08-24): a real 90-round game (delta_me=0.9 for
    # Bob, delta_opp=0.8 for Alice, unlimited horizon, complete information)
    # had our own offer pinned at EXACTLY $69.286 for 44 straight rounds (2
    # through 88) against an opponent proven immovable at $44.44 the entire
    # game, before finally resolving only at the DEADLINE_PANIC_ROUND=90
    # safety net. Replays the exact shape: our own offer at round 88 must
    # now be meaningfully below round 2's value.
    game_id = "barg-90round-offer-no-longer-frozen"
    money = 100.0

    def make_offer_game(round_no, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Split the money.",
            "game_state": {
                "current_player": "bob", "proposer": "bob", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False, "last_offer": None, "history": hist,
            },
            "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
        }

    round2_action = bargaining_strategy(make_offer_game(2, []))
    hist = [
        {"round": r, "proposer": "alice", "offer": {"alice_gain": 55.56, "bob_gain": 44.44, "proposer": "alice", "round": r}, "decision": "reject"}
        for r in range(1, 88, 2)
    ]
    round88_action = bargaining_strategy(make_offer_game(88, hist))
    assert round2_action["bob_gain"] == pytest.approx(69.286, abs=0.01)
    assert round88_action["bob_gain"] < round2_action["bob_gain"] - 4.0


def test_unlimited_horizon_decision_accepts_a_stationary_offer_much_earlier_than_the_deadline_safety_net():
    # CONFIRMED live (2026-08-24): the same real 90-round game never
    # accepted Alice's proven-immovable $44.44 offer (needed >=
    # 0.75*64.29=$48.22 forever, a bar that never adapted) until round 90.
    # With the fix, the not-conceding accept bar erodes over time too, so
    # this must now resolve strictly before the deadline safety net.
    game_id = "barg-90round-decision-accepts-earlier"
    money = 100.0

    def make_decision_game(round_no, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Respond.",
            "game_state": {
                "current_player": "bob", "proposer": "alice", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False,
                "last_offer": {"alice_gain": 55.56, "bob_gain": 44.44, "proposer": "alice", "round": round_no},
                "history": hist,
            },
            "valid_actions": {"type": "decision", "fields": {"decision": "'accept' or 'reject'"}},
        }

    hist = []
    accepted_round = None
    for r in range(1, 90, 2):
        action = bargaining_strategy(make_decision_game(r, list(hist)))
        if action["decision"] == "accept":
            accepted_round = r
            break
        hist.append({"round": r, "proposer": "alice", "offer": {"alice_gain": 55.56, "bob_gain": 44.44, "proposer": "alice", "round": r}, "decision": "reject"})
        hist.append({"round": r + 1, "proposer": "bob", "offer": {"alice_gain": 30.714, "bob_gain": 69.286, "proposer": "bob", "round": r + 1}, "decision": "reject"})

    assert accepted_round is not None
    assert accepted_round < 89  # strictly earlier than the old, undamped live outcome (round 90)


def test_offer_side_erosion_does_not_fire_against_a_genuinely_conceding_opponent():
    # CONFIRMED live (2026-08-24): an earlier version of the erosion fix
    # gated purely on round count -- by round 40 our own offer had already
    # eroded from the true 69.29% target down to 61.12%, and by round 88
    # down to 54.13%, REGARDLESS of whether the opponent had shown any sign
    # of being unmovable. Replays an opponent who genuinely concedes a
    # little every time they're seen (60 -> 55 -> 50 -> 46 -> 42 across
    # rounds 2-80) -- erosion must NOT fire here; our own offer should stay
    # at the exact, un-eroded raw target the whole time.
    game_id = "barg-erosion-not-against-moving-opponent"
    money = 100.0

    def make_offer_game(round_no, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Split the money.",
            "game_state": {
                "current_player": "bob", "proposer": "bob", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False, "last_offer": None, "history": hist,
            },
            "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
        }

    hist = []
    action = None
    for rnd, opp_ask in [(2, 60.0), (20, 55.0), (40, 50.0), (60, 46.0), (80, 42.0)]:
        game = make_offer_game(rnd, list(hist))
        action = bargaining_strategy(game)
        prior_round = rnd - 1 if rnd > 2 else 1
        hist.append({
            "round": prior_round, "proposer": "alice",
            "offer": {"alice_gain": opp_ask, "bob_gain": 100.0 - opp_ask, "proposer": "alice", "round": prior_round},
            "decision": "reject",
        })

    assert action["bob_gain"] == pytest.approx(69.28571428571429, abs=0.01)  # completely un-eroded


def test_offer_side_erosion_fires_once_stationarity_is_strongly_confirmed():
    # Mirror case: once the opponent has proven stationary far beyond the
    # ordinary STATIONARY_OPPONENT_ROUNDS=3 bar (8 identical confirmations
    # here, matching EROSION_CONFIRMATION_ROUNDS), erosion must engage --
    # this is the mechanism that fixes the original 90-round freeze.
    game_id = "barg-erosion-fires-after-strong-confirmation"
    money = 100.0

    def make_offer_game(round_no, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Split the money.",
            "game_state": {
                "current_player": "bob", "proposer": "bob", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False, "last_offer": None, "history": hist,
            },
            "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
        }

    hist = [
        {"round": r, "proposer": "alice", "offer": {"alice_gain": 55.56, "bob_gain": 44.44, "proposer": "alice", "round": r}, "decision": "reject"}
        for r in [2, 4, 6, 8, 10, 12, 14, 16]
    ]
    action = bargaining_strategy(make_offer_game(17, hist))
    assert action["bob_gain"] < 69.28571428571429 - 1.0  # meaningfully eroded below the raw target


def test_decision_side_erosion_does_not_fire_against_a_genuinely_conceding_opponent():
    # Mirror of the offer-side test, for the accept-threshold side.
    game_id = "barg-decision-erosion-not-against-moving-opponent"
    money = 100.0

    def make_decision_game(round_no, offer_price, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Respond.",
            "game_state": {
                "current_player": "bob", "proposer": "alice", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False,
                "last_offer": {"alice_gain": 100.0 - offer_price, "bob_gain": offer_price, "proposer": "alice", "round": round_no},
                "history": hist,
            },
            "valid_actions": {"type": "decision", "fields": {"decision": "'accept' or 'reject'"}},
        }

    from agent.bargaining.strategy import _reject_value, _soften_threshold_ceiling
    hist = []
    action = None
    for rnd, my_gain_offered in [(2, 40.0), (20, 45.0), (40, 50.0), (60, 54.0), (80, 58.0)]:
        game = make_decision_game(rnd, my_gain_offered, hist)
        action = bargaining_strategy(game)
        prior_round = rnd - 1 if rnd > 2 else 1
        prior_gain = my_gain_offered - 1.0  # something clearly below this round's offer -- genuine movement
        hist.append({
            "round": prior_round, "proposer": "alice",
            "offer": {"alice_gain": 100.0 - prior_gain, "bob_gain": prior_gain, "proposer": "alice", "round": prior_round},
            "decision": "reject",
        })

    # The raw (un-eroded) threshold at round 80 for these deltas:
    raw_threshold = _reject_value(money, "bob", 0.9, 0.8, None)
    unfloored = _soften_threshold_ceiling(
        raw_threshold, 30.0, 100.0, money, 80, None, force_soften=False, erode_over_time=False,
    )
    assert action["decision"] == ("accept" if 58.0 >= unfloored else "reject")


def test_offer_side_erosion_uses_a_faster_tau_than_the_ordinary_concession_curve():
    # CONFIRMED live (2026-08-25): once EROSION_CONFIRMATION_ROUNDS worth of
    # proof has already been paid for, reusing CONCESSION_TAU=30 (the pace
    # for cautious, still-uncertain early-game softening) makes erosion
    # itself barely faster than doing nothing -- see EROSION_TAU's own
    # derivation. Directly checks the eroded offer at a fixed round is lower
    # (more converged) under EROSION_TAU than the same scenario would
    # produce under the old CONCESSION_TAU pace.
    game_id = "barg-erosion-tau-is-faster"
    money = 100.0

    def make_offer_game(round_no, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Split the money.",
            "game_state": {
                "current_player": "bob", "proposer": "bob", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False, "last_offer": None, "history": hist,
            },
            "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
        }

    hist = [
        {"round": r, "proposer": "alice", "offer": {"alice_gain": 55.56, "bob_gain": 44.44, "proposer": "alice", "round": r}, "decision": "reject"}
        for r in [2, 4, 6, 8, 10, 12, 14, 16]
    ]
    action = bargaining_strategy(make_offer_game(17, hist))

    from agent.bargaining.strategy import CONCESSION_BETA, CONCESSION_TAU, _soften_offer_floor
    from agent.negotiation.solver import time_fraction

    raw_take = 69.28571428571429 * (1 - 0.03)  # ROBUSTNESS_BUFFER_FRAC applied before softening
    old_pace = _soften_offer_floor(
        raw_take, 30.0, 70.0, 70.0, 17, None, money=money,
        force_soften=False, erode_over_time=True, erosion_tau=CONCESSION_TAU,
    )
    assert action["bob_gain"] < old_pace  # the fix converges further, faster, at the same round


def test_offer_side_damping_does_not_re_slow_a_confirmed_erosion_step():
    # CONFIRMED live (2026-08-25): OFFER_CONCESSION_DAMPING_FRAC used to
    # re-apply its anti-exploitation brake to erosion's own already-gated
    # decrease, halving-then-some a concession that had already survived a
    # much stronger confirmation bar than damping's own. Replays a fully
    # stationary opponent well past EROSION_CONFIRMATION_ROUNDS and checks
    # that a single round's concession step is NOT capped at
    # OFFER_CONCESSION_DAMPING_FRAC (0.3) of the raw erosion step -- i.e.
    # the fix actually reaches a real, undamped erosion target.
    game_id = "barg-damping-does-not-re-slow-erosion"
    money = 10000.0

    def make_offer_game(round_no, hist):
        return {
            "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
            "prompt": "Split the money.",
            "game_state": {
                "current_player": "bob", "proposer": "bob", "round": round_no,
                "max_rounds": None, "horizon_known": False, "money_to_divide": money,
                "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
                "messages_allowed": False, "last_offer": None, "history": hist,
            },
            "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
        }

    stationary_offer = {"alice_gain": 5555.56, "bob_gain": 4444.44}
    hist = []
    actions = []
    for r in range(1, 14, 2):
        hist.append({
            "round": r, "proposer": "alice",
            "offer": {**stationary_offer, "proposer": "alice", "round": r}, "decision": "reject",
        })
        action = bargaining_strategy(make_offer_game(r + 1, list(hist)))
        actions.append(action["bob_gain"])
        hist.append({
            "round": r + 1, "proposer": "bob",
            "offer": {"alice_gain": money - action["bob_gain"], "bob_gain": action["bob_gain"], "proposer": "bob", "round": r + 1},
            "decision": "reject",
        })

    # Once erosion is confirmed (by round 17, per EROSION_CONFIRMATION_ROUNDS=7
    # confirmations), the round-over-round drop must exceed what
    # OFFER_CONCESSION_DAMPING_FRAC=0.3 alone would ever allow through --
    # i.e. damping is genuinely bypassed, not just coincidentally cleared.
    biggest_single_round_drop = max(a - b for a, b in zip(actions, actions[1:]))
    assert biggest_single_round_drop > 200.0  # comfortably beyond a damped-to-30% step


def _hopeless_offer_game(game_id, round_no, hist, opponent_name=None, money=1_000_000.0):
    game = {
        "game_id": game_id, "game_family": "bargaining", "your_player": "bob",
        "prompt": "Split the money.",
        "game_state": {
            "current_player": "bob", "proposer": "bob", "round": round_no,
            "max_rounds": None, "horizon_known": False, "money_to_divide": money,
            "delta_1": 1.0, "delta_2": 0.8, "complete_information": True,
            "messages_allowed": False, "last_offer": None, "history": hist,
        },
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
    }
    if opponent_name:
        game["opponent"] = {"name": opponent_name}
    return game


def _stationary_hopeless_history(rounds):
    return [
        {"round": r, "proposer": "alice", "offer": {"alice_gain": 920_000.0, "bob_gain": 80_000.0, "proposer": "alice", "round": r}, "decision": "reject"}
        for r in rounds
    ]


def test_trusted_hardliner_profile_halves_the_confirmation_requirement(tmp_path, monkeypatch):
    _isolate_concession_profile_store(tmp_path, monkeypatch)
    from agent.rhetoric.opponent_profile import get_concession_profile, record_concession_evidence

    name = "Known Stonewaller"
    # Seed TRUST_THRESHOLD_GAMES (2) prior games that both showed the
    # confirmed-hopeless pattern against this exact name.
    record_concession_evidence(name, "bargaining", True)
    record_concession_evidence(name, "bargaining", True)
    assert get_concession_profile(name, "bargaining").is_trusted_hardliner()

    # Enough opponent confirmations to clear the HALVED bar (trusted) but
    # comfortably short of the FULL HOPELESS_CONFIRMATION_ROUNDS bar
    # (untrusted) -- HOPELESS_CONFIRMATION_ROUNDS=14 halves to 7; 9 rounds
    # of stationary opponent proposals clears 7 but not 14.
    assert HOPELESS_CONFIRMATION_ROUNDS == 14  # pins the assumption this test's round counts rely on
    mid_rounds = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    hist_mid = _stationary_hopeless_history(mid_rounds)
    action_untrusted = bargaining_strategy(
        _hopeless_offer_game("barg-untrusted-mid", mid_rounds[-1] + 2, hist_mid, opponent_name=None)
    )
    # But IS enough once the profile is trusted for this exact name --
    action_trusted = bargaining_strategy(
        _hopeless_offer_game("barg-trusted-mid", mid_rounds[-1] + 2, hist_mid, opponent_name=name)
    )
    # A confirmed_hopeless-triggered offer mirrors the opponent's own ask
    # (80,000 for bob) rather than the ordinary un-eroded/slowly-eroding
    # target -- so the trusted case should already be at (or very near)
    # that mirrored value while the untrusted case is not yet.
    assert action_trusted["bob_gain"] < action_untrusted["bob_gain"]
    assert action_trusted["bob_gain"] == pytest.approx(80_000.0, abs=1.0)


def test_hardliner_confirmation_still_required_even_when_trusted(tmp_path, monkeypatch):
    # The trust bar only ever HALVES the requirement, never skips it --
    # confirm a trusted profile still doesn't fire on essentially zero
    # in-game evidence.
    _isolate_concession_profile_store(tmp_path, monkeypatch)
    from agent.rhetoric.opponent_profile import record_concession_evidence

    name = "Known Stonewaller"
    record_concession_evidence(name, "bargaining", True)
    record_concession_evidence(name, "bargaining", True)

    hist = _stationary_hopeless_history([1])
    action = bargaining_strategy(_hopeless_offer_game("barg-trusted-but-no-evidence", 3, hist, opponent_name=name))
    assert action["bob_gain"] != pytest.approx(80_000.0, abs=1.0)


def test_accepting_records_concession_evidence_for_the_named_opponent(tmp_path, monkeypatch):
    _isolate_concession_profile_store(tmp_path, monkeypatch)
    from agent.rhetoric.opponent_profile import get_concession_profile

    name = "Fresh Opponent"
    money = 100.0
    game = {
        "game_id": "barg-records-evidence-on-accept", "game_family": "bargaining", "your_player": "bob",
        "prompt": "Respond.", "opponent": {"name": name},
        "game_state": {
            "current_player": "bob", "proposer": "alice", "round": 1,
            "max_rounds": None, "horizon_known": False, "money_to_divide": money,
            "delta_1": 0.8, "delta_2": 0.9, "complete_information": True,
            "messages_allowed": False,
            "last_offer": {"alice_gain": 25.0, "bob_gain": 75.0, "proposer": "alice", "round": 1},
            "history": [],
        },
        "valid_actions": {"type": "decision", "fields": {"decision": "'accept' or 'reject'"}},
    }
    action = bargaining_strategy(game)
    assert action["decision"] == "accept"  # a generous 75% offer at round 1 (raw target ~71.4%), clears easily
    profile = get_concession_profile(name, "bargaining")
    assert profile.games_observed == 1
    assert profile.games_confirmed_hopeless == 0  # not a confirmed_hopeless resolution
