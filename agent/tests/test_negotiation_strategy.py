import pytest

from agent.negotiation.strategy import negotiation_strategy


def _buyer_game(game_id, my_value, seller_true_value, round_no=1, max_rounds=10):
    state = {
        "current_player": "player_2",
        "player_1_role": "seller",
        "player_2_role": "buyer",
        "player_1_value": seller_true_value,
        "player_2_value": my_value,
        "round": round_no,
        "max_rounds": max_rounds,
        "horizon_known": True,
        "messages_allowed": False,
        "complete_information": True,
    }
    return {
        "game_id": game_id,
        "game_family": "negotiation",
        "your_player": "player_2",
        "prompt": "Negotiate.",
        "game_state": state,
        "valid_actions": {"type": "offer", "fields": {"product_price": "number"}},
    }


def test_complete_information_opening_offer_anchors_near_opponents_true_value():
    # Buyer (player_2), true reservation 1000; seller's true minimum is 100,
    # known exactly under complete information. An early opening offer
    # should anchor near the seller's true floor (100), not a generic
    # my_value/2-based guess.
    game = _buyer_game("neg-ci-1", my_value=1000.0, seller_true_value=100.0, round_no=1, max_rounds=10)
    action = negotiation_strategy(game)
    assert action["product_price"] < 300.0  # anchored near the true 100 floor, not near 500 (1000/2 guess)


def test_complete_information_never_offers_above_own_reservation():
    game = _buyer_game("neg-ci-2", my_value=1000.0, seller_true_value=100.0, round_no=9, max_rounds=10)
    action = negotiation_strategy(game)
    assert action["product_price"] <= 1000.0


def _seller_game(game_id, my_value, round_no=1, max_rounds=None):
    state = {
        "current_player": "player_1",
        "player_1_role": "seller",
        "player_2_role": "buyer",
        "player_1_value": my_value,
        "round": round_no,
        "max_rounds": max_rounds,
        "horizon_known": max_rounds is not None,
        "messages_allowed": False,
        "complete_information": False,
    }
    return {
        "game_id": game_id,
        "game_family": "negotiation",
        "your_player": "player_1",
        "prompt": "Negotiate.",
        "game_state": state,
        "valid_actions": {"type": "offer", "fields": {"product_price": "number"}},
    }


def test_one_shot_ultimatum_offer_does_not_anchor_at_the_full_speculative_ceiling():
    # Regression: max_rounds=1 as proposer used t=0 (full aggression), pricing
    # at the full 2x speculative ceiling with no information about the
    # counterpart's true value -- a live $80-value seller asked $160 and was
    # rejected outright. A true one-shot has no future round to lean on, so
    # the price should split the difference, not jump straight to 2x.
    game = _seller_game("neg-one-shot", my_value=80.0, round_no=1, max_rounds=1)
    action = negotiation_strategy(game)
    assert 80.0 < action["product_price"] < 160.0


def test_one_shot_ultimatum_offer_lands_near_the_new_lower_target_ratio():
    # CONFIRMED live (2026-08-25): the old T=0.5 (ratio 1.2788x) was accepted
    # only 6/22 times (27%) in a real 150-game sample -- 16 guaranteed-$0
    # losses with no recovery, since a one-shot rejection has no next round.
    # ONE_SHOT_ULTIMATUM_T=0.75 should land the blind-condition ask near
    # 1.1435x (close to the 1.14x ratio that cleared 100% in this same
    # sample's complete-information games) -- not the old 1.2788x.
    game = _seller_game("neg-one-shot-tighter", my_value=100.0, round_no=1, max_rounds=1)
    action = negotiation_strategy(game)
    ratio = action["product_price"] / 100.0
    assert ratio == pytest.approx(1.1435, abs=0.001)
    assert ratio < 1.2  # the old 1.2788x ratio must not still be reachable here


def test_unlimited_horizon_offer_has_not_fully_conceded_by_round_thirty():
    # Regression: the default unlimited_tau=6.0 made an unlimited-horizon
    # offer fully collapse toward the reservation value well before round
    # 30, giving away all bargaining leverage -- a live 32-round buyer
    # negotiation closed at EXACTLY the buyer's own ceiling, zero surplus
    # captured. With the deliberately slower tau (30.0, matching bargaining),
    # real room should still be left at round 30.
    game = _buyer_game("neg-unlimited-tau", my_value=12000.0, seller_true_value=8000.0, round_no=30, max_rounds=None)
    game["game_state"]["horizon_known"] = False
    game["game_state"]["complete_information"] = False
    del game["game_state"]["player_1_value"]
    action = negotiation_strategy(game)
    assert action["product_price"] < 12000.0 * 0.95  # not yet within 5% of full concession


def _buyer_decision_game(game_id, my_value, offer_price, round_no, history=None):
    state = {
        "current_player": "player_2",
        "player_1_role": "seller",
        "player_2_role": "buyer",
        "player_2_value": my_value,
        "round": round_no,
        "max_rounds": None,
        "horizon_known": False,
        "messages_allowed": False,
        "complete_information": False,
        "last_offer": {"price": offer_price, "from_player": "player_1", "round": round_no},
    }
    if history is not None:
        state["history"] = history
    return {
        "game_id": game_id,
        "game_family": "negotiation",
        "your_player": "player_2",
        "prompt": "Respond to the offer.",
        "game_state": state,
        "valid_actions": {"type": "decision", "fields": {"decision": "'AcceptOffer', 'RejectOffer', or 'WalkAway'"}},
    }


def _stationary_seller_history(price: float, rounds: list[int]) -> list[dict]:
    # Real server history shape (see history.py's module docstring): no
    # top-level "proposer" -- the proposer is nested under offer.from_player.
    return [
        {
            "round": r,
            "offer": {"price": price, "from_player": "player_1", "message": "", "round": r},
            "decision": "RejectOffer",
            "decided_by": "player_2",
        }
        for r in rounds
    ]


def test_accepts_a_profitable_offer_once_the_opponent_has_proven_stationary():
    # Regression: a real game had the seller offer an identical $118.80 for
    # three straight rounds (7, 9, 11) while we kept slowly climbing our own
    # counter -- $118.80 was already profitable (my_value=$120) but we kept
    # pushing instead of taking it, and at round 12 the seller simply walked
    # away, turning a locked-in +$1.20 into a guaranteed $0 for both sides.
    # `history` holds only PRIOR, already-decided rounds (7, 9) -- the
    # round-11 offer under decision right now is `last_offer`/offer_price,
    # not yet part of history, matching the real API's actual timing.
    game = _buyer_decision_game(
        "neg-stationary-accept", my_value=120.0, offer_price=118.8, round_no=11,
        history=_stationary_seller_history(118.8, rounds=[7, 9]),
    )
    action = negotiation_strategy(game)
    assert action["decision"] == "AcceptOffer"


def test_stationary_detection_does_not_override_unprofitable_offers():
    # A stationary but genuinely bad offer (above our own reservation, as
    # buyer) must not be accepted just because it hasn't moved.
    game = _buyer_decision_game(
        "neg-stationary-unprofitable", my_value=120.0, offer_price=150.0, round_no=11,
        history=_stationary_seller_history(150.0, rounds=[7, 9]),
    )
    action = negotiation_strategy(game)
    assert action["decision"] != "AcceptOffer"


def test_stationary_detection_does_not_trigger_when_the_opponent_is_actually_moving():
    # Test the detector directly rather than through the full strategy --
    # the ordinary (non-stationary) Boulware logic might independently
    # accept or reject depending on round/curve, which would make an
    # assertion on the full `negotiation_strategy` output ambiguous about
    # what's actually being verified here.
    from agent.negotiation.strategy import _opponent_offer_is_stationary

    game = _buyer_decision_game(
        "neg-not-stationary", my_value=120.0, offer_price=118.8, round_no=11,
        history=[
            {"round": 7, "offer": {"price": 140.0, "from_player": "player_1", "message": "", "round": 7}, "decision": "RejectOffer", "decided_by": "player_2"},
            {"round": 9, "offer": {"price": 130.0, "from_player": "player_1", "message": "", "round": 9}, "decision": "RejectOffer", "decided_by": "player_2"},
        ],
    )
    assert not _opponent_offer_is_stationary(game, "player_1", 120.0, current_price=118.8)


def test_stationary_detection_includes_the_current_offer_not_just_prior_history():
    # The fix's actual point: history alone (rounds 7, 9) is only 2 prior
    # entries -- one short of the threshold. It must be combined with the
    # CURRENT offer (round 11) to reach 3 and fire at the right round,
    # rather than needing a 4th confirming repeat one round too late.
    from agent.negotiation.strategy import _opponent_offer_is_stationary

    game = _buyer_decision_game(
        "neg-stationary-timing", my_value=120.0, offer_price=118.8, round_no=11,
        history=_stationary_seller_history(118.8, rounds=[7, 9]),
    )
    assert _opponent_offer_is_stationary(game, "player_1", 120.0, current_price=118.8)
    # A current offer that breaks the pattern must NOT be treated as stationary.
    assert not _opponent_offer_is_stationary(game, "player_1", 120.0, current_price=95.0)


def test_stationary_detection_requires_the_full_threshold_of_consecutive_offers():
    from agent.negotiation.strategy import _opponent_offer_is_stationary

    game = _buyer_decision_game(
        "neg-not-enough-history", my_value=120.0, offer_price=118.8, round_no=9,
        history=_stationary_seller_history(118.8, rounds=[7]),  # only 1 prior + current = 2, threshold is 3
    )
    assert not _opponent_offer_is_stationary(game, "player_1", 120.0, current_price=118.8)


def _seller_decision_game(game_id, my_value, offer_price, round_no, max_rounds=10):
    state = {
        "current_player": "player_1",
        "player_1_role": "seller",
        "player_2_role": "buyer",
        "player_1_value": my_value,
        "round": round_no,
        "max_rounds": max_rounds,
        "horizon_known": max_rounds is not None,
        "messages_allowed": False,
        "complete_information": False,
        "last_offer": {"price": offer_price, "from_player": "player_2", "round": round_no},
    }
    return {
        "game_id": game_id,
        "game_family": "negotiation",
        "your_player": "player_1",
        "prompt": "Respond to the offer.",
        "game_state": state,
        "valid_actions": {"type": "decision", "fields": {"decision": "'AcceptOffer', 'RejectOffer', or 'WalkAway'"}},
    }


def test_offer_clears_safety_margin_seller_and_buyer_directly():
    from agent.negotiation.strategy import _offer_clears_safety_margin

    # Seller: cost $80. $116.10 is a 45.1% margin -- clears the 30% bar.
    assert _offer_clears_safety_margin("seller", 80.0, 116.10)
    # A thinner, 25% margin ($100 vs $80 cost) must not clear it.
    assert not _offer_clears_safety_margin("seller", 80.0, 100.0)
    # Buyer: reservation $150. Paying $100 is a 33.3% margin below reservation.
    assert _offer_clears_safety_margin("buyer", 150.0, 100.0)
    assert not _offer_clears_safety_margin("buyer", 150.0, 115.0)  # 23.3%, below the bar


def test_early_accepts_a_large_margin_offer_instead_of_holding_out_and_risking_walkaway():
    # CONFIRMED live: seller, my_value=$80 (cost), 10-round game. The buyer's
    # very first counter ($116.10) was already a 45% margin -- genuinely
    # profitable -- but the Boulware curve at round 3 (t=0.2) only wants
    # ~$152-157, far above what was on the table, so the old logic would
    # reject and counter yet again. The buyer walked away the very next
    # round in the real game. This must now take the offer immediately.
    game = _seller_decision_game("neg-early-margin-accept", my_value=80.0, offer_price=116.10, round_no=3, max_rounds=10)
    action = negotiation_strategy(game)
    assert action["decision"] == "AcceptOffer"


def test_does_not_early_accept_a_thin_margin_offer_early_in_the_game():
    # A 25% margin -- real profit, but below the 30% bar -- must NOT trigger
    # the new early-accept path this early (round 3 of 10, Boulware target
    # still far above $100), so it falls through to the ordinary counter
    # logic instead of accepting outright.
    game = _seller_decision_game("neg-thin-margin-no-early-accept", my_value=80.0, offer_price=100.0, round_no=3, max_rounds=10)
    action = negotiation_strategy(game)
    assert action["decision"] == "RejectOffer"


def test_early_accept_margin_does_not_override_unprofitable_offers():
    # A seller offer below cost is never profitable regardless of "margin"
    # sign -- the profitable gate must still block this.
    game = _seller_decision_game("neg-margin-unprofitable", my_value=80.0, offer_price=50.0, round_no=3, max_rounds=10)
    action = negotiation_strategy(game)
    assert action["decision"] != "AcceptOffer"


def test_early_accept_margin_works_symmetrically_for_buyers():
    # Buyer, reservation $150. Paying $95 is a 36.7% margin -- clears the
    # bar -- and should be taken immediately rather than held out on.
    game = _buyer_decision_game("neg-early-margin-accept-buyer", my_value=150.0, offer_price=95.0, round_no=3)
    action = negotiation_strategy(game)
    assert action["decision"] == "AcceptOffer"


def test_recalibrated_concession_curve_accepts_a_thin_margin_offer_sooner():
    # DERIVED (2026-08-23): checks the CONCESSION_BETA/DEFAULT_PRIOR_SPREAD
    # recalibration itself, independent of EARLY_ACCEPT_MARGIN_FRAC (this
    # offer is a 25% margin, below that 30% bar, so the margin path can't be
    # what's firing here). Seller, my_value=$80, round 7 of 10, offer $100:
    # with the OLD curve (anchor=160, beta=0.7) the round-7 target is
    # $121.44 -- still a reject. With the recalibrated curve (anchor=120,
    # beta=0.85) it's $98.07 -- clears $100, so this now accepts several
    # rounds earlier than before via the ordinary target-met path.
    game = _seller_decision_game("neg-recalibrated-curve-accepts-sooner", my_value=80.0, offer_price=100.0, round_no=7, max_rounds=10)
    action = negotiation_strategy(game)
    assert action["decision"] == "AcceptOffer"


def test_style_bandit_arm_is_passed_to_compose_message_on_a_counter_offer(monkeypatch):
    from agent.rhetoric.bandit import StyleBandit

    captured = {}

    def fake_compose_message(**kwargs):
        captured.update(kwargs)
        return "a message"

    monkeypatch.setattr("agent.negotiation.strategy.compose_message", fake_compose_message)
    # Thin margin (25%), early round -- falls through to the ordinary
    # reject+counter path rather than any of the accept shortcuts.
    game = _seller_decision_game("neg-style-counter", my_value=80.0, offer_price=100.0, round_no=3, max_rounds=10)
    game["game_state"]["messages_allowed"] = True
    action = negotiation_strategy(game)
    assert action["decision"] == "RejectOffer"
    assert captured["style"] in StyleBandit.ARMS
    assert action["message"] == "a message"

    from agent.negotiation import strategy as strategy_mod
    assert strategy_mod._LAST_STYLE_ARM["neg-style-counter"] == captured["style"]


def test_style_bandit_arm_is_passed_to_compose_message_on_an_opening_offer(monkeypatch):
    from agent.rhetoric.bandit import StyleBandit

    captured = {}

    def fake_compose_message(**kwargs):
        captured.update(kwargs)
        return "a message"

    monkeypatch.setattr("agent.negotiation.strategy.compose_message", fake_compose_message)
    game = _seller_game("neg-style-opening", my_value=80.0, round_no=1, max_rounds=10)
    game["game_state"]["messages_allowed"] = True
    action = negotiation_strategy(game)
    assert captured["style"] in StyleBandit.ARMS
    assert action["message"] == "a message"


def test_no_style_selected_when_messages_not_allowed():
    game = _seller_decision_game("neg-no-style-counter", my_value=80.0, offer_price=100.0, round_no=3, max_rounds=10)
    game["game_state"]["messages_allowed"] = False
    action = negotiation_strategy(game)
    assert "message" not in action

    from agent.negotiation import strategy as strategy_mod
    assert "neg-no-style-counter" not in strategy_mod._LAST_STYLE_ARM


def _opponent_price_history(price: float, rounds: list[int], from_player: str = "player_2") -> list[dict]:
    decided_by = "player_1" if from_player == "player_2" else "player_2"
    return [
        {
            "round": r,
            "offer": {"price": price, "from_player": from_player, "message": "", "round": r},
            "decision": "RejectOffer",
            "decided_by": decided_by,
        }
        for r in rounds
    ]


def test_style_bandit_credits_success_for_seller_when_opponent_price_rises():
    from agent.negotiation import strategy as strategy_mod

    game_id = "neg-style-credit-seller-success"
    strategy_mod._LAST_STYLE_ARM[game_id] = "care"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    alpha_before = bandit.alpha["care"]

    game = _seller_decision_game(game_id, my_value=80.0, offer_price=110.0, round_no=5, max_rounds=10)
    # Buyer's price rose 100 -> 110 -- moved toward the seller.
    game["game_state"]["history"] = _opponent_price_history(100.0, rounds=[3])
    negotiation_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).alpha["care"] == alpha_before + 1.0
    assert game_id not in strategy_mod._LAST_STYLE_ARM


def test_style_bandit_credits_failure_for_seller_when_opponent_price_does_not_rise():
    from agent.negotiation import strategy as strategy_mod

    game_id = "neg-style-credit-seller-failure"
    strategy_mod._LAST_STYLE_ARM[game_id] = "loyalty"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    beta_before = bandit.beta["loyalty"]

    game = _seller_decision_game(game_id, my_value=80.0, offer_price=100.0, round_no=5, max_rounds=10)
    # Buyer's price held flat 100 -> 100 -- did not move toward the seller.
    game["game_state"]["history"] = _opponent_price_history(100.0, rounds=[3])
    negotiation_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).beta["loyalty"] == beta_before + 1.0
    assert game_id not in strategy_mod._LAST_STYLE_ARM


def test_style_bandit_does_not_credit_success_for_noise_level_movement():
    # DERIVED (2026-08-24): a $0.10 rise on my_value=$80 is well below
    # STATIONARY_OPPONENT_TOLERANCE_FRAC's noise band (0.5% = $0.40) --
    # must not count as clean success, but must still count as a failure.
    from agent.negotiation import strategy as strategy_mod

    game_id = "neg-style-credit-noise-movement"
    strategy_mod._LAST_STYLE_ARM[game_id] = "care"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    alpha_before, beta_before = bandit.alpha["care"], bandit.beta["care"]

    game = _seller_decision_game(game_id, my_value=80.0, offer_price=100.10, round_no=5, max_rounds=10)
    game["game_state"]["history"] = _opponent_price_history(100.0, rounds=[3])
    negotiation_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).alpha["care"] == alpha_before  # not credited as success
    assert strategy_mod._style_bandit({"game_id": game_id}).beta["care"] == beta_before + 1.0  # still counted as failure


def test_style_bandit_credits_success_for_buyer_when_opponent_price_falls():
    from agent.negotiation import strategy as strategy_mod

    game_id = "neg-style-credit-buyer-success"
    strategy_mod._LAST_STYLE_ARM[game_id] = "purity"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    alpha_before = bandit.alpha["purity"]

    game = _buyer_decision_game(game_id, my_value=150.0, offer_price=130.0, round_no=5)
    # Seller's price fell 140 -> 130 -- moved toward the buyer.
    game["game_state"]["history"] = _opponent_price_history(140.0, rounds=[3], from_player="player_1")
    negotiation_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).alpha["purity"] == alpha_before + 1.0
    assert game_id not in strategy_mod._LAST_STYLE_ARM


def test_style_bandit_does_not_credit_with_no_prior_opponent_offer():
    from agent.negotiation import strategy as strategy_mod

    game_id = "neg-style-credit-no-history"
    strategy_mod._LAST_STYLE_ARM[game_id] = "authority"
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    alpha_before, beta_before = bandit.alpha["authority"], bandit.beta["authority"]

    game = _seller_decision_game(game_id, my_value=80.0, offer_price=100.0, round_no=1, max_rounds=10)
    game["game_state"]["history"] = []  # opponent's first-ever offer -- nothing to compare
    negotiation_strategy(game)

    assert strategy_mod._style_bandit({"game_id": game_id}).alpha["authority"] == alpha_before
    assert strategy_mod._style_bandit({"game_id": game_id}).beta["authority"] == beta_before
    assert game_id not in strategy_mod._LAST_STYLE_ARM


def test_stationary_offer_clears_min_margin_directly():
    from agent.negotiation.solver import ValueBounds
    from agent.negotiation.strategy import _stationary_offer_clears_min_margin

    bounds = ValueBounds(lo=12000.0, hi=15000.0)  # $3,000 ZOPA
    # Exact tie -- 0% of the ZOPA -- must not clear under complete information.
    assert not _stationary_offer_clears_min_margin("seller", 12000.0, 12000.0, bounds, complete_information=True)
    # A comfortable margin (12.2%, matching a real live case) clears it.
    assert _stationary_offer_clears_min_margin("seller", 12000.0, 12366.67, bounds, complete_information=True)
    # Under incomplete information the bar is bypassed entirely (bounds is
    # only a guessed range there, not a real ZOPA measurement) -- even a
    # zero margin clears it, preserving the original stationary-accept
    # behavior for the incomplete-information case it was built for.
    assert _stationary_offer_clears_min_margin("seller", 12000.0, 12000.0, bounds, complete_information=False)


def test_stationary_accept_requires_a_real_margin_under_complete_information():
    # CONFIRMED live: seller, my_value=$8,000, opponent (buyer) true value
    # $10,000 (complete information) -- the buyer planted an identical
    # $8,000 offer for 3 straight rounds, an EXACT tie with our reservation
    # value (0% of the known $2,000 ZOPA). The old bare `profitable` check
    # (`>=`) treated this as "profitable" and accepted immediately, handing
    # over 100% of the surplus for nothing -- holding out here has zero
    # downside, since even a walkaway is the same $0 already on the table.
    game = _seller_decision_game("neg-zero-margin-tie", my_value=8000.0, offer_price=8000.0, round_no=6, max_rounds=10)
    game["game_state"]["complete_information"] = True
    game["game_state"]["player_2_value"] = 10000.0
    game["game_state"]["history"] = _opponent_price_history(8000.0, rounds=[2, 4], from_player="player_2")
    action = negotiation_strategy(game)
    assert action["decision"] != "AcceptOffer"


def test_stationary_accept_still_fires_for_a_meaningful_margin_under_complete_information():
    # A 12.2%-of-ZOPA margin (matching a different real live case) is
    # comfortably above STATIONARY_ACCEPT_MIN_MARGIN_FRAC (0.10) and should
    # still trigger the immediate accept -- the new bar only blocks the
    # near-zero-margin cases, not every stationary accept.
    game = _buyer_decision_game("neg-meaningful-margin", my_value=15000.0, offer_price=14633.33, round_no=5)
    game["game_state"]["complete_information"] = True
    game["game_state"]["player_1_value"] = 12000.0
    game["game_state"]["history"] = _opponent_price_history(14700.0, rounds=[1], from_player="player_1") + \
        _opponent_price_history(14692.59, rounds=[3], from_player="player_1")
    action = negotiation_strategy(game)
    assert action["decision"] == "AcceptOffer"


def test_offer_side_concession_is_dampened_against_a_stationary_opponent():
    # CONFIRMED live: same real game as the zero-margin-tie test above --
    # our own counter kept sliding toward the buyer's proven-fixed $8,000
    # every round (10000 -> 9866.78 -> 9514.85 -> ...) purely on the
    # time-based curve, with no regard for the buyer's total lack of
    # reciprocity. Once the buyer is proven stationary, our next counter
    # should only take OFFER_CONCESSION_DAMPING_FRAC of the raw formula's
    # further concession, not the full scheduled amount.
    from agent.negotiation import strategy as strategy_mod
    from agent.negotiation.solver import concession_target, time_fraction

    game_id = "neg-damping-test"
    strategy_mod._LAST_SENT_PRICE[game_id] = 9514.847  # our own round-5 counter, matching the real game

    game = _seller_decision_game(game_id, my_value=8000.0, offer_price=8000.0, round_no=6, max_rounds=10)
    game["game_state"]["complete_information"] = True
    game["game_state"]["player_2_value"] = 10000.0
    game["game_state"]["history"] = _opponent_price_history(8000.0, rounds=[2, 4], from_player="player_2")

    action = negotiation_strategy(game)
    assert action["decision"] == "RejectOffer"

    undamped_t = time_fraction(6, 10, unlimited_tau=strategy_mod.UNLIMITED_TAU)
    undamped_target = concession_target(10000.0, 8000.0, undamped_t, strategy_mod.CONCESSION_BETA)
    # The raw curve wants to keep sliding toward the buyer's proven-fixed
    # $8,000 -- damping should leave the counter well above that undamped
    # target, while still conceding somewhat (less than the prior offer).
    assert action["product_price"] > undamped_target
    assert action["product_price"] < 9514.847
