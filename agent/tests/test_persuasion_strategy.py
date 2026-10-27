"""Tests for the persuasion strategy on both sides: the seller's opening-round
bluff cap, the buyer's reading of signals and exploratory buys, and credit for
the framing bandit.
"""

from agent.persuasion.strategy import SELLER_TRUST_BUILDING_ROUNDS, persuasion_strategy


def _seller_game(game_id, round_no, quality="low", p=0.5, v=100.0, u=0.0, price=50.0):
    return {
        "game_id": game_id,
        "game_family": "persuasion",
        "your_player": "seller",
        "prompt": "Decide whether to recommend.",
        "game_state": {
            "product_price": price,
            "p": p,
            "v": v,
            "u": u,
            "current_quality": quality,
            "round": round_no,
            "total_rounds": 20,
            "seller_total_payoff": 0.0,
            "buyer_total_payoff": 0.0,
            "is_seller_know_cv": True,
        },
        "valid_actions": {"type": "seller_recommendation", "fields": {"decision": "'yes' or 'no'"}},
    }


def test_seller_caps_bluffing_during_the_trust_building_window():
    assert SELLER_TRUST_BUILDING_ROUNDS >= 1
    signaled_buy_count = 0
    trials = 400
    for i in range(trials):
        game = _seller_game(f"pers-trust-build-{i}", round_no=1, quality="low", p=0.5, v=100.0, u=0.0, price=50.0)
        action = persuasion_strategy(game)
        if action["decision"] == "yes":
            signaled_buy_count += 1
    bluff_rate = signaled_buy_count / trials
    assert bluff_rate == 0.0


def test_seller_does_not_bluff_on_round_one_of_the_live_regression_game():
    for i in range(200):
        game = _seller_game(f"pers-live-regression-{i}", round_no=1, quality="low", p=0.5, v=3_000_000.0, u=0.0, price=1_000_000.0)
        action = persuasion_strategy(game)
        assert action["decision"] == "no"


def test_seller_bluffing_returns_to_normal_after_the_trust_building_window():
    game = _seller_game(
        "pers-trust-build-later", round_no=SELLER_TRUST_BUILDING_ROUNDS + 5,
        quality="low", p=0.5, v=100.0, u=0.0, price=50.0,
    )
    signaled_buy_count = 0
    trials = 400
    for i in range(trials):
        game = _seller_game(
            f"pers-trust-build-later-{i}", round_no=SELLER_TRUST_BUILDING_ROUNDS + 5,
            quality="low", p=0.5, v=100.0, u=0.0, price=50.0,
        )
        action = persuasion_strategy(game)
        if action["decision"] == "yes":
            signaled_buy_count += 1
    bluff_rate = signaled_buy_count / trials
    assert 0.6 < bluff_rate < 0.8


def _buyer_game(game_id, seller_message, favorable_prior=True):
    p = 0.9 if favorable_prior else 0.1
    return {
        "game_id": game_id,
        "game_family": "persuasion",
        "your_player": "buyer",
        "prompt": "Decide whether to buy.",
        "game_state": {
            "product_price": 40.0,
            "p": p,
            "v": 100.0,
            "u": 0.0,
            "seller_message": seller_message,
            "seller_message_type": "binary",
            "round": 1,
            "total_rounds": 10,
            "seller_total_payoff": 0.0,
            "buyer_total_payoff": 0.0,
            "is_seller_know_cv": True,
        },
        "valid_actions": {"type": "buyer_decision", "fields": {"decision": "'yes' or 'no'"}},
    }


def test_buyer_declines_when_seller_signals_no_even_with_favorable_prior(monkeypatch):
    monkeypatch.setattr("agent.persuasion.strategy.DECLINE_TEST_UPSIDE_RATIO_THRESHOLD", float("inf"))
    game = _buyer_game("pers-no-signal", seller_message="no", favorable_prior=True)
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_buyer_can_accept_when_seller_signals_yes_with_favorable_prior():
    game = _buyer_game("pers-yes-signal", seller_message="yes", favorable_prior=True)
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_buyer_does_not_lock_out_after_two_unlucky_bootstrap_buys_on_a_comfortable_prior():
    game_id = "pers-comfortable-prior-unlucky-bootstrap"
    price, v, u, p = 100.0, 200.0, 0.0, 0.8
    payoff = 0.0

    def make_game(round_no):
        return {
            "game_id": game_id, "game_family": "persuasion", "your_player": "player_2",
            "prompt": "Decide whether to buy.",
            "game_state": {
                "product_price": price, "p": p, "v": v, "u": u,
                "seller_message": "Good unit this round. Worth taking at the asking price.",
                "seller_message_type": "text", "round": round_no, "total_rounds": 20,
                "seller_total_payoff": 0.0, "buyer_total_payoff": payoff,
                "is_seller_know_cv": True,
            },
            "valid_actions": {"type": "buyer_decision", "fields": {"decision": "'yes' or 'no'"}},
        }

    for round_no in (1, 2):
        action = persuasion_strategy(make_game(round_no))
        assert action["decision"] == "yes"
        payoff += u - price

    action = persuasion_strategy(make_game(3))
    assert action["decision"] == "yes", (
        "buyer must keep buying after 2 unlucky draws on a comfortably profitable, "
        "exactly-known prior with an uninformative constant seller message"
    )


def test_buyer_declines_when_seller_signals_yes_but_prior_is_unfavorable():
    game = _buyer_game("pers-yes-unfavorable", seller_message="yes", favorable_prior=False)
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def _text_mode_buyer_game(game_id, seller_message, favorable_prior=True):
    game = _buyer_game(game_id, seller_message, favorable_prior)
    game["game_state"]["seller_message_type"] = "text"
    return game


def _patch_llm_unavailable(monkeypatch):
    monkeypatch.setattr("agent.persuasion.strategy.classify_discourage_signal", lambda message: None)
    monkeypatch.setattr("agent.persuasion.strategy.DECLINE_TEST_UPSIDE_RATIO_THRESHOLD", float("inf"))


def test_text_mode_buyer_declines_on_explicit_discouraging_message(monkeypatch):
    _patch_llm_unavailable(monkeypatch)
    game = _text_mode_buyer_game(
        "pers-text-discourage",
        seller_message="This product is low quality — you should pass on it.",
        favorable_prior=True,
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_text_mode_buyer_declines_on_comparative_discouraging_message(monkeypatch):
    _patch_llm_unavailable(monkeypatch)
    game = _text_mode_buyer_game(
        "pers-text-comparative-discourage",
        seller_message="I'll be straight with you: this one is ordinary. Wait for the next -- I'll tell you when it's worth it.",
        favorable_prior=True,
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_text_mode_buyer_declines_on_a_soft_hedged_pass_suggestion(monkeypatch):
    _patch_llm_unavailable(monkeypatch)
    game = _text_mode_buyer_game(
        "pers-text-soft-hedge-discourage",
        seller_message="I'll be straight with you -- this one isn't my best unit. You may want to pass today.",
        favorable_prior=True,
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_text_mode_buyer_declines_on_a_relative_quality_comparison_hedge(monkeypatch):
    _patch_llm_unavailable(monkeypatch)
    game = _text_mode_buyer_game(
        "pers-text-relative-quality-discourage",
        seller_message="This unit is below our usual quality this round.",
        favorable_prior=True,
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_text_mode_buyer_can_still_buy_on_generic_positive_pitch(monkeypatch):
    _patch_llm_unavailable(monkeypatch)
    game = _text_mode_buyer_game(
        "pers-text-positive",
        seller_message="This is an exceptional high-value item, strongly recommended.",
        favorable_prior=True,
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_text_mode_buyer_declines_on_confident_llm_verdict_even_with_no_phrase_match(monkeypatch):
    monkeypatch.setattr(
        "agent.persuasion.strategy.classify_discourage_signal",
        lambda message: (True, 0.95),
    )
    monkeypatch.setattr("agent.persuasion.strategy.DECLINE_TEST_UPSIDE_RATIO_THRESHOLD", float("inf"))
    game = _text_mode_buyer_game(
        "pers-text-llm-confident-discourage",
        seller_message="Honestly? I've seen better days from this shelf.",
        favorable_prior=True,
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_text_mode_buyer_buys_on_confident_llm_verdict_even_when_phrase_list_would_have_matched(monkeypatch):
    monkeypatch.setattr(
        "agent.persuasion.strategy.classify_discourage_signal",
        lambda message: (False, 0.9),
    )
    game = _text_mode_buyer_game(
        "pers-text-llm-confident-not-discourage",
        seller_message="Nothing special about the packaging, but the unit itself is great — I'd buy it.",
        favorable_prior=True,
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_text_mode_buyer_falls_back_to_phrases_when_llm_confidence_is_below_threshold(monkeypatch):
    from agent.persuasion.strategy import DISCOURAGE_CONFIDENCE_THRESHOLD

    monkeypatch.setattr(
        "agent.persuasion.strategy.classify_discourage_signal",
        lambda message: (False, DISCOURAGE_CONFIDENCE_THRESHOLD - 0.05),
    )
    monkeypatch.setattr("agent.persuasion.strategy.DECLINE_TEST_UPSIDE_RATIO_THRESHOLD", float("inf"))
    game = _text_mode_buyer_game(
        "pers-text-llm-uncertain",
        seller_message="This product is low quality — you should pass on it.",
        favorable_prior=True,
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_buyer_proceeds_to_explore_on_generic_neutral_filler_not_a_real_hedge(monkeypatch):
    monkeypatch.setattr(
        "agent.persuasion.strategy.classify_discourage_signal",
        lambda message: (False, 0.3),
    )
    game = _text_mode_buyer_game(
        "pers-neutral-filler-explores", seller_message="No strong recommendation this round.",
        favorable_prior=True,
    )
    game["game_state"]["product_price"] = 10000.0
    game["game_state"]["v"] = 30000.0
    game["game_state"]["u"] = 0.0
    game["game_state"]["p"] = 0.8
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def _yes_signal_game(game_id, price, v, u, p):
    return {
        "game_id": game_id,
        "game_family": "persuasion",
        "your_player": "buyer",
        "prompt": "Decide whether to buy.",
        "game_state": {
            "product_price": price,
            "p": p,
            "v": v,
            "u": u,
            "seller_message": "yes",
            "seller_message_type": "binary",
            "round": 1,
            "total_rounds": 20,
            "seller_total_payoff": 0.0,
            "buyer_total_payoff": 0.0,
            "is_seller_know_cv": True,
        },
        "valid_actions": {"type": "buyer_decision", "fields": {"decision": "'yes' or 'no'"}},
    }


def test_buyer_bootstrap_tests_a_buy_signal_at_bare_breakeven():
    game = _yes_signal_game("pers-exploration-tie", price=10000.0, v=30000.0, u=0.0, p=1 / 3)
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_buyer_bootstrap_now_fires_on_the_recurring_096_ratio_config():
    game = _yes_signal_game("pers-bootstrap-096-config", price=1_000_000.0, v=1_200_000.0, u=0.0, p=0.8)
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_buyer_still_declines_a_ratio_below_the_new_threshold():
    game = _yes_signal_game("pers-bootstrap-below-threshold", price=1_000_000.0, v=1_100_000.0, u=0.0, p=0.8)
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_buyer_still_explores_on_a_buy_signal_once_the_safety_margin_is_cleared():
    game = _yes_signal_game("pers-exploration-clears-margin", price=10000.0, v=30000.0, u=0.0, p=0.5)
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_buyer_bootstraps_instead_of_going_0_for_20_on_the_live_regression_game(monkeypatch):
    _patch_llm_unavailable(monkeypatch)
    game = _text_mode_buyer_game(
        "pers-live-regression-0-for-20",
        seller_message="I checked this unit myself and it's high quality — at this price I'd strongly recommend it.",
        favorable_prior=True,
    )
    game["game_state"]["product_price"] = 100.0
    game["game_state"]["v"] = 125.0
    game["game_state"]["u"] = 0.0
    game["game_state"]["p"] = 0.8
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_buyer_does_not_explore_when_prior_ev_is_far_below_price():
    game = _buyer_game("pers-no-explore-bad-prior", seller_message="yes", favorable_prior=False)
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def _no_signal_game(game_id, price, v, u, round_no=1, buyer_total_payoff=0.0, p=0.5):
    return {
        "game_id": game_id,
        "game_family": "persuasion",
        "your_player": "buyer",
        "prompt": "Decide whether to buy.",
        "game_state": {
            "product_price": price,
            "p": p,
            "v": v,
            "u": u,
            "seller_message": "no",
            "seller_message_type": "binary",
            "round": round_no,
            "total_rounds": 20,
            "seller_total_payoff": 0.0,
            "buyer_total_payoff": buyer_total_payoff,
            "is_seller_know_cv": True,
        },
        "valid_actions": {"type": "buyer_decision", "fields": {"decision": "'yes' or 'no'"}},
    }


def test_buyer_tests_a_discourage_signal_when_the_upside_ratio_clears_the_bar():
    game = _no_signal_game("pers-decline-test-fires", price=1_000_000.0, v=4_000_000.0, u=0.0)
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_buyer_does_not_test_a_discourage_signal_when_unconditional_prior_is_unfavorable():
    game = _no_signal_game("pers-decline-test-blocked-by-prior", price=1_000_000.0, v=2_000_000.0, u=0.0, p=0.33)
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_buyer_does_not_test_a_discourage_signal_when_the_upside_ratio_is_too_thin():
    game = _no_signal_game("pers-decline-test-skipped", price=1_000_000.0, v=1_100_000.0, u=0.0)
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_testing_a_discourage_signal_updates_the_decline_model_not_the_yes_model():
    from agent.persuasion import strategy as strategy_mod

    game_id = "pers-decline-test-routing"
    round1 = _no_signal_game(game_id, price=1_000_000.0, v=4_000_000.0, u=0.0, round_no=1)
    action1 = persuasion_strategy(round1)
    assert action1["decision"] == "yes"
    assert strategy_mod._LAST_BUYER_TESTED_NO[game_id] is True

    round2 = _no_signal_game(
        game_id, price=1_000_000.0, v=4_000_000.0, u=0.0, round_no=2,
        buyer_total_payoff=-1_000_000.0,
    )
    persuasion_strategy(round2)

    assert strategy_mod._BUYER_NO_TRUST[game_id].buys == 1
    assert strategy_mod._BUYER_NO_TRUST[game_id].highs == 0
    assert game_id not in strategy_mod._BUYER_TRUST


def test_discourage_signal_trusted_again_after_a_test_confirms_it_is_reliable():
    from agent.persuasion import strategy as strategy_mod

    game_id = "pers-decline-test-confirmed-reliable"
    round1 = _no_signal_game(game_id, price=1_000_000.0, v=4_000_000.0, u=0.0, round_no=1)
    persuasion_strategy(round1)

    round2 = _no_signal_game(
        game_id, price=1_000_000.0, v=4_000_000.0, u=0.0, round_no=2,
        buyer_total_payoff=-1_000_000.0,
    )
    action2 = persuasion_strategy(round2)
    assert action2["decision"] == "no"
    assert strategy_mod._LAST_BUYER_TESTED_NO[game_id] is False


def _seller_text_game(game_id, round_no, quality="low", p=0.5, v=100.0, u=0.0, price=50.0, seller_total_payoff=0.0):
    return {
        "game_id": game_id,
        "game_family": "persuasion",
        "your_player": "seller",
        "prompt": "Decide what to write.",
        "game_state": {
            "product_price": price,
            "p": p,
            "v": v,
            "u": u,
            "current_quality": quality,
            "round": round_no,
            "total_rounds": 20,
            "seller_total_payoff": seller_total_payoff,
            "buyer_total_payoff": 0.0,
            "is_seller_know_cv": True,
        },
        "valid_actions": {"type": "seller_message", "fields": {"message": "string"}},
    }


def test_style_bandit_arm_is_passed_to_compose_message_on_a_buy_pitch(monkeypatch):
    from agent.rhetoric.bandit import StyleBandit

    captured = {}
    monkeypatch.setattr(
        "agent.persuasion.strategy.compose_message",
        lambda **kwargs: captured.update(kwargs) or "a pitch",
    )
    game = _seller_text_game("pers-style-buy", round_no=10, quality="high", price=50.0)
    persuasion_strategy(game)
    assert captured["style"] in StyleBandit.ARMS


def test_no_style_is_passed_on_a_discourage_pitch(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "agent.persuasion.strategy.compose_message",
        lambda **kwargs: captured.update(kwargs) or "a pitch",
    )
    game = _seller_text_game("pers-style-no-buy", round_no=10, quality="low", v=100.0, price=150.0)
    persuasion_strategy(game)
    assert captured["style"] is None


def test_style_bandit_credits_the_arm_used_last_round_after_a_purchase(monkeypatch):
    monkeypatch.setattr("agent.persuasion.strategy.compose_message", lambda **kwargs: "a pitch")
    from agent.persuasion import strategy as strategy_mod

    game_id = "pers-style-credit-buy"
    round1 = _seller_text_game(game_id, round_no=10, quality="high", price=50.0, seller_total_payoff=0.0)
    persuasion_strategy(round1)

    arm_used = strategy_mod._LAST_STYLE_ARM[game_id]
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    baseline_alpha = bandit.alpha[arm_used]

    round2 = _seller_text_game(game_id, round_no=11, quality="high", price=50.0, seller_total_payoff=50.0)
    persuasion_strategy(round2)

    assert bandit.alpha[arm_used] == baseline_alpha + 1.0


def test_style_bandit_credits_the_arm_used_last_round_after_a_decline(monkeypatch):
    monkeypatch.setattr("agent.persuasion.strategy.compose_message", lambda **kwargs: "a pitch")
    from agent.persuasion import strategy as strategy_mod

    game_id = "pers-style-credit-decline"
    round1 = _seller_text_game(game_id, round_no=10, quality="high", price=50.0, seller_total_payoff=0.0)
    persuasion_strategy(round1)

    arm_used = strategy_mod._LAST_STYLE_ARM[game_id]
    bandit = strategy_mod._style_bandit({"game_id": game_id})
    baseline_beta = bandit.beta[arm_used]

    round2 = _seller_text_game(game_id, round_no=11, quality="high", price=50.0, seller_total_payoff=0.0)
    persuasion_strategy(round2)

    assert bandit.beta[arm_used] == baseline_beta + 1.0


def test_style_bandit_starts_from_the_measured_population_prior_not_uniform():
    from agent.persuasion import strategy as strategy_mod

    bandit = strategy_mod._style_bandit({"game_id": "pers-style-prior-check"})
    loyalty_mean = bandit.alpha["loyalty"] / (bandit.alpha["loyalty"] + bandit.beta["loyalty"])
    care_mean = bandit.alpha["care"] / (bandit.alpha["care"] + bandit.beta["care"])
    assert abs(loyalty_mean - strategy_mod.SELLER_FOUNDATION_PRIOR_MEANS["loyalty"]) < 1e-9
    assert abs(care_mean - strategy_mod.SELLER_FOUNDATION_PRIOR_MEANS["care"]) < 1e-9
    assert loyalty_mean > care_mean
