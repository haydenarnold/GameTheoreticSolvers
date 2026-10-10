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
    # Regression: a live game had q* compute to ~1.0 (a breakeven prior,
    # p*v ~= price), the seller bluffed on round 1, the buyer got burned,
    # and never engaged again for the rest of a 20-round game -- roughly
    # ten more honest high-quality rounds all declined. A single early
    # bluff shouldn't be allowed to risk the whole relationship before any
    # track record exists. With q* ~= 1.0 (p=0.5, v=100, price=50), the
    # observed bluff rate during the trust-building window must be well
    # below 1.0, not the full textbook-optimal value.
    assert SELLER_TRUST_BUILDING_ROUNDS >= 1
    signaled_buy_count = 0
    trials = 400
    for i in range(trials):
        game = _seller_game(f"pers-trust-build-{i}", round_no=1, quality="low", p=0.5, v=100.0, u=0.0, price=50.0)
        action = persuasion_strategy(game)
        if action["decision"] == "yes":
            signaled_buy_count += 1
    bluff_rate = signaled_buy_count / trials
    assert bluff_rate == 0.0  # SELLER_TRUST_BUILDING_MAX_Q=0.0: no bluffing at all in this window


def test_seller_does_not_bluff_on_round_one_of_the_live_regression_game():
    # CONFIRMED live (2026-08-25): the exact incident parameters -- p=0.5,
    # v=$3M, u=$0, price=$1M (prior alone clears breakeven, q* clips to
    # 1.0). A round-1 bluff here cost $9M in foregone honest sales against
    # $1M gained. Round 1 specifically must never signal "yes" on a
    # genuinely low-quality item during the trust-building window.
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
    # Reversed on purpose (2026-08-23): past the trust-building window, the
    # rate no longer returns to near-uncapped textbook q* (~1.0) -- it
    # settles near q* * SELLER_BLUFF_DAMPENING instead, per two real games
    # showing a caught/successful bluff can trigger an extended trust
    # collapse costing several times what the bluff itself gained.
    # SELLER_BLUFF_DAMPENING moved 0.5->0.7 (2026-08-26, see its own
    # derivation comment), so with q*=1.0 here (p=0.5, v=100, price=50 is
    # exactly the breakeven-prior case), the settled rate should track ~0.7
    # now, not ~0.5.
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
    # Prior alone (p=0.9) would clear the price easily, but a "no" signal
    # from the seller must still be decisive — the whole point of the fix
    # is that the current round's signal is no longer ignored. This test's
    # fixture (v=100, u=0, price=40) happens to clear the later
    # DECLINE_TEST_UPSIDE_RATIO_THRESHOLD bootstrap-test gate too (ratio
    # 1.5) -- disabled here since that's a different mechanism with its own
    # dedicated tests below; this one verifies the discourage signal is
    # read as decisive at all, independent of whether testing it is ever
    # worthwhile.
    monkeypatch.setattr("agent.persuasion.strategy.DECLINE_TEST_UPSIDE_RATIO_THRESHOLD", float("inf"))
    game = _buyer_game("pers-no-signal", seller_message="no", favorable_prior=True)
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_buyer_can_accept_when_seller_signals_yes_with_favorable_prior():
    game = _buyer_game("pers-yes-signal", seller_message="yes", favorable_prior=True)
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_buyer_does_not_lock_out_after_two_unlucky_bootstrap_buys_on_a_comfortable_prior():
    # CONFIRMED live (2026-08-25) regression: real transcript, p=0.8,
    # v=$200, u=$0, price=$100 (unconditional EV $60/round -- a comfortable
    # 33% margin above the $120 margin-adjusted breakeven). The seller's
    # message was IDENTICAL every round ("Good unit this round..."), so it
    # carries no conditional information at all. The buyer's first 2
    # bootstrap purchases both happened to be low quality (~4% likelihood,
    # not evidence of anything) and the OLD BuyerTrustModel default
    # (prior_pseudo_count=5.0) let that crash the effective posterior below
    # the buy threshold -- passing every one of the remaining 18 rounds and
    # foregoing ~$1,080 in expectation. Replays that exact sequence through
    # the real strategy function end-to-end.
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

    # Rounds 1-2: bootstrap buys, both forced low (the unlucky real draw).
    for round_no in (1, 2):
        action = persuasion_strategy(make_game(round_no))
        assert action["decision"] == "yes"  # bootstrap should still fire
        payoff += u - price  # low-quality outcome

    # Round 3: this is the exact point the old default locked out forever.
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
    # These tests exercise the DISCOURAGE_PHRASES fallback specifically, not
    # the LLM path -- forcing classify_discourage_signal to return None
    # (its own "no usable verdict" signal, same as a real call failing)
    # keeps them hermetic and deterministic instead of attempting a real,
    # slow, non-deterministic network call in the offline test suite.
    monkeypatch.setattr("agent.persuasion.strategy.classify_discourage_signal", lambda message: None)
    # Also disable the decline-testing bootstrap (this fixture's v=100,
    # u=0, price=40 clears its EV-ratio gate too) -- these tests verify
    # discourage-signal DETECTION specifically, a different mechanism with
    # its own dedicated tests below, and shouldn't be coupled to it.
    monkeypatch.setattr("agent.persuasion.strategy.DECLINE_TEST_UPSIDE_RATIO_THRESHOLD", float("inf"))


def test_text_mode_buyer_declines_on_explicit_discouraging_message(monkeypatch):
    # Regression: a live game had the seller write "This product is low
    # quality — you should pass on it" and we bought anyway (losing the
    # full price) because text mode had no signal detection before this fix.
    _patch_llm_unavailable(monkeypatch)
    game = _text_mode_buyer_game(
        "pers-text-discourage",
        seller_message="This product is low quality — you should pass on it.",
        favorable_prior=True,  # prior alone would say buy; the message must override it
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_text_mode_buyer_declines_on_comparative_discouraging_message(monkeypatch):
    # Regression: a live game had the seller write "This one is ordinary.
    # Wait for the next -- I'll tell you when it's worth it." -- no
    # absolute negative phrase at all (it even contains "worth it" in a
    # positive framing), so the old lexicon missed it entirely and we
    # bought anyway, twice, both times genuinely low quality.
    _patch_llm_unavailable(monkeypatch)
    game = _text_mode_buyer_game(
        "pers-text-comparative-discourage",
        seller_message="I'll be straight with you: this one is ordinary. Wait for the next -- I'll tell you when it's worth it.",
        favorable_prior=True,
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_text_mode_buyer_declines_on_a_soft_hedged_pass_suggestion(monkeypatch):
    # Regression: a live 20-round game had the seller write "I'll be
    # straight with you -- this one isn't my best unit. You may want to
    # pass today." on two genuinely low-quality rounds -- neither "isn't my
    # best" nor "may want to pass" matched any existing phrase (closest was
    # "should pass"/"pass on it"), so we bought both times and lost the
    # full price twice in the same game.
    _patch_llm_unavailable(monkeypatch)
    game = _text_mode_buyer_game(
        "pers-text-soft-hedge-discourage",
        seller_message="I'll be straight with you -- this one isn't my best unit. You may want to pass today.",
        favorable_prior=True,
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_text_mode_buyer_declines_on_a_relative_quality_comparison_hedge(monkeypatch):
    # Regression: a real 20-round game had a fully honest seller write "This
    # unit is below our usual quality this round" on 6 genuinely low-quality
    # rounds, and we bought all 6 anyway (a very favorable prior -- EV=$240
    # vs $100 price -- meant the buyer defaulted to "yes" on the unmatched
    # message). Cost $600 of an achievable $2800, over a quarter of the
    # game's profit, in a game otherwise played perfectly.
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
    # The actual point of the structural fix: a message with NO match in
    # DISCOURAGE_PHRASES at all must still be declined if the LLM classifier
    # confidently says it discourages buying.
    monkeypatch.setattr(
        "agent.persuasion.strategy.classify_discourage_signal",
        lambda message: (True, 0.95),
    )
    monkeypatch.setattr("agent.persuasion.strategy.DECLINE_TEST_UPSIDE_RATIO_THRESHOLD", float("inf"))
    game = _text_mode_buyer_game(
        "pers-text-llm-confident-discourage",
        seller_message="Honestly? I've seen better days from this shelf.",  # matches no fixed phrase
        favorable_prior=True,
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_text_mode_buyer_buys_on_confident_llm_verdict_even_when_phrase_list_would_have_matched(monkeypatch):
    # A confident LLM "does not discourage" verdict is trusted outright --
    # the phrase list is only ever a fallback, not a second check run
    # regardless of confidence.
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
        lambda message: (False, DISCOURAGE_CONFIDENCE_THRESHOLD - 0.05),  # below the bar
    )
    monkeypatch.setattr("agent.persuasion.strategy.DECLINE_TEST_UPSIDE_RATIO_THRESHOLD", float("inf"))
    game = _text_mode_buyer_game(
        "pers-text-llm-uncertain",
        seller_message="This product is low quality — you should pass on it.",  # matches a fixed phrase
        favorable_prior=True,
    )
    action = persuasion_strategy(game)
    assert action["decision"] == "no"  # falls back to the phrase list, which catches it


def test_buyer_proceeds_to_explore_on_generic_neutral_filler_not_a_real_hedge(monkeypatch):
    # Regression: a real live game had the seller send the identical "No
    # strong recommendation this round." on all 20 rounds (never varying
    # with true quality -- pure content-free filler), and the buyer
    # blanket-declined every single round against a hugely favorable prior
    # (p=0.8, v=$30,000, u=$0, price=$10,000 -> EV=$24,000). Manually
    # verified against the real model that the corrected prompt now returns
    # (False, 0.3) for this exact message -- low confidence, so it falls
    # back to the phrase list (which also doesn't match), correctly
    # defaulting to "signaled buy" and letting the round-1 bootstrap
    # exploration fire instead of blanket-declining.
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
    assert action["decision"] == "yes"  # round-1 bootstrap exploration, not a blanket decline


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
    # Reversed back on purpose (2026-08-23), a second time the same day.
    # BUYER_SAFETY_MARGIN was first applied to the bootstrap gate too
    # (this test used to assert "no" here), but that made the bootstrap
    # branch mathematically identical to -- and therefore fully subsumed
    # by -- the final decision, since posterior_high_given_signal() returns
    # bare prior_p while buys < min_observations. CONFIRMED live: a real
    # 20-round game (p=0.8, v=$125, u=$0, price=$100 -- ratio 1.0, same
    # tie as this test) declined all 20 rounds against a seller repeating a
    # specific, consistent high-quality claim, reproducing the original
    # 0-for-20 deadlock this file already fixed once. BOOTSTRAP_EV_RATIO_
    # THRESHOLD now gates the one-time bounded bootstrap test (cheap
    # information-gathering under real uncertainty about this seller) at
    # bare breakeven; EXPLORATION_EV_RATIO_THRESHOLD's higher margin still
    # applies to the no-side's unconditional_ev_ok gate and every
    # post-bootstrap decision. No longer needs DISABLE_BUYER_TRUST_MODEL
    # monkeypatched off (2026-08-30): the bootstrap test now fires
    # regardless of that flag -- see its derivation comment in strategy.py.
    game = _yes_signal_game("pers-exploration-tie", price=10000.0, v=30000.0, u=0.0, p=1 / 3)
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_buyer_bootstrap_now_fires_on_the_recurring_096_ratio_config():
    # CONFIRMED live (2026-08-25): p=0.8, v=1.2x price, u=0 (ratio=0.96) is
    # the one sub-1.0 config that actually recurs in real games (21/200
    # buyer-role games in a 400-game pull, 10.5%) -- every one a guaranteed
    # $0 under the old BOOTSTRAP_EV_RATIO_THRESHOLD=1.0, since 0.96 < 1.0
    # meant the bootstrap gate never even tried it, and posterior_high_
    # given_signal() can only move once real purchase data exists. Real
    # p=0.8 yes-signaled purchases elsewhere in that pull cleared this
    # config's 83.3% breakeven empirically (88.0% at n=50, 85.7% at n=321)
    # -- see BOOTSTRAP_EV_RATIO_THRESHOLD's derivation for the full
    # reasoning and the CI caveat. This must now bootstrap-test rather than
    # blanket-decline. RECONFIRMED live (2026-08-30): a fresh 198-game
    # buyer-role pull found this exact config still going 0-for-20, because
    # DISABLE_BUYER_TRUST_MODEL (on by default) short-circuited the
    # bootstrap branch below entirely -- this test only passed before via a
    # DISABLE_BUYER_TRUST_MODEL=False monkeypatch that masked the bug. No
    # longer needed now that the bootstrap fires regardless of that flag.
    game = _yes_signal_game("pers-bootstrap-096-config", price=1_000_000.0, v=1_200_000.0, u=0.0, p=0.8)
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_buyer_still_declines_a_ratio_below_the_new_threshold():
    # A ratio clearly below the new 0.95 threshold (unlike the validated
    # 0.96 config above) must still decline -- the lowered threshold isn't
    # a blanket loosening, only the one empirically-checked config now
    # clears it.
    game = _yes_signal_game("pers-bootstrap-below-threshold", price=1_000_000.0, v=1_100_000.0, u=0.0, p=0.8)
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_buyer_still_explores_on_a_buy_signal_once_the_safety_margin_is_cleared():
    # Comfortably above BOOTSTRAP_EV_RATIO_THRESHOLD too, so this isn't a
    # useful discriminator between the two thresholds on its own -- kept as
    # a basic sanity check that a clearly favorable config still bootstraps.
    game = _yes_signal_game("pers-exploration-clears-margin", price=10000.0, v=30000.0, u=0.0, p=0.5)
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_buyer_bootstraps_instead_of_going_0_for_20_on_the_live_regression_game(monkeypatch):
    # CONFIRMED live (2026-08-23): p=0.8, v=$125, u=$0, price=$100 -- exact
    # EV tie -- with the seller repeating a specific, consistent
    # high-quality claim on 19 of 20 rounds. Declined every round: 0-for-20.
    # No longer needs DISABLE_BUYER_TRUST_MODEL monkeypatched off
    # (2026-08-30) -- see the 096-ratio-config test above.
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
    # The gate must exclude a clearly-bad prior (ratio 0.25 here) -- burning
    # an exploratory purchase can't plausibly change a verdict that far off,
    # so the existing conservative "no" behavior must be unchanged.
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
    # Regression: a real 20-round game (p=0.5, v=$4,000,000, u=$0,
    # price=$1,000,000 -- upside/downside = 3.0, well above the 1.0 bar)
    # trusted the seller's "no" unconditionally on every single round, with
    # no mechanism to ever learn whether this seller's "no" was reliable.
    # The breakeven reliability for obeying "no" here is only 75% (see
    # DECLINE_TEST_UPSIDE_RATIO_THRESHOLD's derivation) -- this must now
    # spend a bounded first test rather than trusting blind. No longer
    # needs DISABLE_BUYER_TRUST_MODEL monkeypatched off (2026-08-30) -- see
    # the 096-ratio-config test above.
    game = _no_signal_game("pers-decline-test-fires", price=1_000_000.0, v=4_000_000.0, u=0.0)
    action = persuasion_strategy(game)
    assert action["decision"] == "yes"


def test_buyer_does_not_test_a_discourage_signal_when_unconditional_prior_is_unfavorable():
    # Regression: a real game (p=0.33, v=$2,000,000, u=$0, price=$1,000,000)
    # had upside/downside = 1.0, clearing the shape-ratio gate on its own,
    # even though the unconditional EV ratio (0.66) is BELOW the yes-side's
    # own EXPLORATION_EV_RATIO_THRESHOLD (0.75) -- so "no" got bootstrap-
    # tested (bought, twice, both genuinely low quality, -$1M each) while
    # every real "yes" endorsement in the same game was declined, because
    # the yes-side bootstrap never got the same chance under that same
    # unfavorable prior. Net -$2,000,000, worse than doing nothing. Testing
    # "no" must never be easier to trigger than testing "yes" would be.
    game = _no_signal_game("pers-decline-test-blocked-by-prior", price=1_000_000.0, v=2_000_000.0, u=0.0, p=0.33)
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_buyer_does_not_test_a_discourage_signal_when_the_upside_ratio_is_too_thin():
    # v barely above price -> upside/downside well under 1.0 -- not worth
    # testing even under a maximally uninformed 50/50 guess, so the old
    # unconditional trust-the-"no" behavior must be unchanged here.
    game = _no_signal_game("pers-decline-test-skipped", price=1_000_000.0, v=1_100_000.0, u=0.0)
    action = persuasion_strategy(game)
    assert action["decision"] == "no"


def test_testing_a_discourage_signal_updates_the_decline_model_not_the_yes_model():
    # No longer needs DISABLE_BUYER_TRUST_MODEL monkeypatched off
    # (2026-08-30) -- see the 096-ratio-config test above.
    from agent.persuasion import strategy as strategy_mod

    game_id = "pers-decline-test-routing"
    round1 = _no_signal_game(game_id, price=1_000_000.0, v=4_000_000.0, u=0.0, round_no=1)
    action1 = persuasion_strategy(round1)
    assert action1["decision"] == "yes"  # bootstrap test
    assert strategy_mod._LAST_BUYER_TESTED_NO[game_id] is True

    # Round 2: the test purchase turned out low quality (payoff delta ==
    # u - price == -$1,000,000).
    round2 = _no_signal_game(
        game_id, price=1_000_000.0, v=4_000_000.0, u=0.0, round_no=2,
        buyer_total_payoff=-1_000_000.0,
    )
    persuasion_strategy(round2)

    assert strategy_mod._BUYER_NO_TRUST[game_id].buys == 1
    assert strategy_mod._BUYER_NO_TRUST[game_id].highs == 0  # confirmed low quality
    assert game_id not in strategy_mod._BUYER_TRUST  # the unrelated "yes"-signal model untouched


def test_discourage_signal_trusted_again_after_a_test_confirms_it_is_reliable():
    from agent.persuasion import strategy as strategy_mod

    game_id = "pers-decline-test-confirmed-reliable"
    round1 = _no_signal_game(game_id, price=1_000_000.0, v=4_000_000.0, u=0.0, round_no=1)
    persuasion_strategy(round1)  # bootstrap test, buys

    round2 = _no_signal_game(
        game_id, price=1_000_000.0, v=4_000_000.0, u=0.0, round_no=2,
        buyer_total_payoff=-1_000_000.0,  # confirmed low quality
    )
    action2 = persuasion_strategy(round2)
    # Past the bootstrap (buys=1 >= min_observations=1) and not yet due for
    # periodic re-exploration: posterior_high_given_signal() should now
    # reflect the confirmed-low observation and correctly go back to
    # trusting "no" rather than testing every single round forever.
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
    # quality="high" forces signal_buy=True deterministically (no randomness).
    game = _seller_text_game("pers-style-buy", round_no=10, quality="high", price=50.0)
    persuasion_strategy(game)
    assert captured["style"] in StyleBandit.ARMS


def test_no_style_is_passed_on_a_discourage_pitch(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "agent.persuasion.strategy.compose_message",
        lambda **kwargs: captured.update(kwargs) or "a pitch",
    )
    # quality="low" with price >= v forces optimal_bluff_prob to exactly 0,
    # so signal_buy is deterministically False (no randomness to control for).
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

    # seller_total_payoff jumped by the full price -> a purchase happened.
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

    # seller_total_payoff unchanged -> no purchase happened.
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
    assert loyalty_mean > care_mean  # matches the measured ranking, not a flat 0.5/0.5
