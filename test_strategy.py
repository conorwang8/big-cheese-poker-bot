"""Correctness checks for the optimized evaluator and tournament state."""
from itertools import combinations

import numpy as np
import pytest
from dataclasses import replace

import agent
import poker_strategy as strategy
from evaluator import cards_from_str, omaha_score, score_five
from game import Action, AgentConfig, Game, Street, make_deal
from test_agent import AUTH, load_agent, play_hands, start_half
from fastapi.testclient import TestClient


def test_vector_scores_match_reference_on_full_and_partial_boards():
    rng = np.random.default_rng(491)
    samples = np.array([rng.choice(36, 10, replace=False) for _ in range(250)])
    holes, boards = samples[:, :5], samples[:, 5:]
    for visible in (3, 4, 5):
        expected = []
        for hole, board in zip(holes, boards[:, :visible]):
            expected.append(max(score_five((*h, *b))
                                for h in combinations(hole.tolist(), 2)
                                for b in combinations(board.tolist(), 3)))
        np.testing.assert_array_equal(strategy.score_batch(holes, boards[:, :visible]), expected)
    # Also verify broadcasting a fixed hero hand across many independent boards.
    fixed = holes[:1]
    safe_boards = np.array([rng.choice([c for c in range(36) if c not in fixed[0]], 5,
                                       replace=False) for _ in range(100)])
    np.testing.assert_array_equal(strategy.score_batch(fixed, safe_boards),
                                  [omaha_score(fixed[0], b) for b in safe_boards])


def test_omaha_requires_two_hole_cards_for_a_flush():
    hole = np.array([cards_from_str("Th 7s 7d 8c 9c")])
    board = np.array([cards_from_str("Ah Kh Qh Jh 6c")])
    score = int(strategy.score_batch(hole, board)[0])
    assert score >> 20 == 3  # king-high straight; a single heart cannot make a flush


def test_precomputed_quality_and_draw_pools_are_valid():
    rng = np.random.default_rng(2718)
    holes = np.array([rng.choice(36, 5, replace=False) for _ in range(500)])
    np.testing.assert_allclose(strategy.preflop_quality(holes), strategy._quality_features(holes))
    for remaining, pool in strategy.DRAW_POOLS.items():
        assert pool.min() >= 0 and pool.max() < remaining
        assert not pool.flags.writeable
        assert np.all(np.diff(np.sort(pool, axis=1), axis=1) > 0)


def test_samples_have_no_repeated_or_visible_cards():
    memory = {}
    agent.new_half(AgentConfig(), memory)
    game = Game(make_deal("sampling", 0), 0)
    game.apply(Action.CHECK_CALL)
    game.apply(Action.CHECK_CALL)
    obs = game.observation(game.to_act)
    agent.choose_action(obs, memory)
    assert memory["fallbacks"] == 0
    cache = memory["cache"]
    for opponent, board in zip(cache["opponents"], cache["boards"]):
        assert len(set((*obs.hole_cards, *opponent, *board))) == 15
        assert tuple(board[:len(obs.board)]) == obs.board


def test_full_hands_never_use_the_exception_fallback():
    module = load_agent()
    module.TOKEN = "test-token"
    client = TestClient(module.app)
    start_half(client)
    play_hands(client, 150, seed="strategy-check", raise_rate=0.65)
    assert module.MEMORY["fallbacks"] == 0
    assert module.MEMORY["hands_played"] == 150


def test_new_half_resets_rng_opponent_model_and_equity_cache():
    module = load_agent()
    module.TOKEN = "test-token"
    client = TestClient(module.app)
    start_half(client)
    play_hands(client, 15, seed="previous-half")
    start_half(client, half=2, seat=1)
    memory = module.MEMORY
    assert memory["cache"] is None
    assert memory["hands_played"] == 0
    assert memory["active_hand"] == -1
    assert memory["spent"] == 0
    fresh = {}
    module.new_half(AgentConfig(), fresh)
    assert memory["stats"] == fresh["stats"]
    obs = Game(make_deal("same-visible-cards", 0), 0).observation(0)
    assert module.choose_action(obs, memory) == module.choose_action(obs, fresh)
    np.testing.assert_array_equal(memory["cache"]["opponents"], fresh["cache"]["opponents"])


def river_bet_fixture():
    game = Game(make_deal("river-learning", 0), 0)
    while game.street != Street.RIVER:
        game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_POT)  # non-button opponent bets first
    obs = game.observation(0)
    memory = {}
    agent.new_half(AgentConfig(), memory)
    agent.choose_action(obs, memory)
    assert memory["fallbacks"] == 0
    actual = omaha_score(game.deal.hole_cards[1], obs.board)
    # Known low-percentile fixture isolates the learning path from MC variance.
    memory["cache"]["villain_scores"] = np.full(128, actual + 1)
    return game, obs, memory


def test_revealed_river_bluff_is_learned_after_direct_call():
    game, obs, memory = river_bet_fixture()
    memory["last_action"] = Action.CHECK_CALL
    game.apply(Action.CHECK_CALL)
    agent.on_hand_end(game.hand_result(0), memory)
    assert memory["river_reads"][0] == [1.0, 0.0]


def test_fold_does_not_invent_a_bluff_read():
    game, obs, memory = river_bet_fixture()
    memory["last_action"] = Action.FOLD
    game.apply(Action.FOLD)
    agent.on_hand_end(game.hand_result(0), memory)
    assert sum(sum(x) for x in memory["river_reads"]) == 0


def test_opponent_calling_our_reraise_is_not_a_bluff_read():
    game, obs, memory = river_bet_fixture()
    memory["last_action"] = Action.RAISE_POT
    game.apply(Action.RAISE_POT)
    game.apply(Action.CHECK_CALL)
    agent.on_hand_end(game.hand_result(0), memory)
    assert sum(sum(x) for x in memory["river_reads"]) == 0


def test_cheap_bluffs_do_not_change_expensive_bet_reads():
    _, obs, memory = river_bet_fixture()
    memory["river_reads"][0] = [20.0, 0.0]
    strength = np.linspace(0, 1, 100)
    default = strategy._selection(strength, 0.3)
    cheap = strategy._river_raise_likelihood(obs, memory, False, strength, 0.3, default)
    expensive_obs = replace(obs, to_call=12 * obs.big_blind)
    expensive = strategy._river_raise_likelihood(expensive_obs, memory, False, strength, 0.3, default)
    np.testing.assert_array_equal(expensive, default)
    assert (cheap[:30].sum() / cheap.sum()) > (default[:30].sum() / default.sum())


def test_old_opponent_behavior_decays_toward_the_prior():
    game, _, memory = river_bet_fixture()
    memory["stats"][0]["open"] = [7.0, 103.0]
    memory["last_action"] = Action.CHECK_CALL
    game.apply(Action.CHECK_CALL)
    agent.on_hand_end(game.hand_result(0), memory)
    assert 3.0 < memory["stats"][0]["open"][1] < 103.0


@pytest.mark.parametrize("win_count,expected", [(250, Action.CHECK_CALL), (100, Action.FOLD),
                                               (1000, Action.CHECK_CALL)])
def test_all_in_response_uses_pot_odds(monkeypatch, win_count, expected):
    game = Game(make_deal("all-in-price", 0), 0)
    for _ in range(4):
        game.apply(Action.RAISE_POT)
    obs = game.observation(0)
    assert obs.opp_stack == 0 and obs.to_call == 26 and obs.pot == 134
    # Calling costs 26 to contest 160: 25% equity is enough; 10% is not.
    memory = {}
    agent.new_half(AgentConfig(), memory)
    outcomes = np.r_[np.ones(win_count), np.zeros(1000-win_count)]
    monkeypatch.setattr(strategy, "_samples", lambda *args: {"outcomes": outcomes})
    monkeypatch.setattr(strategy, "_range_weights", lambda *args: np.ones(1000))
    assert strategy._choose(obs, memory, {}) == expected
