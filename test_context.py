import numpy as np
import poker_strategy as strategy
from game import Action, AgentConfig, Game, make_deal


def memory():
    state = {}
    strategy.start_half(AgentConfig(), state)
    return state


def test_opening_and_big_blind_options_learn_separately():
    state = memory()
    small_blind = Game(make_deal('context-open', 0), 0, hand_index=0)
    small_blind.apply(Action.RAISE_HALF)
    strategy._history(small_blind.observation(1), state)
    big_blind = Game(make_deal('context-open', 1), 0, hand_index=1)
    big_blind.apply(Action.CHECK_CALL)
    big_blind.apply(Action.RAISE_POT)
    obs = big_blind.observation(0)
    strategy._history(obs, state)
    np.testing.assert_array_equal(state['context_nodes'][(0, True, 0, 0)], [0, 0, 1, 0])
    np.testing.assert_array_equal(state['context_nodes'][(0, False, 1, 0)], [0, 0, 0, 1])
    strategy._history(obs, state)
    np.testing.assert_array_equal(state['context_nodes'][(0, False, 1, 0)], [0, 0, 0, 1])


def test_half_and_pot_bet_responses_are_distinct():
    state = memory()
    for index, action in enumerate((Action.RAISE_HALF, Action.RAISE_POT)):
        game = Game(make_deal('context-sizes', index), 0, hand_index=index)
        game.apply(action)
        game.apply(Action.CHECK_CALL)
        game.apply(Action.CHECK_CALL)
        strategy._history(game.observation(0), state)
    assert state['context_nodes'][(0, False, 2, 0)][1] == 1
    assert state['context_nodes'][(0, False, 2, 1)][1] == 1


def test_small_blind_fold_is_learned_without_an_observation():
    state = memory()
    game = Game(make_deal('context-first-fold', 0), 0, hand_index=0)
    game.apply(Action.FOLD)
    strategy.finish_hand(game.hand_result(1), state)
    counts = state['context_nodes'][(0, True, 0, 0)]
    assert 0.9 < counts[0] <= 1.0
    assert counts[1:].sum() == 0


def test_size_specific_fold_evidence_is_partially_pooled():
    state = memory()
    small, large = (3, True, 2, 0), (3, True, 2, 1)
    for _ in range(30):
        strategy._context_note(state, small, Action.FOLD)
        strategy._context_note(state, large, Action.CHECK_CALL)
    assert strategy._context_rates(state, small)[0] > 0.70
    assert strategy._context_rates(state, large)[0] < 0.30


def test_weighted_percentiles_preserve_ties_and_range_mass():
    result = strategy._weighted_percentiles(np.array([1, 1, 2]), np.array([2.0, 4.0, 4.0]))
    np.testing.assert_allclose(result, [0.3, 0.3, 0.8])


def test_new_half_clears_context_learning():
    state = memory()
    strategy._context_note(state, (0, True, 0, 0), Action.RAISE_POT)
    strategy.start_half(AgentConfig(), state)
    assert state['context_nodes'] == state['context_pools'] == {}
