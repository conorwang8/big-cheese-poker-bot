"""Visible-information adversaries for local robustness tests.

These synthetic policies are tests, not equilibrium or champion opponents.
They receive only an Observation and ordinary post-hand HandResult objects.
"""
from __future__ import annotations

import numpy as np

from game import Action, Street
import poker_strategy as scoring


NAMES = ("always_pot", "river_bluffer", "delayed_bluffer", "inverse_bluffer",
         "alternating", "switch_to_bluff", "switch_to_value", "tight_value", "check_fold",
         "check_raise_value", "size_mixer")


def new_half(config, memory):
    memory.update(rng=np.random.default_rng(680731), cache_key=None, equity=None,
                  hands_played=0, num_hands=config.num_hands, fallbacks=0)


def equity(obs, memory):
    """Independent uniform-opponent Monte Carlo; no target strategy ranges.

    Use an independent RNG and sample legal unknown cards without replacement.
    Only the exact, public hand-scoring table is reused from the bot module.
    """
    key = (obs.hand_index, obs.hole_cards, obs.board)
    if key == memory["cache_key"]:
        return memory["equity"]
    known = set(obs.hole_cards + obs.board)
    unseen = np.array([card for card in range(36) if card not in known], dtype=np.int16)
    n = 192 if len(obs.board) < 5 else 384
    missing = 5 - len(obs.board)
    indices = memory["rng"].random((n, len(unseen))).argsort(axis=1)[:, :5 + missing]
    draws = unseen[indices]
    opponent = draws[:, :5]
    boards = np.empty((n, 5), dtype=np.int16)
    boards[:, :len(obs.board)] = obs.board
    boards[:, len(obs.board):] = draws[:, 5:]
    ours = scoring.score_batch(np.array([obs.hole_cards]), boards[:1] if not missing else boards)
    theirs = scoring.score_batch(opponent, boards)
    value = float(((ours > theirs) + 0.5 * (ours == theirs)).mean())
    memory["cache_key"], memory["equity"] = key, value
    return value


def pot_or_call(obs):
    return Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL


def tight_value(obs, memory):
    value = equity(obs, memory)
    threshold = 0.60 if obs.street == Street.PREFLOP else 0.70
    if value >= threshold:
        return pot_or_call(obs)
    if not obs.to_call:
        return Action.CHECK_CALL
    price = min(obs.to_call, obs.my_stack)
    odds = price / (obs.pot + price)
    margin = 0.06 if obs.street != Street.RIVER and obs.opp_stack else 0.025
    return Action.CHECK_CALL if value >= odds + margin else Action.FOLD


def choose_action(name, obs, memory):
    if name == "always_pot":
        return pot_or_call(obs)
    if name == "check_fold":
        return Action.FOLD if obs.to_call else Action.CHECK_CALL
    if name == "river_bluffer":
        return pot_or_call(obs) if obs.street == Street.RIVER else Action.CHECK_CALL
    if name == "delayed_bluffer":
        return pot_or_call(obs) if obs.street != Street.PREFLOP else Action.CHECK_CALL
    if name == "tight_value":
        return tight_value(obs, memory)
    if name == "check_raise_value":
        if obs.street != Street.PREFLOP and not obs.to_call and not obs.is_button:
            return Action.CHECK_CALL
        return tight_value(obs, memory)
    if name == "size_mixer":
        value = equity(obs, memory)
        if value >= (0.61 if obs.street == Street.PREFLOP else 0.72):
            return pot_or_call(obs)
        if obs.raises_this_street == 0 and Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        price = min(obs.to_call, obs.my_stack)
        if price and value < price / (obs.pot + price) + 0.04:
            return Action.FOLD
        return Action.CHECK_CALL
    if name == "inverse_bluffer":
        if obs.street == Street.PREFLOP:
            return Action.CHECK_CALL
        value = equity(obs, memory)
        if value < 0.42 and Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if not obs.to_call:
            return Action.CHECK_CALL
        price = min(obs.to_call, obs.my_stack)
        return Action.CHECK_CALL if value >= price / (obs.pot + price) else Action.FOLD
    if name == "alternating":
        return tight_value(obs, memory) if (obs.hand_index // 300) % 2 == 0 else pot_or_call(obs)
    if name == "switch_to_bluff":
        return tight_value(obs, memory) if obs.hand_index < 200 else pot_or_call(obs)
    if name == "switch_to_value":
        return pot_or_call(obs) if obs.hand_index < 200 else tight_value(obs, memory)
    raise ValueError(f"Unknown opponent: {name}")


def on_hand_end(result, memory):
    memory["hands_played"] += 1
    memory["cache_key"] = None
