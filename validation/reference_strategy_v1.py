"""PocketPilot: short-deck PLO5 equity, range estimates, and legal bet selection.

No network, runtime files, learned cross-half state, or hidden cards are used.
The score table is derived from the bundle's unmodified reference evaluator.
"""

from itertools import combinations
from math import comb
import os
from time import perf_counter

import numpy as np

from evaluator import get_table
from game import Action, Street

PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.intp)
TRIPLES = np.array(list(combinations(range(5), 3)), dtype=np.intp)
C2 = np.array([comb(i, 2) if i >= 2 else 0 for i in range(36)], dtype=np.intp)
C3 = np.array([comb(i, 3) if i >= 3 else 0 for i in range(36)], dtype=np.intp)
C4 = np.array([comb(i, 4) if i >= 4 else 0 for i in range(36)], dtype=np.intp)
C5 = np.array([comb(i, 5) if i >= 5 else 0 for i in range(36)], dtype=np.intp)
DECK = np.arange(36, dtype=np.int16)
SCORES = None
PAIR_IDS = None
QUALITY = None
DRAW_POOLS = None


def initialize():
    """Build a read-only 18 MB array before the first timed request."""
    global SCORES, PAIR_IDS, QUALITY, DRAW_POOLS
    if SCORES is not None:
        return
    pair_ids = np.zeros((36, 36), dtype=np.intp)
    for a, b in combinations(range(36), 2):
        pair_ids[a, b] = pair_ids[b, a] = comb(b, 2) + a

    hands = np.array(list(combinations(range(36), 5)), dtype=np.int16)
    quality = np.empty(len(hands), dtype=float)
    indices = hands[:, 0] + C2[hands[:, 1]] + C3[hands[:, 2]] + C4[hands[:, 3]] + C5[hands[:, 4]]
    quality[indices] = _quality_features(hands)
    quality.flags.writeable = False
    # Public, board-independent Monte Carlo draws. Per-half RNG chooses a
    # different window on each new street. Each row is a uniform ordered draw
    # without replacement; mapping through the remaining deck applies blockers.
    generator = np.random.default_rng(741963)
    pools = {}
    for remaining in (26, 27, 28, 31):
        pool = generator.random((16384, remaining)).argsort(axis=1)[:, :10].astype(np.int16)
        pool.flags.writeable = False
        pools[remaining] = pool
    bits = np.left_shift(np.uint64(1), DECK.astype(np.uint64))
    masks = bits[hands].sum(axis=1)
    reference = get_table()
    values = np.fromiter((reference[int(m)] for m in masks), dtype=np.uint32,
                         count=len(hands))
    scores = np.zeros((630, 7140), dtype=np.uint32)
    for i, j in PAIRS:
        rest = [k for k in range(5) if k != i and k != j]
        a, b, c = (hands[:, k] for k in rest)
        triple_ids = a + C2[b] + C3[c]
        scores[pair_ids[hands[:, i], hands[:, j]], triple_ids] = values
    scores.flags.writeable = False
    pair_ids.flags.writeable = False
    PAIR_IDS, SCORES = pair_ids, scores
    QUALITY, DRAW_POOLS = quality, pools


def score_batch(holes, boards):
    """Exact best hands from 2 hole cards and 3 visible board cards.

    Arrays have shape (samples, 5) and (samples, 3..5). One-row inputs
    broadcast, so a fixed hero hand can be evaluated against many runouts.
    """
    holes = np.asarray(holes, dtype=np.int16)
    boards = np.sort(np.asarray(boards, dtype=np.int16), axis=-1)
    pairs = PAIR_IDS[holes[:, PAIRS[:, 0]], holes[:, PAIRS[:, 1]]]
    triples = TRIPLES if boards.shape[1] == 5 else np.array(
        list(combinations(range(boards.shape[1]), 3)), dtype=np.intp)
    a, b, c = (boards[:, triples[:, k]] for k in range(3))
    triple_ids = a + C2[b] + C3[c]
    return SCORES[pairs[:, :, None], triple_ids[:, None, :]].max(axis=(1, 2))


def _quality_features(holes):
    """A cheap range-ordering feature; actual hero equity uses simulation."""
    ranks, suits = holes >> 2, holes & 3
    r1, r2 = ranks[:, PAIRS[:, 0]], ranks[:, PAIRS[:, 1]]
    distance = np.abs(r1 - r2)
    # Ace also connects with 6, 7, 8, 9 in the short-deck wheel.
    distance = np.where((np.maximum(r1, r2) == 8) & (np.minimum(r1, r2) <= 3),
                        np.minimum(distance, np.minimum(r1, r2) + 1), distance)
    result = ranks.sum(axis=1) * 0.05
    result += ((r1 == r2) * (0.25 + 0.07 * r1)).sum(axis=1)
    result += np.select([distance == 1, distance == 2, distance == 3],
                        [0.14, 0.09, 0.04], default=0.0).sum(axis=1)
    for suit in range(4):
        count = (suits == suit).sum(axis=1)
        high = np.where(suits == suit, ranks, -1).max(axis=1)
        result += np.where(count >= 2, 0.30 + 0.025 * high - 0.15 * (count - 2), 0)
    # Three/four cards of a rank waste Omaha's exactly-two-card combinations.
    for rank in range(9):
        result -= np.maximum(0, (ranks == rank).sum(axis=1) - 2) * 0.65
    return result


def preflop_quality(holes):
    cards = np.sort(holes, axis=1)
    indices = cards[:, 0] + C2[cards[:, 1]] + C3[cards[:, 2]] + C4[cards[:, 3]] + C5[cards[:, 4]]
    return QUALITY[indices]


def percentiles(values):
    """Midranks keep equally strong hands equally likely."""
    ordered = np.sort(values)
    return (np.searchsorted(ordered, values, side="left") +
            np.searchsorted(ordered, values, side="right")) / (2.0 * len(values))


def start_half(config, memory):
    memory.update(
        hands_played=0, rng=np.random.default_rng(20261107),
        stats=[{"open": [7.0, 3.0], "facing": [3.0, 6.0, 1.0]} for _ in range(4)],
        active_hand=-1, seen=0, cache=None, last_obs=None, last_action=None,
        spent=0.0, target_seconds=max(0.0, config.time_bank_seconds) * 0.30,
        num_hands=max(1, config.num_hands), decisions=0, fallbacks=0,
        net_chips=0,
    )


def _count(memory, street, action, facing):
    stat = memory["stats"][int(street)]
    if facing:
        index = 0 if action == Action.FOLD else 1 if action == Action.CHECK_CALL else 2
        stat["facing"][index] += 1
    elif action != Action.FOLD:
        stat["open"][int(action in (Action.RAISE_HALF, Action.RAISE_POT))] += 1


def _history(obs, memory):
    """Count each visible opposing action once, reconstructing its price."""
    if memory["active_hand"] != obs.hand_index:
        memory["active_hand"], memory["seen"], memory["cache"] = obs.hand_index, 0, None
    my_bet = obs.big_blind // 2 if obs.is_button else obs.big_blind
    opp_bet = obs.big_blind if obs.is_button else obs.big_blind // 2
    street = Street.PREFLOP
    evidence = {}
    for i, record in enumerate(obs.action_history):
        if record.street != street:
            street, my_bet, opp_bet = record.street, 0, 0
        if record.is_me:
            my_bet = record.street_total
            continue
        facing = my_bet > opp_bet
        if i >= memory["seen"]:
            _count(memory, street, record.action, facing)
        prior = evidence.get(int(street))
        is_raise = record.action in (Action.RAISE_HALF, Action.RAISE_POT)
        if prior is None or is_raise or prior[0] == Action.CHECK_CALL:
            evidence[int(street)] = (record.action, facing)
        opp_bet = record.street_total
    memory["seen"] = len(obs.action_history)
    return evidence


def _samples(obs, memory):
    """Reuse the same samples within a street so raises don't change the dice."""
    key = (obs.hand_index, obs.hole_cards, obs.board)
    cache = memory["cache"]
    if cache is not None and cache["key"] == key:
        return cache
    # Gradually reduce work if actual measurements approach the reserve.
    progress = max(0.12, (obs.hand_index + 1) / memory["num_hands"])
    projected = memory["spent"] / progress
    target = memory["target_seconds"]
    n = 128 if len(obs.board) < 5 else 256
    if projected > target:
        n //= 2
    if memory["spent"] > target:
        n = 32
    missing = 5 - len(obs.board)
    known = set(obs.hole_cards + obs.board)
    remaining = DECK[[int(c) not in known for c in DECK]]
    pool = DRAW_POOLS[len(remaining)]
    offset = int(memory["rng"].integers(0, len(pool) - n + 1))
    order = pool[offset:offset + n, :5 + missing]
    draw = remaining[order]
    opponents = draw[:, :5]
    boards = np.empty((n, 5), dtype=np.int16)
    boards[:, :len(obs.board)] = obs.board
    boards[:, len(obs.board):] = draw[:, 5:]
    hero = score_batch(np.array([obs.hole_cards]),
                       boards[:1] if len(obs.board) == 5 else boards)
    villain = score_batch(opponents, boards)
    outcomes = (hero > villain).astype(float) + 0.5 * (hero == villain)
    cache = {"key": key, "opponents": opponents, "boards": boards, "outcomes": outcomes,
             "percentiles": {0: percentiles(preflop_quality(opponents))}}
    if len(obs.board) == 5:
        cache["percentiles"][3] = percentiles(villain)
    memory["cache"] = cache
    return cache


def _strength(cache, street, board):
    if street not in cache["percentiles"]:
        count = (0, 3, 4, 5)[street]
        scores = score_batch(cache["opponents"], np.array([board[:count]]))
        cache["percentiles"][street] = percentiles(scores)
    return cache["percentiles"][street]


def _selection(strength, fraction):
    """Soft top-range selection, with some bluffs and imperfect decisions."""
    fraction = float(np.clip(fraction, 0.04, 1.0))
    if fraction >= 0.86:
        return np.ones_like(strength)
    return 0.06 + 0.94 / (1.0 + np.exp(-10.0 * (strength - (1.0 - fraction))))


def _range_weights(obs, memory, cache, evidence):
    weights = np.ones(len(cache["outcomes"]))
    for street, (action, facing) in evidence.items():
        stat = memory["stats"][street]
        if action in (Action.RAISE_HALF, Action.RAISE_POT):
            counts = stat["facing"] if facing else stat["open"]
            fraction = counts[-1] / sum(counts)
            if fraction >= 0.86:
                continue
            strength = _strength(cache, street, obs.board)
            likelihood = _selection(strength, fraction)
        elif facing:
            counts = stat["facing"]
            fraction = 1.0 - counts[0] / sum(counts)
            if fraction >= 0.86:
                continue
            likelihood = _selection(_strength(cache, street, obs.board), fraction)
        else:
            counts = stat["open"]
            fraction = counts[1] / sum(counts)
            if fraction < 0.08:
                continue
            selected = _selection(_strength(cache, street, obs.board), fraction)
            likelihood = np.maximum(0.20, 1.0 - selected * (fraction / selected.mean()))
        # Temper correlated preflop/postflop evidence to avoid false certainty.
        weights *= likelihood ** 0.65
    return weights


def _choose(obs, memory, evidence):
    cache = _samples(obs, memory)
    outcomes = cache["outcomes"]
    weights = _range_weights(obs, memory, cache, evidence)
    mass = weights.sum()
    equity = float(weights @ outcomes / mass)
    # Small prior limits decisions driven by a handful of heavily weighted runs.
    effective_n = mass * mass / float(weights @ weights)
    equity = (equity * effective_n + 1.0) / (effective_n + 2.0)
    call = min(obs.to_call, obs.my_stack)
    spr = min(obs.my_stack, obs.opp_stack) / max(1, obs.pot + call)
    realization = (0.025 if obs.is_button else 0.055) * spr / (spr + 2.0)
    if obs.street == Street.RIVER or obs.opp_stack == 0:
        realization = 0.0
    call_equity = max(0.0, equity - realization * (1.0 - equity))
    call_ev = call_equity * (obs.pot + call) - call
    best_action = Action.CHECK_CALL
    best_ev = call_ev
    if call and call_ev < 0:
        best_action, best_ev = Action.FOLD, 0.0

    stat = memory["stats"][int(obs.street)]
    response = stat["facing"]
    fold_rate = response[0] / sum(response)
    strength = None
    for action in (Action.RAISE_HALF, Action.RAISE_POT):
        if action not in obs.legal_actions:
            continue
        cost = obs.raise_to[action] - obs.my_bet
        opp_call = min(obs.opp_stack, max(0, obs.raise_to[action] - obs.opp_bet))
        # Larger bets induce more folds, unless observations say they never fold.
        size = opp_call / max(1, obs.pot + cost)
        adjusted_fold = min(0.90, fold_rate * (0.75 + size))
        if adjusted_fold <= 0.14:
            continue_probability = np.ones_like(outcomes)
        else:
            if strength is None:
                strength = _strength(cache, int(obs.street), obs.board)
            continue_probability = _selection(strength, 1.0 - adjusted_fold)
        continuing = weights * continue_probability
        continue_mass = float(continuing.sum())
        probability_fold = 1.0 - continue_mass / mass
        called_equity = float(continuing @ outcomes / continue_mass)
        called_equity = (called_equity * effective_n + 1) / (effective_n + 2)
        called_ev = called_equity * (obs.pot + cost + opp_call) - cost
        raise_ev = probability_fold * obs.pot + (1.0 - probability_fold) * called_ev
        # A one-street model overvalues marginal raises when more bets remain.
        margin = (0.035 if obs.street == Street.RIVER else 0.065) * (cost + opp_call)
        if raise_ev > best_ev + margin:
            best_action, best_ev = action, raise_ev
    return best_action


def decide(obs, memory):
    start = perf_counter()
    try:
        evidence = _history(obs, memory)
        action = _choose(obs, memory, evidence)
    except Exception:
        # Count this so development checks can detect a hidden strategy failure.
        memory["fallbacks"] = memory.get("fallbacks", 0) + 1
        action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
    if action not in obs.legal_actions:
        action = Action.CHECK_CALL if Action.CHECK_CALL in obs.legal_actions else obs.legal_actions[0]
    memory["last_obs"], memory["last_action"] = obs, action
    memory["decisions"] += 1
    memory["spent"] += perf_counter() - start
    return action


def finish_hand(result, memory):
    obs, action = memory["last_obs"], memory["last_action"]
    if obs is not None and obs.hand_index == result.hand_index:
        # The final opposing action is absent from our last observation.
        if action in (Action.RAISE_HALF, Action.RAISE_POT):
            if not result.went_to_showdown and result.winner == "me":
                _count(memory, obs.street, Action.FOLD, True)
            elif result.went_to_showdown:
                _count(memory, obs.street, Action.CHECK_CALL, True)
        elif (action == Action.CHECK_CALL and obs.to_call == 0 and
              obs.street == Street.RIVER and not obs.is_button and result.went_to_showdown):
            _count(memory, obs.street, Action.CHECK_CALL, False)
    memory["hands_played"] += 1
    memory["net_chips"] += result.my_net
    memory["cache"] = None
    if ((memory["hands_played"] % 100 == 0 or memory["hands_played"] == memory["num_hands"]) and
            os.environ.get("POCKETPILOT_DIAGNOSTICS") == "1"):
        print(f"PocketPilot diagnostics: strategy_seconds={memory['spent']:.4f} "
              f"decisions={memory['decisions']} fallbacks={memory['fallbacks']}", flush=True)
