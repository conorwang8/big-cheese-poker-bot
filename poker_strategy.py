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
TRIPLES_BY_SIZE = {size: np.array(list(combinations(range(size), 3)), dtype=np.intp)
                   for size in (3, 4, 5)}
TRIPLES = TRIPLES_BY_SIZE[5]
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
    triples = TRIPLES_BY_SIZE[boards.shape[1]]
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
        context_nodes={}, context_pools={}, context_hand=-1, context_seen=0, context_evidence={},
        hands_played=0, rng=np.random.default_rng(20261107),
        stats=[{"open": [7.0, 3.0], "facing": [3.0, 6.0, 1.0]} for _ in range(4)],
        active_hand=-1, seen=0, cache=None, last_obs=None, last_action=None,
        last_evidence=None, river_reads=[[0.0, 0.0] for _ in range(4)],
        spent=0.0, target_seconds=max(0.0, config.time_bank_seconds) * 0.20,
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
    _context_history(obs, memory)
    return evidence


def _samples(obs, memory):
    """Reuse the same samples within a street so raises don't change the dice."""
    key = (obs.hand_index, obs.hole_cards, obs.board)
    cache = memory["cache"]
    if cache is not None and cache["key"] == key:
        return cache
    # React to projected cost before spending the reserve. HTTP bookkeeping
    # and scheduling can consume more time than the numerical strategy itself.
    progress = max(0.06, (obs.hand_index + 1) / memory["num_hands"])
    projected = memory["spent"] / progress
    target = memory["target_seconds"]
    n = 128 if len(obs.board) < 5 else 256
    if projected > target:
        n //= 2
    if projected > 2 * target:
        n = max(32, n // 2)
    if projected > 4 * target:
        n = 32
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
             "villain_scores": villain,
             "percentiles": {}}
    if len(obs.board) == 5:
        cache["percentiles"][3] = percentiles(villain)
    memory["cache"] = cache
    return cache


def _strength(cache, street, board):
    if street not in cache["percentiles"]:
        if street == int(Street.PREFLOP):
            values = preflop_quality(cache["opponents"])
        else:
            count = (0, 3, 4, 5)[street]
            values = score_batch(cache["opponents"], np.array([board[:count]]))
        cache["percentiles"][street] = percentiles(values)
    return cache["percentiles"][street]


def _selection(strength, fraction):
    """Soft top-range selection, with some bluffs and imperfect decisions."""
    fraction = float(np.clip(fraction, 0.04, 1.0))
    if fraction >= 0.86:
        return np.ones_like(strength)
    return 0.06 + 0.94 / (1.0 + np.exp(-10.0 * (strength - (1.0 - fraction))))


def _range_weights(obs, memory, cache, evidence):
    weights = np.ones(len(cache['outcomes']))
    for street, (action, key) in memory['context_evidence'].items():
        rates = _context_rates(memory, key)
        strength = _strength(cache, street, obs.board)
        raising = float(rates[2] + rates[3])
        top = _selection(strength, raising)
        if action == Action.RAISE_POT:
            # A mixture prevents bet-size stereotypes from excluding bluffs.
            likelihood = 0.55 * top + 0.45 * _selection(strength, float(rates[3]))
        elif action == Action.RAISE_HALF:
            likelihood = np.maximum(0.06, top - 0.45 * _selection(strength, float(rates[3])))
        elif key[2] >= 2:
            surviving = _selection(strength, 1.0-float(rates[0]))
            likelihood = np.maximum(0.08, surviving - 0.65 * top)
        elif key[2] == 0:
            surviving = _selection(strength, 1.0-float(rates[0]))
            likelihood = np.maximum(0.08, surviving - 0.65 * top)
        else:
            likelihood = np.maximum(0.20, 1.0-top)
        if street == int(Street.RIVER) and action in (Action.RAISE_HALF, Action.RAISE_POT):
            likelihood = _river_raise_likelihood(obs, memory, key[2] >= 2, strength, raising, likelihood)
        # Earlier streets have useful evidence, but correlated board evolution
        # and draws make that evidence less precise than current-street action.
        weights *= likelihood ** (0.65 if street == int(obs.street) else 0.35)
    return weights



def _river_bucket(obs, facing):
    # A cheap bluff is not evidence that a player bluffs an expensive shove.
    return 2 * int(facing) + int(obs.to_call >= 6 * obs.big_blind)


def _river_raise_likelihood(obs, memory, facing, strength, fraction, default):
    weak, strong = memory["river_reads"][_river_bucket(obs, facing)]
    observed = weak + strong
    if observed < 3:
        return default
    bluff_fraction = min(0.60, (weak + 1.0) / (observed + 7.0))
    value = 0.01 + 0.99 / (1.0 + np.exp(-18.0 * (strength - (1.0 - fraction))))
    bluff = 0.03 + 0.97 / (1.0 + np.exp(12.0 * (strength - 0.35)))
    learned = ((1.0 - bluff_fraction) * value / value.mean() +
               bluff_fraction * bluff / bluff.mean())
    confidence = min(0.80, observed / (observed + 8.0))
    return (1.0 - confidence) * default / default.mean() + confidence * learned


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
    previous_raise_to = None
    for action in (Action.RAISE_HALF, Action.RAISE_POT):
        if action not in obs.legal_actions:
            continue
        raise_to = obs.raise_to[action]
        if raise_to == previous_raise_to:
            continue  # Both raise buttons sometimes name the same all-in.
        previous_raise_to = raise_to
        cost = raise_to - obs.my_bet
        opp_call = min(obs.opp_stack, max(0, raise_to - obs.opp_bet))
        # Larger bets induce more folds, unless observations say they never fold.
        size = opp_call / max(1, obs.pot + cost)
        adjusted_fold = min(0.90, _context_fold_rate(obs, memory, action))
        if adjusted_fold <= 0.14 or opp_call / max(1, obs.pot + cost + opp_call) <= 0.08:
            probability_fold = 0.0
            called_equity = equity
        else:
            if strength is None:
                strength = _strength(cache, int(obs.street), obs.board)
            ranks = _weighted_percentiles(strength, weights) if CONDITIONAL_CONTINUATION else strength
            continue_probability = _selection(ranks, 1.0 - adjusted_fold)
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
        memory["last_evidence"] = evidence
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
    start = perf_counter()
    _context_finish(result, memory)
    obs, action = memory["last_obs"], memory["last_action"]
    # Forget old behavior gradually, retaining the original small-sample priors.
    for stat in memory["stats"]:
        for key, priors in (("open", (7.0, 3.0)), ("facing", (3.0, 6.0, 1.0))):
            stat[key][:] = [prior + 0.99 * (value - prior)
                            for value, prior in zip(stat[key], priors)]
    for reads in memory["river_reads"]:
        reads[0] *= 0.995
        reads[1] *= 0.995
    if obs is not None and obs.hand_index == result.hand_index:
        cache = memory["cache"]
        history = obs.action_history
        # Only classify a revealed river bet that we directly called. A later
        # call of our re-raise would select stronger hands and bias these reads.
        if (result.went_to_showdown and result.opp_hole is not None and
                obs.street == Street.RIVER and obs.to_call > 0 and
                action == Action.CHECK_CALL and history and not history[-1].is_me and
                history[-1].street == Street.RIVER and
                history[-1].action in (Action.RAISE_HALF, Action.RAISE_POT) and
                cache is not None and cache["key"] == (obs.hand_index, obs.hole_cards, obs.board)):
            actual = score_batch(np.array([result.opp_hole]), np.array([obs.board]))[0]
            scores = cache["villain_scores"]
            percentile = float(((scores < actual) + 0.5 * (scores == actual)).mean())
            facing = memory["last_evidence"].get(int(Street.RIVER), (None, False))[1]
            bucket = memory["river_reads"][_river_bucket(obs, facing)]
            if percentile < 0.40:
                bucket[0] += 1.0
            elif percentile > 0.65:
                bucket[1] += 1.0
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
    memory["spent"] += perf_counter() - start


# Context-specific opponent modeling; learned state lives in memory.
CONDITIONAL_CONTINUATION = False

def _context_key(street, opponent_button, price, pot, raised_before, first=False):
    # 0: small-blind opening; 1: free option/check; 2: face first bet;
    # 3: face a raise after already betting/raising on this street.
    kind = 0 if first else 1 if price <= 0 else 3 if raised_before else 2
    size = int(price / max(1, pot + price) >= 0.29) if kind >= 2 else 0
    return int(street), bool(opponent_button), kind, size


def _context_note(memory, key, action):
    index = (Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT).index(action)
    pool_key = (key[0], key[2])
    if key not in memory['context_nodes']:
        memory['context_nodes'][key] = np.zeros(4)
    if pool_key not in memory['context_pools']:
        memory['context_pools'][pool_key] = np.zeros(4)
    memory['context_nodes'][key][index] += 1.0
    memory['context_pools'][pool_key][index] += 1.0


def _context_rates(memory, key):
    if key[2] == 0:
        prior = np.array((0.10, 0.45, 0.30, 0.15))
    elif key[2] == 1:
        prior = np.array((0.0, 0.65, 0.25, 0.10))
    elif key[2] == 2:
        prior = np.array((0.30, 0.55, 0.10, 0.05))
    else:
        prior = np.array((0.20, 0.65, 0.08, 0.07))
    pool = memory['context_pools'].get((key[0], key[2]))
    if pool is not None:
        prior = (pool + 8.0 * prior) / (pool.sum() + 8.0)
    local = memory['context_nodes'].get(key)
    if local is None:
        return prior
    return (local + 12.0 * prior) / (local.sum() + 12.0)


def _response_context(obs, action):
    cost = obs.raise_to[action] - obs.my_bet
    price = min(obs.opp_stack, max(0, obs.raise_to[action]-obs.opp_bet))
    raised = any(not r.is_me and r.street == obs.street and
                 r.action in (Action.RAISE_HALF, Action.RAISE_POT) for r in obs.action_history)
    return _context_key(obs.street, not obs.is_button, price, obs.pot+cost, raised)


def _context_fold_rate(obs, memory, action):
    return float(_context_rates(memory, _response_context(obs, action))[0])


def _weighted_percentiles(values, weights):
    order = np.argsort(values)
    ordered = values[order]
    prefix = np.r_[0.0, np.cumsum(weights[order])]
    left = np.searchsorted(ordered, values, side='left')
    right = np.searchsorted(ordered, values, side='right')
    return (prefix[left] + prefix[right]) / (2.0 * prefix[-1])


def _context_history(obs, memory):
    if memory['context_hand'] != obs.hand_index:
        memory['context_hand'] = obs.hand_index
        memory['context_seen'] = 0
    mine = obs.big_blind // 2 if obs.is_button else obs.big_blind
    theirs = obs.big_blind if obs.is_button else obs.big_blind // 2
    pot, street, raised = 3 * obs.big_blind // 2, Street.PREFLOP, False
    detailed = {}
    for i, record in enumerate(obs.action_history):
        if record.street != street:
            street, mine, theirs, raised = record.street, 0, 0, False
        if record.is_me:
            mine = record.street_total
        else:
            key = _context_key(street, not obs.is_button, mine-theirs, pot, raised,
                               first=i == 0 and street == Street.PREFLOP)
            if i >= memory['context_seen']:
                _context_note(memory, key, record.action)
            prior = detailed.get(int(street))
            is_raise = record.action in (Action.RAISE_HALF, Action.RAISE_POT)
            if prior is None or is_raise or prior[0] == Action.CHECK_CALL:
                detailed[int(street)] = (record.action, key)
            raised |= is_raise
            theirs = record.street_total
        pot += record.amount
    memory['context_seen'] = len(obs.action_history)
    memory['context_evidence'] = detailed


def _context_finish(result, memory):
    obs, action = memory['last_obs'], memory['last_action']
    if obs is not None and obs.hand_index == result.hand_index:
        if action in (Action.RAISE_HALF, Action.RAISE_POT):
            if result.went_to_showdown:
                _context_note(memory, _response_context(obs, action), Action.CHECK_CALL)
            elif result.winner == 'me':
                _context_note(memory, _response_context(obs, action), Action.FOLD)
        elif (action == Action.CHECK_CALL and obs.to_call == 0 and obs.street == Street.RIVER
              and not obs.is_button and result.went_to_showdown):
            key = _context_key(Street.RIVER, True, 0, obs.pot, False)
            _context_note(memory, key, Action.CHECK_CALL)
    elif not result.went_to_showdown and result.winner == 'me' and not result.board:
        # Opponent folded their small blind; we never received an observation.
        _context_note(memory, (0, True, 0, 0), Action.FOLD)
    for counts in memory['context_nodes'].values():
        counts *= 0.995
    for counts in memory['context_pools'].values():
        counts *= 0.995
