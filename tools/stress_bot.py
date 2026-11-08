"""Paired in-process strategy tests, including self-play and bluff adversaries.

Examples:
  python tools/stress_bot.py --deals 1000 --seeds 101 202
  python tools/stress_bot.py --opponents self always_pot inverse_bluffer
  python tools/stress_bot.py --candidate /path/to/strategy.py --opponents reference --reference /path/to/old_strategy.py

By default a generous configured time bank keeps sample scheduling fixed for
repeatable strategy comparisons. This does NOT test the official HTTP time
bank, startup limit, Docker isolation, or prove equilibrium/unexploitability.
Use the official judge separately for those runtime limits. Self-play checks
symmetry and robustness; a draw against itself is not evidence of strength.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from game import Action, AgentConfig, BIG_BLIND, CHOP_Z, Game, make_deal
from judge_engine import coerce_action
import poker_strategy
import stress_opponents


def load_strategy(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "initialize"):
        module.initialize()
    return module


@dataclass
class Player:
    name: str
    module: object | None = None
    memory: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)
    cpu_seconds: float = 0.0
    end_seconds: float = 0.0
    actions: Counter = field(default_factory=Counter)
    all_in_facing: int = 0
    all_in_calls: int = 0
    all_in_folds: int = 0

    def start(self, config):
        if self.module is None:
            stress_opponents.new_half(config, self.memory)
        else:
            function = getattr(self.module, "start_half", None) or self.module.new_half
            function(config, self.memory)

    def choose(self, obs):
        begin, cpu_begin = time.perf_counter(), time.process_time()
        try:
            if self.module is None:
                raw = stress_opponents.choose_action(self.name, obs, self.memory)
            else:
                function = getattr(self.module, "decide", None) or self.module.choose_action
                raw = function(obs, self.memory)
            action = coerce_action(raw)
            if action not in obs.legal_actions:
                raise ValueError(f"Illegal action {raw!r}; legal={obs.legal_actions}")
        except Exception:
            self.errors.append(traceback.format_exc())
            action = Action.FOLD if obs.to_call else Action.CHECK_CALL
        self.cpu_seconds += time.process_time() - cpu_begin
        self.elapsed.append(time.perf_counter() - begin)
        self.actions[f"{obs.street.name}:{action.name}"] += 1
        if obs.opp_stack == 0 and obs.to_call:
            self.all_in_facing += 1
            self.all_in_calls += int(action == Action.CHECK_CALL)
            self.all_in_folds += int(action == Action.FOLD)
        return action

    def finish(self, result):
        begin = time.perf_counter()
        try:
            if self.module is None:
                stress_opponents.on_hand_end(result, self.memory)
            else:
                function = getattr(self.module, "finish_hand", None) or self.module.on_hand_end
                function(result, self.memory)
        except Exception:
            self.errors.append(traceback.format_exc())
        self.end_seconds += time.perf_counter() - begin

    def report(self):
        times = sorted(self.elapsed)
        return {
            "decisions": len(times), "errors": len(self.errors),
            "error_messages": self.errors[:5],
            "internal_fallbacks": self.memory.get("fallbacks", 0),
            "decision_cpu_seconds": self.cpu_seconds,
            "decision_wall_seconds": sum(times),
            "hand_end_wall_seconds": self.end_seconds,
            "mean_decision_ms": 1000 * statistics.mean(times) if times else 0,
            "p99_decision_ms": 1000 * times[min(len(times)-1, int(0.99 * len(times)))] if times else 0,
            "max_decision_ms": 1000 * max(times, default=0),
            "action_counts": dict(self.actions),
            "facing_all_in": self.all_in_facing,
            "all_in_calls": self.all_in_calls, "all_in_folds": self.all_in_folds,
        }


def play_pair(candidate, opponent_name, reference, seed, deals, time_bank):
    halves, details = [], []
    config = AgentConfig(num_hands=deals, time_bank_seconds=time_bank)
    for half in (1, 2):
        # Every player/half gets a new memory. Only immutable public lookup
        # tables are reused in-process. No hand result crosses a half boundary.
        hero = Player("candidate", candidate)
        if opponent_name == "self":
            villain = Player("self", candidate)
        elif opponent_name == "reference":
            villain = Player("reference", reference)
        else:
            villain = Player(opponent_name)
        hero.start(config)
        villain.start(config)
        hero_seat = half - 1
        players = [hero, villain] if hero_seat == 0 else [villain, hero]
        nets = []
        for hand in range(deals):
            game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
            decisions = 0
            while not game.is_terminal:
                seat = game.to_act
                game.apply(players[seat].choose(game.observation(seat)))
                decisions += 1
                if decisions > 128:
                    raise RuntimeError("Hand failed to terminate within the legal action bound")
            result = game.hand_result(hero_seat)
            nets.append(result.my_net)
            # HandResult reveals opposing cards only at a real showdown.
            for seat, player in enumerate(players):
                player.finish(game.hand_result(seat))
        halves.append(nets)
        details.append({"half": half, "candidate_seat": hero_seat,
                        "candidate_chips": sum(nets), "candidate": hero.report(),
                        "opponent": villain.report()})
    paired = [(a+b) / BIG_BLIND for a, b in zip(*halves)]
    mean = statistics.mean(paired)
    error = statistics.stdev(paired) / math.sqrt(deals) if deals > 1 else 0.0
    interval = [mean-CHOP_Z*error, mean+CHOP_Z*error]
    verdict = "WIN" if interval[0] > 0 else "LOSS" if interval[1] < 0 else "CHOP"
    if any(h[side]["errors"] or h[side]["internal_fallbacks"] for h in details for side in ("candidate", "opponent")):
        verdict = "ERROR"
    return {"opponent": opponent_name, "seed": seed, "deals": deals,
            "hands_played": 2 * deals, "mean_bb_per_duplicate_deal": mean,
            "ci95": interval, "verdict": verdict, "halves": details,
            "paired_bb": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", default=["101", "202"])
    parser.add_argument("--opponents", nargs="+", choices=("self", "reference", *stress_opponents.NAMES),
                        default=["self", *stress_opponents.NAMES])
    parser.add_argument("--candidate", type=Path, default=ROOT / "poker_strategy.py")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--time-bank", type=float, default=1e9,
                        help="Configured strategy budget only, NOT enforced as an HTTP bank. Default fixes full sample scheduling.")
    parser.add_argument("--output", type=Path, default=Path("stress-results.json"))
    args = parser.parse_args()
    if args.deals < 2:
        parser.error("--deals must be at least 2")
    if "reference" in args.opponents and args.reference is None:
        parser.error("--opponents reference needs --reference")
    poker_strategy.initialize()
    candidate = load_strategy(args.candidate, "stress_candidate")
    reference = load_strategy(args.reference, "stress_reference") if args.reference else None
    summary = {
        "kind": "in_process_strategy_stress_test",
        "candidate": str(args.candidate.resolve()),
        "reference": str(args.reference.resolve()) if args.reference else None,
        "configured_time_bank_seconds": args.time_bank,
        "timing_note": "Direct function CPU/wall measurements, excluding HTTP. No official time bank enforced; use the official judge for runtime eligibility.",
        "isolation_note": "Independent memory per player and half; public immutable lookup tables shared. No hidden cards passed to strategies.",
        "self_play_note": "Self-play checks symmetry and stability, not strength or unexploitability.",
        "matches": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for opponent in args.opponents:
        for seed in args.seeds:
            result = play_pair(candidate, opponent, reference, seed, args.deals, args.time_bank)
            summary["matches"].append(result)
            args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            max_ms = max(h["candidate"]["max_decision_ms"] for h in result["halves"])
            print(f"{opponent:20s} seed={seed:>5s} {result['verdict']:5s} "
                  f"{result['mean_bb_per_duplicate_deal']:+.3f} bb/deal "
                  f"CI [{result['ci95'][0]:+.3f}, {result['ci95'][1]:+.3f}] "
                  f"max decision={max_ms:.2f} ms", flush=True)
    return int(any(r["verdict"] == "ERROR" for r in summary["matches"]))


if __name__ == "__main__":
    raise SystemExit(main())
