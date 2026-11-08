"""Play fresh-process, duplicate matches against each bundled opponent.

Run: python tools/benchmark_bot.py --deals 1000 --seeds 7 29
Every half restarts both servers, matching the tournament's isolation.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judge_engine import HalfResult, combine_halves, format_report


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", default=["7", "29"])
    parser.add_argument("--opponents", nargs="+", default=["random", "always_half", "always_pot", "calling_station"])
    parser.add_argument("--output", default="benchmark-results")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for opponent in args.opponents:
        for seed in args.seeds:
            halves = []
            for half in (1, 2):
                name = f"{opponent}-{seed}-half{half}"
                processes, files = [], []
                ports = [free_port(), free_port()]
                while ports[1] == ports[0]:
                    ports[1] = free_port()
                opponent_path = (ROOT / "agent.py" if opponent == "self" else
                                 ROOT / "tools" / "stress_opponent.py" if opponent.startswith("stress_") else
                                 ROOT / "baseline-bots" / f"baseline_{opponent}.py")
                try:
                    for path, port, label in [(ROOT / "agent.py", ports[0], "bot"),
                                              (opponent_path, ports[1], "opponent")]:
                        log = open(output / f"{name}-{label}.log", "w", encoding="utf-8")
                        files.append(log)
                        env = dict(os.environ, PORT=str(port), PYTHONDONTWRITEBYTECODE="1",
                                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                   POCKETPILOT_DIAGNOSTICS="1")
                        env.pop("AGENT_TOKEN", None)
                        env.pop("STRESS_OPPONENT", None)
                        if label == "opponent" and opponent.startswith("stress_"):
                            env["STRESS_OPPONENT"] = opponent.removeprefix("stress_")
                        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                        processes.append(subprocess.Popen([sys.executable, str(path)], cwd=ROOT,
                                                          env=env, stdout=log, stderr=log, creationflags=flags))
                    result_path = output / f"{name}.json"
                    command = [sys.executable, str(ROOT / "judge_engine.py"), "--half", str(half),
                               "--deals", str(args.deals), "--seed", str(seed),
                               "--agent1", f"http://127.0.0.1:{ports[0]}",
                               "--agent2", f"http://127.0.0.1:{ports[1]}",
                               "--json", str(result_path)]
                    judge_env = dict(os.environ)
                    judge_env.pop("AGENT1_TOKEN", None)
                    judge_env.pop("AGENT2_TOKEN", None)
                    run = subprocess.run(command, cwd=ROOT, env=judge_env, capture_output=True, text=True, timeout=240)
                    (output / f"{name}.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
                    if run.returncode or not result_path.exists():
                        raise RuntimeError(run.stdout + run.stderr)
                    halves.append(HalfResult.from_dict(json.loads(result_path.read_text(encoding="utf-8"))))
                finally:
                    for process in processes:
                        process.terminate()
                    for process in processes:
                        process.wait(timeout=10)
                    for log in files:
                        log.close()
            result = combine_halves(*halves)
            report = format_report(result)
            print(f"\nOPPONENT: {opponent}, SEED: {seed}\n{report}", flush=True)
            payload = result.to_dict()
            payload.update(opponent=opponent, test_seed=str(seed))
            summary.append(payload)
            (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (output / f"{opponent}-{seed}.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
