"""Serve a synthetic opponent through the unmodified participant HTTP routes.

PowerShell example:
  $env:STRESS_OPPONENT='inverse_bluffer'
  $env:PORT='5009'
  python tools/stress_opponent.py

The official judge can then play it exactly like a bundled baseline.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agent
import uvicorn
import stress_opponents

OPPONENT = os.environ.get("STRESS_OPPONENT", "always_pot")
if OPPONENT not in stress_opponents.NAMES:
    raise ValueError(f"STRESS_OPPONENT must be one of {stress_opponents.NAMES}")


def new_half(config, memory):
    stress_opponents.new_half(config, memory)


def choose_action(obs, memory):
    return stress_opponents.choose_action(OPPONENT, obs, memory)


def on_hand_end(result, memory):
    stress_opponents.on_hand_end(result, memory)


agent.PARTICIPANT = "Synthetic robustness tests"
agent.AGENT_NAME = OPPONENT
agent.new_half = new_half
agent.choose_action = choose_action
agent.on_hand_end = on_hand_end

if __name__ == "__main__":
    uvicorn.run(agent.app, host="0.0.0.0", port=int(os.environ.get("PORT", "5009")),
                log_level="warning", access_log=False)
