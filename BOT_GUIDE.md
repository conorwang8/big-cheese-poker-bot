# PocketPilot

Team: **big-cheese**. Bot: **PocketPilot**.

This bot uses the TAMU Datathon Heads Up participant bundle. It needs no API
key, internet connection, GPU, external model, or runtime file writes.

## Run it

Open a terminal at this repository's root. Use Python 3.10 or newer
(the tournament uses Python 3.12):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python agent.py
```

If PowerShell blocks environment activation, use `.\.venv\Scripts\python.exe`
in place of `python` in subsequent commands; activation is optional.

The server listens on port 5008. To play a practice match, open two more
terminals in this folder, activate the same environment, and run:

```powershell
# Terminal 2
python baseline-bots/baseline_random.py
```

```powershell
# Terminal 3
python judge_engine.py --deals 1000 --seed 7
```

The judge handles the two halves and swaps seats. The final report includes
the verdict, thinking time, and errors. `PORT` and `AGENT_TOKEN` are supported
by the unchanged participant server.

## Strategy

- **Exact rules:** An optimized NumPy lookup table reproduces the bundle's
  reference evaluator, including flushes above full houses, trips above
  straights, the ace-low short-deck straight, and exactly two hole cards.
- **Unseen-card simulation:** Sample possible opposing hands and future
  boards without replacement. Compare actual five-card Omaha scores, and
  count tied outcomes as half a win. Reuse samples when facing a re-raise
  on the same street so a changed random sample does not drive a decision.
- **Opponent estimates:** Learn how often this opponent bets, calls,
  raises, and folds. Weight possible hands using only actions and cards
  already visible. Older observations gradually lose weight so the bot can
  react when an opponent changes tactics. These are approximate ranges,
  not solved poker strategy.
- **Bluff evidence:** After directly calling a river bet, use the revealed
  hand to cautiously update the opponent's value/bluff mix. Keep cheap bets
  separate from expensive bets, and initial bets separate from re-raises.
  Never infer hidden cards from a fold or treat an opponent's call of our
  re-raise as evidence about their original betting range.
- **Pot odds and sizing:** Compare folding, calling, and each legal raise
  using the engine's actual chip amounts, estimated fold probability, and
  equity against an estimated continuing range. Allow for position and
  future betting when evaluating marginal calls and raises. All-in calls
  use the actual price: owing 26 chips into a 134-chip pot requires 16.25%
  equity. Tiny re-raises assume no fold equity when the opponent gets an
  extremely cheap call.
- **Time reserve:** Reduce sample counts when measured strategy time
  approaches a conservative share of the configured time bank. HTTP overhead
  is measured by the official judge; internal timing alone cannot guarantee
  a tournament time-bank margin.
- **Fresh halves:** All random state, samples, opponent statistics, and
  timing counters live in `memory` and reset each half. Only immutable,
  publicly derived scoring tables are shared at module scope.

`agent.py` contains the required entry point and team identity. The strategy
is in `poker_strategy.py`, which must be included in your submission. The
reference game, evaluator, referee, Docker recipes, and server routes are
unchanged.

## Reproduce evaluation

```powershell
python -m pytest -q
python tools/benchmark_bot.py --deals 1000 --seeds 7 29
```

The benchmark runner restarts both bots between halves and saves the official
judge's JSON results and reports in `benchmark-results/`. It tests the random,
always-half-pot, always-pot, and calling-station opponents. Run matches one at
a time for meaningful timing. These simple opponents do not establish an
edge against other teams or guarantee tournament results.

The extended test tools cover self-play and 11 scripted opponent styles:
constant maximum raises, delayed and selective bluffs, tight value betting,
check-raises, mixed bet sizes, and changes of tactics. Every deal is replayed
with seats swapped. The direct strategy tests use generous computation time
for repeatable comparisons; run the HTTP tests separately to check the actual
8-second time bank. A self-play draw checks consistency, not playing strength.

```powershell
# Direct strategy tests (no official HTTP time-bank enforcement)
python tools/stress_bot.py --deals 1000 --seeds 733 1907

# Official judge, fresh servers for each half
python tools/benchmark_bot.py --deals 1000 --seeds 2903 --opponents self always_pot always_half stress_river_bluffer stress_inverse_bluffer stress_switch_to_bluff

# Compare this revision with the preserved earlier strategy
python tools/stress_bot.py --deals 1000 --seeds 733 1907 --opponents reference --reference validation/reference_strategy_v1.py
```

Raw match results are in `validation/`. Slumbot uses two-card
Texas Hold'em, so it cannot serve as a direct opponent for this short-deck
five-card Omaha submission; no Slumbot match was run.

From Git Bash or WSL, with Docker Desktop running:

```bash
./tools/check_build.sh
```

This is the organizer's image-build and restricted-container smoke check.
The first build downloads the shared CPU PyTorch base even though PocketPilot
only needs NumPy and a small HTTP parser. Keep `requirements.txt` and `agent.py`
at the repository root.

## Submit

1. Confirm your team is registered as **big-cheese**.
2. Use [conorwang8/big-cheese-poker-bot](https://github.com/conorwang8/big-cheese-poker-bot)
   as the team repository. This repository has `agent.py`, `poker_strategy.py`,
   and `requirements.txt` at its root, as required by the tournament.
3. Run the tests, full practice matches, and Docker build check on your final
   revision. Check that neither half exhausts its time bank or reports errors.
4. Commit and push the source. Get the full 40-character commit with
   `git rev-parse HEAD`.
5. Enter the registered team name, public repository URL, and full pushed
   commit in the event's submission form.

The source is based on the [organizer bundle](https://github.com/tamu-datathon-org/heads-up-bundle).
`origin` points to `conorwang8/big-cheese-poker-bot`.
Uploading the source to GitHub does not register or submit your team.
