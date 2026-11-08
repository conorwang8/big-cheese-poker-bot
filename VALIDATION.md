# PocketPilot validation

Team: **big-cheese**. Updated November 7, 2026.

This revision adds decaying opponent statistics, cautious learning from revealed river bets, separate cheap/expensive bluff reads, and conservative fold estimates for tiny re-raises.

**26 test cases passed** in the full suite. The official participant smoke check passed locally on this revision; see `validation/smoke-result.txt`.

## Direct strategy robustness tests

**52,000 played hands**, across 26 duplicate matches. Each match uses 1,000 deals, replayed with seats swapped. All 22 matches against the 11 scripted opponent styles won; both self-play controls drew. Against the earlier strategy, one seed won and one was inconclusive. There were no illegal actions, harness errors, or internal strategy fallbacks for either player.

The candidate faced 8,578 all-in decisions, calling 7,851 and folding 727. The counts describe behavior, not whether each choice was optimal.

The seeds below were evaluated after the strategy changes were fixed. Direct tests keep a generous configured strategy budget so timing does not alter the comparison. They do **not** enforce the official HTTP time bank. Opponents use only visible information and normal showdown results. These full-sample results preceded the final timing adjustment and equivalent calculation optimizations. A separate 2,000-hand duplicate comparison of the final revision with that snapshot had identical outcomes on every paired deal at the generous budget. The final revision was separately tested under the actual bank, below.

| Opponent | Seed | bb / duplicate deal | 95% interval | Verdict |
|---|---:|---:|---|---|
| self | 733 | +0.000 | [+0.000, +0.000] | CHOP |
| self | 1907 | +0.000 | [+0.000, +0.000] | CHOP |
| always_pot | 733 | +6.774 | [+5.220, +8.328] | WIN |
| always_pot | 1907 | +7.108 | [+5.571, +8.645] | WIN |
| river_bluffer | 733 | +20.338 | [+19.205, +21.472] | WIN |
| river_bluffer | 1907 | +19.608 | [+18.483, +20.732] | WIN |
| delayed_bluffer | 733 | +11.635 | [+10.163, +13.107] | WIN |
| delayed_bluffer | 1907 | +11.954 | [+10.508, +13.399] | WIN |
| inverse_bluffer | 733 | +10.454 | [+9.102, +11.805] | WIN |
| inverse_bluffer | 1907 | +11.027 | [+9.689, +12.365] | WIN |
| alternating | 733 | +2.651 | [+1.457, +3.845] | WIN |
| alternating | 1907 | +2.558 | [+1.442, +3.673] | WIN |
| switch_to_bluff | 733 | +5.689 | [+4.261, +7.118] | WIN |
| switch_to_bluff | 1907 | +6.036 | [+4.591, +7.481] | WIN |
| switch_to_value | 733 | +2.703 | [+1.791, +3.614] | WIN |
| switch_to_value | 1907 | +3.561 | [+2.656, +4.466] | WIN |
| tight_value | 733 | +2.321 | [+1.687, +2.955] | WIN |
| tight_value | 1907 | +2.177 | [+1.579, +2.775] | WIN |
| check_fold | 733 | +1.500 | [+1.500, +1.500] | WIN |
| check_fold | 1907 | +1.500 | [+1.500, +1.500] | WIN |
| check_raise_value | 733 | +2.361 | [+1.757, +2.964] | WIN |
| check_raise_value | 1907 | +1.500 | [+0.902, +2.097] | WIN |
| size_mixer | 733 | +2.989 | [+2.076, +3.901] | WIN |
| size_mixer | 1907 | +2.083 | [+1.178, +2.989] | WIN |
| reference | 733 | +0.596 | [+0.014, +1.178] | WIN |
| reference | 1907 | +0.341 | [-0.170, +0.851] | CHOP |

Intervals use the organizer's per-deal normal approximation (z=1.96); they are individual match intervals, without correction for testing multiple opponents. A duplicate deal includes two played hands. Self-play is a consistency check, not evidence of strength. Scripted adversaries are not equilibrium strategies, and finite tests do not prove the bot unexploitable.

## Official HTTP matches

**12,000 played hands on the final revision**, under the actual judge and 8-second time bank. Both servers restarted before every half. Five adversary matches won; self-play drew. Neither player had errors, forfeits, skipped decisions, or an exhausted time bank. PocketPilot diagnostics reported no internal fallbacks.

An earlier HTTP run exhausted the time bank against the river bluffer in one half (8.02 seconds charged, 85 skipped actions), despite winning the match. That run is preserved in `validation/matches-before-timing-fix.json`. The fix reduces samples earlier when projected computation is high, includes hand-end computation in the estimate, and removes redundant calculations. The complete final suite below was rerun after the fix.

| Opponent | Seed | bb / duplicate deal | 95% interval | Max bot time / half | Verdict |
|---|---:|---:|---|---:|---|
| stress_river_bluffer | 2903 | +19.183 | [+18.036, +20.329] | 7.024 s / 8 s | AGENT1_WIN |
| self | 2903 | -0.016 | [-0.759, +0.727] | 2.271 s / 8 s | CHOP |
| always_pot | 2903 | +7.218 | [+5.666, +8.771] | 1.823 s / 8 s | AGENT1_WIN |
| always_half | 2903 | +11.136 | [+9.853, +12.418] | 2.176 s / 8 s | AGENT1_WIN |
| stress_inverse_bluffer | 2903 | +11.636 | [+10.266, +13.006] | 2.433 s / 8 s | AGENT1_WIN |
| stress_switch_to_bluff | 2903 | +6.133 | [+4.692, +7.573] | 1.676 s / 8 s | AGENT1_WIN |

The official self-play score need not be exactly zero: adaptive sample counts depend on measured execution time, which varies between fresh processes. Windows timings are local measurements and do not guarantee the same margin on tournament hardware.

Raw results: `validation/stress-matches.json` (direct tests) and `validation/matches.json` (official HTTP matches). `validation/source-sha256.json` identifies this revision. The earlier eight baseline wins are preserved separately in `validation/matches-v1-baselines.json` with `validation/source-v1-sha256.json`; they are historical results, not additional tests of this revision.

## Slumbot

Slumbot uses two-card Texas Hold'em, with 200-big-blind stacks and no-limit betting. This event uses short-deck five-card pot-limit Omaha at 40 big blinds, with different hand rankings. After reviewing the mismatch, the user chose to focus on the datathon bot. No Slumbot match was run. Source: [Slumbot public game and API documentation](https://slumbot.com/).

## Container check

The organizer's tools/check_build.sh was attempted, but it stopped with: "Docker isn't running". Docker Desktop was asked to start, but its Linux engine remained unavailable. The local HTTP smoke check and all matches passed; the Linux container build, image-size check, and restricted-container smoke test still need to be run once Docker is available.

The Linux container build, size check, and restricted-container smoke test remain unverified. The game, evaluator, judge, official smoke checker, Docker recipes, and server routes are unchanged from the participant bundle. No runtime network access or API key is required. The team repository is [conorwang8/big-cheese-poker-bot](https://github.com/conorwang8/big-cheese-poker-bot); uploading the source does not submit it to the event.

## Reproduce

```text
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tools/stress_bot.py --deals 1000 --seeds 733 1907
python tools/stress_bot.py --deals 1000 --seeds 733 1907 --opponents reference --reference validation/reference_strategy_v1.py
python tools/stress_bot.py --deals 1000 --seeds 733 --opponents reference --reference validation/strategy_before_timing_fix.py
python tools/benchmark_bot.py --deals 1000 --seeds 2903 --opponents self always_pot always_half stress_river_bluffer stress_inverse_bluffer stress_switch_to_bluff
./tools/check_build.sh
```

Run match suites sequentially for meaningful timings. The final command needs Git Bash or WSL and a working Docker Desktop Linux engine.
