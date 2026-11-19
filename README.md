# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nunfixx/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nunfixx/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of GTA VI's 55 PlayStation trophies, the earn rate (percent of tracked players) at 2026-11-26T00:00:00Z — one week after launch — as five quantiles (5%, 25%, 50%, 75%, 95%) on the logit scale, from the Exophase read history in each case file.

**Model.** Per trophy, OLS of logit(earn rate) on `ln(1 + effective hours since launch)` over the last 24 h of distinct reads, read at the target; spread = random-walk term (`0.5 · √horizon in days`) plus the fit's residual. The one idea that beats the house curve baseline: *effective* hours, not calendar hours. Launch was Thursday 19 Nov 2026, so the forecast window (Sat 21 Nov 11:30 UTC → Thu 26 Nov 00:00 UTC) contains the entire first weekend, when players accumulate playtime roughly twice as fast as on weekdays — weekend hours count 2×. A flat history (practice cases: every read identical) means the next refresh will dilute rates as the denominator explodes, so the median is cut (tapered to no cut at 100%, where fresh players arrive as earners too) and the spread widened. Medians are clamped to ±3 logit units of the last read rate and to a per-tier floor.

**Validation.** Built and tuned against a launch simulator calibrated on the real launch-night data (638 players, first refresh to 3,868 at 04:30 UTC 19 Nov) and the live Exophase page: 216 final-like cases across earn-speed regimes, weekend strengths, and weekend-window definitions. Mean pinball loss on the logit scale: **0.39** (shipped model) vs **0.42** for the house curve baseline and **1.22** for the flat house baseline; 90% coverage 0.98, typical miss 4.3 points. Practice-like cases: **0.52** vs 1.45 (curve) and 1.58 (baseline). Ablated and rejected: E/N decomposition (0.74), quadratic logit trend (0.68), and blends with either — the simple trend on effective time wins because it is the lowest-variance estimator of a smooth launch curve.

## Run it

```sh
MAKEFUN_INPUT_DIR=/path/to/cases MAKEFUN_OUTPUT_DIR=/path/to/out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` in the input directory (or `/input` by default) and writes one `<id>.out` per case to the output directory (or `/output` by default). Deterministic: two runs are byte-identical. Runs in ~0.1 s per case, far under the 8 s limit.

---

<sub>Entered on [make.fun](https://make.fun/?ref=pqrphsn3&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
