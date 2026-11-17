# GTA VI Twitch CCU forecaster

[![make.fun](https://img.shields.io/badge/make.fun-bounty-blue)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)

Forecasts Twitch concurrent viewers in the **Grand Theft Auto VI** category,
hour by hour, for the 24 UTC hours of release day (**2026-11-19**), from
everything recorded up to the end of hour 1 — as five quantiles per hour.

Bounty: [gta6-twitch-ccu-forecast-01 on make.fun](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
— $250 over 20 places. This is the **round-2** entry: a better forecaster
than the round-1 entry in the same repo family (round 1: LOO 0.201 /
0.177; this entry: LOO 0.180 / 0.164 — see Validation below).

## How to run

```sh
python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` in `python:3.13-slim`, standard
library only, no network. `main.py` reads every `<id>.in` in `/input` (or
`$MAKEFUN_INPUT_DIR`) and writes one `<id>.out` per case to `/output` (or
`$MAKEFUN_OUTPUT_DIR`):

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Output format per case: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75,
q95]}, ...]}`, exactly 23 rows, hours 2..24. Deterministic, ~0.06 s per
case (limit 8 s).

## Model

Each past launch (a *comparable*) carries its full release-day curve. For
a launch, the hour-1 print fixes the scale and the rest of the day is a
shape: the ratios `viewers_h / viewers_1`. GTA VI unlocks at local
midnight, rolling around the world, so hour 1 is small and the curve ramps
hard into US prime time. Only comparables whose own release day already
had a real hour-1 print share that shape.

Per forecast hour, two ridge-shrunk regressions of the log-ratio are fit
on the shape pool:

- **base:** `log-ratio ~ log(print_1) + jump`, where `jump =
  log(print_1 / hour-0 viewers)`;
- **extended:** `log-ratio ~ log(print_1) + jump + preweek`, where
  `preweek` is the mean log viewers in the category 14–7 days before
  release — the category's baseline popularity before launch week.

What the practice data shows: bigger hour-1 prints ramp less (slope about
−0.05 early, −0.35 at the evening peak); a bigger jump into hour 1 predicts
a stronger ramp; and a category that was already popular two weeks out
ramps more (the preweek slope is positive at every hour). Early-access
launches, whose categories were quiet two weeks out, ramp least.

The fitted coefficients are lightly smoothed across neighbouring hours
(the diurnal shape is smooth: ramp, peak, decline), except at the edges
(hours 2 and 24 keep their own fit). The final centre blends the two fits,
75% extended / 25% base, hedging the third predictor's extrapolation. The
spread is the empirical SD of the pool's log-ratios at that hour (floored
0.25, capped 0.9, scaled 0.85, widened a little for hours 2–8 and narrowed
a little for hours 17–24). Quantiles are log-normal, rounded to whole
viewers, at least 1, forced non-decreasing. The slope of the minute reads
inside hour 1 nudges the centres by at most 0.10 log units.

Fallbacks: no usable comparables → median ramp shape with a wide spread;
missing hour-0 viewers → pool median jump; missing pre-release history →
pool median preweek; missing/zero hour-1 print → floored at 1.

## Validation

Leave-one-out over the comparables' own release days, scored with the
contest's pinball loss on the log scale:

| population | this entry | round 1 | house baseline |
|---|---|---|---|
| all 37 comparables | 0.629 | 0.695 | 0.930 |
| midnight-like (26) | **0.180** | 0.201 | 0.284 |
| closest analogues (16) | **0.164** | 0.177 | 0.288 |

Ideas tried and kept only when they validated better on LOO: the preweek
predictor (+blend), coefficient smoothing across hours with un-smoothed
edges, ridge shrinkage, weak-tier launches in the pool, and per-band
spread multipliers. Ideas tried and dropped: similarity weighting by
scale/jump/platform, channels-based predictors, pre-release diurnal tilt,
polynomial terms, empirical residual quantiles, and an early-peak /
evening-peak mixture — none of them beat the model above on LOO.

## Files

- `main.py` — I/O driver: reads `/input/*.in`, writes `/output/*.out`.
- `forecast.py` — the model (standard library only).
- `LICENSE` — MIT.
- `.makefun/owner.json` — entry owner wallet.
