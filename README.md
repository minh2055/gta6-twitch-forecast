# GTA VI Twitch CCU forecaster

![gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/minh2055/gta6-twitch-forecast/rank.svg)
![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/minh2055/gta6-twitch-forecast/reward.svg)

Entry for the make.fun bounty [gta6-twitch-ccu-forecast-01](https://make.fun/c/gta6-twitch-ccu-forecast-01):
forecast, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch over the 24 UTC hours of its release day, 2026-11-19, from everything
recorded up to the end of its first hour.

## How it works

The program reads one case file (`/input/<id>.in`) and writes one forecast
(`/output/<id>.out`): for hours 2..24, five quantiles (5%, 25%, 50%, 75%,
95%) of that hour's mean Twitch viewers in the Grand Theft Auto VI
category.

GTA VI unlocks at local midnight, rolling — New Zealand first, then Europe,
then the US East at 05:00Z and Pacific at 08:00Z — so hour 1 (00:00–01:00
UTC) is before most of the world unlocks: its print is low and viewership
ramps steeply into US prime time. The forecaster learns that diurnal shape
from the case's comparables (past big launches' hourly curves):

- **Pool:** only comparables whose own release day already had a real
  hour-1 print (peak at most 6x the hour-1 print), with no early-access
  contamination. Launches whose unlock was a single simultaneous global
  moment later in the day have a different shape and are excluded.
- **Centre:** per hour, a least-squares fit of
  `log(viewers_h / viewers_1) = a + b * log(print_1) + c * jump`, where
  `jump = log(print_1 / hour-0 viewers)` is the launch's momentum into
  hour 1. Bigger hour-1 prints ramp less (later hours scale sub-linearly
  with the print); a bigger jump predicts a stronger ramp. The target's own
  print and jump are plugged in, clamped to the pool's observed ranges.
- **Spread:** the empirical standard deviation of the pool's log-ratios at
  that hour (floored at 0.25, capped at 0.9 log units); quantiles are
  log-normal around the fitted centre.
- **Minute-read momentum:** when the case carries minute reads inside hour
  1 (the final case does), their log-slope nudges every hour's centre by at
  most 0.10 log units.

Validation by leave-one-out over the comparables' own release days
(contest pinball loss on the log scale): 0.201 on midnight-local-like
launches vs 0.284 for the house persistence baseline (-29%); 0.177 vs
0.288 on the closest analogues, midnight-local with no early access
(-38%). Practice-case runs are deterministic and take ~0.05 s per case.

## Run it

Python 3.13, standard library only — no network, no dependencies.

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` with one case file in
`/input`; the program writes `/output/<id>.out` as
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Files

- `main.py` — I/O driver: reads every `<id>.in`, writes `<id>.out`.
- `forecast.py` — the forecasting model (ratio pool, regression centre,
  log-normal quantiles, minute-read momentum, fallbacks).
- `LICENSE` — MIT.
- `.makefun/owner.json` — entry owner wallet.

## License

MIT — see [LICENSE](LICENSE).
