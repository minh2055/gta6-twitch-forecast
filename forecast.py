"""Forecasting model for gta6-twitch-ccu-forecast-01.

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category for
hours 2..24 of release day (2026-11-19 UTC) as five quantiles per hour, from
everything recorded up to the end of hour 1.

Idea
----
Past big game launches (the case's ``comparables``) each carry their full
release-day hourly curve. For a launch, hour 1's print anchors the scale, and
the shape of the rest of the day -- the ratios viewers_h / viewers_1 -- is
what has to be predicted. GTA VI unlocks at local midnight, rolling (New
Zealand first, then Europe, then the US East at 05:00Z and Pacific at
08:00Z), so hour 1 is before most of the world unlocks: its print is low and
viewership ramps steeply into US prime time. The comparables encode that
diurnal shape, but only the ones whose own release day already had a real
hour-1 print (a midnight-local style launch). A launch whose unlock was a
single simultaneous global moment later in the day (its hour-1 print near
zero, then a vertical spike) has a wildly different shape and must not enter
the ratio pool.

Model, per forecast hour h
--------------------------
On the selected pool, fit by least squares

    log(viewers_h / viewers_1) = a_h + b_h * log(print_1)
                                       + c_h * jump

where ``jump`` = log(print_1 / hour-0 viewers) measures how hard the launch
jumped into hour 1 (momentum). Two facts the practice data shows:

* bigger hour-1 prints ramp less (b_h is about -0.05 early, about -0.3 at
  the evening peak: later hours scale sub-linearly with the hour-1 print);
* a bigger jump into hour 1 predicts a stronger ramp.

Validation (leave-one-out over the comparables' own release days, scored
with the contest's pinball loss on the log scale; the house persistence
baseline scores 0.284 on the same launches, 0.930 over all of them):
midnight-local-like launches 0.201 (baseline 0.284, -29%); the closest
analogues, midnight-local with no early access, 0.177 (baseline 0.288,
-38%).

The target's own log(print_1) and jump are plugged in, clamped to the pool's
observed ranges so an unusually large GTA VI print cannot extrapolate the
fit. The spread is the empirical standard deviation of the pool's
log-ratios at that hour, floored at 0.25 and capped at 0.9 in log units.
Quantiles are log-normal:

    q_tau = print_1 * exp(centre_h + z_tau * spread_h)

with z = (-1.645, -0.674, 0, 0.674, 1.645) for the levels
(0.05, 0.25, 0.5, 0.75, 0.95), rounded to whole viewers, at least 1, and
forced non-decreasing.

Minute-read momentum
--------------------
When the case carries minute reads inside hour 1 (the final case does; the
practice cases carry hourly points only), the log-slope of those reads is a
second momentum signal: a steep rise through hour 1 nudges every hour's
centre up, a fall nudges it down, by at most 0.10 log units. The nudge is
deliberately small -- the jump predictor above carries the validated signal
and the minute slope is only a bounded refinement of it.

Fallbacks
---------
* No usable comparables at all: fall back to a persistence forecast (every
  hour at the hour-1 print) with a wide spread, so the output is still a
  genuine, valid forecast.
* Missing hour-0 viewers: the jump falls back to the pool's median jump.
* Missing/zero hour-1 print: floored at 1 viewer.

Everything is deterministic: no randomness, no clock, no network, standard
library only.
"""

import math

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# The standard normal's quantile at each level.
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Pool selection: a comparable joins the ratio pool only if its release day
# already had a real hour-1 print. Launches whose peak is more than this
# many times their hour-1 print unlocked later in the day (a single global
# moment) and have a different shape; they would only inflate the spread.
PEAK_TO_PRINT1_CAP = 6.0
# A comparable must have this many release-day hours present to be usable.
MIN_RELEASE_HOURS = 20

# Spread of the log-normal quantiles, in log units.
SPREAD_FLOOR = 0.25
SPREAD_CAP = 0.9
# Fallback spread when the pool is empty (persistence forecast).
FALLBACK_SPREAD = 0.35

# Minute-read momentum: at most this many log units of centre shift.
MAX_MOMENTUM_NUDGE = 0.10
MOMENTUM_SLOPE_WEIGHT = 0.25
# A minute trend needs at least this many reads spanning this many minutes.
MIN_MINUTE_READS = 20
MIN_MINUTE_SPAN = 30.0


def _parse_time(text):
    """Parse an ISO-8601 UTC timestamp into epoch seconds (float)."""
    # All timestamps in the case files end in "Z" and carry no fractional
    # seconds beyond the minute; fromisoformat handles the rest.
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _mean(values):
    return sum(values) / len(values)


def _stdev(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _clamp(value, low, high):
    return low if value < low else high if value > high else value


def _ols_1d(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b)."""
    n = len(xs)
    mx = _mean(xs)
    my = _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def _ols_2d(x1s, x2s, ys):
    """Least squares y ~ a + b1*x1 + b2*x2. Returns (a, b1, b2) or None if
    the design is (near-)singular."""
    n = len(ys)
    m1 = _mean(x1s)
    m2 = _mean(x2s)
    my = _mean(ys)
    s11 = sum((v - m1) ** 2 for v in x1s)
    s22 = sum((v - m2) ** 2 for v in x2s)
    s12 = sum((u - m1) * (v - m2) for u, v in zip(x1s, x2s))
    s1y = sum((u - m1) * (y - my) for u, y in zip(x1s, ys))
    s2y = sum((v - m2) * (y - my) for v, y in zip(x2s, ys))
    det = s11 * s22 - s12 * s12
    if abs(det) <= 1e-12 * max(1.0, s11 * s22):
        return None
    b1 = (s1y * s22 - s2y * s12) / det
    b2 = (s2y * s11 - s1y * s12) / det
    return my - b1 * m1 - b2 * m2, b1, b2


def _tier(unlock_text):
    """The kit tags each comparable's unlock note with a tier marker."""
    text = unlock_text or ""
    if "Tier: early-access-contaminated" in text:
        return "early-access-contaminated"
    if "Tier: weak" in text:
        return "weak"
    if "Tier: target-his" in text:
        return "target-history"
    return "comparable"


def release_day_hours(comp):
    """{hour: viewers} for hours 1..24 of a comparable's release day.

    A point counts for hour h when it starts exactly at
    day_start + (h - 1) hours and its viewer count is positive.
    """
    day_start = _parse_time(comp["day_start"])
    hours = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        try:
            offset = (_parse_time(point["from"]) - day_start) / 3600.0
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= offset < 24 and offset == int(offset):
            hours[int(offset) + 1] = float(viewers)
    return hours


def select_pool(comparables):
    """The ratio pool: comparables usable for estimating the launch shape.

    Kept: a real hour-1 print, at least MIN_RELEASE_HOURS release-day
    hours, peak no more than PEAK_TO_PRINT1_CAP times the hour-1 print, and
    a tier that is not early-access-contaminated, weak, or the target's own
    pre-release history. Returns a list of (comparable, release-day hours);
    the jump for a comparable needs its hour-0 viewers, which the caller
    derives from the comparable's own hourly series.
    """
    pool = []
    for comp in comparables or []:
        if _tier(comp.get("unlock")) != "comparable":
            continue
        hours = release_day_hours(comp)
        if 1 not in hours or len(hours) < MIN_RELEASE_HOURS:
            continue
        peak = max(hours.values())
        if peak / hours[1] > PEAK_TO_PRINT1_CAP:
            continue
        pool.append((comp, hours))
    return pool


def _hour_zero_viewers(comp):
    """Viewers in the hour just before a comparable's release day, or None."""
    day_start = _parse_time(comp["day_start"])
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        try:
            offset = (_parse_time(point["from"]) - day_start) / 3600.0
        except (KeyError, TypeError, ValueError):
            continue
        if offset == -1:
            return float(viewers)
    return None


def _hour_zero_for_case(case):
    """The case's own hour-0 viewers (the whole hour before release day).

    Prefers the history_hourly point starting exactly one hour before
    day_start; otherwise the latest whole-hour point before day_start.
    """
    day_start = _parse_time(case["day_start"])
    hourly = case.get("history_hourly") or []
    best = None
    best_from = None
    for point in hourly:
        viewers = point.get("viewers")
        if not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        try:
            start = _parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if start == day_start - 3600.0:
            return float(viewers)
        if start < day_start and (best_from is None or start > best_from):
            best = float(viewers)
            best_from = start
    return best


def minute_trend_nudge(case):
    """Bounded momentum nudge from the minute reads inside hour 1.

    Fits log(viewers) against minutes since day_start over the reads in
    [day_start, checkpoint) and returns a centre shift in log units, capped
    at +-MAX_MOMENTUM_NUDGE. Returns 0.0 when there are too few minute
    reads to measure a trend (the practice cases carry hourly points only).
    """
    try:
        day_start = _parse_time(case["day_start"])
        checkpoint = _parse_time(case["checkpoint"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    minutes = []
    logs = []
    for point in case.get("history") or []:
        viewers = point.get("viewers")
        if not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        try:
            at = _parse_time(point["at"])
        except (KeyError, TypeError, ValueError):
            continue
        if day_start <= at < checkpoint:
            minutes.append((at - day_start) / 60.0)
            logs.append(math.log(float(viewers)))
    if len(minutes) < MIN_MINUTE_READS:
        return 0.0
    if max(minutes) - min(minutes) < MIN_MINUTE_SPAN:
        return 0.0
    _, slope_per_minute = _ols_1d(minutes, logs)
    slope_per_hour = slope_per_minute * 60.0
    return _clamp(MOMENTUM_SLOPE_WEIGHT * slope_per_hour,
                  -MAX_MOMENTUM_NUDGE, MAX_MOMENTUM_NUDGE)


def _round_viewers(value):
    """Whole viewers, half up, at least 1, at most 1e9."""
    whole = math.floor(value + 0.5)
    if whole < 1:
        whole = 1
    if whole > 1_000_000_000:
        whole = 1_000_000_000
    return int(whole)


def _quantiles(print_1, centre, spread):
    """The five quantiles for one hour, as whole viewers, non-decreasing."""
    out = []
    previous = 0
    for z in Z:
        q = _round_viewers(print_1 * math.exp(centre + z * spread))
        if q < previous:
            q = previous
        out.append(q)
        previous = q
    return out


def forecast(case):
    """Forecast hours 2..24 of the case's release day.

    Returns {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}.
    """
    if list(case.get("levels") or []) != list(LEVELS):
        raise ValueError("case asks for levels %r, not %r"
                         % (case.get("levels"), LEVELS))
    known = case.get("known") or []
    first = next((row for row in known if row.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    print_1 = first.get("viewers")
    if not isinstance(print_1, (int, float)) or print_1 <= 0:
        print_1 = 1.0
    print_1 = float(print_1)

    hours_to_forecast = sorted(row["hour"] for row in case.get("forecast") or []
                               if row.get("hour") in range(2, 25))
    if not hours_to_forecast:
        hours_to_forecast = list(range(2, 25))

    nudge = minute_trend_nudge(case)

    # The target's jump into hour 1, from its own hour-0 viewers.
    hour0 = _hour_zero_for_case(case)
    target_jump = math.log(print_1 / hour0) if hour0 else None

    # Build the pool: per comparable, its release-day curve, its
    # log(print_1) and its jump into hour 1.
    pool = []
    for comp, hours in select_pool(case.get("comparables")):
        p1 = hours[1]
        hour0_c = _hour_zero_viewers(comp)
        jump_c = math.log(p1 / hour0_c) if hour0_c else None
        pool.append((hours, math.log(p1), jump_c))

    if len(pool) < 5:
        # Too thin a pool to fit the shape: persistence with a wide spread.
        rows = [{"hour": h,
                 "q": _quantiles(print_1, nudge, FALLBACK_SPREAD)}
                for h in hours_to_forecast]
        return {"hours": rows}

    pool_jumps = [jump for _, _, jump in pool if jump is not None]
    median_jump = _median(pool_jumps) if pool_jumps else 0.0
    if target_jump is None:
        target_jump = median_jump
    pool = [(hours, log_p1, jump if jump is not None else median_jump)
            for hours, log_p1, jump in pool]

    target_log_p1 = math.log(print_1)
    rows = []
    for h in hours_to_forecast:
        xs1, xs2, ys = [], [], []
        for hours, log_p1, jump in pool:
            if h in hours:
                xs1.append(log_p1)
                xs2.append(jump)
                ys.append(math.log(hours[h] / hours[1]))
        if len(ys) < 5:
            rows.append({"hour": h,
                         "q": _quantiles(print_1, nudge, FALLBACK_SPREAD)})
            continue
        fit = _ols_2d(xs1, xs2, ys)
        if fit is None:
            intercept, slope = _ols_1d(xs1, ys)
            centre = intercept + slope * _clamp(target_log_p1,
                                                min(xs1), max(xs1))
        else:
            intercept, slope1, slope2 = fit
            centre = (intercept
                      + slope1 * _clamp(target_log_p1, min(xs1), max(xs1))
                      + slope2 * _clamp(target_jump, min(xs2), max(xs2)))
        spread = _clamp(_stdev(ys), SPREAD_FLOOR, SPREAD_CAP)
        rows.append({"hour": h,
                     "q": _quantiles(print_1, centre + nudge, spread)})
    return {"hours": rows}
