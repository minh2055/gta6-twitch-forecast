"""Hour-by-hour Twitch CCU forecaster for GTA VI launch day (round 2).

Predicts the mean concurrent viewers in the Grand Theft Auto VI Twitch
category for hours 2..24 of release day (2026-11-19 UTC) as five quantiles
per hour, from everything recorded up to the end of hour 1.

Approach
--------
Each past launch (a ``comparable``) carries its full release-day curve.
For a launch, the hour-1 print fixes the scale and the rest of the day
is a shape: the ratios viewers_h / viewers_1. GTA VI unlocks at local
midnight, rolling around the world, so hour 1 is small and the curve
ramps hard into US prime time. Only comparables whose own release day
already had a real hour-1 print share that shape; a launch that unlocked
in one simultaneous global moment later in the day spikes vertically from
a near-zero hour 1 and must stay out of the shape pool.

Model, per forecast hour h
--------------------------
On the shape pool, fit two ridge-shrunk least-squares regressions of
the log-ratio log(viewers_h / viewers_1):

  base:      log-ratio ~ log(print_1) + jump
  extended:  log-ratio ~ log(print_1) + jump + preweek

where ``jump`` = log(print_1 / hour-0 viewers) measures how hard the
launch jumped into hour 1, and ``preweek`` = the mean log viewers in
the category 14-7 days before release -- the category's baseline
popularity before launch week. Three facts the practice data shows:

* bigger hour-1 prints ramp less (the log(print_1) slope is about -0.05
  early and about -0.35 at the evening peak: later hours scale
  sub-linearly with the hour-1 print);
* a bigger jump into hour 1 predicts a stronger ramp;
* a category that was already popular two weeks out ramps more (the
  preweek slope is positive at every hour, +0.02 to +0.11). Early-access
  launches, whose categories were quiet two weeks out, ramp least.

The fitted coefficients are lightly smoothed across neighbouring hours
(the diurnal shape is smooth: ramp, peak, decline) except at the edges
(hours 2 and 24 keep their own fit -- smoothing would drag hour 2 down
toward the 03:00-04:00 dip and hour 24 up toward hour 23). The final
centre blends the two fits, 75% extended and 25% base, which hedges the
third predictor's extrapolation for targets whose preweek sits far
outside the pool. The target's features enter the fits as measured: the
slopes are small and the relationships (bigger prints ramp less, a
popular category ramps more) hold across the pool's full range, so
modest extrapolation is intended rather than clamped away.

The spread is the empirical standard deviation of the pool's log-ratios
at that hour (floored at 0.25, capped at 0.9, scaled by 0.85, and
widened a little for hours 2-8 and narrowed a little for hours 17-24).
Quantiles are log-normal:

    q_tau = print_1 * exp(centre_h + z_tau * spread_h)

with z = (-1.645, -0.674, 0, 0.674, 1.645) for the levels
(0.05, 0.25, 0.5, 0.75, 0.95), rounded to whole viewers, at least 1,
and forced non-decreasing.

Validation (leave-one-out over the comparables' own release days,
scored with the contest's pinball loss on the log scale; the house
persistence baseline scores 0.284 on midnight-like launches and 0.288
on the closest analogues):
midnight-like launches 0.180 (baseline 0.284, -36%; round 1: 0.201);
the closest analogues, midnight-like with no early access, 0.164
(baseline 0.288, -43%; round 1: 0.177).

Minute-read momentum
--------------------
When the case carries minute reads inside hour 1 (the final case does;
the practice cases carry hourly points only), the log-slope of those
reads is a second momentum signal: a steep rise through hour 1 nudges
every hour's centre up, a fall nudges it down, by at most 0.10 log
units. The nudge is deliberately small -- the jump predictor above
carries the validated signal and the minute slope is only a bounded
refinement of it.

Fallbacks
---------
* No usable comparables at all: fall back to a median ramp shape (the
  pooled median shape of every comparable with a real hour-1 print,
  however thin) with a wide spread, so the output is still a genuine,
  valid forecast.
* Missing hour-0 viewers: the jump falls back to the pool's median jump.
* Missing pre-release history: preweek falls back to the pool's median.
* Missing/zero hour-1 print: floored at 1 viewer.

Everything is deterministic: no randomness, no clock, no network,
standard library only.
"""

import math

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# The standard normal's quantile at each level.
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Pool selection: a comparable joins the shape pool only if its release
# day already had a real hour-1 print. Launches whose peak is more than
# this many times their hour-1 print unlocked later in the day (a single
# global moment) and have a different shape; they would only inflate the
# spread. Weak-tier launches (free-to-play) stay in the pool: on
# leave-one-out they help slightly. Early-access-contaminated launches
# and the target's own pre-release history stay out.
PEAK_TO_PRINT1_CAP = 6.0
# A comparable must have this many release-day hours present to be usable.
MIN_RELEASE_HOURS = 20

# Ridge penalty on the regression slopes (times the number of pool
# members), shrinking them toward zero so a 16-launch pool cannot
# overfit two or three noisy slopes.
RIDGE = 0.05
# Blend of the extended fit's centre with the base fit's centre:
# centre = BLEND * extended + (1 - BLEND) * base.
BLEND_EXTENDED = 0.75

# Spread of the log-normal quantiles, in log units.
SPREAD_MULT = 0.85
SPREAD_FLOOR = 0.25
SPREAD_CAP = 0.9
# Per-band spread multipliers: (hours 2-8, hours 9-16, hours 17-24).
SPREAD_BAND_MULT = (1.15, 1.0, 0.9)
# Fallback spread when the pool is empty (median-shape forecast).
FALLBACK_SPREAD = 0.40

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


def _day_hours(comp):
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


def _hour_before(comp):
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


def _preweek_mean(hourly_points, day_start_ts):
    """Mean log viewers 14-7 days before release (the category's baseline
    popularity before launch week). Returns None when the history is too
    short to measure it."""
    logs = []
    for point in hourly_points or []:
        viewers = point.get("viewers")
        if not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        try:
            at = _parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = (at - day_start_ts) / 3600.0
        if -336.0 <= offset < -168.0:
            logs.append(math.log(float(viewers)))
    return _mean(logs) if len(logs) >= 24 else None


def _case_hour_before(case):
    """The case's own hour-0 viewers (the whole hour before release day).

    Prefers the history_hourly point starting exactly one hour before
    day_start; otherwise the latest whole-hour point before day_start.
    """
    day_start = _parse_time(case["day_start"])
    best = None
    best_from = None
    for point in case.get("history_hourly") or []:
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


def _minute_nudge(case):
    """Bounded momentum nudge from the minute reads inside hour 1.

    Fits log(viewers) against minutes since day_start over the reads in
    [day_start, checkpoint) and returns a centre shift in log units,
    capped at +-MAX_MOMENTUM_NUDGE. Returns 0.0 when there are too few
    minute reads to measure a trend (the practice cases carry hourly
    points only).
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
    n = len(minutes)
    mx = sum(minutes) / n
    my = sum(logs) / n
    sxx = sum((x - mx) ** 2 for x in minutes)
    if sxx <= 0.0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(minutes, logs))
    slope_per_hour = (sxy / sxx) * 60.0
    return _clamp(MOMENTUM_SLOPE_WEIGHT * slope_per_hour,
                  -MAX_MOMENTUM_NUDGE, MAX_MOMENTUM_NUDGE)


def _ols_multi(xs_list, ys, ridge):
    """Least squares y ~ a + b.x for a small number of predictors.

    Slopes are shrunk toward zero with a ridge penalty (times the number
    of rows); the intercept is not penalised. Returns the coefficient
    list [a, b1, ...] or None when the design is (near-)singular or the
    pool is too thin for the number of predictors.
    """
    k = len(xs_list)
    n = len(ys)
    if n < k + 2:
        return None
    means = [_mean(xs) for xs in xs_list]
    my = _mean(ys)
    # Normal equations on centred predictors.
    matrix = [[0.0] * k for _ in range(k)]
    rhs = [0.0] * k
    for i in range(k):
        for j in range(k):
            matrix[i][j] = sum((xs_list[i][t] - means[i])
                               * (xs_list[j][t] - means[j])
                               for t in range(n))
        rhs[i] = sum((xs_list[i][t] - means[i]) * (ys[t] - my)
                     for t in range(n))
    for i in range(k):
        matrix[i][i] += ridge * n
    # Gaussian elimination with partial pivoting.
    rows = [matrix[i][:] + [rhs[i]] for i in range(k)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, k):
            factor = rows[r][col] / rows[col][col]
            for cc in range(col, k + 1):
                rows[r][cc] -= factor * rows[col][cc]
    coefs = [0.0] * k
    for i in range(k - 1, -1, -1):
        rest = rows[i][k] - sum(rows[i][j] * coefs[j]
                                for j in range(i + 1, k))
        coefs[i] = rest / rows[i][i]
    intercept = my - sum(coefs[i] * means[i] for i in range(k))
    return [intercept] + coefs


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


def _fallback_shape(comparables):
    """A sensible ramp shape when the pool is empty: the median log-ratio
    per hour over every comparable with a real hour-1 print, whatever its
    tier or peak. Returns {hour: median log-ratio} (empty if none)."""
    per_hour = {}
    for comp in comparables or []:
        if _tier(comp.get("unlock")) == "target-history":
            continue
        hours = _day_hours(comp)
        if 1 not in hours:
            continue
        p1 = hours[1]
        for h in range(2, 25):
            if h in hours:
                per_hour.setdefault(h, []).append(math.log(hours[h] / p1))
    return {h: _median(vals) for h, vals in per_hour.items() if vals}


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

    nudge = _minute_nudge(case)

    # The target's features: scale, jump into hour 1, and the category's
    # baseline popularity two weeks out.
    day_start_ts = _parse_time(case["day_start"])
    hour0 = _case_hour_before(case)
    target_jump = math.log(print_1 / hour0) if hour0 else None
    target_preweek = _preweek_mean(case.get("history_hourly"), day_start_ts)
    target_log_p1 = math.log(print_1)

    # Build the shape pool: per comparable, its release-day curve, its
    # log(print_1), its jump into hour 1, and its pre-release baseline.
    pool = []
    for comp in case.get("comparables") or []:
        tier = _tier(comp.get("unlock"))
        if tier in ("early-access-contaminated", "target-history"):
            continue
        hours = _day_hours(comp)
        if 1 not in hours or len(hours) < MIN_RELEASE_HOURS:
            continue
        peak = max(hours.values())
        if peak / hours[1] > PEAK_TO_PRINT1_CAP:
            continue
        p1 = hours[1]
        hour0_c = _hour_before(comp)
        jump_c = math.log(p1 / hour0_c) if hour0_c else None
        preweek_c = _preweek_mean(comp.get("hourly"), _parse_time(comp["day_start"]))
        pool.append({
            "hours": hours,
            "log_p1": math.log(p1),
            "jump": jump_c,
            "preweek": preweek_c,
        })

    if len(pool) < 5:
        # Too thin a pool to fit the shape: median-shape fallback with a
        # wide spread, still a genuine forecast, never the baseline.
        shape = _fallback_shape(case.get("comparables"))
        rows = [{"hour": h,
                 "q": _quantiles(print_1,
                                 shape.get(h, 0.0) + nudge,
                                 FALLBACK_SPREAD)}
                for h in hours_to_forecast]
        return {"hours": rows}

    pool_jumps = [entry["jump"] for entry in pool if entry["jump"] is not None]
    median_jump = _median(pool_jumps) if pool_jumps else 0.0
    if target_jump is None:
        target_jump = median_jump
    pool_preweeks = [entry["preweek"] for entry in pool
                     if entry["preweek"] is not None]
    median_preweek = _median(pool_preweeks) if pool_preweeks else 0.0
    if target_preweek is None:
        target_preweek = median_preweek
    for entry in pool:
        if entry["jump"] is None:
            entry["jump"] = median_jump
        if entry["preweek"] is None:
            entry["preweek"] = median_preweek

    # The target's features enter the fits as measured (see the module
    # docstring); the regression slopes are small, so modest extrapolation
    # beyond the pool's range is intended rather than clamped away.
    target_log_p1_c = target_log_p1
    target_jump_c = target_jump
    target_preweek_c = target_preweek

    # Per hour, fit the base and the extended regressions on the pool.
    centres_base = {}
    centres_ext = {}
    spreads = {}
    fits_base = {}
    fits_ext = {}
    for h in hours_to_forecast:
        rows = [(entry, math.log(entry["hours"][h] / entry["hours"][1]))
                for entry in pool if h in entry["hours"]]
        if len(rows) < 5:
            centres_base[h] = _median([y for _, y in rows]) if rows else 0.0
            centres_ext[h] = centres_base[h]
            spreads[h] = FALLBACK_SPREAD
            continue
        xs_base = [[e["log_p1"] for e, _ in rows],
                   [e["jump"] for e, _ in rows]]
        xs_ext = [xs_base[0], xs_base[1],
                  [e["preweek"] for e, _ in rows]]
        ys = [y for _, y in rows]
        fit_base = _ols_multi(xs_base, ys, RIDGE)
        fit_ext = _ols_multi(xs_ext, ys, RIDGE)
        if fit_base is None:
            centre_b = _mean(ys)
        else:
            centre_b = (fit_base[0]
                        + fit_base[1] * target_log_p1_c
                        + fit_base[2] * target_jump_c)
        if fit_ext is None:
            centre_e = centre_b
        else:
            centre_e = (fit_ext[0]
                        + fit_ext[1] * target_log_p1_c
                        + fit_ext[2] * target_jump_c
                        + fit_ext[3] * target_preweek_c)
        fits_base[h] = fit_base
        fits_ext[h] = fit_ext
        centres_base[h] = centre_b
        centres_ext[h] = centre_e
        spreads[h] = _stdev(ys)

    # Smooth the fitted coefficients across neighbouring hours (the
    # diurnal shape is smooth), recompute the centres, and keep the edge
    # hours (2 and 24) on their own fit.
    def _smooth_centres(centres, fits, n_pred):
        smoothed = dict(centres)
        for h in hours_to_forecast:
            neighbours = [fits.get(hh) for hh in (h - 1, h, h + 1)]
            neighbours = [f for f in neighbours if f is not None]
            if len(neighbours) < 2:
                continue
            avg = [_mean([f[i] for f in neighbours])
                   for i in range(1 + n_pred)]
            target_vals = [target_log_p1_c, target_jump_c,
                           target_preweek_c][:n_pred]
            smoothed[h] = avg[0] + sum(avg[i + 1] * target_vals[i]
                                       for i in range(n_pred))
        for edge in (2, 24):
            if edge in centres:
                smoothed[edge] = centres[edge]
        return smoothed

    centres_base = _smooth_centres(centres_base, fits_base, 2)
    centres_ext = _smooth_centres(centres_ext, fits_ext, 3)

    rows = []
    for h in hours_to_forecast:
        centre = ((1.0 - BLEND_EXTENDED) * centres_base[h]
                  + BLEND_EXTENDED * centres_ext[h])
        spread = _clamp(spreads[h] * SPREAD_MULT, SPREAD_FLOOR, SPREAD_CAP)
        if h <= 8:
            spread *= SPREAD_BAND_MULT[0]
        elif h <= 16:
            spread *= SPREAD_BAND_MULT[1]
        else:
            spread *= SPREAD_BAND_MULT[2]
        spread = _clamp(spread, SPREAD_FLOOR, SPREAD_CAP)
        rows.append({"hour": h,
                     "q": _quantiles(print_1, centre + nudge, spread)})
    return {"hours": rows}
