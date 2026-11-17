"""GTA VI Twitch CCU forecaster.

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category,
hour by hour, for the 24 UTC hours of release day (2026-11-19), from
everything recorded up to the end of hour 1, as five quantiles per hour.

Model (round 2): a launch's hour-1 print fixes the scale; the rest of the
day is a shape, the ratios viewers_h / viewers_1, modelled in log space.
Per forecast hour, two ridge-shrunk regressions of the log-ratio are fit
on a pool of past big launches (comparables):

  base:     log-ratio ~ log(print_1) + jump
  extended: log-ratio ~ log(print_1) + jump + preweek

where jump = log(print_1 / hour-0 viewers) and preweek is the mean log
viewers in the category 14-7 days before release.  The fitted
coefficients are smoothed across neighbouring hours (edges hours 2 and 24
keep their own fit), the centre blends the two fits 75% extended /
25% base, and the spread is the empirical SD of the pool's log-ratios at
that hour (floored, capped, scaled, widened early / narrowed late).
Quantiles are log-normal.  The slope of the minute reads inside hour 1
nudges the centres by at most 0.10 log units.

Standard library only.  Deterministic: no randomness, no clock, no
network.  Reads /input/*.in (or $MAKEFUN_INPUT_DIR), writes one
/output/<id>.out per case (or $MAKEFUN_OUTPUT_DIR).
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Tuned constants (validated on leave-one-out over the comparables).
# ---------------------------------------------------------------------------

HOURS = list(range(2, 25))
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Pool membership: a comparable joins the shape pool when its own release
# day has a real hour-1 print, at least this many release-day hours, and a
# peak no more than this multiple of the hour-1 print (midnight-rolling
# launches ramp from a real hour 1; simultaneous-global unlocks spike from
# near zero and do not share the shape).
MIN_DAY_HOURS = 20
PEAK_RATIO_MAX = 6.0

# Pre-release baseline window: mean log viewers this many days before
# release (from 14 days out up to 7 days out).
PREWEEK_LO_DAYS = 14
PREWEEK_HI_DAYS = 7
# Ridge penalty.  It applies to the RAW predictors (no standardisation):
# with these predictor scales that shrinks the jump slope most -- the jump
# is the noisiest predictor, built from one hype-driven hour-0 print --
# while the log-print and preweek slopes are barely penalised.  Validated
# on LOO: raw ridge at lambda 1.0 beats standardised ridge clearly
# (midnight-like 0.172 vs 0.187).
RIDGE_LAMBDA = 1.0
# Blend of the two fits: centre = BLEND_EXT * extended + (1-BLEND_EXT) * base.
BLEND_EXTENDED = 0.75
# Coefficient smoothing across neighbouring hours: interior hours take
# 0.25 / 0.5 / 0.25 of (previous, own, next); hours 2 and 24 are untouched.
# SMOOTH_PASSES applies the smoothing pass that many times.
SMOOTH_SIDE = 0.25
SMOOTH_PASSES = 1
# Ridge standardisation: when True, predictors are standardised before the
# ridge penalty is applied (the usual choice); when False, the penalty
# applies to the raw predictors.  Kept False: see the RIDGE_LAMBDA note.
RIDGE_STANDARDISE = False
# Spread mode: "empirical" uses the SD of the pool's log-ratios at each
# hour; "residual" uses the SD of the extended fit's in-sample residuals
# at each hour (how much the ratios vary after the predictors explain
# what they can).  Both are then floored, capped, scaled and banded the
# same way.
SPREAD_MODE = "empirical"
# Spread: empirical SD of the pool's log-ratios, floored, capped, scaled,
# narrowed a little for late hours.  (LOO found no benefit in widening
# the early hours, so the early multiplier is 1.0.)
SPREAD_FLOOR = 0.25
SPREAD_CAP = 0.9
SPREAD_SCALE = 0.80
SPREAD_EARLY_MULT = 1.0    # hours 2..8
SPREAD_EARLY_LAST = 8
SPREAD_LATE_MULT = 0.9     # hours 17..24
SPREAD_LATE_FIRST = 17
# Minute-read slope nudge: nudge = clip(SLOPE_COEF * slope_per_hour).
SLOPE_COEF = 0.25
SLOPE_CAP = 0.10
# Regression is only trusted with a pool of at least this size; below it the
# per-hour median shape is used instead.
MIN_POOL_FOR_REGRESSION = 8
# Fallback spread when there is no pool at all.
FALLBACK_SPREAD = 0.6

# Median ramp shape (log viewers_h / viewers_1) of a typical midnight
# rolling launch, used only when no comparable is usable at all.  It is the
# observed median shape of such launches: a night dip, then a ramp into the
# US evening peak, then a late decline.
FALLBACK_SHAPE = {
    2: -0.12, 3: -0.30, 4: -0.33, 5: -0.28, 6: -0.25, 7: -0.25,
    8: -0.27, 9: -0.31, 10: -0.30, 11: -0.20, 12: -0.09, 13: -0.04,
    14: 0.04, 15: 0.15, 16: 0.22, 17: 0.27, 18: 0.32, 19: 0.38,
    20: 0.40, 21: 0.39, 22: 0.47, 23: 0.37, 24: 0.16,
}

MAX_VIEWERS = 1e9


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _parse_ts(text):
    """Parse an ISO-8601 UTC timestamp like 2026-11-19T00:00:00Z."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None


def _log_viewers(value):
    """log(viewers) with a floor of 1 viewer so zero reads stay finite."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return math.log(v if v > 1.0 else 1.0)


def _median(values):
    if not values:
        return None
    return statistics.median(values)


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _stdev(values):
    """Sample standard deviation; None when fewer than two values."""
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def _solve_linear(matrix, rhs):
    """Solve matrix * x = rhs by Gaussian elimination with pivoting.

    matrix is a list of n lists of length n, rhs a list of length n.
    Returns x as a list, or None when the system is singular.
    """
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / piv
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = aug[i][n]
        for k in range(i + 1, n):
            acc -= aug[i][k] * out[k]
        out[i] = acc / aug[i][i]
    return out


# ---------------------------------------------------------------------------
# Pool building
# ---------------------------------------------------------------------------

def _comparable_day_curve(comp):
    """Return the comparable's release-day data, or None when unusable.

    Returns a dict with print1, hour-0 viewers, preweek baseline and the
    per-hour log-ratios, or None when the comparable has no release-day
    curve, no real hour-1 print, too few release-day hours, or a peak far
    above the hour-1 print.
    """
    hourly = comp.get("hourly") or []
    day_start = _parse_ts(comp.get("day_start"))
    if day_start is None or not hourly:
        return None
    start_idx = None
    for i, row in enumerate(hourly):
        if _parse_ts(row.get("from")) == day_start:
            start_idx = i
            break
    if start_idx is None:
        return None
    day = hourly[start_idx:start_idx + 24]
    if len(day) < MIN_DAY_HOURS:
        return None
    print1 = day[0].get("viewers")
    if not isinstance(print1, (int, float)) or print1 <= 0:
        return None
    peak = 0.0
    for row in day:
        v = row.get("viewers")
        if isinstance(v, (int, float)) and v > peak:
            peak = v
    if peak / print1 > PEAK_RATIO_MAX:
        return None
    hour0 = None
    if start_idx >= 1:
        prev = hourly[start_idx - 1].get("viewers")
        if isinstance(prev, (int, float)) and prev > 0:
            hour0 = float(prev)
    # Pre-release baseline: mean log viewers PREWEEK_LO_DAYS to
    # PREWEEK_HI_DAYS days before release.
    pre_lo = day_start - timedelta(days=PREWEEK_LO_DAYS)
    pre_hi = day_start - timedelta(days=PREWEEK_HI_DAYS)
    pre_logs = []
    for row in hourly:
        ts = _parse_ts(row.get("from"))
        if ts is None:
            continue
        if pre_lo <= ts < pre_hi:
            pre_logs.append(_log_viewers(row.get("viewers")))
    preweek = _mean(pre_logs) if pre_logs else None
    ratios = {}
    for h in HOURS:
        idx = h - 1
        if idx < len(day):
            v = day[idx].get("viewers")
            if isinstance(v, (int, float)):
                ratios[h] = _log_viewers(v) - math.log(print1)
    return {
        "name": comp.get("name"),
        "print1": float(print1),
        "hour0": hour0,
        "preweek": preweek,
        "ratios": ratios,
        "tier": _tier_of(comp),
    }


def _tier_of(comp):
    """Tier label from the unlock note: comparable / early-access / weak."""
    unlock = comp.get("unlock") or ""
    if "Tier: early-access-contaminated" in unlock:
        return "early-access"
    if "Tier: weak" in unlock:
        return "weak"
    if "Tier: comparable" in unlock:
        return "comparable"
    return "other"


def build_pool(comparables):
    """Build the shape pool from the case's comparables.

    Each entry carries its own predictors (log print_1, jump, preweek) and
    its release-day log-ratios.  Missing jump / preweek values are filled
    with the pool medians.
    """
    pool = []
    for comp in comparables or []:
        entry = _comparable_day_curve(comp)
        if entry is not None:
            pool.append(entry)
    jumps = [math.log(e["print1"] / e["hour0"]) for e in pool
             if e["hour0"] is not None]
    preweeks = [e["preweek"] for e in pool if e["preweek"] is not None]
    med_jump = _median(jumps)
    med_preweek = _median(preweeks)
    for e in pool:
        e["jump"] = (math.log(e["print1"] / e["hour0"])
                     if e["hour0"] is not None else med_jump)
        e["preweek"] = e["preweek"] if e["preweek"] is not None else med_preweek
        e["log_print1"] = math.log(e["print1"])
    return pool


def pool_medians(pool):
    """(median jump, median preweek) of a pool, for filling missing values."""
    jumps = [e["jump"] for e in pool if e.get("jump") is not None]
    preweeks = [e["preweek"] for e in pool if e.get("preweek") is not None]
    return _median(jumps), _median(preweeks)


# ---------------------------------------------------------------------------
# Ridge regression on standardised predictors
# ---------------------------------------------------------------------------

def _standardise(rows):
    """Return (means, sds) of each predictor column over rows."""
    ncols = len(rows[0])
    means = []
    sds = []
    for j in range(ncols):
        col = [r[j] for r in rows]
        m = _mean(col)
        sd = _stdev(col) or 0.0
        means.append(m)
        sds.append(sd if sd > 1e-9 else 1.0)
    return means, sds


def _ridge_fit(rows, targets, lam):
    """Ridge fit of targets on rows (list of predictor lists).

    Returns (coef, means, sds) where coef[0] is the intercept in
    standardised-predictor space.  Returns None when unsolvable.
    """
    n = len(rows)
    if n == 0:
        return None
    if RIDGE_STANDARDISE:
        means, sds = _standardise(rows)
    else:
        means = [0.0] * len(rows[0])
        sds = [1.0] * len(rows[0])
    p = len(rows[0])
    # Normal equations in standardised space; intercept unpenalised.
    size = p + 1
    ata = [[0.0] * size for _ in range(size)]
    atb = [0.0] * size
    for i in range(n):
        z = [1.0] + [(rows[i][j] - means[j]) / sds[j] for j in range(p)]
        y = targets[i]
        for a in range(size):
            atb[a] += z[a] * y
            for b in range(size):
                ata[a][b] += z[a] * z[b]
    for j in range(1, size):
        ata[j][j] += lam
    coef = _solve_linear(ata, atb)
    if coef is None:
        return None
    return coef, means, sds


def _ridge_predict(fit, predictors):
    coef, means, sds = fit
    total = coef[0]
    for j, x in enumerate(predictors):
        total += coef[j + 1] * (x - means[j]) / sds[j]
    return total


# ---------------------------------------------------------------------------
# Case features
# ---------------------------------------------------------------------------

def case_features(case):
    """Extract (print1, hour0, preweek, slope) for the case itself.

    print1 is the hour-1 print (floored at 1).  hour0 is the hour-0 viewers
    from the hourly history when available.  preweek is the mean log
    viewers 14-7 days before release.  slope is the per-hour slope of log
    viewers over the minute reads inside hour 1 (0 when unavailable).
    """
    known = case.get("known") or []
    print1 = None
    for row in known:
        if row.get("hour") == 1:
            print1 = row.get("viewers")
            break
    if print1 is None and known:
        print1 = known[0].get("viewers")
    if not isinstance(print1, (int, float)) or print1 <= 0:
        print1 = 1.0
    print1 = float(print1)

    day_start = _parse_ts(case.get("day_start"))
    checkpoint = _parse_ts(case.get("checkpoint"))

    hourly_hist = case.get("history_hourly") or case.get("history") or []
    minute_hist = case.get("history") or []

    hour0 = None
    if day_start is not None:
        prev_ts = day_start - timedelta(hours=1)
        for row in hourly_hist:
            ts = _parse_ts(row.get("from") or row.get("at"))
            if ts == prev_ts:
                v = row.get("viewers")
                if isinstance(v, (int, float)) and v > 0:
                    hour0 = float(v)
                break

    preweek = None
    if day_start is not None:
        pre_lo = day_start - timedelta(days=PREWEEK_LO_DAYS)
        pre_hi = day_start - timedelta(days=PREWEEK_HI_DAYS)
        pre_logs = []
        for row in hourly_hist:
            ts = _parse_ts(row.get("from") or row.get("at"))
            if ts is None:
                continue
            if pre_lo <= ts < pre_hi:
                pre_logs.append(_log_viewers(row.get("viewers")))
        if pre_logs:
            preweek = _mean(pre_logs)

    slope = 0.0
    if day_start is not None and checkpoint is not None:
        points = []
        for row in minute_hist:
            ts = _parse_ts(row.get("at"))
            if ts is None:
                continue
            if day_start <= ts < checkpoint:
                points.append(((ts - day_start).total_seconds() / 3600.0,
                               _log_viewers(row.get("viewers"))))
        if len(points) >= 5:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            mx = _mean(xs)
            my = _mean(ys)
            denom = sum((x - mx) ** 2 for x in xs)
            if denom > 1e-9:
                slope = sum((x - mx) * (y - my)
                            for x, y in points) / denom
    return print1, hour0, preweek, slope


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def _quantiles(print1, centre, spread, levels):
    """Log-normal quantiles for one hour, rounded, >= 1, non-decreasing."""
    dist = statistics.NormalDist()
    out = []
    prev = 0
    for tau in levels:
        z = dist.inv_cdf(tau)
        value = print1 * math.exp(centre + z * spread)
        q = int(round(value))
        if q < 1:
            q = 1
        if q > MAX_VIEWERS:
            q = int(MAX_VIEWERS)
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def _spread_from_sd(sd, hour):
    """Floor, cap, scale and band a per-hour log-ratio SD into a spread."""
    if sd is None:
        return FALLBACK_SPREAD
    spread = sd * SPREAD_SCALE
    if spread < SPREAD_FLOOR:
        spread = SPREAD_FLOOR
    if spread > SPREAD_CAP:
        spread = SPREAD_CAP
    if hour <= SPREAD_EARLY_LAST:
        spread *= SPREAD_EARLY_MULT
    if hour >= SPREAD_LATE_FIRST:
        spread *= SPREAD_LATE_MULT
    return spread


def _spread_for(pool, hour):
    """Per-hour spread from the pool's log-ratio dispersion."""
    vals = [e["ratios"][hour] for e in pool if hour in e["ratios"]]
    return _spread_from_sd(_stdev(vals), hour)


def _median_centres(pool):
    """Per-hour median log-ratio of the pool (small-pool fallback)."""
    centres = {}
    for h in HOURS:
        vals = [e["ratios"][h] for e in pool if h in e["ratios"]]
        centres[h] = _median(vals)
    return centres


def forecast_from_pool(pool, print1, jump, preweek, slope, levels):
    """Forecast hours 2..24 from a pool and the case's features.

    jump / preweek may be None (filled with the pool medians); slope is the
    hour-1 minute-read slope (0 when unavailable).  Returns a list of
    {"hour": h, "q": [...]} rows.
    """
    if print1 is None or print1 <= 0:
        print1 = 1.0
    med_jump, med_preweek = pool_medians(pool)
    if jump is None:
        jump = med_jump if med_jump is not None else 0.0
    if preweek is None:
        preweek = med_preweek if med_preweek is not None else 0.0
    log_print1 = math.log(print1)
    nudge = max(-SLOPE_CAP, min(SLOPE_CAP, SLOPE_COEF * slope))

    if not pool:
        centres = dict(FALLBACK_SHAPE)
        spreads = {h: FALLBACK_SPREAD for h in HOURS}
    elif len(pool) < MIN_POOL_FOR_REGRESSION:
        centres = _median_centres(pool)
        spreads = {}
        for h in HOURS:
            vals = [e["ratios"][h] for e in pool if h in e["ratios"]]
            sd = _stdev(vals)
            spreads[h] = sd if sd is not None else FALLBACK_SPREAD
            if spreads[h] < SPREAD_FLOOR:
                spreads[h] = SPREAD_FLOOR
    else:
        centres = {}
        spreads = {}
        base_coefs = {}
        ext_coefs = {}
        for h in HOURS:
            members = [e for e in pool if h in e["ratios"]]
            base_rows = [[e["log_print1"], e["jump"]] for e in members]
            ext_rows = [[e["log_print1"], e["jump"], e["preweek"]]
                        for e in members]
            targets = [e["ratios"][h] for e in members]
            base_fit = _ridge_fit(base_rows, targets, RIDGE_LAMBDA)
            ext_fit = _ridge_fit(ext_rows, targets, RIDGE_LAMBDA)
            if base_fit is None or ext_fit is None:
                centres[h] = _median(targets)
            else:
                base_coefs[h] = base_fit
                ext_coefs[h] = ext_fit
        # Smooth the fitted coefficients across neighbouring hours; the
        # edge hours (2 and 24) keep their own fit.  Repeat SMOOTH_PASSES
        # times for a wider smoothing kernel.
        for _pass in range(SMOOTH_PASSES):
            for h in HOURS:
                if (h in base_coefs and h - 1 in base_coefs
                        and h + 1 in base_coefs):
                    for store in (base_coefs, ext_coefs):
                        c_prev, c_own, c_next = (store[h - 1][0],
                                                 store[h][0], store[h + 1][0])
                        store[h] = (
                            [SMOOTH_SIDE * a
                             + (1.0 - 2.0 * SMOOTH_SIDE) * b
                             + SMOOTH_SIDE * c
                             for a, b, c in zip(c_prev, c_own, c_next)],
                            store[h][1], store[h][2])
        for h in HOURS:
            if h in base_coefs:
                base_pred = _ridge_predict(base_coefs[h],
                                           [log_print1, jump])
                ext_pred = _ridge_predict(ext_coefs[h],
                                          [log_print1, jump, preweek])
                centres[h] = (BLEND_EXTENDED * ext_pred
                              + (1.0 - BLEND_EXTENDED) * base_pred)
            if SPREAD_MODE == "residual" and h in ext_coefs:
                members = [e for e in pool if h in e["ratios"]]
                resid = []
                for e in members:
                    pred = _ridge_predict(
                        ext_coefs[h],
                        [e["log_print1"], e["jump"], e["preweek"]])
                    resid.append(e["ratios"][h] - pred)
                # In-sample residual SD with a degrees-of-freedom
                # correction (4 fitted parameters); fall back to the
                # empirical spread when the pool is too small.
                dof = len(resid) - 4
                if dof > 0 and resid:
                    var = sum(r * r for r in resid) / dof
                    spreads[h] = _spread_from_sd(math.sqrt(var), h)
                else:
                    spreads[h] = _spread_for(pool, h)
            else:
                spreads[h] = _spread_for(pool, h)

    rows = []
    for h in HOURS:
        centre = centres.get(h)
        if centre is None:
            centre = FALLBACK_SHAPE[h]
        centre += nudge
        spread = spreads.get(h, FALLBACK_SPREAD)
        rows.append({"hour": h,
                     "q": _quantiles(print1, centre, spread, levels)})
    return rows


def forecast_case(case):
    """Forecast one case dict, returning {"hours": [...]}."""
    levels = case.get("levels") or DEFAULT_LEVELS
    levels = [float(t) for t in levels]
    pool = build_pool(case.get("comparables"))
    print1, hour0, preweek, slope = case_features(case)
    jump = None
    if hour0 is not None and hour0 > 0:
        jump = math.log(print1 / hour0)
    rows = forecast_from_pool(pool, print1, jump, preweek, slope, levels)
    return {"hours": rows}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        in_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, case_id + ".out")
        try:
            with open(in_path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
            result = forecast_case(case)
        except Exception:
            # Never fail the whole run on one bad case: emit the fallback
            # shape at a print of 1 viewer so the output stays valid.
            result = {"hours": [
                {"hour": h,
                 "q": _quantiles(1.0, FALLBACK_SHAPE[h], FALLBACK_SPREAD,
                                 DEFAULT_LEVELS)}
                for h in HOURS]}
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
