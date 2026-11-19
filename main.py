"""Forecast how many GTA VI players earn each trophy in week one.

Reads every <id>.in case file in the input directory and writes one
<id>.out forecast per case: for each trophy, five quantiles (0.05, 0.25,
0.5, 0.75, 0.95) of its Exophase earn rate at the case's target.

How the forecast is made
------------------------
An earn rate is a ratio, rate = earners / players_tracked, and both sides
move for different reasons in a launch week, so the model forecasts them
separately and divides:

- Players tracked N(t) grows as more players buy, play and sync. It is
  fitted on the log scale against log(1 + hours since launch) with a
  quadratic term, so the fit can bend as the sync wave saturates, and
  extrapolated to the target with a cap on total growth.
- Earners E_j(t) for each trophy are fitted two ways: a power law in
  log(1 + t), and a saturating log-logistic curve E_inf / (1 + (tm/t)^k)
  fitted by grid search. Their geometric mean is the earner forecast,
  clamped so earners never fall and never explode past a tier-based cap.
- The median rate is 100 * E / N at the target. The spread on the logit
  scale combines the fits' residual scatter, a random-walk term that grows
  with the horizon, and an allowance for Exophase's own refresh jumps
  (the page is refreshed on its own schedule, and a refresh can move every
  rate at once). The five quantiles are the median shifted by the normal
  quantiles times that spread.

When the history is flat (no refresh yet, as on launch night), the fits
have nothing to learn from, so the model falls back to a widened version
of the last rate: the next refresh is expected to multiply the tracked
players several-fold while earners grow more slowly, which is exactly
what the house's launch-night recording showed.

Standard library only; deterministic (no randomness, no network).
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LAUNCH = 1795046400  # 2026-11-19T00:00:00Z as a Unix timestamp.


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    x = clamp_pct(pct)
    return math.log(x / (100.0 - x))


def logistic(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    """ISO 8601 UTC string -> Unix timestamp (seconds)."""
    y = int(s[0:4])
    mo = int(s[5:7])
    d = int(s[8:10])
    h = int(s[11:13])
    mi = int(s[14:16])
    sec = int(s[17:19])
    import calendar

    return calendar.timegm((y, mo, d, h, mi, sec, 0, 0, 0))


def hours_since_launch(ts):
    return (ts - LAUNCH) / 3600.0


def weighted_linfit(xs, ys, ws):
    """Weighted least squares for y = a + b*x. Returns (a, b, resid_std)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    return (a, b, math.sqrt(var))


def weighted_quadfit(xs, ys, ws):
    """Weighted least squares for y = a + b*x + c*x^2. Returns (a, b, c, resid_std)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    # Solve the 3x3 normal equations by Gaussian elimination.
    s = [0.0] * 5
    t = [0.0] * 3
    for w, x, y in zip(ws, xs, ys):
        p = [1.0, x, x * x]
        for i in range(3):
            t[i] += w * p[i] * y
            for k in range(3):
                s[i + k] += w * p[i] * p[k]
    # s[0]=S0 s[1]=S1 s[2]=S2 s[3]=S3 s[4]=S4; matrix rows use S0..S4.
    m = [[s[0], s[1], s[2], t[0]], [s[1], s[2], s[3], t[1]], [s[2], s[3], s[4], t[2]]]
    n = 3
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    coef = [0.0] * 3
    for r in range(n - 1, -1, -1):
        coef[r] = (m[r][n] - sum(m[r][c] * coef[c] for c in range(r + 1, n))) / m[r][r]
    resid = [y - (coef[0] + coef[1] * x + coef[2] * x * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    return (coef[0], coef[1], coef[2], math.sqrt(var))


def distinct_reads(case):
    """Collapse consecutive identical reads (one Exophase refresh) to one
    observation each: (hours since launch, players, [rates])."""
    out = []
    prev = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev:
            continue
        prev = key
        out.append(
            (
                hours_since_launch(parse_time(read["at"])),
                float(read["players_tracked"]),
                [float(r) if r is not None else None for r in read["rates"]],
            )
        )
    return out


def fit_loglogistic(ts, es, ws):
    """Fit E(t) = E_inf / (1 + (tm/t)^k) on log E by grid search over
    (tm, k); E_inf is profiled out. Returns (E_at_target_fn, resid_std)."""
    best = None
    for k in (1.2, 1.6, 2.0, 2.5, 3.0):
        tm = 5.0
        while tm <= 6000.0:
            # log F for each observation; F = 1/(1+(tm/t)^k)
            lfs = []
            ok = True
            for t in ts:
                if t <= 0:
                    ok = False
                    break
                f = 1.0 / (1.0 + (tm / t) ** k)
                lfs.append(math.log(max(f, 1e-12)))
            if not ok:
                tm *= 1.35
                continue
            sw = sum(ws)
            # profile log E_inf = weighted mean of (log E - log F)
            le = sum(w * (math.log(e) - lf) for w, e, lf in zip(ws, es, lfs)) / sw
            sse = sum(
                w * (math.log(e) - (le + lf)) ** 2 for w, e, lf in zip(ws, es, lfs)
            )
            if best is None or sse < best[0]:
                best = (sse, k, tm, le, sw)
            tm *= 1.35
    if best is None:
        return None
    _, k, tm, le, sw = best
    resid = math.sqrt(best[0] / sw)

    def predict(t):
        f = 1.0 / (1.0 + (tm / max(t, 1e-9)) ** k)
        return math.exp(le) * f

    return (predict, resid)


def tier_cap(trophy):
    """How many times a trophy's earners may grow from the last read to the
    target before the forecast is capped (guards degenerate fits)."""
    d = trophy["description"].lower()
    if trophy["tier"] == "platinum" or "100% completion" in d:
        return 10.0
    if any(
        k in d
        for k in [
            "250 stunts",
            "50 challenges",
            "100 locations",
            "collectible card sets",
            "meditate at every spot",
            "explore all of leonida",
            "8 mission strands",
            "gold medal",
            "top fun",
            "20 different races",
            "5 bank vaults",
        ]
    ):
        return 30.0
    return 100.0


def tier_floor(trophy):
    """A floor for the median rate (percent) when a trophy has essentially
    no earners yet: the rarest trophies are still earned by somebody."""
    if trophy["tier"] == "platinum":
        return 0.02
    if "100% completion" in trophy["description"].lower():
        return 0.03
    return 0.05


def forecast_trophy(obs, trophy, t_target, horizon_days):
    """obs: list of (t_hours, N, rate) for this trophy, oldest first.
    Returns (median_rate, sigma_logit)."""
    # Recency weights: 12-hour half-life.
    t_last = obs[-1][0]
    ws = [0.5 ** ((t_last - t) / 12.0) for t, _, _ in obs]
    xs = [math.log1p(max(t, 0.0)) for t, _, _ in obs]

    # --- players tracked N(t): quadratic in log(1+t), else linear.
    n_vals = [n for _, n, _ in obs]
    ln_n = [math.log(max(n, 1.0)) for n in n_vals]
    n_target = None
    n_resid = 0.4
    quad = weighted_quadfit(xs, ln_n, ws)
    if quad is not None:
        a, b, c, r = quad
        xt = math.log1p(max(t_target, 0.0))
        n_target = math.exp(a + b * xt + c * xt * xt)
        n_resid = r
    lin = weighted_linfit(xs, ln_n, ws)
    if lin is not None:
        a, b, r = lin
        xt = math.log1p(max(t_target, 0.0))
        n_lin = math.exp(a + b * xt)
        n_target = n_lin if n_target is None else math.sqrt(max(n_target, 1.0) * n_lin)
        n_resid = max(n_resid, r) if quad is not None else r
    n_last = n_vals[-1]
    if n_target is None or not (n_target > 0):
        n_target = n_last
    n_target = min(max(n_target, n_last), n_last * 10.0)

    # --- earners E(t): power law and saturating log-logistic, averaged.
    es = []
    for (_, n, r) in obs:
        es.append(max(r / 100.0 * n, 0.5))
    ln_e = [math.log(e) for e in es]
    e_last = es[-1]
    e_fit = weighted_linfit(xs, ln_e, ws)
    e_pl = None
    e_resid = 0.4
    if e_fit is not None:
        a, b, r = e_fit
        xt = math.log1p(max(t_target, 0.0))
        e_pl = math.exp(a + b * xt)
        e_resid = r
    ll = fit_loglogistic([t for t, _, _ in obs], es, ws)
    e_ll = None
    if ll is not None:
        predict, r = ll
        e_ll = predict(max(t_target, 0.0))
        e_resid = max(e_resid, r)
    if e_pl is not None and e_ll is not None:
        e_target = math.sqrt(max(e_pl, 1e-9) * max(e_ll, 1e-9))
        disagree = abs(math.log(max(e_pl, 1e-9) / max(e_ll, 1e-9)))
    elif e_pl is not None:
        e_target = e_pl
        disagree = 0.0
    elif e_ll is not None:
        e_target = e_ll
        disagree = 0.0
    else:
        e_target = e_last
        disagree = 0.0
    cap = tier_cap(trophy)
    e_target = min(max(e_target, e_last), e_last * cap)

    # --- median rate and spread.
    med = 100.0 * e_target / n_target
    med = max(med, tier_floor(trophy))
    med = clamp_pct(med)
    sigma = math.sqrt(
        e_resid**2 + n_resid**2 + (0.30 * math.sqrt(horizon_days)) ** 2 + 0.35**2
        + (0.5 * disagree) ** 2
    )
    sigma = min(max(sigma, 0.55), 2.2)
    return med, sigma


def forecast_decomp(case):
    """The E/N decomposition forecast for a whole case."""
    obs_all = distinct_reads(case)
    t_target = hours_since_launch(parse_time(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    rows = []
    flat = len(obs_all) <= 1
    for j, trophy in enumerate(case["trophies"]):
        series = [(t, n, rates[j]) for (t, n, rates) in obs_all if rates[j] is not None]
        if not series:
            med, sigma = tier_floor(trophy), 1.5
        else:
            med, sigma = forecast_trophy(series, trophy, t_target, horizon_days)
            if flat:
                # No refresh seen yet: the next refresh multiplies the
                # tracked players several-fold while earners grow slowly,
                # so rates fall. Widen and shift the median down.
                med = max(med * 0.45, 0.01)
                sigma = max(sigma, 1.2)
        lg = logit(med)
        qs = [round4(clamp_pct(logistic(lg + z * sigma))) for z in Z]
        qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_quad(case):
    """Quadratic logit trend in log(1+t), per trophy."""
    obs_all = distinct_reads(case)
    t_target = hours_since_launch(parse_time(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    rows = []
    for j, trophy in enumerate(case["trophies"]):
        series = [(t, rates[j]) for (t, _, rates) in obs_all if rates[j] is not None]
        t_last = series[-1][0] if series else 0.0
        ws = [0.5 ** ((t_last - t) / 12.0) for t, _ in series]
        xs = [math.log1p(max(t, 0.0)) for t, _ in series]
        ys = [logit(r) for _, r in series]
        med = series[-1][1] if series else 0.5
        sigma = 1.0
        quad = weighted_quadfit(xs, ys, ws) if len(series) >= 4 else None
        if quad is not None:
            a, b, c, r = quad
            xt = math.log1p(max(t_target, 0.0))
            pred = a + b * xt + c * xt * xt
            # Guard against a runaway quadratic: never move the median
            # more than 3 logit units from the last rate.
            last_lg = logit(series[-1][1])
            pred = min(max(pred, last_lg - 3.0), last_lg + 3.0)
            med = clamp_pct(logistic(pred))
            sigma = math.sqrt(r**2 + (0.30 * math.sqrt(horizon_days)) ** 2 + 0.35**2)
        else:
            lin = weighted_linfit(xs, ys, ws) if len(series) >= 2 else None
            if lin is not None:
                a, b, r = lin
                xt = math.log1p(max(t_target, 0.0))
                pred = a + b * xt
                last_lg = logit(series[-1][1])
                pred = min(max(pred, last_lg - 3.0), last_lg + 3.0)
                med = clamp_pct(logistic(pred))
                sigma = math.sqrt(r**2 + (0.30 * math.sqrt(horizon_days)) ** 2 + 0.35**2)
        sigma = min(max(sigma, 0.55), 2.2)
        lg = logit(med)
        qs = sorted(round4(clamp_pct(logistic(lg + z * sigma))) for z in Z)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


WEEKEND_W = 2.0


def eff_hours(t):
    """Effective playing hours since launch: calendar hours, except the
    first weekend (Saturday 00:00 to Monday 00:00 UTC, t in [48, 96))
    counts WEEKEND_W times, because that is when players actually
    accumulate playtime. Launch was Thursday 19 Nov 2026, so the forecast
    window (deadline Saturday to target Thursday) contains that whole
    weekend. WEEKEND_W = 2.0: weekend gaming hours run about twice
    weekday hours, and the ablation tournament picked 2.0 over 1.3/1.6
    (and over no weekend correction at all) across simulated launch
    regimes and weekend-window definitions."""
    if t <= 48.0:
        return t
    if t >= 96.0:
        return t + (WEEKEND_W - 1.0) * 48.0
    return 48.0 + WEEKEND_W * (t - 48.0)


def story_chain(case):
    """Ids of the story-progression trophies in order: Chapter 1, 2, 3, 4,
    then the final mission. Their earn rates cannot increase along the
    chain: nobody completes Chapter 2 without Chapter 1."""
    chain = []
    want = [
        "complete chapter 1.",
        "complete chapter 2.",
        "complete chapter 3.",
        "complete chapter 4.",
        "complete the final mission.",
    ]
    for w in want:
        for trophy in case["trophies"]:
            if trophy["description"].lower().strip() == w:
                chain.append(trophy["id"])
                break
    return chain


def forecast_ship(case):
    """The shipped forecaster: the house curve baseline, fitted on
    effective playing time instead of calendar time.

    Per trophy, OLS of logit(earn rate) on ln(1 + effective hours since
    launch) over the last 24 h of distinct reads, read at the target;
    spread = random-walk term (0.5 * sqrt(horizon in days)) plus the
    fit's residual, floored at 0.3 and capped at 2.5. The median is
    clamped to within 3 logit units of the last read rate.

    Why effective time: the house curve fits against calendar hours, so
    it treats the first weekend as ordinary time. But the forecast
    window (Sat 21 Nov 11:30 UTC to Thu 26 Nov 00:00 UTC) contains the
    whole first weekend, when players accumulate playtime roughly twice
    as fast. Fitting on effective time lets the trend extrapolate at
    weekend speed. In the ablation tournament (launch simulator,
    final-like cases, 216 cases across earn-speed regimes, weekend
    strengths and weekend-window definitions) this scored mean pinball
    0.40 against 0.42 for the plain calendar-time curve and 1.23 for
    the flat house baseline; the E/N decomposition (0.74), the
    quadratic trend (0.68) and blends with them all scored worse, so
    they are kept below only as the ablated variants.

    Flat history (practice cases: every read identical): there is no
    trend to fit, and the next refresh drops rates hard as the
    denominator explodes, so the median is halved and the spread is
    widened to at least 1.2, matching the launch-night pattern."""
    obs_all = distinct_reads(case)
    t_target = hours_since_launch(parse_time(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    flat = len(obs_all) <= 1
    tau_target = eff_hours(max(t_target, 0.0))
    rows = []
    for j, trophy in enumerate(case["trophies"]):
        series = [(t, rates[j]) for (t, _, rates) in obs_all if rates[j] is not None]
        if not series:
            rows.append({"id": trophy["id"], "q": [0.05, 0.05, 0.05, 0.05, 0.05]})
            continue
        last_rate = series[-1][1]
        med = last_rate
        sigma = 0.5 * math.sqrt(horizon_days)
        recent = [(t, r) for (t, r) in series if t >= series[-1][0] - 24.0]
        if len(recent) >= 3:
            xs = [math.log1p(eff_hours(max(t, 0.0))) for t, _ in recent]
            ys = [logit(r) for _, r in recent]
            fit = weighted_linfit(xs, ys, [1.0] * len(xs))
            if fit is not None:
                a, b, resid = fit
                pred = a + b * math.log1p(tau_target)
                last_lg = logit(last_rate)
                pred = min(max(pred, last_lg - 3.0), last_lg + 3.0)
                med = clamp_pct(logistic(pred))
                sigma = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2 + resid**2)
        if flat:
            # The refresh dilutes earners with fresh syncers, but a
            # trophy near 100% cannot dilute: fresh players complete the
            # intro mission too, so they arrive as earners. Taper the
            # drop from 0.45x at low rates to 1.0x at 100%.
            drop = 0.45 + 0.55 * (last_rate / 100.0) ** 2
            med = max(med * drop, 0.01)
            sigma = max(sigma, 1.2)
        med = max(med, tier_floor(trophy))
        sigma = min(max(sigma, 0.3), 2.5)
        lg = logit(med)
        qs = sorted(round4(clamp_pct(logistic(lg + z * sigma))) for z in Z)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_v2(case):
    """Two low-variance views of each trophy, both fitted on the last 36 h
    of reads only (the launch-night collapse must not flatten the slope):
    the rate's logit trend on effective time, and the earners' trend
    divided by a single global linear fit of the tracked-players curve
    (the denominator is shared by all 55 trophies). Story-chain order is
    enforced on the medians."""
    obs_all = distinct_reads(case)
    t_target = hours_since_launch(parse_time(case["target"]))
    tau_target = eff_hours(max(t_target, 0.0))
    horizon_days = case["horizon_minutes"] / 1440.0
    flat = len(obs_all) <= 1
    t_last = obs_all[-1][0]

    # --- global players-tracked fit: linear in log(1+effective hours),
    # last 48 h only, growth damped (log N is concave) and clamped.
    n_last = obs_all[-1][1]
    n_target = n_last
    n_resid = 0.4
    recent_n = [(t, n) for (t, n, _) in obs_all if t >= t_last - 48.0]
    if len(recent_n) >= 2:
        xs = [math.log1p(eff_hours(max(t, 0.0))) for t, _ in recent_n]
        ys = [math.log(max(n, 1.0)) for _, n in recent_n]
        lin = weighted_linfit(xs, ys, [1.0] * len(xs))
        if lin is not None:
            a, b, n_resid = lin
            growth = math.exp(b * (math.log1p(tau_target) - math.log1p(eff_hours(max(t_last, 0.0)))))
            growth = max(growth, 1.0) ** 0.85
            n_target = min(n_last * growth, n_last * 10.0)

    medians = {}
    sigmas = {}
    for j, trophy in enumerate(case["trophies"]):
        series = [(t, n, rates[j]) for (t, n, rates) in obs_all if rates[j] is not None]
        if not series:
            medians[trophy["id"]] = tier_floor(trophy)
            sigmas[trophy["id"]] = 1.5
            continue
        t_s = series[-1][0]
        recent = [(t, n, r) for (t, n, r) in series if t >= t_s - 36.0]
        if len(recent) < 2:
            recent = series[-2:]
        xs = [math.log1p(eff_hours(max(t, 0.0))) for t, _, _ in recent]
        xt = math.log1p(tau_target)
        # Rate view: logit(rate) linear in log(1+effective hours).
        ys_r = [logit(r) for _, _, r in recent]
        fit_r = weighted_linfit(xs, ys_r, [1.0] * len(xs))
        lg_rate = logit(recent[-1][2])
        resid_r = 0.3
        if fit_r is not None:
            a, b, resid_r = fit_r
            lg_rate = a + b * xt
            last_lg = logit(recent[-1][2])
            lg_rate = min(max(lg_rate, last_lg - 3.0), last_lg + 3.0)
        # Earner view: log(earners) linear in log(1+effective hours).
        es = [max(r / 100.0 * n, 0.5) for _, n, r in recent]
        ys_e = [math.log(e) for e in es]
        fit_e = weighted_linfit(xs, ys_e, [1.0] * len(xs))
        lg_earn = logit(recent[-1][2])
        resid_e = 0.3
        if fit_e is not None:
            a, b, resid_e = fit_e
            e_t = math.exp(a + b * xt)
            e_t = min(max(e_t, es[-1]), es[-1] * tier_cap(trophy))
            lg_earn = logit(clamp_pct(100.0 * e_t / n_target))
        lg = 0.5 * (lg_rate + lg_earn)
        med = clamp_pct(logistic(lg))
        med = max(med, tier_floor(trophy))
        sigma = math.sqrt(
            resid_r**2 + resid_e**2 + n_resid**2
            + (0.30 * math.sqrt(horizon_days)) ** 2 + 0.35**2
            + (0.5 * abs(lg_rate - lg_earn)) ** 2
        )
        sigma = min(max(sigma, 0.5), 2.0)
        if flat:
            med = max(med * 0.45, 0.01)
            sigma = max(sigma, 1.2)
        medians[trophy["id"]] = med
        sigmas[trophy["id"]] = sigma

    # Enforce the story chain: rates cannot rise along it.
    chain = story_chain(case)
    for a, b in zip(chain, chain[1:]):
        if medians[b] > medians[a]:
            medians[b] = medians[a]
    # The platinum cannot exceed the 100%-completion gold.
    plat = None
    comp = None
    for trophy in case["trophies"]:
        if trophy["tier"] == "platinum":
            plat = trophy["id"]
        if "100% completion" in trophy["description"].lower():
            comp = trophy["id"]
    if plat is not None and comp is not None and medians[plat] > medians[comp]:
        medians[plat] = medians[comp]

    rows = []
    for trophy in case["trophies"]:
        lg = logit(medians[trophy["id"]])
        sigma = sigmas[trophy["id"]]
        qs = sorted(round4(clamp_pct(logistic(lg + z * sigma))) for z in Z)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_ensemble(case):
    """Median: average of the decomposition and quadratic medians on the
    logit scale. Spread: the wider of the two, plus their disagreement."""
    d = {r["id"]: r["q"] for r in forecast_decomp(case)["trophies"]}
    q = {r["id"]: r["q"] for r in forecast_quad(case)["trophies"]}
    rows = []
    for trophy in case["trophies"]:
        tid = trophy["id"]
        lg_d = logit(d[tid][2])
        lg_q = logit(q[tid][2])
        lg = 0.5 * (lg_d + lg_q)
        # Recover each model's sigma from its 5%-95% width.
        sig_d = (logit(d[tid][4]) - logit(d[tid][0])) / (2 * 1.645)
        sig_q = (logit(q[tid][4]) - logit(q[tid][0])) / (2 * 1.645)
        sigma = math.sqrt(max(sig_d, sig_q) ** 2 + (0.5 * abs(lg_d - lg_q)) ** 2)
        sigma = min(max(sigma, 0.55), 2.2)
        qs = sorted(round4(clamp_pct(logistic(lg + z * sigma))) for z in Z)
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


# The model that ships: chosen by the backtest tournament in dev/.
FORECAST = forecast_ship


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the contest's five")
    if not case["history"]:
        raise ValueError("case has no history")
    return FORECAST(case)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
