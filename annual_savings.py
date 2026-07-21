#!/usr/bin/env python3
"""
Full-coverage battery-savings estimate — every hour of 2024-2025, both markets.

Per hour, from the day-ahead bid stack, this caches everything needed to apply
any scarcity-price treatment afterward without rebuilding stacks:

  - the anchored counterfactual price on the full stack (paper-parallel variant)
    and on the battery-offer-removed stack (primary variant; batteries don't
    exist in the no-battery world, so their offers shouldn't either);
  - stack totals + exhaustion flags (did q* + power_storage run off the top?);
  - the production-cost integrand along the battery-free stack between the two
    operating points, cached as the exact traversed (price, dq) segments so any
    hour-specific cap — e.g. the empirical scarcity price — is priced later,
    exactly, without rebuilding the stack.

Treatments applied in `aggregate` (post-processing, seconds):
  floor    counterfactual price capped at the max observed DAM price in that
           market-month (the LOWESS-like "prices stay at observed levels" bound)
  central  capped at max(empirical pre-buildout price at the counterfactual net
           load [scarcity_price.py], month-max observed) — the headline
  ceiling  the administrative offer cap ($5,000 ERCOT / $1,000 CAISO)

Metrics: consumer savings (dprice x load) and production-cost savings (stack
integral). Bootstrap CIs resample days within each month.

Commands:
    python annual_savings.py compute caiso     (~10 min from cached zips)
    python annual_savings.py compute ercot     (~1-3 h from cached parquets)
    python annual_savings.py aggregate         (tables + figures)
"""

import calendar
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIG_DIR = os.path.join(HERE, "figs")

# Bump when the per-day cache schema or the numerical method changes. Caches are
# written under data/annual/v{CACHE_VERSION}/ and every file also carries a
# `cache_version` column, so a bump forces a clean rebuild instead of silently
# reusing stale results (see compute()); older versions stay on disk untouched
# for before/after comparison.
CACHE_VERSION = 2
ANNUAL_DIR = os.path.join(REPO, "data", "annual", f"v{CACHE_VERSION}")

YEARS = [2024, 2025]
ADMIN_CAP = {"ercot": 5000.0, "caiso": 1000.0}


# ── numpy stack helpers ─────────────────────────────────────────────────────────

def _arrays(stack):
    """stack DataFrame [price, mw, cum_mw] sorted by price -> numpy arrays."""
    p = stack["price"].to_numpy(dtype=float)
    mw = stack["mw"].to_numpy(dtype=float)
    cum = stack["cum_mw"].to_numpy(dtype=float)
    return p, mw, cum


def anchor_np(p, cum, lam, edge="right"):
    """Cumulative quantity where the stack first reaches the observed price `lam`.

    edge="right" returns the right edge of the first block priced >= lam (cum[i]);
    edge="left" returns that block's left edge (cum[i-1], or 0 for the first
    block). The two bracket the flat marginal block; the choice is a methodology
    decision (see the aggregate() left-vs-right anchoring sensitivity report).
    """
    i = np.searchsorted(p, lam, side="left")
    if i >= len(p):
        return float(cum[-1])
    if edge == "left":
        return float(cum[i - 1]) if i > 0 else 0.0
    return float(cum[i])


def price_np(p, cum, q, fill):
    i = np.searchsorted(cum, q, side="left")
    if i >= len(p):
        return fill
    return float(p[i])


def step_xy(cum, price):
    """Staircase plot coordinates matching the price_np lookup convention:
    price[i] occupies the quantity interval (cum[i-1], cum[i]], with the first
    block running from 0. Use as ax.step(*step_xy(cum, price), where="pre") or
    ax.plot(*step_xy(cum, price)). Callers add any base offset to the x array."""
    cum = np.asarray(cum, dtype=float)
    price = np.asarray(price, dtype=float)
    x = np.concatenate([[0.0], cum])
    y = np.concatenate([price[:1], price])
    return x, y


def traversed_np(p, mw, cum, q1, q2, fill):
    """Segments the signed sweep q1 -> q2 spans on the stack.

    Returns (seg_price, seg_dq), where seg_dq carries the sweep sign and any
    quantity beyond the top of the stack is priced at `fill`. For any cap,
        sum(minimum(seg_price, cap) * seg_dq)
    is the signed integral of min(price, cap) dq from q1 to q2. This is the exact
    replacement for the old fixed cap grid: it counts the top partial segment
    (which the searchsorted-slice integral dropped) and prices any cap directly,
    including caps below the former $100 grid floor.
    """
    if q1 == q2:
        return np.empty(0), np.empty(0)
    sign = 1.0 if q2 > q1 else -1.0
    lo, hi = (q1, q2) if q2 > q1 else (q2, q1)
    lo = max(lo, 0.0)
    left = cum - mw
    top = float(cum[-1])
    prices, dqs = [], []
    hi_in = min(hi, top)
    if hi_in > lo:
        i0 = np.searchsorted(cum, lo, side="left")
        i1 = np.searchsorted(cum, hi_in, side="left")   # segment containing hi_in
        sl = slice(i0, i1 + 1)
        ov = (np.minimum(cum[sl], hi_in) - np.maximum(left[sl], lo)).clip(min=0.0)
        keep = ov > 0
        prices.append(p[sl][keep])
        dqs.append(ov[keep])
    if hi > top:                                        # beyond the offered supply
        prices.append(np.array([float(fill)]))
        dqs.append(np.array([hi - max(lo, top)]))
    if not prices:
        return np.empty(0), np.empty(0)
    return np.concatenate(prices), np.concatenate(dqs) * sign


def integral_np(p, mw, cum, q1, q2, cap, fill):
    """Signed integral of min(price, cap) dq from q1 to q2 (cap=None -> uncapped);
    thin wrapper over traversed_np so both share the corrected segment logic."""
    pr, dq = traversed_np(p, mw, cum, q1, q2, fill)
    if pr.size == 0:
        return 0.0
    if cap is not None:
        pr = np.minimum(pr, cap)
    return float(pr @ dq)


# ── per-day computation ─────────────────────────────────────────────────────────

def _hour_record(stacks_full, stacks_nb, hour, lam, ps, load, nl, market):
    fill = ADMIN_CAP[market]
    rec = dict(hour=hour, lam=lam, ps=ps, load=load, net_load=nl)

    # full (paper-parallel) stack — right-edge anchor only
    p, mw, cum = _arrays(stacks_full)
    q = anchor_np(p, cum, lam)
    rec["q_full"] = q
    rec["tot_full"] = float(cum[-1])
    rec["p_full"] = min(price_np(p, cum, q + ps, fill), fill)
    rec["exh_full"] = bool(q + ps > cum[-1])

    # battery-free stack — the primary variant. Compute under both anchor
    # conventions (right = headline, left = sensitivity) and cache the exact
    # traversed (price, dq) segments in place of the old int_c{cap} grid, so any
    # hour-specific cap is priced later without rebuilding the stack.
    p, mw, cum = _arrays(stacks_nb)
    rec["tot_nb"] = float(cum[-1])
    for edge, suf in [("right", ""), ("left", "_left")]:
        q = anchor_np(p, cum, lam, edge=edge)
        seg_p, seg_dq = traversed_np(p, mw, cum, q, q + ps, fill)
        rec[f"q_nb{suf}"] = q
        rec[f"p_nb{suf}"] = min(price_np(p, cum, q + ps, fill), fill)
        rec[f"exh_nb{suf}"] = bool(q + ps > cum[-1])
        rec[f"seg_price{suf}"] = seg_p
        rec[f"seg_dq{suf}"] = seg_dq
        rec[f"int_raw{suf}"] = float(seg_p @ seg_dq) if seg_p.size else 0.0
    return rec


def _day_caiso(date, day_csv):
    import caiso_dam_counterfactual as cc
    df = cc.load_day(date)
    stacks_f = cc.hourly_stacks(df, date)
    stacks_n = cc.hourly_stacks(df, date, exclude_battery=True)
    rows = []
    for h in range(24):
        if h not in stacks_f or h not in stacks_n or h not in day_csv.index:
            continue
        r = day_csv.loc[h]
        if not (np.isfinite(r["dam_price"]) and np.isfinite(r["power_storage"])):
            continue
        rows.append(_hour_record(stacks_f[h], stacks_n[h], h,
                                 float(r["dam_price"]), float(r["power_storage"]),
                                 float(r["total_load_mw"]), float(r["net_load_mw"]),
                                 "caiso"))
    return rows


def _ercot_hour_stacks(g, e):
    """Parse each offer curve once; return (full, battery-free) stacks."""
    import dam_counterfactual as dc
    cap = g["hsl"] - g[dc.AS_GEN].fillna(0).sum(axis=1)
    cap = cap.where(cap > 0, g["hsl"])
    segs = []
    for c, curve, rtype in zip(cap.to_numpy(),
                               g["qse_submitted_curve"].to_numpy(),
                               g["resource_type"].to_numpy()):
        bat = rtype == "PWRSTR"
        for mw, price in dc._curve_segments(dc.parse_curve(curve),
                                            c if np.isfinite(c) else None):
            segs.append((mw, price, bat))
    for v in e["energy_only_offer_curve"].to_numpy():
        for mw, price in dc._curve_segments(dc.parse_curve(v)):
            segs.append((mw, price, False))
    df = pd.DataFrame(segs, columns=["mw", "price", "bat"]).sort_values(
        "price", kind="mergesort")
    full = df[["mw", "price"]].reset_index(drop=True)
    full["cum_mw"] = full["mw"].cumsum()
    nb = df.loc[~df["bat"], ["mw", "price"]].reset_index(drop=True)
    nb["cum_mw"] = nb["mw"].cumsum()
    return full, nb


def _day_ercot(date, day_csv):
    import dam_counterfactual as dc
    tag = f"{date:%Y%m%d}"
    gpath = os.path.join(REPO, "data", "dam", "days", f"gen_{tag}.parquet")
    epath = os.path.join(REPO, "data", "dam", "days", f"eo_{tag}.parquet")
    if not (os.path.exists(gpath) and os.path.exists(epath)):
        return None
    gen = pd.read_parquet(gpath)
    eo = pd.read_parquet(epath)
    for df in (gen, eo):
        df["interval_start_utc"] = pd.to_datetime(df["interval_start_utc"], utc=True)
    for c in ["hsl", "awarded_quantity"] + dc.AS_GEN:
        if c in gen:
            gen[c] = pd.to_numeric(gen[c], errors="coerce")
    eo_by_ts = dict(tuple(eo.groupby("interval_start_utc")))
    rows = []
    for ts, g in gen.groupby("interval_start_utc"):
        h = pd.Timestamp(ts).tz_convert(dc.LOCAL_TZ).hour
        if h not in day_csv.index:
            continue
        r = day_csv.loc[h]
        if not (np.isfinite(r["dam_price"]) and np.isfinite(r["power_storage"])):
            continue
        e = eo_by_ts.get(ts, eo.iloc[0:0])
        st_f, st_n = _ercot_hour_stacks(g, e)
        if st_f.empty or st_n.empty:
            continue
        rows.append(_hour_record(st_f, st_n, h,
                                 float(r["dam_price"]), float(r["power_storage"]),
                                 float(r["total_load_mw"]), float(r["net_load_mw"]),
                                 "ercot"))
    return rows


def _cache_ok(path):
    """A cached day is reusable only if it is readable, non-empty, and stamped
    with the current CACHE_VERSION."""
    try:
        meta = pd.read_parquet(path, columns=["cache_version"])
    except Exception:
        return False
    return len(meta) > 0 and int(meta["cache_version"].iloc[0]) == CACHE_VERSION


def compute(market, shard=0, nshards=1, force=False, force_date=None):
    if market == "caiso":
        from caiso_dam_counterfactual import load_paper_csv
        csv = load_paper_csv()
        day_fn = _day_caiso
    else:
        from dam_counterfactual import load_ercot_csv
        csv = load_ercot_csv()
        day_fn = _day_ercot
    out_dir = os.path.join(ANNUAL_DIR, market)
    os.makedirs(out_dir, exist_ok=True)

    dates = []
    for y in YEARS:
        d = dt.date(y, 1, 1)
        while d.year == y:
            dates.append(d)
            d += dt.timedelta(days=1)
    dates = [d for i, d in enumerate(dates) if i % nshards == shard]
    done = skipped = stale = 0
    for i, date in enumerate(dates):
        path = os.path.join(out_dir, f"{date:%Y%m%d}.parquet")
        do_force = force or (force_date is not None and date == force_date)
        if os.path.exists(path) and not do_force:
            if _cache_ok(path):
                skipped += 1
                continue
            stale += 1        # exists but wrong version / unreadable -> rebuild
        day_csv = csv[csv["date"] == date].set_index("hour")
        # DST fall-back days repeat a local hour; keep the first occurrence
        day_csv = day_csv[~day_csv.index.duplicated(keep="first")]
        if day_csv.empty:
            continue
        try:
            rows = day_fn(date, day_csv)
        except FileNotFoundError:
            print(f"  {date}: no bid data, skipped")
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.insert(0, "date", pd.Timestamp(date))
        df["cache_version"] = CACHE_VERSION
        tmp = f"{path}.tmp"
        df.to_parquet(tmp)
        os.replace(tmp, path)        # atomic: never leave a partial cache behind
        done += 1
        if done % 25 == 0:
            print(f"[{i+1}/{len(dates)}] {date} done={done} skipped={skipped}",
                  flush=True)
    print(f"{market}: wrote {done} days ({stale} rebuilt stale), "
          f"{skipped} reused (cache v{CACHE_VERSION} at {out_dir})")


# ── aggregation ─────────────────────────────────────────────────────────────────

def _load_hourly(market):
    import glob
    files = sorted(glob.glob(os.path.join(ANNUAL_DIR, market, "*.parquet")))
    if not files:
        sys.exit(f"no cached days for {market} — run compute first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def apply_treatments(df, market, variant="nb"):
    """Add per-hour treated counterfactual prices + savings columns.

    Three scarcity-price assumptions, each a cap on the raw stack price:
      floor    max observed DAM price in that market-month ("prices never
               exceed what the month demonstrably produced")
      central  the empirical pre-buildout price at the counterfactual net load,
               guarded hour-by-hour at the observed price (removing supply
               cannot lower the price) — NOT guarded at month-max, so one
               extreme observed hour does not set the cap for the whole month
      ceiling  the administrative offer cap
    floor and central are alternative assumptions, not nested bounds: where the
    empirical price is below the month's max, central < floor.
    """
    from scarcity_price import ScarcityModel
    model = ScarcityModel(market)
    p_raw = df[f"p_{variant}"]

    monthmax = df.groupby(["year", "month"])["lam"].transform("max")

    # empirical scarcity price at the counterfactual net load
    emp = np.array([model.price(m, nl + ps) for m, nl, ps in
                    zip(df["month"], df["net_load"], df["ps"])])
    cap_central = np.maximum(emp, df["lam"].to_numpy())
    cap_ceiling = ADMIN_CAP[market]

    df = df.copy()
    df["p_floor"] = np.minimum(p_raw, monthmax)
    df["p_central"] = np.minimum(p_raw, cap_central)
    df["p_ceiling"] = np.minimum(p_raw, cap_ceiling)
    df["cap_central"] = cap_central

    for t in ["floor", "central", "ceiling"]:
        df[f"cons_{t}"] = (df[f"p_{t}"] - df["lam"]) * df["load"]

    # production-cost savings: exact capped integral from the cached traversed
    # segments, prod_t = sum(min(seg_price, cap_t) * seg_dq) per hour. This prices
    # sub-$100 caps correctly (the old CAP_GRID/np.interp path clamped every cap
    # below $100 up to the $100 integral).
    def prod_at(cap_arr, price_col="seg_price", dq_col="seg_dq"):
        cap_arr = np.asarray(cap_arr, dtype=float)
        scalar = cap_arr.ndim == 0
        sp = df[price_col].to_numpy()
        sd = df[dq_col].to_numpy()
        out = np.empty(len(df))
        for i in range(len(df)):
            pr = np.asarray(sp[i], dtype=float)
            if pr.size == 0:
                out[i] = 0.0
                continue
            c = float(cap_arr) if scalar else cap_arr[i]
            out[i] = float(np.minimum(pr, c) @ np.asarray(sd[i], dtype=float))
        return out

    df["prod_floor"] = prod_at(monthmax.to_numpy())
    df["prod_central"] = prod_at(cap_central)
    df["prod_ceiling"] = prod_at(np.full(len(df), cap_ceiling))

    # anchoring sensitivity: the central treatment under the left-edge anchor,
    # when those columns were cached (see aggregate()'s comparison table).
    if "p_nb_left" in df and "seg_price_left" in df:
        df["cons_central_left"] = (np.minimum(df["p_nb_left"], cap_central)
                                   - df["lam"]) * df["load"]
        df["prod_central_left"] = prod_at(cap_central, "seg_price_left",
                                          "seg_dq_left")
    return df


def bootstrap_ci(df, col, n_boot=1000, seed=0):
    """95% CI on the annual sum of `col`, resampling days within each month."""
    rng = np.random.default_rng(seed)
    out = {}
    for y in YEARS:
        sub = df[df["year"] == y]
        daily = sub.groupby(["month", "date"])[col].sum().reset_index()
        sums = np.zeros(n_boot)
        for m, g in daily.groupby("month"):
            v = g[col].to_numpy()
            idx = rng.integers(0, len(v), size=(n_boot, len(v)))
            sums += v[idx].sum(axis=1)
        out[y] = (float(np.percentile(sums, 2.5)),
                  float(np.percentile(sums, 97.5)))
    return out


def drop_exhausted():
    """Savings computed on the surviving hours only — hours in which the
    no-battery counterfactual demand (q* + power_storage) exceeds the top of
    the offered supply stack are dropped entirely, not priced.

    For every surviving hour the counterfactual price is a genuine interior
    point on the stack (< offer cap by construction), so NO scarcity-price
    assumption enters. This is the estimate the bid stack can make without any
    extrapolation: it simply declines to speak for the ~1-3% of hours where the
    offered supply runs out.

    Note this is NOT a lower bound: it comes out ABOVE the empirical-central
    estimate. Dropping the exhausted hours removes only the offer-cap-pinned
    tail; every surviving hour is still priced at its raw, uncapped stack offer,
    which on tight evenings climbs the steep upper region toward the cap. The
    empirical-central estimate instead REPLACES those steep offered prices with
    what the pre-buildout market actually cleared at, which is much lower. So
    this variant answers a specific question — how much of the bid-stack total
    survives once we discard the hours the stack literally cannot represent —
    and the answer (still ~10-30x LOWESS) shows the gap is a property of the
    whole steep upper stack, not just the administrative cap."""
    os.makedirs(FIG_DIR, exist_ok=True)
    import matplotlib.pyplot as plt

    rows = []
    for market in ["ercot", "caiso"]:
        df = _load_hourly(market)
        surv = df[~df["exh_nb"]].copy()
        surv["cons"] = (surv["p_nb"] - surv["lam"]) * surv["load"]
        surv["prod"] = surv["int_raw"]
        if market == "ercot":
            from dam_counterfactual import load_ercot_csv
            from dam_monthly_savings import lowess_monthly
            low = lowess_monthly(load_ercot_csv())
        else:
            from caiso_dam_counterfactual import load_paper_csv, lowess_monthly
            low = lowess_monthly(load_paper_csv())
        for y in YEARS:
            s = surv[surv["year"] == y]
            allh = df[df["year"] == y]
            ne = int(allh["exh_nb"].sum())
            cc = bootstrap_ci(s, "cons")[y]
            pc = bootstrap_ci(s, "prod")[y]
            rows.append(dict(
                market=market, year=y,
                hours_total=len(allh), hours_dropped=ne,
                pct_dropped=100 * ne / len(allh),
                cons_drop=s["cons"].sum() / 1e6,
                cons_drop_lo=cc[0] / 1e6, cons_drop_hi=cc[1] / 1e6,
                prod_drop=s["prod"].sum() / 1e6,
                prod_drop_lo=pc[0] / 1e6, prod_drop_hi=pc[1] / 1e6,
                lowess=sum(v for (yy, m), v in low.items() if yy == y) / 1e6))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:9.1f}"))
    out.to_csv(os.path.join(FIG_DIR, "drop_exhausted_summary.csv"), index=False)
    print(f"saved {os.path.join(FIG_DIR, 'drop_exhausted_summary.csv')}")

    # figure: drop-exhausted vs central vs LOWESS (consumer side)
    smry = pd.read_csv(os.path.join(FIG_DIR, "annual_savings_summary.csv"))
    fig, ax = plt.subplots(figsize=(11, 6))
    xt, xl = [], []
    for i, r in out.iterrows():
        c = smry[(smry.market == r.market) & (smry.year == r.year)].iloc[0]
        ax.errorbar([i - 0.15], [r.cons_drop],
                    yerr=[[r.cons_drop - r.cons_drop_lo],
                          [r.cons_drop_hi - r.cons_drop]],
                    fmt="D", color="#009E73", ms=9, capsize=4, lw=2,
                    label="drop exhausted hours (no scarcity assumption)" if i == 0 else None)
        ax.plot([i + 0.05], [c.cons_central], "o", color="#D55E00", ms=9,
                label="central (empirical scarcity)" if i == 0 else None)
        ax.plot([i + 0.22], [c.lowess], "s", color="#0072B2", ms=8,
                label="LOWESS (paper)" if i == 0 else None)
        xt.append(i)
        xl.append(f"{r.market.upper()}\n{r.year}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_ylabel("Annual consumer savings ($M)")
    ax.set_title("Dropping the hours the bid stack cannot price\n"
                 "(no-battery demand exceeds the offered supply) vs the priced estimates")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    outp = os.path.join(FIG_DIR, "drop_exhausted_comparison.png")
    fig.savefig(outp, dpi=200)
    plt.close(fig)
    print(f"saved {outp}")


def aggregate():
    os.makedirs(FIG_DIR, exist_ok=True)
    import matplotlib.pyplot as plt

    summary = []
    monthly_frames = {}
    diag = []
    for market in ["ercot", "caiso"]:
        df = _load_hourly(market)
        df = apply_treatments(df, market)
        monthly_frames[market] = df

        # LOWESS benchmark (validated reproductions)
        if market == "ercot":
            from dam_counterfactual import load_ercot_csv
            from dam_monthly_savings import lowess_monthly
            low = lowess_monthly(load_ercot_csv())
        else:
            from caiso_dam_counterfactual import load_paper_csv, lowess_monthly
            low = lowess_monthly(load_paper_csv())

        for y in YEARS:
            sub = df[df["year"] == y]
            row = dict(market=market, year=y,
                       hours=len(sub),
                       lowess=sum(v for (yy, m), v in low.items() if yy == y))
            for metric in ["cons", "prod"]:
                for t in ["floor", "central", "ceiling"]:
                    row[f"{metric}_{t}"] = sub[f"{metric}_{t}"].sum()
            ci = bootstrap_ci(sub, "cons_central")
            row["cons_central_lo"], row["cons_central_hi"] = ci[y]
            ci = bootstrap_ci(sub, "prod_central")
            row["prod_central_lo"], row["prod_central_hi"] = ci[y]
            summary.append(row)

            # scarcity diagnostics
            scarce = sub[sub["p_nb"] > sub["cap_central"]]
            diag.append(dict(
                market=market, year=y, n_hours=len(sub),
                n_capped=len(scarce),
                pct_capped=100 * len(scarce) / max(len(sub), 1),
                capped_share_of_central=(
                    scarce["cons_central"].sum() / sub["cons_central"].sum()
                    if sub["cons_central"].sum() else np.nan),
                n_exhausted=int(sub["exh_nb"].sum())))

    smry = pd.DataFrame(summary)
    for c in smry.columns:
        if c not in ("market", "year", "hours"):
            smry[c] = smry[c] / 1e6
    out = os.path.join(FIG_DIR, "annual_savings_summary.csv")
    smry.to_csv(out, index=False, float_format="%.1f")
    print(smry.to_string(index=False, float_format=lambda v: f"{v:9.0f}"))
    print(f"saved {out}")
    dg = pd.DataFrame(diag)
    print(dg.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    dg.to_csv(os.path.join(FIG_DIR, "scarcity_diagnostics.csv"), index=False)

    # ── anchoring sensitivity: left- vs right-edge marginal-block anchor ──────
    # How much the central estimate moves if the operating point is anchored at
    # the left edge of the marginal offer block instead of the right (current)
    # edge. Consumer side gauges movement in the headline (Table 1) numbers.
    anch = []
    for market in ["ercot", "caiso"]:
        df = monthly_frames[market]
        if "cons_central_left" not in df:
            continue
        for y in YEARS:
            sub = df[df["year"] == y]
            anch.append(dict(
                market=market, year=y,
                cons_right=sub["cons_central"].sum() / 1e6,
                cons_left=sub["cons_central_left"].sum() / 1e6,
                prod_right=sub["prod_central"].sum() / 1e6,
                prod_left=sub["prod_central_left"].sum() / 1e6))
    if anch:
        ad = pd.DataFrame(anch)
        ad["cons_delta"] = ad["cons_left"] - ad["cons_right"]
        ad["prod_delta"] = ad["prod_left"] - ad["prod_right"]
        print("\nanchoring sensitivity — central estimate, $M "
              "(delta = left-edge minus right-edge anchor):")
        print(ad.to_string(index=False, float_format=lambda v: f"{v:9.1f}"))
        ad.to_csv(os.path.join(FIG_DIR, "anchor_sensitivity.csv"), index=False)
        print(f"saved {os.path.join(FIG_DIR, 'anchor_sensitivity.csv')}")

    # ── headline range figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, metric, label in [(axes[0], "cons", "Consumer savings"),
                              (axes[1], "prod", "Production-cost savings")]:
        xt, xl = [], []
        for i, (market, y) in enumerate([(m, y) for m in ["ercot", "caiso"]
                                         for y in YEARS]):
            r = smry[(smry.market == market) & (smry.year == y)].iloc[0]
            vals = [r[f"{metric}_floor"], r[f"{metric}_central"],
                    r[f"{metric}_ceiling"]]
            c = r[f"{metric}_central"]
            ax.plot([i, i], [min(vals), max(vals)], color="#999999", lw=6,
                    alpha=0.5, solid_capstyle="round",
                    label="range across scarcity assumptions" if i == 0 else None)
            ax.plot([i], [r[f"{metric}_floor"]], "v", color="#555555", ms=7,
                    label="observed-month-max cap" if i == 0 else None)
            ax.plot([i], [r[f"{metric}_ceiling"]], "^", color="#555555", ms=7,
                    label="offer-cap (administrative)" if i == 0 else None)
            ax.errorbar([i], [c],
                        yerr=[[c - r[f"{metric}_central_lo"]],
                              [r[f"{metric}_central_hi"] - c]],
                        fmt="o", color="#D55E00", ms=9, capsize=5, lw=2,
                        label="central (empirical scarcity) ±95% CI" if i == 0 else None)
            if metric == "cons":
                ax.plot([i], [r["lowess"]], "s", color="#0072B2", ms=8,
                        label="LOWESS (paper method)" if i == 0 else None)
            xt.append(i)
            xl.append(f"{market.upper()}\n{y}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(xt)
        ax.set_xticklabels(xl)
        ax.set_ylabel(f"{label} ($M/yr)")
        ax.set_yscale("symlog", linthresh=1000)
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9, loc="best")
    fig.suptitle("Battery savings, full-coverage bid-stack estimate (all hours "
                 "2024–2025, battery offers removed)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "annual_defensible_range.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")

    # ── monthly central vs LOWESS per market ────────────────────────────────
    for market in ["ercot", "caiso"]:
        df = monthly_frames[market]
        if market == "ercot":
            from dam_counterfactual import load_ercot_csv
            from dam_monthly_savings import lowess_monthly
            low = lowess_monthly(load_ercot_csv())
        else:
            from caiso_dam_counterfactual import load_paper_csv, lowess_monthly
            low = lowess_monthly(load_paper_csv())
        mo = df.groupby(["year", "month"])[["cons_floor", "cons_central",
                                            "cons_ceiling"]].sum().reset_index()
        mo["lowess"] = [low.get((y, m), np.nan) for y, m in
                        zip(mo["year"], mo["month"])]
        mo["ym"] = (mo["year"].astype(str) + "-"
                    + mo["month"].astype(str).str.zfill(2))
        fig, ax = plt.subplots(figsize=(14, 6.5))
        x = np.arange(len(mo))
        lo = mo[["cons_floor", "cons_central", "cons_ceiling"]].min(axis=1)
        hi = mo[["cons_floor", "cons_central", "cons_ceiling"]].max(axis=1)
        ax.fill_between(x, lo / 1e6, hi / 1e6, color="#999999", alpha=0.3,
                        label="range across scarcity assumptions")
        ax.plot(x, mo["cons_central"] / 1e6, "o-", color="#D55E00", lw=2,
                label="central (empirical scarcity)")
        ax.plot(x, mo["lowess"] / 1e6, "s-", color="#0072B2", lw=1.5,
                label="LOWESS (paper method)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(mo["ym"], rotation=90, fontsize=8)
        ax.set_ylabel("Monthly consumer savings ($M)")
        ax.set_yscale("symlog", linthresh=100)
        ax.set_title(f"{market.upper()}: monthly consumer savings — full-coverage "
                     "bid stack vs LOWESS")
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"{market}_monthly_central_vs_lowess.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "aggregate"
    if cmd == "compute":
        rest = sys.argv[2:]
        force = "--force" in rest
        force_date = None
        for a in rest:
            if a.startswith("--force-date="):
                force_date = dt.date.fromisoformat(a.split("=", 1)[1])
        pos = [a for a in rest if not a.startswith("--")]
        market = pos[0]
        shard = int(pos[1]) if len(pos) > 1 else 0
        nshards = int(pos[2]) if len(pos) > 2 else 1
        compute(market, shard, nshards, force=force, force_date=force_date)
    elif cmd == "aggregate":
        aggregate()
    elif cmd == "drop-exhausted":
        drop_exhausted()
    else:
        sys.exit(f"unknown command {cmd}")
