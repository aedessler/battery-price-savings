#!/usr/bin/env python3
"""
CAISO DAM bid-stack counterfactual — the ERCOT-parallel analysis.

Builds the day-ahead supply curve for CAISO from OASIS public bid data
(PUB_DAM_GRP GroupZip; masked bids, 90-day lag, no API key), then applies the
same anchoring method as ERCOT's dam_counterfactual.py / dam_monthly_savings.py:

  1. Supply stack per hour = all EN (energy) bid-curve segments with MW >= 0
     from GENERATOR + INTERTIE resources, plus self-scheduled GENERATOR/INTERTIE
     MW as price-takers at the bid floor (-$150). NGR batteries' negative-MW
     (charging) segments are demand, not supply, and are excluded.
  2. Anchor: find q* where the stack price reaches the observed DAM price
     (paper CSV dam_price). Do NOT trust the absolute clearing reconstruction.
  3. Counterfactual: delta = S(q* + power_storage) - dam_price, where
     power_storage is the fleet net output from the paper's CSV (+ = discharge).
     Same construction as the paper's LOWESS net-load shift.

Sampled days (5, 12, 19, 26 of each month, 2024-2025) mirror ERCOT's
dam_monthly_savings.py so the two markets are directly comparable.

Commands:
    python caiso_dam_counterfactual.py download          # 96 sampled days (~12 min)
    python caiso_dam_counterfactual.py run [YYYY-MM-DD]  # one-day validation figs
    python caiso_dam_counterfactual.py compute           # monthly rollup vs LOWESS
"""

import calendar
import datetime as dt
import glob
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO      = os.path.dirname(HERE)
from core import interp_lowess   # local copy in this folder  # noqa: E402
BID_DIR   = os.path.join(REPO, "data", "caiso", "pub_bids")
FIG_DIR   = os.path.join(HERE, "figs")
CAISO_CSV = os.path.join(HERE, "data", "caiso", "complete_caiso_2020_2025.csv")

SAMPLE_DAYS  = [5, 12, 19, 26]
YEARS_MONTHS = [(y, m) for y in (2024, 2025) for m in range(1, 13)]
PRICE_FLOOR  = -150.0          # CAISO energy bid floor ($/MWh)
PRICE_CAP    = 1000.0          # CAISO soft offer cap


# ── Download ─────────────────────────────────────────────────────────────────────

def download():
    from download_caiso_bids import download_day
    import time
    os.makedirs(BID_DIR, exist_ok=True)
    from pathlib import Path
    dates = [dt.date(y, m, d) for (y, m) in YEARS_MONTHS for d in SAMPLE_DAYS]
    print(f"downloading {len(dates)} sampled DAM days -> {BID_DIR}")
    failures = []
    for i, date in enumerate(dates):
        if i:
            time.sleep(5.0)
        print(f"{date}:")
        if not download_day("dam", date, Path(BID_DIR), 6, extract=False):
            failures.append(date)
    if failures:
        print(f"\nFAILED: {failures}")
        sys.exit(1)
    print("\nall days downloaded")


# ── Stack construction ───────────────────────────────────────────────────────────

def load_day(date):
    """Read one trade day's public-bid CSV (from the cached zip)."""
    path = os.path.join(BID_DIR, f"{date:%Y%m%d}_PUB_BID_DAM_csv.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as zf:
        name = [n for n in zf.namelist() if "PUB_BID" in n][0]
        with zf.open(name) as f:
            df = pd.read_csv(f, low_memory=False,
                             usecols=["RESOURCE_TYPE", "RESOURCEBID_SEQ",
                                      "MARKETPRODUCTTYPE", "SELFSCHEDMW",
                                      "TIMEINTERVALSTART", "TIMEINTERVALEND",
                                      "SCH_BID_TIMEINTERVALSTART",
                                      "SCH_BID_TIMEINTERVALSTOP",
                                      "SCH_BID_XAXISDATA", "SCH_BID_Y1AXISDATA"])
    return df[df["MARKETPRODUCTTYPE"] == "EN"]


def _expand_hours(df, start_col, stop_col, date):
    """Bids cover multi-hour blocks (up to the whole trade day, recorded once);
    replicate each row into every local hour in [start, stop)."""
    day0 = pd.Timestamp(date)
    t0 = pd.to_datetime(df[start_col])
    t1 = pd.to_datetime(df[stop_col])
    h0 = ((t0 - day0).dt.total_seconds() // 3600).astype(int)
    n = ((t1 - t0).dt.total_seconds() // 3600).clip(lower=1).astype(int)
    ex = df.loc[df.index.repeat(n)].copy()
    ex["hour"] = h0.loc[ex.index] + ex.groupby(level=0).cumcount()
    return ex[(ex["hour"] >= 0) & (ex["hour"] <= 23)]


def storage_seqs(df):
    """RESOURCEBID_SEQs whose EN bid curve spans negative MW anywhere in the day —
    the NGR (battery) signature: the curve covers charging (x<0) through
    discharging (x>0)."""
    cur = df[df["SCH_BID_XAXISDATA"].notna()]
    return set(cur.loc[cur["SCH_BID_XAXISDATA"] < 0, "RESOURCEBID_SEQ"].unique())


def hourly_stacks(df, date, exclude_battery=False):
    """dict hour_local(0-23) -> stack DataFrame [price, mw, cum_mw] sorted by price.

    exclude_battery drops all bid segments of storage resources (identified by
    negative-MW curve support) — the no-battery supply stack. Self-schedules
    cannot be attributed to storage and are kept in both variants (caveat).

    Segment construction is vectorized (groupby.shift) so a full day builds in
    well under a second — required for the 731-day full-coverage run.
    """
    supply_types = ["GENERATOR", "INTERTIE"]

    cur = df[(df["SCH_BID_XAXISDATA"].notna())
             & (df["RESOURCE_TYPE"].isin(supply_types))].copy()
    if exclude_battery:
        cur = cur[~cur["RESOURCEBID_SEQ"].isin(storage_seqs(df))]
    cur = _expand_hours(cur, "SCH_BID_TIMEINTERVALSTART",
                        "SCH_BID_TIMEINTERVALSTOP", date)

    ss = df[(df["SELFSCHEDMW"].notna()) & (df["SELFSCHEDMW"] > 0)
            & (df["RESOURCE_TYPE"].isin(supply_types))].copy()
    ss = _expand_hours(ss, "TIMEINTERVALSTART", "TIMEINTERVALEND", date)
    ss_by_hour = ss.groupby("hour")["SELFSCHEDMW"].sum()

    # consecutive curve points -> (lo, hi, price) segments, vectorized
    cur = cur.sort_values(["hour", "RESOURCEBID_SEQ", "SCH_BID_XAXISDATA"],
                          kind="stable")
    grp = cur.groupby(["hour", "RESOURCEBID_SEQ"], sort=False)
    cur["hi"] = grp["SCH_BID_XAXISDATA"].shift(-1)
    cur["npts"] = grp["SCH_BID_XAXISDATA"].transform("size")
    # single-point curves = flat offer of size x at price y
    single = cur["npts"] == 1
    cur.loc[single, "hi"] = cur.loc[single, "SCH_BID_XAXISDATA"]
    cur.loc[single, "SCH_BID_XAXISDATA"] = 0.0
    seg = cur.dropna(subset=["hi"]).copy()
    seg["lo"] = seg["SCH_BID_XAXISDATA"].clip(lower=0.0)  # clip charging range
    seg["mw"] = seg["hi"] - seg["lo"]
    seg = seg[seg["mw"] > 1e-9]
    seg = seg.rename(columns={"SCH_BID_Y1AXISDATA": "price"})[
        ["hour", "price", "mw"]]

    stacks = {}
    for hour, gh in seg.groupby("hour"):
        stack = gh[["price", "mw"]]
        ss_mw = float(ss_by_hour.get(hour, 0.0))
        if ss_mw > 0:
            stack = pd.concat([pd.DataFrame({"price": [PRICE_FLOOR], "mw": [ss_mw]}),
                               stack], ignore_index=True)
        stack = stack.sort_values("price", kind="stable").reset_index(drop=True)
        stack["cum_mw"] = stack["mw"].cumsum()
        stacks[hour] = stack
    return stacks


def price_at(stack, q):
    """Stack price at cumulative quantity q; offer cap if the stack is exhausted."""
    if q <= 0:
        return float(stack["price"].iloc[0])
    hit = stack[stack["cum_mw"] >= q]
    if hit.empty:
        return PRICE_CAP
    return float(hit["price"].iloc[0])


def anchor_quantity(stack, price):
    """Smallest cumulative MW at which the stack reaches `price`."""
    hit = stack[stack["price"] >= price]
    if hit.empty:
        return float(stack["cum_mw"].iloc[-1])
    return float(hit["cum_mw"].iloc[0])


def counterfactual(stack, lam, power_storage_mw):
    q_anchor = anchor_quantity(stack, lam)
    p_without = price_at(stack, q_anchor + power_storage_mw)
    return q_anchor, p_without, p_without - lam


# ── Paper CSV ───────────────────────────────────────────────────────────────────

def load_paper_csv():
    df = pd.read_csv(CAISO_CSV, parse_dates=["timestamp_local"])
    for c in ["wind", "solar", "power_storage"]:
        df[c] = df[c].fillna(0.0)
    df["hour"]  = df["timestamp_local"].dt.hour
    df["month"] = df["timestamp_local"].dt.month
    df["year"]  = df["timestamp_local"].dt.year
    df["date"]  = df["timestamp_local"].dt.date
    df["net_load_mw"] = (df["total_load_mw"] - df["wind"] - df["solar"]
                         - df["power_storage"].clip(lower=0.0))
    return df


def day_slice(paper, date):
    d = paper[paper["date"] == date]
    return d.set_index("hour")[["dam_price", "power_storage", "total_load_mw"]]


# ── One-day validation ──────────────────────────────────────────────────────────

def run_day(date):
    os.makedirs(FIG_DIR, exist_ok=True)
    paper = load_paper_csv()
    obs = day_slice(paper, date)
    stacks = hourly_stacks(load_day(date), date)

    rows = []
    for hour in range(24):
        if hour not in stacks or hour not in obs.index:
            continue
        lam = obs.loc[hour, "dam_price"]
        ps = obs.loc[hour, "power_storage"]
        q, p_wo, dp = counterfactual(stacks[hour], lam, ps)
        rows.append(dict(hour=hour, lam=lam, power_storage=ps,
                         q_anchor=q, p_without=p_wo, dprice=dp))
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    # figure 1: the 8pm stack with anchor + shift annotated
    hr = 20
    stack, lam, ps = stacks[hr], obs.loc[hr, "dam_price"], obs.loc[hr, "power_storage"]
    q, p_wo, dp = counterfactual(stack, lam, ps)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.step(np.concatenate([[0.0], stack["cum_mw"] / 1000]),
            np.concatenate([stack["price"].iloc[:1], stack["price"]]),
            where="pre", color="#333", lw=1.5)
    ax.axhline(lam, color="#0072B2", ls="--", lw=1.2,
               label=f"observed DAM price ${lam:.0f}")
    ax.axvline(q / 1000, color="#0072B2", ls=":", lw=1.2,
               label=f"anchor q* = {q/1000:.1f} GW")
    ax.axvline((q + ps) / 1000, color="#D55E00", ls=":", lw=1.2,
               label=f"q* + storage ({ps/1000:+.1f} GW)")
    ax.axhline(p_wo, color="#D55E00", ls="--", lw=1.2,
               label=f"counterfactual ${p_wo:.0f} (Δ {dp:+.0f})")
    ax.set_xlabel("Cumulative supply (GW)")
    ax.set_ylabel("Offer price ($/MWh)")
    ax.set_ylim(-175, min(PRICE_CAP, max(200, p_wo * 1.5)))
    ax.set_title(f"CAISO DAM supply stack, {date} 20:00 PT — anchored counterfactual")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"caiso_dam_curve_{date:%Y%m%d}_20PT.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")

    # figure 2: hourly delta-price with storage overlay
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(res["hour"], res["dprice"], color=np.where(res["dprice"] >= 0,
                                                      "#D55E00", "#0072B2"))
    ax.set_xlabel("Hour (PT)")
    ax.set_ylabel("Counterfactual − actual DAM price ($/MWh)")
    ax2 = ax.twinx()
    ax2.plot(res["hour"], res["power_storage"] / 1000, "k.-", alpha=0.6)
    ax2.set_ylabel("Battery net output (GW, + = discharge)")
    ax.set_title(f"CAISO no-battery price impact by hour, {date} (bid stack, anchored)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"caiso_dprice_hourly_{date:%Y%m%d}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")
    return res


# ── Monthly rollup vs LOWESS ────────────────────────────────────────────────────

LOWESS_FRAC = 0.5


def lowess_delta_for_day(csv, date):
    """Paper-method LOWESS price impact per hour for `date` (for comparison).
    Identical construction to dam_counterfactual.lowess_delta_for_day (ERCOT)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    month = csv[(csv["year"] == date.year) & (csv["month"] == date.month)]
    month = month[np.isfinite(month["dam_price"]) & np.isfinite(month["net_load_mw"])]
    dis = month[month["power_storage"] > 0]
    chg = month[month["power_storage"] < 0]

    def fit(sub):
        if len(sub) < 50:
            return None, None
        lw = lowess(sub["dam_price"].to_numpy(), sub["net_load_mw"].to_numpy(),
                    frac=LOWESS_FRAC, return_sorted=True)
        return lw[:, 0], lw[:, 1]
    xd, yd = fit(dis)
    xc, yc = fit(chg)
    out = {}
    for _, r in month[month["date"] == date].iterrows():
        ps = r["power_storage"]
        xs, ys = (xd, yd) if ps >= 0 else (xc, yc)
        if xs is None:
            continue
        p_fit = interp_lowess(xs, ys, r["net_load_mw"])
        p_adj = interp_lowess(xs, ys, r["net_load_mw"] + ps)
        out[int(r["hour"])] = p_adj - p_fit
    return out


def compare_day(date):
    """Overlay hourly bid-stack Delta vs LOWESS Delta for one day."""
    os.makedirs(FIG_DIR, exist_ok=True)
    csv = load_paper_csv()
    stack_d = _day_hourly_delta(date, csv)
    low_d = lowess_delta_for_day(csv, date)
    hours = [h for h in range(24) if np.isfinite(stack_d[h]) and h in low_d]
    s = np.array([stack_d[h] for h in hours])
    l = np.array([low_d[h] for h in hours])
    r = np.corrcoef(s, l)[0, 1]
    print(f"{date}: hourly bid-stack vs LOWESS correlation r = {r:.3f}")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(hours, s, "o-", color="#D55E00", lw=2, label="bid stack (anchored)")
    ax.plot(hours, l, "s-", color="#0072B2", lw=2, label="LOWESS (paper method)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Hour (PT)")
    ax.set_ylabel("No-battery price impact ($/MWh)")
    ax.set_title(f"CAISO {date}: hourly counterfactual price impact — "
                 f"bid stack vs LOWESS (r = {r:.3f})")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"caiso_dam_compare_{date:%Y%m%d}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def _day_hourly_delta(date, csv):
    """24-vector of bid-stack Delta price for a sampled day (NaN where missing).
    Mirrors dam_monthly_savings._day_hourly_delta for ERCOT."""
    try:
        stacks = hourly_stacks(load_day(date), date)
    except FileNotFoundError:
        return None
    day_csv = csv[csv["date"] == date].set_index("hour")
    out = np.full(24, np.nan)
    for h in range(24):
        if h not in stacks or h not in day_csv.index:
            continue
        lam = float(day_csv.loc[h, "dam_price"])
        ps = float(day_csv.loc[h, "power_storage"])
        stack = stacks[h]
        if stack.empty:
            continue
        q = anchor_quantity(stack, lam)
        out[h] = price_at(stack, q + ps) - lam
    return out


def bidstack_monthly(csv):
    """{(year, month): (delta_profile[24], n_days_sampled)} from cached days."""
    prof = {}
    for (y, m) in YEARS_MONTHS:
        deltas = []
        for d in SAMPLE_DAYS:
            if d > calendar.monthrange(y, m)[1]:
                continue
            dd = _day_hourly_delta(dt.date(y, m, d), csv)
            if dd is not None:
                deltas.append(dd)
        if deltas:
            prof[(y, m)] = (np.nanmean(np.vstack(deltas), axis=0), len(deltas))
    return prof


def bidstack_savings_dollars(csv, prof):
    """Convert bid-stack Delta profiles to monthly $ using month-average hourly load."""
    out = {}
    for (y, m), (delta, _n) in prof.items():
        month = csv[(csv["year"] == y) & (csv["month"] == m)]
        daily = 0.0
        for h in range(24):
            if not np.isfinite(delta[h]):
                continue
            load = float(month[month["hour"] == h]["total_load_mw"].mean())
            daily += delta[h] * load
        out[(y, m)] = daily * calendar.monthrange(y, m)[1]
    return out


def lowess_monthly(csv):
    """{(year, month): monthly_savings_$} using the paper's LOWESS construction.
    Identical to dam_monthly_savings.lowess_monthly (ERCOT)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    out = {}
    for (y, m) in YEARS_MONTHS:
        month = csv[(csv["year"] == y) & (csv["month"] == m)]
        month = month[np.isfinite(month["dam_price"]) & np.isfinite(month["net_load_mw"])]
        if month.empty:
            continue
        dis = month[month["power_storage"] > 0]
        chg = month[month["power_storage"] < 0]

        def fit(sub):
            if len(sub) < 200:
                return None, None
            lw = lowess(sub["dam_price"].to_numpy(), sub["net_load_mw"].to_numpy(),
                        frac=LOWESS_FRAC, return_sorted=True)
            return lw[:, 0], lw[:, 1]
        xd, yd = fit(dis)
        xc, yc = fit(chg)
        nl_floor = float(month["net_load_mw"].min())

        daily = 0.0
        for h in range(24):
            hh = month[month["hour"] == h]
            if len(hh) < 10:
                continue
            nl = float(hh["net_load_mw"].mean())
            ps = float(hh["power_storage"].mean())
            load = float(hh["total_load_mw"].mean())
            xs, ys = (xd, yd) if ps >= 0 else (xc, yc)
            if xs is None:
                continue
            p_fit = interp_lowess(xs, ys, max(nl_floor, nl))
            p_adj = interp_lowess(xs, ys, max(nl_floor, nl + ps))
            daily += (p_adj - p_fit) * load
        out[(y, m)] = daily * calendar.monthrange(y, m)[1]
    return out


def _hourly_table(csv):
    """One row per sampled hour: date, hour, lam, power_storage, p_without.
    Cached to figs/caiso_dam_hourly_deltas.csv (parsing 96 days is slow)."""
    cache = os.path.join(FIG_DIR, "caiso_dam_hourly_deltas.csv")
    if os.path.exists(cache):
        t = pd.read_csv(cache, parse_dates=["date"])
        t["date"] = t["date"].dt.date
        return t
    rows = []
    for (y, m) in YEARS_MONTHS:
        for d in SAMPLE_DAYS:
            if d > calendar.monthrange(y, m)[1]:
                continue
            date = dt.date(y, m, d)
            try:
                stacks = hourly_stacks(load_day(date), date)
            except FileNotFoundError:
                continue
            day_csv = csv[csv["date"] == date].set_index("hour")
            for h in range(24):
                if h not in stacks or h not in day_csv.index:
                    continue
                lam = float(day_csv.loc[h, "dam_price"])
                ps = float(day_csv.loc[h, "power_storage"])
                q = anchor_quantity(stacks[h], lam)
                p_wo = price_at(stacks[h], q + ps)
                rows.append(dict(date=date, year=y, month=m, hour=h,
                                 lam=lam, power_storage=ps, p_without=p_wo))
    t = pd.DataFrame(rows)
    os.makedirs(FIG_DIR, exist_ok=True)
    t.to_csv(cache, index=False)
    return t


def caps():
    """Cap-sensitivity of annual savings + tail diagnostics — the CAISO parallel
    of ERCOT's scarcity-tail analysis."""
    csv = load_paper_csv()
    t = _hourly_table(csv)
    low = lowess_monthly(csv)
    low_annual = {y: sum(v for (yy, m), v in low.items() if yy == y) / 1e6
                  for y in (2024, 2025)}

    month_load = csv.groupby(["year", "month", "hour"])["total_load_mw"].mean()

    def annual_dollars(tt):
        """Same typical-day aggregation as compute()/ERCOT."""
        out = {2024: 0.0, 2025: 0.0}
        prof = tt.groupby(["year", "month", "hour"])["dprice"].mean()
        for (y, m, h), dp in prof.items():
            if not np.isfinite(dp):
                continue
            load = month_load.get((y, m, h), np.nan)
            if not np.isfinite(load):
                continue
            out[y] += dp * load * calendar.monthrange(y, m)[1]
        return {y: v / 1e6 for y, v in out.items()}

    print(f"cap on counterfactual price -> annual savings ($M)   "
          f"[LOWESS: {low_annual[2024]:.0f} / {low_annual[2025]:.0f}; "
          f"paper: ~270 / -420]")
    for cap in [None, 500, 300, 200, 150, 100, 75]:
        tt = t.copy()
        p = tt["p_without"] if cap is None else tt["p_without"].clip(upper=cap)
        tt["dprice"] = p - tt["lam"]
        a = annual_dollars(tt)
        lab = "None" if cap is None else f"{cap:4d}"
        print(f"  cap={lab:>5}:  2024 ${a[2024]:8.0f}M   2025 ${a[2025]:8.0f}M")

    # tail diagnostics (uncapped)
    t["dprice"] = t["p_without"] - t["lam"]
    for y in (2024, 2025):
        s = t[t["year"] == y].copy()
        pos = s[s["dprice"] > 0]
        thresh = s["dprice"].quantile(0.99)
        top = s[s["dprice"] >= thresh]
        print(f"{y}: hours={len(s)}  median Δ={s['dprice'].median():.2f}  "
              f"p99 Δ={thresh:.0f}  hours with p_without>=999: "
              f"{(s['p_without'] >= 999).sum()}  "
              f"top-1%-hours Δ-sum share: "
              f"{top['dprice'].sum() / s[s.dprice > 0]['dprice'].sum() * 100:.0f}% of positive Δ")


def overlay_8pm():
    """Overlay the 20:00 PT supply stacks of every sampled day, 2024 vs 2025 —
    the bid-level view of the hockey stick disappearing (paper Fig. 8)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    for (y, m) in YEARS_MONTHS:
        for d in SAMPLE_DAYS:
            if d > calendar.monthrange(y, m)[1]:
                continue
            date = dt.date(y, m, d)
            try:
                stacks = hourly_stacks(load_day(date), date)
            except FileNotFoundError:
                continue
            if 20 not in stacks:
                continue
            st = stacks[20]
            c = "#9ecae1" if y == 2024 else "#08519c"
            ax.step(np.concatenate([[0.0], st["cum_mw"] / 1000]),
                    np.concatenate([st["price"].iloc[:1], st["price"]]),
                    where="pre", color=c, lw=0.8, alpha=0.5)
    ax.plot([], [], color="#9ecae1", lw=2, label="2024 (4 days/month)")
    ax.plot([], [], color="#08519c", lw=2, label="2025 (4 days/month)")
    ax.set_xlabel("Cumulative supply (GW)")
    ax.set_ylabel("Offer price ($/MWh)")
    ax.set_ylim(-175, 400)
    ax.set_title("CAISO DAM supply stacks at 20:00 PT — all sampled days 2024 vs 2025")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "caiso_dam_monthly_curves_20PT.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def compute():
    os.makedirs(FIG_DIR, exist_ok=True)
    csv = load_paper_csv()

    prof = bidstack_monthly(csv)
    if not prof:
        sys.exit("No cached sampled days — run `download` first.")
    bid = bidstack_savings_dollars(csv, prof)
    low = lowess_monthly(csv)

    rows = []
    for (y, m) in YEARS_MONTHS:
        if (y, m) not in bid:
            continue
        rows.append({"year": y, "month": m,
                     "n_days": prof[(y, m)][1],
                     "bidstack_$M": bid[(y, m)] / 1e6,
                     "lowess_$M": low.get((y, m), np.nan) / 1e6})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False,
                       formatters={"bidstack_$M": "{:.1f}".format,
                                   "lowess_$M": "{:.1f}".format}))
    for y in (2024, 2025):
        s = df[df.year == y]
        print(f"{y}: bid-stack ${s['bidstack_$M'].sum():8.0f}M | "
              f"LOWESS ${s['lowess_$M'].sum():8.0f}M")

    df["ym"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["bidstack_$M"], width=0.4,
           color="#D55E00", label="bid stack (anchored)")
    ax.bar(x + 0.2, df["lowess_$M"], width=0.4,
           color="#0072B2", label="LOWESS (paper method)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["ym"], rotation=90, fontsize=8)
    ax.set_ylabel("Monthly consumer savings ($M)")
    ax.set_title("CAISO: battery consumer savings by month — bid-stack vs LOWESS "
                 "(4 sampled days/month)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "caiso_dam_vs_lowess_monthly_2024_2025.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")
    df.to_csv(os.path.join(FIG_DIR, "caiso_dam_vs_lowess_monthly_2024_2025.csv"),
              index=False)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "download":
        download()
    elif cmd == "run":
        date = (dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2
                else dt.date(2025, 8, 15))
        run_day(date)
    elif cmd == "compare":
        date = (dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2
                else dt.date(2025, 8, 15))
        compare_day(date)
    elif cmd == "overlay":
        overlay_8pm()
    elif cmd == "caps":
        caps()
    elif cmd == "compute":
        compute()
    else:
        sys.exit(f"Unknown command: {cmd}")
