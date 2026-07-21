#!/usr/bin/env python3
"""
Monthly ERCOT battery savings: DAM bid-stack (anchored) vs the paper's LOWESS.

For each month of 2024-2025 we sample a few days, download the DAM offer stack
for each hour of those days, and compute the anchored bid-stack price impact
  Delta_h = S(q_anchor + power_storage_h) - dam_price_h
where the curve is anchored at the observed DAM price (= dam_price in the CSV).
The sampled days give a monthly 24-hour Delta profile; monthly savings are that
profile times the month's average hourly load times days-in-month — the same
aggregation the paper uses.

The LOWESS monthly savings are recomputed the paper's way (bs_lowess core) for a
head-to-head comparison, to see whether the mechanistic method moves the numbers.

Everything except the DAM offer stacks comes from the paper's CSV (dam_price,
power_storage, total_load) so the two methods share identical prices and battery
data.

Commands:
    python dam_monthly_savings.py download      # fetch + cache sampled days (slow)
    python dam_monthly_savings.py download-all  # fetch + cache EVERY day 2024-2025
    python dam_monthly_savings.py compute       # build comparison table + figure
"""

import os
import sys
import time
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dam_counterfactual import (
    build_stack, anchor_quantity, price_at, load_ercot_csv,
    GEN_COLS, EO_COLS, LOCAL_TZ, FIG_DIR,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from core import interp_lowess   # local copy in this folder  # noqa: E402

DAYS_DIR = os.path.join(REPO, "data", "dam", "days")
YEARS_MONTHS = [(y, m) for y in (2024, 2025) for m in range(1, 13)]
SAMPLE_DAYS = [5, 12, 19, 26]        # days-of-month sampled per month
LOWESS_FRAC = 0.5


def sampled_dates():
    for y, m in YEARS_MONTHS:
        ndays = calendar.monthrange(y, m)[1]
        for d in SAMPLE_DAYS:
            if d <= ndays:
                yield pd.Timestamp(f"{y}-{m:02d}-{d:02d}")


# ── Download ─────────────────────────────────────────────────────────────────────

def all_dates():
    for y in (2024, 2025):
        for day in pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D"):
            yield day


def download(dates=None):
    from gridstatusio import GridStatusClient
    api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        sys.exit("Error: set the GRIDSTATUS_API_KEY environment variable.")
    client = GridStatusClient(api_key=api_key)
    os.makedirs(DAYS_DIR, exist_ok=True)

    dates = list(dates) if dates is not None else list(sampled_dates())
    print(f"{len(dates)} days to fetch")
    for i, day in enumerate(dates):
        tag = f"{day:%Y%m%d}"
        gpath = os.path.join(DAYS_DIR, f"gen_{tag}.parquet")
        epath = os.path.join(DAYS_DIR, f"eo_{tag}.parquet")
        if os.path.exists(gpath) and os.path.exists(epath):
            continue
        start = pd.Timestamp(f"{day:%Y-%m-%d}", tz=LOCAL_TZ).tz_convert("UTC")
        S = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        E = (start + pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for dataset, cols, path in [
            ("ercot_dam_gen_resource_60_day", GEN_COLS, gpath),
            ("ercot_dam_energy_only_offers_60_day", EO_COLS, epath),
        ]:
            if os.path.exists(path):
                continue
            for attempt in range(4):
                try:
                    df = client.get_dataset(dataset=dataset, columns=cols,
                                            start=S, end=E, timezone="utc", limit=100000)
                    break
                except Exception as e:
                    print(f"    retry {attempt+1} ({day:%Y-%m-%d} {dataset}): {e}")
                    time.sleep(10)
            else:
                sys.exit(f"failed on {day:%Y-%m-%d} {dataset}")
            df["interval_start_utc"] = pd.to_datetime(df["interval_start_utc"], utc=True)
            df.to_parquet(path)
            time.sleep(2.0)
        print(f"[{i+1}/{len(dates)}] {day:%Y-%m-%d} cached")


# ── Bid-stack monthly savings ────────────────────────────────────────────────────

def _day_hourly_delta(day, csv):
    """24-vector of bid-stack Delta price for a sampled day (NaN where missing)."""
    tag = f"{day:%Y%m%d}"
    gpath = os.path.join(DAYS_DIR, f"gen_{tag}.parquet")
    epath = os.path.join(DAYS_DIR, f"eo_{tag}.parquet")
    if not (os.path.exists(gpath) and os.path.exists(epath)):
        return None
    gen = pd.read_parquet(gpath); eo = pd.read_parquet(epath)
    for df in (gen, eo):
        df["interval_start_utc"] = pd.to_datetime(df["interval_start_utc"], utc=True)
    for c in ["hsl", "awarded_quantity",
              "regup_awarded", "rrspfr_awarded", "rrsffr_awarded",
              "rrsufr_awarded", "nonspin_awarded", "ecrssd_awarded"]:
        if c in gen:
            gen[c] = pd.to_numeric(gen[c], errors="coerce")

    day_csv = csv[csv["date"] == day.date()].set_index("hour")
    out = np.full(24, np.nan)
    for ts in gen["interval_start_utc"].unique():
        h = pd.Timestamp(ts).tz_convert(LOCAL_TZ).hour
        if h not in day_csv.index:
            continue
        lam = float(day_csv.loc[h, "dam_price"])
        ps = float(day_csv.loc[h, "power_storage"])
        stack = build_stack(gen[gen["interval_start_utc"] == ts],
                            eo[eo["interval_start_utc"] == ts])
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
            dd = _day_hourly_delta(pd.Timestamp(f"{y}-{m:02d}-{d:02d}"), csv)
            if dd is not None:
                deltas.append(dd)
        if deltas:
            prof[(y, m)] = (np.nanmean(np.vstack(deltas), axis=0), len(deltas))
    return prof


# ── LOWESS monthly savings (paper method, point estimate) ───────────────────────

def lowess_monthly(csv):
    """{(year, month): monthly_savings_$} using the paper's LOWESS construction."""
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


# ── Compute + compare ────────────────────────────────────────────────────────────

def compute():
    csv = load_ercot_csv()
    csv["date"] = csv["timestamp_local"].dt.date

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
        b = df[df.year == y]["bidstack_$M"].sum()
        l = df[df.year == y]["lowess_$M"].sum()
        print(f"\n{y} annual:  bid-stack ${b:.0f}M   |   LOWESS ${l:.0f}M")

    # Figure
    fig, ax = plt.subplots(figsize=(15, 6.5))
    df["label"] = df.apply(lambda r: f"{int(r.year)}-{int(r.month):02d}", axis=1)
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["bidstack_$M"], width=0.4, color="#0072B2",
           label="DAM bid-stack (anchored, sampled days)")
    ax.bar(x + 0.2, df["lowess_$M"], width=0.4, color="#000",
           label="LOWESS (paper method)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=90, fontsize=8)
    ax.set_ylabel("Monthly consumer savings ($M)")
    ax.set_title("ERCOT monthly battery savings: DAM bid-stack vs LOWESS (2024–2025)")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "dam_vs_lowess_monthly_2024_2025.png")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print(f"\nsaved {out}")
    df.to_csv(os.path.join(FIG_DIR, "dam_vs_lowess_monthly_2024_2025.csv"), index=False)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "compute"
    if cmd == "download":
        download()
    elif cmd == "download-all":
        download(all_dates())
    elif cmd == "compute":
        compute()
    else:
        sys.exit(f"Unknown command: {cmd}")
