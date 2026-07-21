#!/usr/bin/env python3
"""
CAISO battery bid-behavior analysis (day-ahead / IFM), 2024 vs 2025.

CAISO's full supply-side bid stack is not available through GridStatus (only
aggregated storage bids are), so we cannot rebuild the merit order the way we
did for ERCOT. What we CAN show is the battery fleet's own bidding behavior —
which is the mechanism behind the paper's CAISO result: rising midday charging
demand lifts midday prices, and the charging cost grows enough to overwhelm
evening discharge savings, flipping consumers to a net loss by 2025.

Datasets (GridStatus, aggregated over the CAISO storage fleet):
  caiso_storage_awards_ifm       cleared MW by product (Energy +=discharge/
                                 -=charge, plus Reg/Spin/Non-Spin) and type.
  caiso_storage_energy_bids_ifm  submitted energy MW by price band (bid_range),
                                 Charge/Discharge, Hybrid/Standalone.

All times converted to US/Pacific. Prices/net-storage cross-checks use the
paper's CSV (data/caiso/complete_caiso_2020_2025.csv).

Commands:
    python caiso_storage_bids.py download
    python caiso_storage_bids.py plot
"""

import os
import re
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOCAL_TZ  = "US/Pacific"
DATA_DIR  = os.path.join(REPO, "data", "caiso")
FIG_DIR   = os.path.join(os.path.dirname(__file__), "figs")
REPO      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAISO_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "caiso", "complete_caiso_2020_2025.csv")
YEARS     = [2024, 2025]


# ── Download ─────────────────────────────────────────────────────────────────────

def _fetch(client, dataset, start, end):
    return client.get_dataset(dataset=dataset, start=start, end=end,
                              timezone="utc", limit=500000)


def download():
    from gridstatusio import GridStatusClient
    api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        sys.exit("Error: set the GRIDSTATUS_API_KEY environment variable.")
    client = GridStatusClient(api_key=api_key)
    os.makedirs(DATA_DIR, exist_ok=True)

    for dataset, name in [("caiso_storage_awards_ifm", "awards"),
                          ("caiso_storage_energy_bids_ifm", "bids")]:
        cache = os.path.join(DATA_DIR, f"storage_{name}.parquet")
        if os.path.exists(cache):
            print(f"cached: {cache}"); continue
        frames = []
        for y in YEARS:
            for q in range(4):
                start = pd.Timestamp(f"{y}-{1+3*q:02d}-01", tz="UTC")
                end = start + pd.offsets.MonthBegin(3)
                for attempt in range(4):
                    try:
                        df = _fetch(client, dataset,
                                    start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    end.strftime("%Y-%m-%dT%H:%M:%SZ"))
                        break
                    except Exception as e:
                        print(f"  retry {attempt+1} {y}Q{q+1}: {e}"); time.sleep(10)
                else:
                    sys.exit(f"failed {dataset} {y}Q{q+1}")
                print(f"  {name} {y}Q{q+1}: {len(df)} rows")
                frames.append(df)
                time.sleep(1.5)
        out = pd.concat(frames, ignore_index=True)
        out["interval_start_utc"] = pd.to_datetime(out["interval_start_utc"], utc=True)
        out.to_parquet(cache)
        print(f"saved {len(out)} rows -> {cache}")


# ── Load + helpers ───────────────────────────────────────────────────────────────

def _load(name):
    df = pd.read_parquet(os.path.join(DATA_DIR, f"storage_{name}.parquet"))
    df["interval_start_utc"] = pd.to_datetime(df["interval_start_utc"], utc=True)
    t = df["interval_start_utc"].dt.tz_convert(LOCAL_TZ)
    df["hour_pt"] = t.dt.hour
    df["year"] = t.dt.year
    df["month"] = t.dt.month
    df["mw"] = pd.to_numeric(df["mw"], errors="coerce")
    return df


def _band_low(s):
    """Signed lower edge of a bid_range like '($100, $200]' or '(-$100,-$50]'."""
    nums = re.findall(r"-?\d+", s.replace("$", ""))
    return float(nums[0]) if nums else np.nan


def _band_mid(s):
    nums = [float(x) for x in re.findall(r"-?\d+", s)]
    return float(np.mean(nums[:2])) if len(nums) >= 2 else np.nan


# ── Plots ────────────────────────────────────────────────────────────────────────

def plot_diurnal_awards(aw):
    """Energy awards by hour: charging (negative) vs discharging, 2024 vs 2025."""
    en = aw[aw["product"] == "Energy"].groupby(["year", "hour_pt"])["mw"].sum().unstack(0)
    ndays = {y: aw[aw.year == y]["interval_start_utc"].dt.normalize().nunique() for y in YEARS}
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for y, c in [(2024, "#9ecae1"), (2025, "#08519c")]:
        if y in en.columns:
            ax.plot(en.index, en[y] / ndays[y] / 1000, "o-", color=c, lw=2,
                    label=f"{y} (net Energy award)")
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between([-0.5, 23.5], 0, ax.get_ylim()[1], color="#0072B2", alpha=0.05)
    ax.text(3.5, ax.get_ylim()[1]*0.9, "discharge (net +)", color="#08519c", fontsize=11)
    ax.text(3.5, ax.get_ylim()[0]*0.9, "charge (net −)", color="#CC6600", fontsize=11)
    ax.set_xlabel("Hour of day (Pacific)"); ax.set_ylabel("Avg fleet Energy award (GW)")
    ax.set_title("CAISO battery day-ahead Energy awards by hour — 2024 vs 2025\n"
                 "(+ = discharge, − = charge; midday charging deepens as the fleet grows)")
    ax.set_xticks(range(0, 24, 2)); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "caiso_diurnal_awards.png")
    fig.savefig(out, dpi=200); plt.close(fig); print(f"saved {out}")


def plot_charge_bid_curve(bids):
    """Fleet charge-bid volume by price band, midday hours, 2024 vs 2025.
    A charge bid at a higher price band = willingness to pay more to charge,
    which lifts midday clearing prices."""
    HRS = list(range(9, 16))                       # 9am–3pm PT (7 midday hours)
    midday = bids[(bids["hour_pt"].isin(HRS)) & (bids["operation"] == "Charge")].copy()
    midday["band_low"] = midday["bid_range"].map(_band_low)
    midday = midday.dropna(subset=["band_low"])
    g = midday.groupby(["year", "bid_range"]).agg(
        mw=("mw", "sum"), band_low=("band_low", "first")).reset_index()
    # normalize to average GW offered per midday hour
    nobs = {y: (bids[bids.year == y]["interval_start_utc"].dt.normalize().nunique() * len(HRS))
            for y in YEARS}

    bands = (g[["bid_range", "band_low"]].drop_duplicates()
             .sort_values("band_low")["bid_range"].tolist())
    x = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, (y, c) in enumerate([(2024, "#9ecae1"), (2025, "#08519c")]):
        vals = []
        for b in bands:
            row = g[(g.year == y) & (g.bid_range == b)]
            vals.append((row["mw"].iloc[0] / nobs[y] / 1000) if len(row) else 0.0)
        ax.bar(x + (i - 0.5) * 0.4, vals, width=0.4, color=c, label=str(y))
    ax.set_xticks(x); ax.set_xticklabels(bands, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Charge bid price band ($/MWh — max price the battery will pay to charge, low→high)")
    ax.set_ylabel("Avg charge-bid volume (GW per midday hour)")
    ax.set_title("CAISO battery charge bids by price band, midday (9am–3pm PT) — 2024 vs 2025\n"
                 "(more volume at higher bands = batteries willing to pay more to charge → lifts midday prices)")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "caiso_charge_bid_curve.png")
    fig.savefig(out, dpi=200); plt.close(fig); print(f"saved {out}")


def plot_monthly_energy(aw):
    """Monthly charge vs discharge energy award volume, 2024–2025."""
    en = aw[(aw["product"] == "Energy") & (aw["year"].isin(YEARS))].copy()
    en["ym"] = en["interval_start_utc"].dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m")
    chg = en[en.mw < 0].groupby("ym")["mw"].sum() / 1000
    dis = en[en.mw > 0].groupby("ym")["mw"].sum() / 1000
    idx = sorted(set(chg.index) | set(dis.index))
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(idx))
    ax.bar(x, [dis.get(i, 0) for i in idx], color="#08519c", label="Discharge (GWh-scale, + )")
    ax.bar(x, [chg.get(i, 0) for i in idx], color="#CC6600", label="Charge ( − )")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(idx, rotation=90, fontsize=8)
    ax.set_ylabel("Monthly fleet Energy award (GWh, signed)")
    ax.set_title("CAISO battery monthly day-ahead energy: charge vs discharge (2024–2025)")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "caiso_monthly_energy.png")
    fig.savefig(out, dpi=200); plt.close(fig); print(f"saved {out}")


def plot():
    os.makedirs(FIG_DIR, exist_ok=True)
    aw = _load("awards"); bids = _load("bids")
    # quick stats
    en = aw[aw["product"] == "Energy"]
    for y in YEARS:
        e = en[en.year == y]
        nd = e["interval_start_utc"].dt.normalize().nunique()
        print(f"{y}: {nd} days | avg daily charge {e[e.mw<0].mw.sum()/nd/1000:6.1f} GWh | "
              f"discharge {e[e.mw>0].mw.sum()/nd/1000:6.1f} GWh")
    plot_diurnal_awards(aw)
    plot_charge_bid_curve(bids)
    plot_monthly_energy(aw)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if cmd == "download":
        download()
    elif cmd == "plot":
        plot()
    else:
        sys.exit(f"Unknown command: {cmd}")
