#!/usr/bin/env python3
"""
Construct the ERCOT price-demand (supply) curve directly from SCED offer data
and compute a no-battery counterfactual price (bid-stack method, cf. Aurora
Energy Research CAISO note).

Instead of fitting a statistical LOWESS curve to (net load, price) scatter,
this builds the actual merit-order supply curve from each generator's SCED
offer curve for a single 5-minute interval, finds where it clears against
observed dispatched demand, and then re-clears a counterfactual stack with
battery offers removed and battery-charging load subtracted.

Commands:
    python stack_counterfactual.py download [YYYY-MM-DD]   # fetch + cache
    python stack_counterfactual.py run      [YYYY-MM-DD]   # analyze + plot

API key (download only) is read from GRIDSTATUS_API_KEY. Downloads cache under
data/stack_counterfactual/; analysis runs entirely from cache.
"""

import os
import re
import sys
import ast
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DEFAULT_DAY = "2025-08-15"          # local ERCOT (US/Central) calendar day
LOCAL_TZ    = "US/Central"
REPO        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(REPO, "data", "stack_counterfactual")
FIG_DIR     = os.path.join(os.path.dirname(__file__), "figs")

# Telemetered statuses that are online and producing energy into the stack.
ONLINE_STATUS = {
    "ON", "ONOS", "ONTEST", "ONREG", "ONFFRRRS", "ONRR", "ONECRS", "ONEMR",
    "EMR", "EMRSWGR",
}
# Offline but available to be committed (quick-start / non-spin reserve). These
# are the dispatchable resources that would replace battery discharge in the
# counterfactual. "OUT" (on outage) and transitional states are excluded.
AVAILABLE_OFFLINE = {"OFF", "OFFNS", "OFFQS"}

# Resource-type → fuel grouping.
TYPE_MAP = {
    "CCGT90": "Natural Gas", "SCGT90": "Natural Gas", "GSREH": "Natural Gas",
    "GSNONR": "Natural Gas", "GSSUP": "Natural Gas", "CCLE90": "Natural Gas",
    "SCLE90": "Natural Gas", "SCLE95": "Natural Gas",
    "CLLIG": "Coal", "NUC": "Nuclear", "HYDRO": "Hydro",
    "WIND": "Wind", "PVGR": "Solar", "PWRSTR": "Battery",
    "RENEW": "Other", "DSL": "Other",
}
COLORS = {
    "Natural Gas": "#e08030", "Coal": "#5a5a5a", "Nuclear": "#9b59b6",
    "Hydro": "#2980b9", "Wind": "#27ae60", "Solar": "#f1c40f",
    "Battery": "#16a085", "Other": "#bdc3c7",
}
FUEL_ORDER = ["Nuclear", "Coal", "Hydro", "Wind", "Solar", "Battery",
              "Natural Gas", "Other"]

PRICE_MIN = -250   # ERCOT offer floor
PRICE_MAX = 5000   # ERCOT offer cap (HCAP)


# ── Offer-curve parsing ────────────────────────────────────────────────────────

def parse_curve(s):
    """Return list of [MW, price] pairs from a string or list cell."""
    if isinstance(s, list):
        return s
    if not isinstance(s, str) or not s.strip().startswith("["):
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def resource_segments(row, cap_col):
    """
    Convert one resource's SCED offer curve into (MW_increment, price, fuel)
    segments, capped at cap_col (HASL for online units = HSL minus ancillary
    service set-aside; HSL for offline units, which carry no AS) and floored
    at 0.
    """
    curve = parse_curve(row["sced1_offer_curve"])
    if len(curve) < 2:
        return []
    cap = row[cap_col]
    if pd.isna(cap) or cap <= 0:
        cap = row["hsl"]
    if pd.isna(cap):
        return []
    fuel = TYPE_MAP.get(row["resource_type"], "Other")

    segs = []
    for i in range(len(curve) - 1):
        mw_lo, _p_lo = curve[i]
        mw_hi, p_hi  = curve[i + 1]
        mw_hi = min(mw_hi, cap)
        dmw = mw_hi - mw_lo
        if dmw > 0 and PRICE_MIN <= p_hi <= PRICE_MAX:
            segs.append((dmw, float(p_hi), fuel))
    return segs


def build_stack(gen_snap, exclude_battery=False, include_offline=False):
    """
    Build the merit-order supply stack for one SCED interval.

    Args:
        exclude_battery: drop all PWRSTR (battery) resources — used for the
            no-battery counterfactual supply.
        include_offline: also include offline-but-available resources
            (AVAILABLE_OFFLINE statuses), i.e. quick-start / non-spin peakers
            that would be committed to replace battery discharge. Offline units
            are capped at HSL (they carry no ancillary-service obligation).

    Returns a DataFrame sorted by price with columns [mw, price, fuel, cum_mw].
    """
    segs = []
    online = gen_snap[gen_snap["telemetered_resource_status"].isin(ONLINE_STATUS)]
    for _, row in online.iterrows():
        if exclude_battery and row["resource_type"] == "PWRSTR":
            continue
        segs.extend(resource_segments(row, "hasl"))

    if include_offline:
        offline = gen_snap[gen_snap["telemetered_resource_status"].isin(AVAILABLE_OFFLINE)]
        for _, row in offline.iterrows():
            if row["resource_type"] == "PWRSTR":   # never revive batteries here
                continue
            segs.extend(resource_segments(row, "hsl"))

    stack = pd.DataFrame(segs, columns=["mw", "price", "fuel"])
    stack = stack.sort_values("price", kind="mergesort").reset_index(drop=True)
    stack["cum_mw"] = stack["mw"].cumsum()
    return stack


def clear_price(stack, demand_mw):
    """Offer price of the marginal segment when cumulative supply reaches
    demand_mw. Returns NaN for an empty stack."""
    if stack.empty:
        return np.nan
    idx = int(stack["cum_mw"].searchsorted(demand_mw))
    idx = min(idx, len(stack) - 1)
    return float(stack.iloc[idx]["price"])


# ── Battery-charging identification ─────────────────────────────────────────────

def _site(name):
    """Strip a resource's trailing unit/load designator to get its site id."""
    return re.sub(r"_(LD|BESS|ESS|UNIT|BES|LDU|BATT|BE|SLR)?\d*$", "", str(name))


def battery_charge_by_interval(gen_df, load_df):
    """
    Battery charging (MW, positive) per SCED interval. A load resource is a
    battery if its site id matches a PWRSTR generation resource's site id.
    """
    bat_sites = {_site(n) for n in
                 gen_df.loc[gen_df["resource_type"] == "PWRSTR", "resource_name"].unique()}
    is_bat = load_df["resource_name"].map(lambda n: _site(n) in bat_sites)
    bat_load = load_df[is_bat]
    return bat_load.groupby("sced_timestamp_utc")["real_power_consumption"].sum()


# ── Per-interval analysis ────────────────────────────────────────────────────────

def analyze_day(gen_df, load_df, lambda_df):
    """
    For each hourly SCED snapshot: reconstruct the with-battery clearing price,
    build the no-battery counterfactual, and record the price impact.

    Counterfactual accounting (power balance):
        Sigma base_point (all online gens) = native load + battery charge + losses,
        and includes battery discharge on the supply side.
      Without batteries:
        - supply loses all Battery segments (no discharge available),
        - demand loses the battery charging load.
        => clear the non-battery stack at (Sigma base_point - battery_charge).
    """
    charge = battery_charge_by_interval(gen_df, load_df)
    rows = []
    for ts, snap in gen_df.groupby("sced_timestamp_utc"):
        stack     = build_stack(snap)                                   # observed: online, incl battery
        cf_online = build_stack(snap, exclude_battery=True)             # online non-battery only
        cf_commit = build_stack(snap, exclude_battery=True, include_offline=True)  # + offline available
        demand = float(snap["base_point"].sum())
        discharge = float(snap.loc[snap["resource_type"] == "PWRSTR", "base_point"].sum())
        chg = float(charge.get(ts, 0.0))
        demand_nobat = demand - chg

        p_with = clear_price(stack, demand)

        # Two counterfactual bounds — the truth lies between them because a
        # single SCED snapshot cannot re-solve unit commitment:
        #   scarcity bound: only currently-online thermal ramps up. If it can't
        #     cover demand, price hits scarcity (upper bound on savings).
        #   committed bound: offline quick-start peakers are freely available at
        #     their energy offers (lower bound on savings; ignores startup cost).
        p_cf_scarcity  = clear_price(cf_online, demand_nobat)
        p_cf_committed = clear_price(cf_commit, demand_nobat)
        online_short = demand_nobat > cf_online["cum_mw"].max()

        lam = lambda_df.iloc[(lambda_df["sced_timestamp_utc"] - ts).abs().argmin()]["system_lambda"]

        rows.append({
            "ts_utc":       ts,
            "hour_ct":      ts.tz_convert(LOCAL_TZ).hour,
            "demand_mw":    demand,
            "discharge_mw": discharge,
            "charge_mw":    chg,
            "p_with":       p_with,
            "p_cf_scarcity":  p_cf_scarcity,
            "p_cf_committed": p_cf_committed,
            "dprice_scarcity":  p_cf_scarcity  - p_with,   # + = savings, − = charging cost
            "dprice_committed": p_cf_committed - p_with,
            "online_short": bool(online_short),
            "system_lambda": float(lam),
            "stack_gw":     stack["mw"].sum() / 1000,
        })
    out = pd.DataFrame(rows).sort_values("ts_utc").reset_index(drop=True)
    return out


# ── Plots ────────────────────────────────────────────────────────────────────────

def plot_validation(res, day, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(res["hour_ct"], res["system_lambda"], "o-", color="#333",
            lw=2, label="Observed SCED system lambda")
    ax.plot(res["hour_ct"], res["p_with"], "s--", color="#0072B2",
            lw=2, label="Reconstructed from bid stack")
    ax.set_xlabel("Hour (CT)")
    ax.set_ylabel("Price ($/MWh)")
    ax.set_title(f"Bid-stack reconstruction vs. observed price — ERCOT {day}")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend()
    corr = res["p_with"].corr(res["system_lambda"])
    ax.text(0.02, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
            va="top", fontsize=13, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}  (r={corr:.3f})")


def plot_dprice(res, day, out_path):
    """Battery price impact by hour, as a bracket between the two counterfactual
    bounds. Charging hours (both bounds agree) are robust; discharge hours show
    the unit-commitment uncertainty as a spread."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    h = res["hour_ct"].to_numpy()
    lo = res["dprice_committed"].to_numpy()   # lower bound on savings
    hi = res["dprice_scarcity"].to_numpy()    # upper bound (scarcity, may be capped)
    hi_disp = np.clip(hi, None, 300)          # clip the $5000 cap for readability

    ax.vlines(h, np.minimum(lo, hi_disp), np.maximum(lo, hi_disp),
              color="#888", lw=6, alpha=0.5)
    ax.plot(h, lo, "o", color="#CC6600", label="Lower bound (offline peakers freely available)")
    ax.plot(h, hi_disp, "s", color="#0072B2", label="Upper bound (only online thermal; scarcity)")
    for hh, val, capped in zip(h, hi, res["online_short"]):
        if capped:
            ax.annotate("→ scarcity\n(capped)", (hh, min(val, 300)),
                        fontsize=8, ha="center", va="bottom", color="#0072B2")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Hour (CT)")
    ax.set_ylabel("Counterfactual − actual price ($/MWh)")
    ax.set_title(f"Bid-stack battery price impact by hour — ERCOT {day}\n"
                 f"(+ = savings from discharge, − = cost of charging; band = unit-commitment uncertainty)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_stack_snapshot(gen_df, res, ts, out_path):
    """The price-demand (merit-order supply) curve for one interval, built from
    the bid stack. Shows the observed stack, the demand line and clearing price,
    and the no-battery counterfactual clearing (as a bracket)."""
    snap = gen_df[gen_df["sced_timestamp_utc"] == ts]
    stack = build_stack(snap)
    row = res[res["ts_utc"] == ts].iloc[0]

    fig, ax = plt.subplots(figsize=(13, 7))
    x = 0.0
    seen = set()
    for _, s in stack.iterrows():
        w = s["mw"] / 1000
        ax.bar(x + w / 2, s["price"], width=w, bottom=0,
               color=COLORS.get(s["fuel"], "#bdc3c7"), linewidth=0)
        x += w
        seen.add(s["fuel"])

    d_with    = row["demand_mw"] / 1000
    d_nobat   = (row["demand_mw"] - row["charge_mw"]) / 1000
    ax.axvline(d_with, color="black", lw=2,
               label=f"Dispatched demand = {d_with:.1f} GW  →  ${row['p_with']:.0f}/MWh")
    if abs(d_nobat - d_with) > 0.05:
        ax.axvline(d_nobat, color="black", lw=2, ls=":",
                   label=f"Demand w/o battery charging = {d_nobat:.1f} GW")
    ax.axhline(row["p_with"], color="black", lw=0.8, ls="--", alpha=0.5)

    patches = [mpatches.Patch(color=COLORS[f], label=f) for f in FUEL_ORDER if f in seen]
    demand_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=patches + demand_handles, loc="upper left", fontsize=10)

    ts_ct = ts.tz_convert(LOCAL_TZ)
    hi = min(row["p_cf_scarcity"], 5000)
    cf_txt = (f"counterfactual ${row['p_cf_committed']:.0f}"
              f"–{'scarcity' if row['online_short'] else f'${hi:.0f}'}/MWh")
    ax.set_title(f"ERCOT bid-stack price-demand curve — {ts_ct:%Y-%m-%d %H:%M %Z}\n"
                 f"battery discharge {row['discharge_mw']:.0f} MW, "
                 f"charging {row['charge_mw']:.0f} MW  |  {cf_txt}")
    ax.set_xlabel("Cumulative capacity (GW)")
    ax.set_ylabel("Offer price ($/MWh)")
    ax.set_xlim(0, min(stack["cum_mw"].max() / 1000, d_with + 25))
    ax.set_ylim(-30, max(120, row["p_cf_committed"] * 1.4))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


# ── Figure 1 (report): cleaned merit-stack snapshot ──────────────────────────────

# Merge low-variable-cost baseload + miscellaneous into a single "Other" band.
FIG1_MERGE  = {"Nuclear", "Coal", "Hydro", "Other"}
FIG1_ORDER  = ["Wind", "Solar", "Natural Gas", "Battery", "Other"]
FIG1_COLORS = {"Wind": "#27ae60", "Solar": "#f1c40f", "Natural Gas": "#e08030",
               "Battery": "#16a085", "Other": "#7f8c8d"}


def _fig1_fuel(f):
    return "Other" if f in FIG1_MERGE else f


def plot_fig1(gen_df, res, ts, out_path):
    """Report Figure 1: the merit-order supply stack for one interval, with
    (1) coal/hydro/nuclear/other merged into one 'Other' band, and (2) segments
    grouped by fuel within each tied price so each fuel is contiguous (no
    confetti). Merit order and clearing are unchanged — the regrouping only
    reorders equal-price segments."""
    snap = gen_df[gen_df["sced_timestamp_utc"] == ts]
    stack = build_stack(snap).copy()
    stack["fuel"] = stack["fuel"].map(_fig1_fuel)
    # group equal-price segments by fuel: stable sort on (price, fuel rank)
    rank = {f: i for i, f in enumerate(FIG1_ORDER)}
    stack["_r"] = stack["fuel"].map(lambda f: rank.get(f, len(FIG1_ORDER)))
    stack = stack.sort_values(["price", "_r"], kind="mergesort").reset_index(drop=True)
    stack["cum_mw"] = stack["mw"].cumsum()
    row = res[res["ts_utc"] == ts].iloc[0]

    fig, ax = plt.subplots(figsize=(13, 7))
    x = 0.0
    seen = set()
    for _, s in stack.iterrows():
        w = s["mw"] / 1000
        ax.bar(x + w / 2, s["price"], width=w, bottom=0,
               color=FIG1_COLORS.get(s["fuel"], "#7f8c8d"), linewidth=0)
        x += w
        seen.add(s["fuel"])

    d_with  = row["demand_mw"] / 1000
    d_nobat = (row["demand_mw"] - row["charge_mw"]) / 1000
    ax.axvline(d_with, color="black", lw=2,
               label=f"Dispatched demand = {d_with:.1f} GW  →  ${row['p_with']:.0f}/MWh")
    if abs(d_nobat - d_with) > 0.05:
        ax.axvline(d_nobat, color="black", lw=2, ls=":",
                   label=f"Demand w/o battery charging = {d_nobat:.1f} GW")
    ax.axhline(row["p_with"], color="black", lw=0.8, ls="--", alpha=0.5)

    patches = [mpatches.Patch(color=FIG1_COLORS[f], label=f)
               for f in FIG1_ORDER if f in seen]
    demand_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=patches + demand_handles, loc="upper left", fontsize=10)

    ts_ct = ts.tz_convert(LOCAL_TZ)
    ax.set_title(f"ERCOT bid-stack price-demand curve — {ts_ct:%Y-%m-%d %H:%M %Z}\n"
                 f"battery discharge {row['discharge_mw']:.0f} MW, "
                 f"charging {row['charge_mw']:.0f} MW")
    ax.set_xlabel("Cumulative capacity (GW)")
    ax.set_ylabel("Offer price ($/MWh)")
    ax.set_xlim(0, min(stack["cum_mw"].max() / 1000, d_with + 8))
    ax.set_ylim(-30, 120)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


def fig1(day="2025-08-15", hour=16):
    """Regenerate report Figure 1 for a chosen local hour (default 16:00 CT)."""
    gen, load, lam = load_cached(day)
    res = analyze_day(gen, load, lam)
    os.makedirs(FIG_DIR, exist_ok=True)
    match = res[res["hour_ct"] == hour]
    if match.empty:
        sys.exit(f"no SCED snapshot at {hour:02d}:00 CT for {day}")
    ts = match.iloc[0]["ts_utc"]
    tag = day.replace("-", "")
    plot_fig1(gen, res, ts,
              os.path.join(FIG_DIR, f"stack_fig1_{hour:02d}00_{tag}.png"))


# ── Load cached data ─────────────────────────────────────────────────────────────

def load_cached(day):
    tag = day.replace("-", "")
    gen = pd.read_csv(os.path.join(DATA_DIR, f"gen_resource_{tag}.csv"))
    load = pd.read_csv(os.path.join(DATA_DIR, f"load_resource_{tag}.csv"))
    lam = pd.read_csv(os.path.join(DATA_DIR, f"system_lambda_{tag}.csv"))
    for df in (gen, load, lam):
        df["sced_timestamp_utc"] = pd.to_datetime(df["sced_timestamp_utc"], utc=True)
    return gen, load, lam


def run(day):
    gen, load, lam = load_cached(day)
    res = analyze_day(gen, load, lam)

    os.makedirs(FIG_DIR, exist_ok=True)
    tag = day.replace("-", "")
    res.to_csv(os.path.join(FIG_DIR, f"stack_results_{tag}.csv"), index=False)

    show = res[["hour_ct", "demand_mw", "discharge_mw", "charge_mw", "p_with",
                "p_cf_committed", "p_cf_scarcity", "dprice_committed",
                "dprice_scarcity", "online_short", "system_lambda"]].copy()
    print("\n" + show.to_string(index=False))
    corr = res["p_with"].corr(res["system_lambda"])
    print(f"\nreconstruction vs lambda: r = {corr:.3f}")

    plot_validation(res, day, os.path.join(FIG_DIR, f"validation_{tag}.png"))
    plot_dprice(res, day, os.path.join(FIG_DIR, f"dprice_hourly_{tag}.png"))

    # Peak discharge snapshot + a midday charging snapshot.
    peak_ts = res.loc[res["discharge_mw"].idxmax(), "ts_utc"]
    chg_ts  = res.loc[res["charge_mw"].idxmax(), "ts_utc"]
    plot_stack_snapshot(gen, res, peak_ts,
                        os.path.join(FIG_DIR, f"stack_peak_{tag}.png"))
    plot_stack_snapshot(gen, res, chg_ts,
                        os.path.join(FIG_DIR, f"stack_charge_{tag}.png"))
    return res


# ── Download (see module docstring) ──────────────────────────────────────────────

def _download(day):
    from gridstatusio import GridStatusClient
    api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        sys.exit("Error: set the GRIDSTATUS_API_KEY environment variable.")
    client = GridStatusClient(api_key=api_key)
    tag = day.replace("-", "")

    gen_cols = [
        "sced_timestamp_utc", "resource_name", "resource_type",
        "telemetered_resource_status", "telemetered_net_output",
        "output_schedule", "hsl", "hasl", "hdl", "lsl", "lasl", "ldl",
        "base_point", "sced1_offer_curve",
        "as_responsibility_for_regup", "as_responsibility_for_regdown",
        "as_responsibility_for_rrs", "as_responsibility_for_rrsffr",
        "as_responsibility_for_nonspin", "as_responsibility_for_ecrs",
    ]
    load_cols = [
        "sced_timestamp_utc", "resource_name", "telemetered_resource_status",
        "max_power_consumption", "low_power_consumption",
        "real_power_consumption", "base_point", "sced_bid_to_buy_curve",
    ]

    start = pd.Timestamp(day, tz=LOCAL_TZ).tz_convert("UTC")
    hours = [start + pd.Timedelta(hours=h) for h in range(24)]
    os.makedirs(DATA_DIR, exist_ok=True)

    # system lambda (whole day, one call)
    lam_cache = os.path.join(DATA_DIR, f"system_lambda_{tag}.csv")
    if not os.path.exists(lam_cache):
        df = client.get_dataset(
            dataset="ercot_sced_system_lambda",
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=(start + pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            timezone="utc", limit=2000)
        df.to_csv(lam_cache, index=False)
        print(f"saved {len(df)} rows -> {lam_cache}")

    for dataset, cols, name in [
        ("ercot_sced_gen_resource_60_day",  gen_cols,  f"gen_resource_{tag}.csv"),
        ("ercot_sced_load_resource_60_day", load_cols, f"load_resource_{tag}.csv"),
    ]:
        cache = os.path.join(DATA_DIR, name)
        if os.path.exists(cache):
            print(f"cached: {cache}")
            continue
        frames = []
        for ts in hours:
            df = client.get_dataset(
                dataset=dataset, columns=cols,
                start=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end=(ts + pd.Timedelta(minutes=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                timezone="utc", limit=20000)
            if df.empty:
                continue
            df["sced_timestamp_utc"] = pd.to_datetime(df["sced_timestamp_utc"], utc=True)
            first = df["sced_timestamp_utc"].min()
            frames.append(df[df["sced_timestamp_utc"] == first])
            time.sleep(0.3)
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(cache, index=False)
        print(f"saved {len(out)} rows -> {cache}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    day = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DAY
    if cmd == "download":
        _download(day)
    elif cmd == "run":
        run(day)
    elif cmd == "fig1":
        hour = int(sys.argv[3]) if len(sys.argv) > 3 else 16
        fig1(day, hour)
    else:
        sys.exit(f"Unknown command: {cmd}")
