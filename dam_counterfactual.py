#!/usr/bin/env python3
"""
Day-ahead-market (DAM) bid-stack price-demand curve + battery counterfactual.

Builds the DAM merit-order supply curve from submitted offers (three-part gen
offers + energy-only offers), then estimates the battery price impact the same
way the paper's LOWESS method does — but using the real supply curve instead of
a smoothed scatter:

  1. Anchor the curve to the observed DAM system lambda: find the cumulative
     supply quantity q* where the curve equals the observed price. This absorbs
     the level ambiguity of a system-level merit reconstruction (virtual bids,
     nodal congestion) — we only use the curve for its *shape*.
  2. Shift the operating point by the actual battery net output (power_storage
     from the paper's CSV, + = discharge, − = charge), exactly as the paper
     shifts net load, and read the counterfactual price off the curve.
  3. Delta = price_without - price_with (= observed lambda). + = savings.

Because commitment happens in the DAM, offline peakers' offers are already in
the stack — so stepping up the curve dispatches exactly the units that would
replace battery discharge, avoiding the real-time scarcity-cap problem.

Commands:
    python dam_counterfactual.py download [YYYY-MM-DD]   # one local day, all hours
    python dam_counterfactual.py run      [YYYY-MM-DD]   # analyze + plot (from cache)

API key (download only) from GRIDSTATUS_API_KEY.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from core import interp_lowess   # local copy in this folder  # noqa: E402

DEFAULT_DAY = "2025-08-15"
LOCAL_TZ    = "US/Central"
DATA_DIR    = os.path.join(REPO, "data", "dam")
FIG_DIR     = os.path.join(os.path.dirname(__file__), "figs")
ERCOT_CSV   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "ercot", "complete_ercot_2020_2025.csv")

PRICE_MIN, PRICE_MAX = -250, 5000
AS_GEN = ["regup_awarded", "rrspfr_awarded", "rrsffr_awarded", "rrsufr_awarded",
          "nonspin_awarded", "ecrssd_awarded"]
GEN_COLS = ["interval_start_utc", "resource_name", "resource_type", "hsl",
            "awarded_quantity", "qse_submitted_curve"] + AS_GEN
EO_COLS  = ["interval_start_utc", "energy_only_offer_curve"]


# ── Offer-curve parsing ────────────────────────────────────────────────────────

def parse_curve(s):
    """Return [[MW, price], ...] from a list / numpy array / string cell."""
    if s is None:
        return []
    if isinstance(s, np.ndarray):
        s = s.tolist()
    if isinstance(s, str):
        if not s.strip().startswith("["):
            return []
        import ast
        try:
            s = ast.literal_eval(s)
        except Exception:
            return []
    if isinstance(s, list):
        try:
            return [[float(a), float(b)] for a, b in s]
        except Exception:
            return []
    return []


def _curve_segments(curve, cap=None):
    """Piecewise offer curve → (MW_increment, price) segments, capped at `cap`."""
    segs = []
    if len(curve) == 1:                       # single price-quantity point
        mw, p = curve[0]
        if mw > 0 and PRICE_MIN <= p <= PRICE_MAX:
            segs.append((mw if cap is None else min(mw, cap), p))
        return segs
    for i in range(len(curve) - 1):
        mw_lo = curve[i][0]
        mw_hi = curve[i + 1][0]
        price = curve[i + 1][1]
        if cap is not None:
            mw_hi = min(mw_hi, cap)
        if mw_hi - mw_lo > 0 and PRICE_MIN <= price <= PRICE_MAX:
            segs.append((mw_hi - mw_lo, price))
    return segs


def build_stack(gen_snap, eo_snap, exclude_battery=False):
    """Combined DAM priced supply stack (three-part gen offers + energy-only
    offers), sorted by price. Returns DataFrame [mw, price, cum_mw] (MW).

    exclude_battery drops PWRSTR three-part offers — the no-battery stack.
    (Energy-only offers are anonymous and cannot be attributed to batteries.)
    """
    segs = []
    if exclude_battery and "resource_type" in gen_snap.columns:
        gen_snap = gen_snap[gen_snap["resource_type"] != "PWRSTR"]
    for _, r in gen_snap.iterrows():
        cap = r["hsl"] - r[AS_GEN].fillna(0).sum()
        if pd.isna(cap) or cap <= 0:
            cap = r["hsl"]
        segs.extend(_curve_segments(parse_curve(r["qse_submitted_curve"]),
                                    cap if pd.notna(cap) else None))
    for v in eo_snap["energy_only_offer_curve"].values:
        segs.extend(_curve_segments(parse_curve(v)))
    stack = pd.DataFrame(segs, columns=["mw", "price"]).sort_values("price",
                                                                    kind="mergesort")
    stack = stack.reset_index(drop=True)
    stack["cum_mw"] = stack["mw"].cumsum()
    return stack


# ── Anchored counterfactual ────────────────────────────────────────────────────

def price_at(stack, q_mw):
    """Offer price at cumulative quantity q_mw (MW)."""
    if stack.empty:
        return np.nan
    i = min(int(stack["cum_mw"].searchsorted(q_mw)), len(stack) - 1)
    return float(stack.iloc[i]["price"])


def anchor_quantity(stack, price):
    """Cumulative MW where the stack first reaches `price` (the operating point
    consistent with the observed clearing price)."""
    hit = stack[stack["price"] >= price]
    if hit.empty:
        return float(stack["cum_mw"].iloc[-1])
    return float(hit["cum_mw"].iloc[0])


def counterfactual(stack, lam, power_storage_mw):
    """Battery price impact at one hour.

    power_storage_mw > 0 = discharging (shift right/up), < 0 = charging.
    Returns (q_anchor_mw, p_without, delta).
    """
    q_anchor = anchor_quantity(stack, lam)
    p_without = price_at(stack, q_anchor + power_storage_mw)
    return q_anchor, p_without, p_without - lam


# ── Battery output + LOWESS reference from the paper's CSV ──────────────────────

def load_ercot_csv():
    df = pd.read_csv(ERCOT_CSV, parse_dates=["timestamp_local"])
    for c in ["wind", "solar", "power_storage"]:
        df[c] = df[c].fillna(0.0)
    df["hour"]  = df["timestamp_local"].dt.hour
    df["month"] = df["timestamp_local"].dt.month
    df["year"]  = df["timestamp_local"].dt.year
    df["date"]  = df["timestamp_local"].dt.date
    df["net_load_mw"] = (df["total_load_mw"] - df["wind"] - df["solar"]
                         - df["power_storage"].clip(lower=0.0))
    return df


def lowess_delta_for_day(csv, day):
    """Paper-method LOWESS price impact per hour for `day` (for comparison)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    d = pd.Timestamp(day)
    month = csv[(csv["year"] == d.year) & (csv["month"] == d.month)]
    month = month[np.isfinite(month["dam_price"]) & np.isfinite(month["net_load_mw"])]
    dis = month[month["power_storage"] > 0]
    chg = month[month["power_storage"] < 0]

    def fit(sub):
        if len(sub) < 50:
            return None, None
        lw = lowess(sub["dam_price"].to_numpy(), sub["net_load_mw"].to_numpy(),
                    frac=0.5, return_sorted=True)
        return lw[:, 0], lw[:, 1]
    xd, yd = fit(dis)
    xc, yc = fit(chg)
    out = {}
    for _, r in month[month["date"] == d.date()].iterrows():
        ps = r["power_storage"]
        xs, ys = (xd, yd) if ps >= 0 else (xc, yc)
        if xs is None:
            continue
        p_fit = interp_lowess(xs, ys, r["net_load_mw"])
        p_adj = interp_lowess(xs, ys, r["net_load_mw"] + ps)
        out[int(r["hour"])] = p_adj - p_fit
    return out


# ── Analyze one day ─────────────────────────────────────────────────────────────

def load_cached(day):
    tag = day.replace("-", "")
    gen = pd.read_parquet(os.path.join(DATA_DIR, f"gen_{tag}.parquet"))
    eo  = pd.read_parquet(os.path.join(DATA_DIR, f"eo_{tag}.parquet"))
    lam = pd.read_parquet(os.path.join(DATA_DIR, f"lambda_{tag}.parquet"))
    for df in (gen, eo, lam):
        df["interval_start_utc"] = pd.to_datetime(df["interval_start_utc"], utc=True)
    for c in ["hsl", "awarded_quantity"] + AS_GEN:
        gen[c] = pd.to_numeric(gen[c], errors="coerce")
    lam["system_lambda"] = pd.to_numeric(lam["system_lambda"], errors="coerce")
    return gen, eo, lam


def analyze_day(day):
    gen, eo, lam = load_cached(day)
    csv = load_ercot_csv()
    day_csv = csv[csv["date"] == pd.Timestamp(day).date()].set_index("hour")

    rows = []
    for ts in sorted(gen["interval_start_utc"].unique()):
        hour_ct = pd.Timestamp(ts).tz_convert(LOCAL_TZ).hour
        g = gen[gen["interval_start_utc"] == ts]
        e = eo[eo["interval_start_utc"] == ts]
        L = lam[lam["interval_start_utc"] == ts]["system_lambda"]
        if L.empty or hour_ct not in day_csv.index:
            continue
        L = float(L.iloc[0])
        ps = float(day_csv.loc[hour_ct, "power_storage"])
        stack = build_stack(g, e)
        q_anchor, p_without, delta = counterfactual(stack, L, ps)
        rows.append({
            "hour_ct": hour_ct, "dam_lambda": L, "power_storage_mw": ps,
            "q_anchor_gw": q_anchor / 1000, "p_without": p_without,
            "dprice_bidstack": delta, "stack_gw": stack["cum_mw"].max() / 1000,
        })
    res = pd.DataFrame(rows).sort_values("hour_ct").reset_index(drop=True)
    res["dprice_lowess"] = res["hour_ct"].map(lowess_delta_for_day(csv, day))
    return res, (gen, eo, lam)


# ── Plots ────────────────────────────────────────────────────────────────────────

def plot_curve_snapshot(gen, eo, res, day, hour_ct, out_path):
    ts = [t for t in gen["interval_start_utc"].unique()
          if pd.Timestamp(t).tz_convert(LOCAL_TZ).hour == hour_ct][0]
    stack = build_stack(gen[gen["interval_start_utc"] == ts],
                        eo[eo["interval_start_utc"] == ts])
    row = res[res["hour_ct"] == hour_ct].iloc[0]
    q_a = row["q_anchor_gw"]
    q_cf = q_a + row["power_storage_mw"] / 1000

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.step(np.concatenate([[0], stack["cum_mw"] / 1000]),
            np.concatenate([[stack["price"].iloc[0]], stack["price"]]),
            where="post", color="#333", lw=1.2)
    ax.axhline(row["dam_lambda"], color="#0072B2", lw=1, ls="--", alpha=0.7)
    ax.plot(q_a, row["dam_lambda"], "o", color="#0072B2", ms=9, zorder=5,
            label=f"With batteries (observed λ) = ${row['dam_lambda']:.0f}")
    ax.plot(q_cf, row["p_without"], "s", color="#D55E00", ms=9, zorder=5,
            label=f"Without batteries = ${row['p_without']:.0f}  (Δ ${row['dprice_bidstack']:+.0f})")
    ax.annotate("", xy=(q_cf, row["p_without"]), xytext=(q_a, row["dam_lambda"]),
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.5))
    ax.set_xlabel("Cumulative offered supply (GW)")
    ax.set_ylabel("Offer price ($/MWh)")
    ax.set_title(f"ERCOT DAM bid-stack counterfactual — {day} {hour_ct:02d}:00 CT\n"
                 f"battery net {row['power_storage_mw']/1000:+.1f} GW → step along the curve")
    ax.set_ylim(-60, max(150, row["p_without"] * 1.3))
    ax.set_xlim(0, stack["cum_mw"].max() / 1000)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"saved {out_path}")


def plot_compare(res, day, out_path):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    h = res["hour_ct"]
    ax.bar(h - 0.2, res["dprice_bidstack"], width=0.4, color="#0072B2",
           label="DAM bid-stack (anchored)")
    ax.bar(h + 0.2, res["dprice_lowess"], width=0.4, color="#000",
           label="LOWESS (paper method)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Hour (CT)"); ax.set_ylabel("Battery price impact ($/MWh)")
    ax.set_title(f"DAM bid-stack vs LOWESS battery price impact — ERCOT {day}\n"
                 f"(+ = discharge savings, − = charging cost)")
    ax.set_xticks(range(0, 24, 2)); ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"saved {out_path}")


def run(day):
    res, (gen, eo, lam) = analyze_day(day)
    os.makedirs(FIG_DIR, exist_ok=True)
    tag = day.replace("-", "")
    print(res.to_string(index=False))
    corr = res[["dprice_bidstack", "dprice_lowess"]].dropna()
    if len(corr) > 2:
        print(f"\ncorr(bidstack Δ, LOWESS Δ) = "
              f"{corr['dprice_bidstack'].corr(corr['dprice_lowess']):.3f}")
    peak = int(res.loc[res["power_storage_mw"].idxmax(), "hour_ct"])
    plot_curve_snapshot(gen, eo, res, day, peak,
                        os.path.join(FIG_DIR, f"dam_curve_{tag}_{peak:02d}CT.png"))
    plot_compare(res, day, os.path.join(FIG_DIR, f"dam_compare_{tag}.png"))
    res.to_csv(os.path.join(FIG_DIR, f"dam_results_{tag}.csv"), index=False)
    return res


# ── Download ─────────────────────────────────────────────────────────────────────

def _download(day):
    from gridstatusio import GridStatusClient
    api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        sys.exit("Error: set the GRIDSTATUS_API_KEY environment variable.")
    client = GridStatusClient(api_key=api_key)
    os.makedirs(DATA_DIR, exist_ok=True)
    tag = day.replace("-", "")
    start = pd.Timestamp(day, tz=LOCAL_TZ).tz_convert("UTC")
    end = start + pd.Timedelta(hours=24)
    S, E = start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")

    specs = [
        ("ercot_dam_gen_resource_60_day", GEN_COLS, f"gen_{tag}.parquet"),
        ("ercot_dam_energy_only_offers_60_day", EO_COLS, f"eo_{tag}.parquet"),
        ("ercot_dam_system_lambda", None, f"lambda_{tag}.parquet"),
    ]
    for dataset, cols, name in specs:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            print(f"cached: {path}")
            continue
        kw = dict(dataset=dataset, start=S, end=E, timezone="utc", limit=100000)
        if cols:
            kw["columns"] = cols
        df = client.get_dataset(**kw)
        df["interval_start_utc"] = pd.to_datetime(df["interval_start_utc"], utc=True)
        df.to_parquet(path)
        print(f"saved {len(df)} rows -> {path}")


# ── Monthly overlay of the 8 PM DAM curve ───────────────────────────────────────

MONTH_YEAR, MONTH_M, MONTH_HOUR = 2025, 8, 20


def _month_cache():
    return (os.path.join(DATA_DIR, f"month_gen_{MONTH_YEAR}{MONTH_M:02d}_{MONTH_HOUR:02d}CT.parquet"),
            os.path.join(DATA_DIR, f"month_eo_{MONTH_YEAR}{MONTH_M:02d}_{MONTH_HOUR:02d}CT.parquet"),
            os.path.join(DATA_DIR, f"month_lambda_{MONTH_YEAR}{MONTH_M:02d}.parquet"))


def _download_month():
    from gridstatusio import GridStatusClient
    api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        sys.exit("Error: set the GRIDSTATUS_API_KEY environment variable.")
    client = GridStatusClient(api_key=api_key)
    os.makedirs(DATA_DIR, exist_ok=True)
    gen_c, eo_c, lam_c = _month_cache()

    days = pd.date_range(f"{MONTH_YEAR}-{MONTH_M:02d}-01", periods=31, freq="D")
    days = days[days.month == MONTH_M]

    for cache, dataset, cols in [(gen_c, "ercot_dam_gen_resource_60_day", GEN_COLS),
                                 (eo_c, "ercot_dam_energy_only_offers_60_day", EO_COLS)]:
        if os.path.exists(cache):
            print(f"cached: {cache}"); continue
        import time
        frames = []
        for d in days:
            ts = pd.Timestamp(f"{d:%Y-%m-%d} {MONTH_HOUR:02d}:00", tz=LOCAL_TZ).tz_convert("UTC")
            df = client.get_dataset(dataset=dataset, columns=cols,
                start=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end=(ts + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                timezone="utc", limit=40000)
            df["interval_start_utc"] = pd.to_datetime(df["interval_start_utc"], utc=True)
            frames.append(df)
            print(f"  {d:%Y-%m-%d} {MONTH_HOUR}:00 CT -> {len(df)} rows")
            time.sleep(1.5)
        pd.concat(frames, ignore_index=True).to_parquet(cache)
        print(f"saved -> {cache}")

    if not os.path.exists(lam_c):
        start = pd.Timestamp(f"{MONTH_YEAR}-{MONTH_M:02d}-01", tz=LOCAL_TZ).tz_convert("UTC")
        end = (start + pd.offsets.MonthBegin(1))
        lam = client.get_dataset(dataset="ercot_dam_system_lambda",
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"), end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            timezone="utc", limit=2000)
        lam["interval_start_utc"] = pd.to_datetime(lam["interval_start_utc"], utc=True)
        lam.to_parquet(lam_c)
        print(f"saved -> {lam_c}")


def plot_month():
    from matplotlib import cm
    from matplotlib.colors import Normalize
    gen_c, eo_c, lam_c = _month_cache()
    gen = pd.read_parquet(gen_c); eo = pd.read_parquet(eo_c); lam = pd.read_parquet(lam_c)
    for df in (gen, eo, lam):
        df["interval_start_utc"] = pd.to_datetime(df["interval_start_utc"], utc=True)
    for c in ["hsl", "awarded_quantity"] + AS_GEN:
        gen[c] = pd.to_numeric(gen[c], errors="coerce")
    lam["system_lambda"] = pd.to_numeric(lam["system_lambda"], errors="coerce")

    fig, ax = plt.subplots(figsize=(13, 8))
    cmap, norm = cm.viridis, Normalize(vmin=1, vmax=31)
    for ts in sorted(gen["interval_start_utc"].unique()):
        day = pd.Timestamp(ts).tz_convert(LOCAL_TZ).day
        stack = build_stack(gen[gen["interval_start_utc"] == ts],
                            eo[eo["interval_start_utc"] == ts])
        if stack.empty:
            continue
        color = cmap(norm(day))
        ax.step(np.concatenate([[0], stack["cum_mw"] / 1000]),
                np.concatenate([[stack["price"].iloc[0]], stack["price"]]),
                where="post", color=color, lw=1.0, alpha=0.7)
        L = lam[lam["interval_start_utc"] == ts]["system_lambda"]
        if not L.empty:
            q = anchor_quantity(stack, float(L.iloc[0])) / 1000
            ax.plot(q, float(L.iloc[0]), "o", color=color, ms=5,
                    markeredgecolor="black", markeredgewidth=0.4, zorder=5)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, pad=0.01).set_label(f"Day of {MONTH_YEAR}-{MONTH_M:02d}")
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel("Cumulative offered supply (GW)")
    ax.set_ylabel("Offer price ($/MWh)")
    ax.set_title(f"ERCOT DAM bid-stack price-demand curves at {MONTH_HOUR}:00 CT — "
                 f"every day of {MONTH_YEAR}-{MONTH_M:02d}\n"
                 f"(dots = anchor at each day's observed DAM λ)")
    ax.set_ylim(-60, 300); ax.set_xlim(0, None); ax.grid(alpha=0.25)
    out = os.path.join(FIG_DIR, f"dam_monthly_curves_{MONTH_YEAR}{MONTH_M:02d}_{MONTH_HOUR:02d}CT.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print(f"saved {out}")


def overlay_8pm(all_days=False, add_base=True):
    """Overlay the 20:00 CT supply stacks of every sampled day, 2024 vs 2025 —
    the ERCOT twin of CAISO's Fig. 7 (caiso_dam_counterfactual.overlay_8pm).

    Batteries are left in the stack (build_stack's default), matching the CAISO
    figure. The default 4-days-per-month sample mirrors CAISO for apples-to-
    apples comparison; all_days=True uses every cached day of 2024-2025.

    add_base shifts each priced stack right by its inferred price-taker base
    B = total_load - q*, where q* anchors the stack at the observed DAM price
    (the same construction as manuscript Fig. 1's ~36 GW base). Each curve then
    crosses the observed price at x = total load, placing the stacks at their
    real operating quantities instead of starting at zero offered MW.
    """
    import calendar
    import datetime as dt

    days_dir = os.path.join(DATA_DIR, "days")
    sample_days = [5, 12, 19, 26]
    os.makedirs(FIG_DIR, exist_ok=True)

    lut = None
    if add_base:
        csv = load_ercot_csv()
        csv = csv[~csv.duplicated(["date", "hour"], keep="first")]
        lut = csv.set_index(["date", "hour"])[["total_load_mw", "dam_price"]]

    fig, ax = plt.subplots(figsize=(13, 7))
    n, bases = 0, []
    for y in (2024, 2025):
        color = "#9ecae1" if y == 2024 else "#08519c"
        for m in range(1, 13):
            ndays = calendar.monthrange(y, m)[1]
            days = (range(1, ndays + 1) if all_days
                    else [d for d in sample_days if d <= ndays])
            for d in days:
                date = dt.date(y, m, d)
                tag = f"{date:%Y%m%d}"
                gp = os.path.join(days_dir, f"gen_{tag}.parquet")
                ep = os.path.join(days_dir, f"eo_{tag}.parquet")
                if not (os.path.exists(gp) and os.path.exists(ep)):
                    continue
                gen, eo = pd.read_parquet(gp), pd.read_parquet(ep)
                for df in (gen, eo):
                    df["interval_start_utc"] = pd.to_datetime(
                        df["interval_start_utc"], utc=True)
                for c in ["hsl", "awarded_quantity"] + AS_GEN:
                    if c in gen:
                        gen[c] = pd.to_numeric(gen[c], errors="coerce")
                ts20 = [ts for ts in gen["interval_start_utc"].unique()
                        if pd.Timestamp(ts).tz_convert(LOCAL_TZ).hour == 20]
                if not ts20:
                    continue
                ts = ts20[0]
                stack = build_stack(gen[gen["interval_start_utc"] == ts],
                                    eo[eo["interval_start_utc"] == ts])
                if stack.empty:
                    continue

                base_mw = 0.0
                if add_base:
                    try:
                        load, price = lut.loc[(date, 20)]
                    except KeyError:
                        continue
                    if not (np.isfinite(load) and np.isfinite(price)):
                        continue
                    q_star = anchor_quantity(stack, price)
                    base_mw = max(load - q_star, 0.0)
                    bases.append(base_mw / 1000)

                x = (stack["cum_mw"] + base_mw) / 1000
                ax.step(x, stack["price"], where="post",
                        color=color, lw=0.8, alpha=0.5)
                n += 1

    if add_base and bases:
        b_med = float(np.median(bases))
        ax.axvspan(0, b_med, color="#d9dbdd", alpha=0.35, zorder=0)
        ax.text(b_med / 2, 3000,
                f"price-taker base\n≈ {b_med:.0f} GW (median)\n"
                "self-scheduled · must-run · wind at floor\n(not in offer data)",
                ha="center", va="center", fontsize=9, color="#555")

    suffix = "" if all_days else " (4 days/month)"
    ax.plot([], [], color="#9ecae1", lw=2, label="2024" + suffix)
    ax.plot([], [], color="#08519c", lw=2, label="2025" + suffix)
    xlabel = ("Cumulative supply / quantity (GW)" if add_base
              else "Cumulative offered supply (GW)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Offer price ($/MWh)")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_ylim(-60, 5000)
    ax.set_yticks([-50, 0, 50, 100, 300, 1000, 3000, 5000])
    ax.get_yaxis().set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    base_note = " on price-taker base" if add_base else ""
    ax.set_title("ERCOT DAM supply stacks at 20:00 CT" + base_note + " — "
                 f"{'all days' if all_days else 'all sampled days'} 2024 vs 2025")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ercot_dam_monthly_curves_20CT.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}  ({n} stacks"
          + (f", median base {np.median(bases):.0f} GW)" if bases else ")"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    day = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DAY
    if cmd == "download":
        _download(day)
    elif cmd == "run":
        run(day)
    elif cmd == "download-month":
        _download_month()
    elif cmd == "month":
        plot_month()
    elif cmd == "overlay":
        overlay_8pm(all_days=("all" in sys.argv[2:]))
    else:
        sys.exit(f"Unknown command: {cmd}")
