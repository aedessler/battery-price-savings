#!/usr/bin/env python3
"""Illustrative supply-demand figure for the anchored bid-stack counterfactual.

Draws the full merit order for a single hour (peak battery discharge on
2025-08-15) the way the method actually treats it:

  * a wide price-taker BASE (self-scheduled + must-run + renewables at the
    floor) that serves most of the load but never appears as a rising priced
    offer -- drawn as a schematic block because its width is inferred, not
    measured;
  * the reconstructed PRICED top (three-part + energy-only offers), fuel-
    coloured, sitting on the base;
  * demand fixed at the observed total load;
  * the battery discharge b as the slice at the top of the dispatch that holds
    the marginal offer down at the observed DAM price (point A). Remove it and
    the margin climbs the no-battery curve to the counterfactual price (point
    C).

Every price and quantity is taken from the SAME functions annual_savings.py
uses for the headline numbers (anchor on the no-battery stack, read at
q* + power_storage), so the figure matches the manuscript exactly. Writes fig1_anchor.jpg in this directory.

Manuscript Figure 1.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
os.chdir(OUT)

import dam_counterfactual as dc
import annual_savings as A
from stack_counterfactual import TYPE_MAP, FIG1_MERGE, FIG1_ORDER, FIG1_COLORS

DAY = "2025-08-15"
HOUR_CT = 20
PRICE_CLIP = 150
BASE_COLOR = "#d9dbdd"
BASE_TOP = 6.0                   # schematic drawn height of the price-taker base
OBS_C = FIG1_COLORS["Battery"]    # teal, observed / with-battery
CF_C = "#D55E00"                  # vermillion, counterfactual / no-battery


def fig1_fuel(rt):
    f = TYPE_MAP.get(rt, "Other")
    return "Other" if f in FIG1_MERGE else f


def colored_nb_segments(g, e):
    """No-battery priced stack carrying a fuel label per segment, fuel-contiguous
    within tied prices (same merit order as annual_savings._ercot_hour_stacks)."""
    segs = []
    cap = g["hsl"] - g[dc.AS_GEN].fillna(0).sum(axis=1)
    cap = cap.where(cap > 0, g["hsl"])
    for c, curve, rt in zip(cap.to_numpy(), g["qse_submitted_curve"].to_numpy(),
                            g["resource_type"].to_numpy()):
        if rt == "PWRSTR":
            continue
        for mw, p in dc._curve_segments(dc.parse_curve(curve),
                                        c if np.isfinite(c) else None):
            segs.append((mw, p, fig1_fuel(rt)))
    for v in e["energy_only_offer_curve"].to_numpy():
        for mw, p in dc._curve_segments(dc.parse_curve(v)):
            segs.append((mw, p, "Energy-only offers"))
    rank = {f: i for i, f in enumerate(FIG1_ORDER + ["Energy-only offers"])}
    st = pd.DataFrame(segs, columns=["mw", "price", "fuel"])
    st["_r"] = st["fuel"].map(lambda f: rank.get(f, 99))
    st = st.sort_values(["price", "_r"], kind="mergesort").reset_index(drop=True)
    st["cum_mw"] = st["mw"].cumsum()
    return st


def main():
    gen, eo, lam = dc.load_cached(DAY)
    csv = dc.load_ercot_csv()
    r = csv[(csv["date"] == pd.Timestamp(DAY).date()) & (csv["hour"] == HOUR_CT)].iloc[0]
    ts = [t for t in gen["interval_start_utc"].unique()
          if pd.Timestamp(t).tz_convert(dc.LOCAL_TZ).hour == HOUR_CT][0]
    g = gen[gen["interval_start_utc"] == ts]
    e = eo[eo["interval_start_utc"] == ts]
    L = float(lam[lam["interval_start_utc"] == ts]["system_lambda"].iloc[0])
    ps = float(r["power_storage"])            # + = discharge (MW)
    load = float(r["total_load_mw"])

    # ── numbers straight from the headline method ────────────────────────────
    st_f, st_n = A._ercot_hour_stacks(g, e)
    pn, mn, cn = A._arrays(st_n)
    fill = A.ADMIN_CAP["ercot"]
    q_nb = A.anchor_np(pn, cn, L)             # anchor the no-battery stack at obs price
    p_nb = min(A.price_np(pn, cn, q_nb + ps, fill), fill)
    delta = p_nb - L

    B = (load - (q_nb + ps)) / 1000           # inferred price-taker base (GW)
    Ax = B + q_nb / 1000                       # with-battery operating point (GW)
    Cx = load / 1000                           # demand / no-battery operating point (GW)

    print(f"observed price      = ${L:.1f}")
    print(f"battery discharge b = {ps/1000:+.2f} GW")
    print(f"total load          = {load/1000:.1f} GW")
    print(f"inferred base B     = {B:.1f} GW")
    print(f"A (with batt)       = ({Ax:.1f} GW, ${L:.0f})")
    print(f"C (no batt)         = ({Cx:.1f} GW, ${p_nb:.0f})   delta {delta:+.0f}")

    # ── figure ───────────────────────────────────────────────────────────────
    stc = colored_nb_segments(g, e)
    fig, ax = plt.subplots(figsize=(14.5, 8))
    XHI = Cx + 4.5
    YLO, YHI = -8, PRICE_CLIP

    # price-taker base (schematic block)
    ax.add_patch(mpatches.Rectangle((0, 0), B, BASE_TOP, facecolor=BASE_COLOR,
                 edgecolor="#b7babd", hatch="////", lw=0.8, zorder=1))
    ax.annotate(f"price-taker base ≈ {B:.0f} GW\nself-scheduled · must-run · wind at the floor\n",
                # f"(serves most of the load, sets no price — not in the offer data)",
                xy=(B / 2, BASE_TOP), xytext=(B / 2, 5),
                ha="center", fontsize=10.5, color="#555",
                arrowprops=None)#dict(arrowstyle="-", color="#b7babd", lw=0.8))

    # priced top (fuel-coloured), shifted right by the base
    seen = set()
    x = B
    for _, s in stc.iterrows():
        w = s["mw"] / 1000
        ax.bar(x + w / 2, min(s["price"], PRICE_CLIP), width=w, bottom=0,
               color=FIG1_COLORS.get(s["fuel"], "#c7ccd1"), linewidth=0, zorder=2)
        x += w
        seen.add(s["fuel"])

    # battery slice A->C : the discharge that holds the margin down
    ax.axvspan(Ax, Cx, color=OBS_C, alpha=0.16, zorder=1.5)
    ax.annotate("", xy=(Cx, L), xytext=(Ax, L),
                arrowprops=dict(arrowstyle="<->", color=OBS_C, lw=2))
    # ax.annotate(f"battery discharge  b = {ps/1000:.1f} GW\n"
    #             f"fills the top of the dispatch — remove it and\n"
    #             f"the same load is served by climbing the\n"
    #             f"offers from A up to C",
    #             xy=((Ax + Cx) / 2, (L + p_nb) / 2), xytext=(30, 96),
    #             ha="left", va="center", fontsize=10, color=OBS_C, fontweight="bold",
    #             arrowprops=dict(arrowstyle="->", color=OBS_C, lw=1.3),
    #             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=OBS_C, alpha=0.95))

    # demand
    ax.plot([Cx, Cx], [0, YHI], color="black", lw=1.6, ls=(0, (6, 4)))
    ax.text(Cx + 0.15, YHI - 6, f"demand = total load {load/1000:.0f} GW",
            rotation=90, va="top", ha="left", fontsize=10.5)

    # observed price line + point A
    ax.axhline(L, color="black", lw=0.9, ls="--", alpha=0.6)
    ax.plot(Ax, L, "o", color=OBS_C, ms=14, mec="black", mew=0.6, zorder=6)
    ax.annotate(f"A  with batteries\nmarginal offer = observed DAM price  ${L:.0f}",
                xy=(Ax, L), xytext=(Ax - 2, L + 24), ha="right",
                fontsize=10.5, color=OBS_C, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=OBS_C, lw=1.5))

    # counterfactual point C
    ax.plot(Cx, p_nb, "o", color=CF_C, ms=14, mec="black", mew=0.6, zorder=6)
    ax.annotate(f"C  batteries removed\nmargin climbs to  ${p_nb:.0f}/MWh",
                xy=(Cx, p_nb), xytext=(Cx - 2.0, p_nb + 8), ha="right",
                fontsize=10.5, color=CF_C, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CF_C, lw=1.5))

    # savings bracket
    ax.annotate("", xy=(Cx + 1.4, L), xytext=(Cx + 1.4, p_nb),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.5))
    ax.text(Cx + 1.7, (L + p_nb) / 2, f"Δ = +${delta:.0f}/MWh\nprice batteries\nsuppressed",
            ha="left", va="center", fontsize=10.5)

    # anchor note
    # ax.text(B + 0.3, YHI - 6,
    #         "ANCHOR: the base width is unknown, so we slide the priced top until\n"
    #         "the marginal offer at A equals the observed DAM price. Everything is\n"
    #         "then measured from A — the counterfactual is the step A → C.",
    #         ha="left", va="top", fontsize=9.5, style="italic", color="#333",
    #         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))

    patches = [mpatches.Patch(facecolor=BASE_COLOR, edgecolor="#b7babd",
                              hatch="////", label="price-taker base (schematic)")]
    patches += [mpatches.Patch(color=FIG1_COLORS[f], label=f)
                for f in FIG1_ORDER if f in seen]
    patches.append(mpatches.Patch(color="#c7ccd1", label="Energy-only offers"))
    ax.legend(handles=patches, loc="upper left", fontsize=9.5, framealpha=0.95)

    ax.set_xlim(0, XHI)
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel("Cumulative supply / quantity (GW)")
    ax.set_ylabel("Offer price ($/MWh)")
    ax.grid(alpha=0.22, zorder=0)
    ts_ct = pd.Timestamp(ts).tz_convert(dc.LOCAL_TZ)
    ax.set_title(f"Anchored bid-stack counterfactual — ERCOT DAM, "
                 f"{ts_ct:%Y-%m-%d %H:%M %Z} (peak battery discharge)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT, "fig1_anchor.jpg")
    fig.savefig(out, dpi=200, pil_kwargs={"quality": 92})
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
