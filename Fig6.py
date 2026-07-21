#!/usr/bin/env python3
"""Manuscript Figure 6 — CAISO 20:00 PT supply stacks for all sampled days
(4 per month, 2024-25), colored by date, each day's anchored operating point
marked.  Annotates the two features referred to throughout the text: "the
wall" (the near-vertical run of high-priced offers at the top of every stack)
and "the drop" (where offers collapse to zero and below at low demand).

Was Fig. 7 through manuscript V3; renumbered to Fig. 6 in V4 (the old annual-
range Fig. 6 was dropped).  Writes fig6_stacks.jpg.
"""
import calendar
import datetime as dt
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from _common import save  # noqa: F401  (sets sys.path + cwd)

warnings.filterwarnings("ignore")

import annual_savings as A
import caiso_dam_counterfactual as cc

HOUR = 20
YLIM = (-175, 400)


def caiso_curves():
    paper = cc.load_paper_csv()
    out = []
    for (y, m) in cc.YEARS_MONTHS:
        for d in cc.SAMPLE_DAYS:
            if d > calendar.monthrange(y, m)[1]:
                continue
            date = dt.date(y, m, d)
            try:
                stacks = cc.hourly_stacks(cc.load_day(date), date)
            except FileNotFoundError:
                continue
            if HOUR not in stacks:
                continue
            day = paper[paper["date"] == date].set_index("hour")
            day = day[~day.index.duplicated(keep="first")]
            if HOUR not in day.index:
                continue
            lam = float(day.loc[HOUR, "dam_price"])
            if not np.isfinite(lam):
                continue
            st = stacks[HOUR]
            p, cum = st["price"].to_numpy(), st["cum_mw"].to_numpy()
            out.append((date, p, cum, lam, A.anchor_np(p, cum, lam)))
    return out


def main():
    curves = caiso_curves()
    t0 = dt.date(2024, 1, 1).toordinal()
    t1 = dt.date(2025, 12, 31).toordinal()
    norm = Normalize(vmin=t0, vmax=t1)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(13, 7))
    qs, ls, ts = [], [], []
    for date, p, cum, lam, q0 in curves:
        ax.step(cum / 1000, p, where="post",
                color=cmap(norm(date.toordinal())), lw=0.8, alpha=0.45)
        qs.append(q0 / 1000)
        ls.append(lam)
        ts.append(date.toordinal())
    ax.scatter(qs, ls, s=24, c=[cmap(norm(t)) for t in ts],
               edgecolors="black", linewidths=0.6, zorder=5,
               label="operating point (anchored at observed price)")

    arrow = dict(arrowstyle="-|>", color="black", lw=1.6,
                 shrinkA=4, shrinkB=2)
    ax.annotate("“the wall”", xy=(43.5, 240), xytext=(27, 245),
                fontsize=16, ha="center", va="center",
                arrowprops=arrow, zorder=6)
    ax.annotate("“the drop”", xy=(16.3, 10), xytext=(15.5, 115),
                fontsize=16, ha="center", va="bottom",
                arrowprops=arrow, zorder=6)

    ax.set_ylim(*YLIM)
    ax.set_xlabel("Cumulative supply (GW)")
    ax.set_ylabel("Offer price ($/MWh)")
    ax.set_title("CAISO DAM supply stacks at 20:00 PT — all sampled days, "
                 "colored by date; dots mark each day's operating point")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    ticks = [dt.date(yy, mm, 1).toordinal()
             for yy, mm in [(2024, 1), (2024, 7), (2025, 1),
                            (2025, 7), (2025, 12)]]
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.03)
    cb.set_ticks(ticks)
    cb.set_ticklabels([dt.date.fromordinal(t).strftime("%b %Y")
                       for t in ticks])

    print(f"{len(curves)} days, median operating point "
          f"{np.median(qs):.1f} GW, median price ${np.median(ls):.0f}")
    save(fig, "fig6_stacks.jpg", bbox_inches="tight")


if __name__ == "__main__":
    main()
