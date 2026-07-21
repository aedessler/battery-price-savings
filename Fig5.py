#!/usr/bin/env python3
"""Manuscript Figure 5 — monthly consumer savings under the central treatment
and the observed month-max cap, for each market.

Split out of the former make_ms_figs.py (archive/); the assumption-range band
was replaced by an explicit month-max line in July 2026.  Writes
fig5_monthly.jpg.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import C2025, MARKET_LABEL, save, treated


def main():
    t = treated()
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, market in zip(axes, ["ercot", "caiso"]):
        df = t[market]
        mo = df.groupby(["year", "month"])[
            ["cons_floor", "cons_central"]].sum().reset_index()
        mo["ym"] = (mo["year"].astype(str) + "-"
                    + mo["month"].astype(str).str.zfill(2))
        x = np.arange(len(mo))
        ax.plot(x, mo["cons_central"] / 1e6, "o-", color=C2025, lw=2, ms=5,
                label="central estimate (empirical scarcity price)")
        ax.plot(x, mo["cons_floor"] / 1e6, "s--", color="#555555", lw=1.6, ms=4,
                label="observed month-max cap")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(mo["ym"], rotation=90, fontsize=8)
        ax.set_yscale("symlog", linthresh=100)
        ax.set_ylabel("Monthly consumer savings ($M)")
        ax.set_title(MARKET_LABEL[market])
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig5_monthly.jpg")


if __name__ == "__main__":
    main()
