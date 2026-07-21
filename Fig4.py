#!/usr/bin/env python3
"""Manuscript Figure 4 — mean hourly price impact of the battery fleet, by
hour of day, for each market and year (central estimate).

Split out of the former make_ms_figs.py (archive/).  Writes fig4_diurnal.jpg.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import C2024, C2025, MARKET_LABEL, save, treated


def main():
    t = treated()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    for ax, market in zip(axes, ["ercot", "caiso"]):
        df = t[market]
        for y, c in [(2024, C2024), (2025, C2025)]:
            sub = df[df["year"] == y]
            prof = sub.groupby("hour").apply(
                lambda g: (g["p_central"] - g["lam"]).mean(),
                include_groups=False)
            ax.plot(prof.index, prof.values, "o-", ms=5, lw=2, color=c,
                    label=str(y))
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Hour of day (local)")
        ax.set_title(MARKET_LABEL[market])
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean price impact of removing batteries ($/MWh)")
    axes[0].legend()
    fig.tight_layout()
    save(fig, "fig4_diurnal.jpg")

    # peak values quoted in the manuscript text
    for market in ["ercot", "caiso"]:
        df = t[market]
        for y in [2024, 2025]:
            sub = df[df["year"] == y]
            prof = sub.groupby("hour").apply(
                lambda g: (g["p_central"] - g["lam"]).mean(),
                include_groups=False)
            print(f"{market} {y}: peak +{prof.max():.1f} at hour {prof.idxmax()}, "
                  f"trough {prof.min():.1f} at hour {prof.idxmin()}")


if __name__ == "__main__":
    main()
