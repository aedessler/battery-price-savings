#!/usr/bin/env python3
"""
Empirical scarcity-price model for the no-battery counterfactual.

Question it answers: when the bid-stack counterfactual runs into the top of the
offered supply curve (the ~1% of hours that dominate annual totals), what price
should the no-battery world be assigned? Administrative offer caps ($5,000
ERCOT / $1,000 CAISO) are an assumption, not an observation. This module builds
the empirical alternative: what DAM prices *actually did* at comparable net
loads in the pre-battery-buildout years, when tight hours had to clear without
storage.

Fit: for each market and season, take the pre-buildout years (ERCOT 2021-2023,
CAISO 2020-2023; battery output was <100 MW on average in every one of those
ERCOT years), keep hours in the top decile of net load, bin net load into
equal-count bins, and record the median (p50) and 90th-percentile (p90) DAM
price per bin. The result is an interpolable price-vs-net-load curve for the
scarcity region, grounded entirely in observed market outcomes. Evaluation
clamps at the historical net-load support (np.interp end behavior) — no
extrapolation beyond what the pre-battery market ever experienced, which makes
the central estimate conservative at the extreme top.

ERCOT's Feb 2021 Winter Storm Uri week is excluded by default (administrative
$9,000 pricing under emergency conditions); include_uri=True adds it back as a
sensitivity.

Usage:
    from scarcity_price import ScarcityModel
    m = ScarcityModel("ercot")           # or "caiso"
    m.price(month, net_load_mw)          # p50 central estimate
    m.price(month, net_load_mw, q="p90") # upper sensitivity

    python scarcity_price.py             # build both + validation figures
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figs")

FIT_YEARS = {"ercot": [2021, 2022, 2023], "caiso": [2020, 2021, 2022, 2023]}
SEASONS = {"summer": [6, 7, 8, 9], "winter": [12, 1, 2],
           "shoulder": [3, 4, 5, 10, 11]}
URI = (pd.Timestamp("2021-02-13"), pd.Timestamp("2021-02-21"))
TOP_QUANTILE = 0.90     # fit on the top decile of net load
N_BINS = 12


def _month_season(month):
    for s, months in SEASONS.items():
        if month in months:
            return s
    raise ValueError(month)


def _load_csv(market):
    if market == "ercot":
        from dam_counterfactual import load_ercot_csv
        return load_ercot_csv()
    from caiso_dam_counterfactual import load_paper_csv
    return load_paper_csv()


class ScarcityModel:
    def __init__(self, market, include_uri=False):
        self.market = market
        self.include_uri = include_uri
        csv = _load_csv(market)
        sub = csv[csv["year"].isin(FIT_YEARS[market])]
        if market == "ercot" and not include_uri:
            sub = sub[~((sub["timestamp_local"] >= URI[0])
                        & (sub["timestamp_local"] < URI[1]))]
        sub = sub[np.isfinite(sub["dam_price"]) & np.isfinite(sub["net_load_mw"])]
        self.curves = {}
        for season, months in SEASONS.items():
            ss = sub[sub["month"].isin(months)]
            thresh = ss["net_load_mw"].quantile(TOP_QUANTILE)
            top = ss[ss["net_load_mw"] >= thresh].copy()
            top["bin"] = pd.qcut(top["net_load_mw"], N_BINS, duplicates="drop")
            g = top.groupby("bin", observed=True).agg(
                nl=("net_load_mw", "mean"),
                p50=("dam_price", "median"),
                p90=("dam_price", lambda s: s.quantile(0.90)),
                n=("dam_price", "size"))
            self.curves[season] = g.reset_index(drop=True)

    def price(self, month, net_load_mw, q="p50"):
        """Empirical pre-buildout DAM price at `net_load_mw` (clamped to the
        historical net-load support)."""
        c = self.curves[_month_season(month)]
        return float(np.interp(net_load_mw, c["nl"], c[q]))


def validation_figure(market, include_uri=False):
    import matplotlib.pyplot as plt
    csv = _load_csv(market)
    sub = csv[csv["year"].isin(FIT_YEARS[market])]
    if market == "ercot" and not include_uri:
        sub = sub[~((sub["timestamp_local"] >= URI[0])
                    & (sub["timestamp_local"] < URI[1]))]
    model = ScarcityModel(market, include_uri=include_uri)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, (season, months) in zip(axes, SEASONS.items()):
        ss = sub[sub["month"].isin(months)]
        thresh = ss["net_load_mw"].quantile(TOP_QUANTILE)
        top = ss[ss["net_load_mw"] >= thresh]
        ax.scatter(top["net_load_mw"] / 1000, top["dam_price"], s=3,
                   alpha=0.15, color="#666666", label="pre-buildout hours")
        c = model.curves[season]
        ax.plot(c["nl"] / 1000, c["p50"], "o-", color="#0072B2", lw=2,
                label="p50 (central)")
        ax.plot(c["nl"] / 1000, c["p90"], "s--", color="#D55E00", lw=2,
                label="p90 (sensitivity)")
        ax.set_yscale("symlog", linthresh=100)
        ax.set_title(f"{season} (months {months})")
        ax.set_xlabel("Net load (GW)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("DAM price ($/MWh, symlog)")
    axes[0].legend(loc="upper left", fontsize=9)
    years = FIT_YEARS[market]
    uri = " (incl. Uri)" if (market == "ercot" and include_uri) else ""
    fig.suptitle(f"{market.upper()}: empirical scarcity price from pre-buildout "
                 f"years {years[0]}–{years[-1]}{uri} — top net-load decile",
                 fontsize=13)
    fig.tight_layout()
    tag = "_uri" if include_uri else ""
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"scarcity_empirical_fit_{market}{tag}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    for market in ["ercot", "caiso"]:
        m = ScarcityModel(market)
        print(f"--- {market} ---")
        for season in SEASONS:
            c = m.curves[season]
            print(f"{season:>9}: nl {c.nl.iloc[0]/1000:.1f}-{c.nl.iloc[-1]/1000:.1f} GW"
                  f" | p50 {c.p50.iloc[0]:.0f} -> {c.p50.iloc[-1]:.0f}"
                  f" | p90 {c.p90.iloc[0]:.0f} -> {c.p90.iloc[-1]:.0f}"
                  f" | n/bin ~{int(c.n.mean())}")
        validation_figure(market)
    validation_figure("ercot", include_uri=True)
