"""Shared setup for the Fig*.py manuscript figure scripts.

Each FigN.py builds exactly the figure numbered N in the manuscript and can be
run on its own:  python Fig6.py
"""
import os
import sys

OUT = os.path.dirname(os.path.abspath(__file__))   # the manuscript/ folder
FIGS = os.path.join(OUT, "figs")
sys.path.insert(0, OUT)     # the analysis modules now live alongside these scripts
os.chdir(OUT)

# Okabe-Ito, consistent across every manuscript figure
C2024 = "#0072B2"
C2025 = "#D55E00"
CGRAY = "#999999"
MARKET_LABEL = {"ercot": "ERCOT", "caiso": "CAISO"}
YEARS = [2024, 2025]

# 95% bootstrap CIs for the month-max ("floor") treatment, computed 2026-07-18
# by annual_savings.bootstrap_ci on cons_floor / prod_floor.  Cached here
# because the resampling takes several minutes and the inputs are frozen.
FLOOR_CI = {
    ("cons", "ercot", 2024): (2626, 4461), ("cons", "ercot", 2025): (3798, 4915),
    ("cons", "caiso", 2024): (2080, 2754), ("cons", "caiso", 2025): (-227, -124),
    ("prod", "ercot", 2024): (87, 138),    ("prod", "ercot", 2025): (248, 306),
    ("prod", "caiso", 2024): (571, 649),   ("prod", "caiso", 2025): (631, 677),
}

_treated = None


def treated():
    """{market: hourly dataframe with treatments applied}, loaded once."""
    global _treated
    if _treated is None:
        from annual_savings import _load_hourly, apply_treatments
        _treated = {m: apply_treatments(_load_hourly(m), m)
                    for m in ["ercot", "caiso"]}
        print("treated data loaded:", {m: len(d) for m, d in _treated.items()})
    return _treated


def save(fig, name, **kw):
    import matplotlib.pyplot as plt
    path = os.path.join(OUT, name)
    if name.lower().endswith((".jpg", ".jpeg")):
        kw.setdefault("pil_kwargs", {"quality": 92})
    fig.savefig(path, dpi=200, **kw)
    plt.close(fig)
    print("saved", path)
    return path
