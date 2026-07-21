"""
config.py — All parameters and market configuration dicts.
"""

import os

# ── Bootstrap params ──────────────────────────────────────────────────────────
N_BOOT    = 500
CI_LO     = 2.5
CI_HI     = 97.5
BOOT_SEED = 42
N_GRID    = 300

# ── LOWESS params ─────────────────────────────────────────────────────────────
LOWESS_FRAC = 0.5
EXTRAPOLATE  = True

# ── Methodology ───────────────────────────────────────────────────────────────
MIN_POINTS_CURVE = 200
MIN_POINTS_HOUR  = 10
ZERO_FALLBACK    = "discharge"   # curve to use when power_storage == 0

# ── Bounds ────────────────────────────────────────────────────────────────────
BOUND_NETLOAD_MONTHLY_MIN = True    # floor x at observed monthly min net load
BOUND_NETLOAD_ZERO        = False   # clamp x at 0 before evaluating curves
BOUND_PRICE_YEAR          = False   # clamp price at year-min DAM price

# ── Output root ───────────────────────────────────────────────────────────────
BASE_OUTPUT = os.path.join(os.path.dirname(__file__), "paper_figs")

# ── Plot styling ──────────────────────────────────────────────────────────────
SCATTER_ALPHA  = 0.35
SCATTER_S      = 30
LINE_WIDTH     = 2.5           # standardised (CAISO was 1.0, ERCOT was 5.0)
BAND_ALPHA     = 0.25
SAVE_DPI       = 350

MONTH_COLORS = {
    12: "#2171b5", 1: "#1f77b4", 2: "#08306b",
    3:  "#b2df8a", 4: "#33a02c", 5: "#006400",
    6:  "#d32f2f", 7: "#b71c1c", 8: "#7f0000",
    9:  "#d2b48c", 10: "#8b5a2b", 11: "#5c4033",
}
DEC_2025_COLOR = "orange"

# ── Typography ────────────────────────────────────────────────────────────────
RCPARAMS = {
    "font.size":           18,
    "axes.labelsize":      20,
    "axes.labelweight":    "bold",
    "axes.titlesize":      20,
    "axes.titleweight":    "bold",
    "xtick.labelsize":     16,
    "ytick.labelsize":     16,
    "legend.fontsize":     18,
    "figure.titlesize":    30,
    "axes.titlepad":       8,
    "axes.linewidth":      1.6,
    "xtick.major.width":   1.4,
    "ytick.major.width":   1.4,
    "xtick.major.size":    6,
    "ytick.major.size":    6,
}

# ── Market dicts ──────────────────────────────────────────────────────────────
ERCOT = dict(
    name         = "ERCOT",
    data_file    = "/Users/austinsabol/master/data/ercot/complete_ercot_2020_2025.csv",
    years        = [2024, 2025],
    sub_batteries = True,
    sub_imports   = False,
    imports_col   = None,
    colors        = {2024: "#9ecae1", 2025: "#1f77b4"},
    scatter_color_discharge = "#0072B2",
    scatter_color_charge    = "#CC6600",
    line_color_discharge    = "#0072B2",
    line_color_charge       = "#CC6600",
    scatter_alpha           = 0.08,
    out_curves  = os.path.join(BASE_OUTPUT, "ercot", "curves"),
    out_hourly  = os.path.join(BASE_OUTPUT, "ercot", "hourly"),
    out_savings = os.path.join(BASE_OUTPUT, "ercot", "savings"),
)

CAISO = dict(
    name         = "CAISO",
    data_file    = "/Users/austinsabol/master/data/caiso/complete_caiso_2020_2025.csv",
    years        = [2021, 2022, 2023, 2024, 2025],
    sub_batteries = True,
    sub_imports   = False,
    imports_col   = None,
    # Cold snap / grid stress event — remove from fitting
    remove_jan13thru16 = True,
    event_exclusions   = ["2024-01-13", "2024-01-14", "2024-01-15", "2024-01-16"],
    colors        = {2021: "#feedde", 2022: "#fdbe85", 2023: "#fd8d3c", 2024: "#e6550d", 2025: "#a63603"},
    scatter_color_discharge = "#0072B2",
    scatter_color_charge    = "#CC6600",
    line_color_discharge    = "#0072B2",
    line_color_charge       = "#CC6600",
    scatter_alpha           = 0.08,
    out_curves  = os.path.join(BASE_OUTPUT, "caiso", "curves"),
    out_hourly  = os.path.join(BASE_OUTPUT, "caiso", "hourly"),
    out_savings = os.path.join(BASE_OUTPUT, "caiso", "savings"),
)
