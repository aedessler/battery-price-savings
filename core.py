"""
core.py — Pure math functions: bootstrap, interpolation, clamping, savings.
No I/O, no matplotlib.
"""

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

from config import (
    CI_LO, CI_HI, LOWESS_FRAC,
    BOUND_NETLOAD_ZERO, BOUND_PRICE_YEAR,
    EXTRAPOLATE, ZERO_FALLBACK,
    MIN_POINTS_HOUR,
)


# ── Interpolation / clamping ──────────────────────────────────────────────────

def interp_lowess(xs, ys, x_val, extrapolate=EXTRAPOLATE):
    """Interpolate (or linearly extrapolate) a LOWESS curve at x_val."""
    if extrapolate and len(xs) >= 2:
        if x_val < xs[0]:
            dx = xs[1] - xs[0]
            slope = (ys[1] - ys[0]) / dx if dx != 0 else 0.0
            return float(ys[0] + slope * (x_val - xs[0]))
        if x_val > xs[-1]:
            dx = xs[-1] - xs[-2]
            slope = (ys[-1] - ys[-2]) / dx if dx != 0 else 0.0
            return float(ys[-1] + slope * (x_val - xs[-1]))
    return float(np.interp(x_val, xs, ys))


def clamp_x(x_val):
    """Clamp net-load at 0 if BOUND_NETLOAD_ZERO is enabled."""
    if BOUND_NETLOAD_ZERO:
        return float(max(0.0, x_val))
    return float(x_val)


def clamp_price(p_val, year_floor):
    """Clamp price at year-minimum DAM price if BOUND_PRICE_YEAR is enabled."""
    if BOUND_PRICE_YEAR:
        return float(max(year_floor, p_val))
    return float(p_val)


# ── Curve selection ───────────────────────────────────────────────────────────

def pick_curve(ps_net):
    """Return 'discharge' or 'charge' based on ps_net sign."""
    if ps_net > 0:
        return "discharge"
    if ps_net < 0:
        return "charge"
    return ZERO_FALLBACK


# ── Bootstrap helpers ─────────────────────────────────────────────────────────

def bootstrap_curves(x, y, frac, n_boot, rng):
    """
    Bootstrap a single regime: resample (x, y) i.i.d. n_boot times and fit
    a LOWESS curve each time.

    Returns:
        list of (xs_b, ys_b) tuples (sorted arrays).
        Failed fits stored as (None, None).
    """
    n = len(x)
    curves = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        try:
            lw = lowess(yb, xb, frac=frac, return_sorted=True)
            curves.append((lw[:, 0], lw[:, 1]))
        except Exception:
            curves.append((None, None))
    return curves


def bootstrap_grid(x, y, frac, n_boot, rng, n_grid):
    """
    Bootstrap LOWESS and interpolate each fit onto a common x-grid.

    Returns:
        x_grid  : (n_grid,) array
        y_boots : (n_boot, n_grid) matrix — NaN for failed fits
    """
    n = len(x)
    x_grid = np.linspace(np.nanmin(x), np.nanmax(x), n_grid)
    y_boots = np.full((n_boot, n_grid), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        try:
            lw = lowess(yb, xb, frac=frac, return_sorted=True)
            y_boots[b, :] = np.interp(x_grid, lw[:, 0], lw[:, 1])
        except Exception:
            pass
    return x_grid, y_boots


def percentile_band(y_boots):
    """
    Compute (median, lo, hi) percentile curves from a (n_boot, n_grid) matrix.
    Uses CI_LO / CI_HI from config.
    """
    median = np.nanpercentile(y_boots, 50, axis=0)
    lo     = np.nanpercentile(y_boots, CI_LO, axis=0)
    hi     = np.nanpercentile(y_boots, CI_HI, axis=0)
    return median, lo, hi


# ── Hour-level inputs ─────────────────────────────────────────────────────────

def compute_hour_inputs(df_month):
    """
    Compute fixed per-hour summary statistics from the original data.
    These are used as the evaluation points across all bootstrap iterations.

    Returns:
        dict: hour (0–23) → {nl_fit, ps_net, nl_adj, curve, n_h, avg_load} | None
    """
    inputs = {}
    for h in range(24):
        df_h = df_month[df_month["hour"] == h]
        n_h = len(df_h)
        if n_h < MIN_POINTS_HOUR:
            inputs[h] = None
            continue
        nl_fit   = float(df_h["net_load_mw"].mean())
        ps_net   = float(df_h["power_storage"].mean())
        nl_adj   = nl_fit + ps_net
        avg_load = float(df_h["total_load_mw"].mean())
        inputs[h] = {
            "nl_fit":   nl_fit,
            "ps_net":   ps_net,
            "nl_adj":   nl_adj,
            "curve":    pick_curve(ps_net),
            "n_h":      n_h,
            "avg_load": avg_load,
        }
    return inputs


# ── Savings per iteration ─────────────────────────────────────────────────────

def savings_one_iteration(hour_inputs, xs_dis, ys_dis, xs_chg, ys_chg, yfloor,
                          x_floor=0.0):
    """
    Given one bootstrap curve-pair and fixed hour inputs, compute the per-hour
    price savings [$/MWh] for all 24 hours.

    Args:
        x_floor: minimum x value used when evaluating curves (default 0.0).
                 When BOUND_NETLOAD_MONTHLY_MIN is True, callers pass the
                 observed monthly minimum net load, stopping evaluation at
                 the left edge of the curve's observed data support.

    Returns:
        np.ndarray of shape (24,) — NaN for hours with no data or missing curve.
    """
    out = np.full(24, np.nan)
    for h in range(24):
        inp = hour_inputs[h]
        if inp is None:
            continue
        if inp["curve"] == "discharge":
            xs_use, ys_use = xs_dis, ys_dis
        else:
            xs_use, ys_use = xs_chg, ys_chg
        if xs_use is None or ys_use is None:
            continue
        x_fit = float(max(x_floor, inp["nl_fit"]))
        x_adj = float(max(x_floor, inp["nl_adj"]))
        p_fit = clamp_price(interp_lowess(xs_use, ys_use, x_fit), yfloor)
        p_adj = clamp_price(interp_lowess(xs_use, ys_use, x_adj), yfloor)
        out[h] = p_adj - p_fit
    return out
