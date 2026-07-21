"""Tests for stack price lookup, anchor conventions, and step-plot coordinates.

Runs under pytest, or standalone: `python tests/test_stack_lookup.py`.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from annual_savings import anchor_np, price_np, step_xy  # noqa: E402

FILL = 5000.0

# three 100-MW blocks priced 20 / 30 / 40 $/MWh
P = np.array([20.0, 30.0, 40.0])
CUM = np.array([100.0, 200.0, 300.0])


def _close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6), f"{a} != {b}"


def test_price_np_interior_and_boundaries():
    _close(price_np(P, CUM, 50.0, FILL), 20.0)     # inside block 0
    _close(price_np(P, CUM, 150.0, FILL), 30.0)    # inside block 1
    _close(price_np(P, CUM, 250.0, FILL), 40.0)    # inside block 2
    _close(price_np(P, CUM, 100.0, FILL), 20.0)    # exact right edge of block 0
    _close(price_np(P, CUM, 350.0, FILL), FILL)    # beyond the top -> fill


def test_anchor_left_right_bracket_the_block():
    # observed price 25 first reached in block 1 (price 30), spanning (100, 200]
    _close(anchor_np(P, CUM, 25.0, edge="right"), 200.0)
    _close(anchor_np(P, CUM, 25.0, edge="left"), 100.0)
    # default is the right edge (the current headline convention)
    _close(anchor_np(P, CUM, 25.0), 200.0)


def test_anchor_first_block_left_edge_is_zero():
    # price 20 is reached in block 0; its left edge is 0
    _close(anchor_np(P, CUM, 20.0, edge="right"), 100.0)
    _close(anchor_np(P, CUM, 20.0, edge="left"), 0.0)


def test_anchor_above_top_returns_stack_top_both_edges():
    _close(anchor_np(P, CUM, 45.0, edge="right"), 300.0)
    _close(anchor_np(P, CUM, 45.0, edge="left"), 300.0)


def test_step_xy_coordinates():
    x, y = step_xy(CUM, P)
    assert np.allclose(x, [0.0, 100.0, 200.0, 300.0])
    assert np.allclose(y, [20.0, 20.0, 30.0, 40.0])


def test_step_xy_draws_price_over_correct_interval():
    # with where="pre", segment y[i] is drawn from x[i-1] to x[i]; verify the
    # first block covers (0, cum0] at price[0] and each later block aligns with
    # the price_np lookup at that block's interior.
    x, y = step_xy(CUM, P)
    for qi, want in [(50.0, 20.0), (150.0, 30.0), (250.0, 40.0)]:
        # find the drawn step interval (x[i-1], x[i]] containing qi
        i = int(np.searchsorted(x, qi, side="left"))
        _close(y[i], price_np(P, CUM, qi, FILL))
        _close(y[i], want)


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
