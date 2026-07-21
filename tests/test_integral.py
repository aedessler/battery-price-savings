"""Exact-value tests for the production-cost stack integral.

Covers the two confirmed numerical bugs the audit found in the old code:
  #1  the searchsorted-slice integral dropped the top partial segment;
  #2  caps below the $100 grid floor were silently evaluated at $100.
Both are exercised directly below (the boundary-to-interior case must return
250, not the old 100; a $60 cap must differ from a $100 cap).

Runs under pytest, or standalone: `python tests/test_integral.py`.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from annual_savings import integral_np, traversed_np  # noqa: E402

FILL = 5000.0

# canonical stack: three 100-MW blocks priced 20 / 30 / 40 $/MWh
P = np.array([20.0, 30.0, 40.0])
MW = np.array([100.0, 100.0, 100.0])
CUM = np.array([100.0, 200.0, 300.0])


def _close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6), f"{a} != {b}"


def test_inside_first_segment():
    # single 100-MW segment priced 10, integrate 0..50 -> 500 (audit repro)
    _close(integral_np(np.array([10.0]), np.array([100.0]), np.array([100.0]),
                       0.0, 50.0, None, FILL), 500.0)
    # inside the first block of the canonical stack
    _close(integral_np(P, MW, CUM, 0.0, 50.0, None, FILL), 20.0 * 50)


def test_boundary_to_interior_bug1():
    # segments (0,10],(10,20],(20,30] priced 20/30/40; integrate 5..15.
    # correct = 20*5 + 30*5 = 250; the old slice bug returned 100.
    p = np.array([20.0, 30.0, 40.0])
    mw = np.array([10.0, 10.0, 10.0])
    cum = np.array([10.0, 20.0, 30.0])
    _close(integral_np(p, mw, cum, 5.0, 15.0, None, FILL), 250.0)


def test_multi_segment_complete_and_partial():
    # 0..250 -> 20*100 + 30*100 + 40*50 = 7000
    _close(integral_np(P, MW, CUM, 0.0, 250.0, None, FILL), 7000.0)


def test_exact_block_boundary_endpoints():
    # 100..200 -> exactly the middle block, 30*100
    _close(integral_np(P, MW, CUM, 100.0, 200.0, None, FILL), 3000.0)
    # 0..300 -> whole stack
    _close(integral_np(P, MW, CUM, 0.0, 300.0, None, FILL), 9000.0)


def test_reverse_charging_interval():
    # q2 < q1 flips the sign
    _close(integral_np(P, MW, CUM, 250.0, 0.0, None, FILL), -7000.0)


def test_interval_below_zero_is_clamped():
    # negative start clamps to 0, so -50..50 == 0..50
    _close(integral_np(P, MW, CUM, -50.0, 50.0, None, FILL), 20.0 * 50)


def test_beyond_stack_uses_fill():
    # 0..400 -> whole 9000 + fill over the last 100 MW
    _close(integral_np(P, MW, CUM, 0.0, 400.0, None, FILL), 9000.0 + FILL * 100)
    # fill is itself capped when a cap below fill is supplied
    _close(integral_np(P, MW, CUM, 0.0, 400.0, 100.0, FILL),
           (20 + 30 + 40) * 100 + 100.0 * 100)


def test_cap_below_100_differs_from_cap_100_bug2():
    # stack priced 50 / 80; 0..200. cap 100 keeps both offers; cap 60 clips 80->60.
    p = np.array([50.0, 80.0])
    mw = np.array([100.0, 100.0])
    cum = np.array([100.0, 200.0])
    at100 = integral_np(p, mw, cum, 0.0, 200.0, 100.0, FILL)
    at60 = integral_np(p, mw, cum, 0.0, 200.0, 60.0, FILL)
    _close(at100, 50 * 100 + 80 * 100)          # 13000
    _close(at60, 50 * 100 + 60 * 100)           # 11000, NOT clamped to 13000
    assert at60 < at100


def test_zero_and_negative_caps_with_negative_prices():
    # a negative-price offer below a positive one
    p = np.array([-10.0, 20.0])
    mw = np.array([100.0, 100.0])
    cum = np.array([100.0, 200.0])
    _close(integral_np(p, mw, cum, 0.0, 200.0, 0.0, FILL), -10.0 * 100)     # cap 0
    _close(integral_np(p, mw, cum, 0.0, 200.0, -5.0, FILL),
           -10.0 * 100 + -5.0 * 100)                                        # cap -5


def test_empty_interval():
    _close(integral_np(P, MW, CUM, 100.0, 100.0, None, FILL), 0.0)
    pr, dq = traversed_np(P, MW, CUM, 100.0, 100.0, FILL)
    assert pr.size == 0 and dq.size == 0


def test_traversed_matches_integral_for_arbitrary_caps():
    # the cached-segment invariant: sum(min(price,cap)*dq) == integral_np
    rng = np.random.default_rng(0)
    for _ in range(200):
        q1, q2 = sorted(rng.uniform(-20, 360, size=2))
        cap = rng.choice([None, -5.0, 0.0, 25.0, 55.0, 100.0, 1e9])
        pr, dq = traversed_np(P, MW, CUM, q1, q2, FILL)
        seg = float(np.minimum(pr, cap) @ dq) if (pr.size and cap is not None) \
            else (float(pr @ dq) if pr.size else 0.0)
        _close(seg, integral_np(P, MW, CUM, q1, q2, cap, FILL))


def test_int_raw_equals_uncapped_integral():
    # int_raw is cached as sum(seg_price*seg_dq); must equal the None-cap integral
    pr, dq = traversed_np(P, MW, CUM, 40.0, 265.0, FILL)
    _close(float(pr @ dq), integral_np(P, MW, CUM, 40.0, 265.0, None, FILL))


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
