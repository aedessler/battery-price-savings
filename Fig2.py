#!/usr/bin/env python3
"""Manuscript Figure 2 — the empirical scarcity-price model for ERCOT.

Thin wrapper: the figure is produced by scarcity_price.validation_figure,
which writes a PNG into ./figs/; this script re-encodes it as
fig2_scarcity_fit_ercot.jpg in this directory (what the manuscript embeds).
"""
import os

from PIL import Image

from _common import FIGS, OUT
from scarcity_price import validation_figure

if __name__ == "__main__":
    validation_figure("ercot")
    src = os.path.join(FIGS, "scarcity_empirical_fit_ercot.png")
    dst = os.path.join(OUT, "fig2_scarcity_fit_ercot.jpg")
    Image.open(src).convert("RGB").save(dst, quality=92)
    print("saved", dst)
