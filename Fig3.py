#!/usr/bin/env python3
"""Manuscript Figure 3 — the empirical scarcity-price model for CAISO.

Thin wrapper: the figure is produced by scarcity_price.validation_figure,
which writes a PNG into ./figs/; this script re-encodes it as
fig3_scarcity_fit_caiso.jpg in this directory (what the manuscript embeds).
"""
import os

from PIL import Image

from _common import FIGS, OUT
from scarcity_price import validation_figure

if __name__ == "__main__":
    validation_figure("caiso")
    src = os.path.join(FIGS, "scarcity_empirical_fit_caiso.png")
    dst = os.path.join(OUT, "fig3_scarcity_fit_caiso.jpg")
    Image.open(src).convert("RGB").save(dst, quality=92)
    print("saved", dst)
