#!/usr/bin/env python3
"""Render the exact L24 C10 boundary-source counts at manuscript size."""
from __future__ import annotations
from publication_io import save_figure_atomic

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
record = json.loads((ROOT / "reproducibility/results/common_carrier/COMMON_CARRIER_AUDIT.json").read_text())
g = record["geometry"]
count_all = g["sources_with_all_probability_to_open_boundary"]
count_some = g["sources_touching_open_boundary"] - count_all
count_none = g["vertices"] - g["sources_touching_open_boundary"]
labels = ["No open-boundary transition", "Some probability outward", "All probability outward"]
counts = [count_none, count_some, count_all]
assert counts == [3059, 24, 842] and sum(counts) == g["vertices"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.0,
    "axes.titlesize": 10.0, "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5, "ytick.labelsize": 9.0,
    "axes.edgecolor": "#5E6462", "text.color": "#171717",
    "axes.labelcolor": "#171717", "xtick.color": "#5E6462",
    "ytick.color": "#171717", "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(178 / 25.4, 73 / 25.4), facecolor="white")
fig.subplots_adjust(left=.31, right=.95, bottom=.23, top=.78)
colors = ["#B6B6B0", "#5E6462", "#1F6F5F"]
ys = [2, 1, 0]
ax.barh(ys, counts, height=.46, color=colors, linewidth=0)
ax.set_yticks(ys, labels)
ax.set_xlim(0, 3600)
ax.xaxis.set_major_locator(MultipleLocator(1000))
ax.set_xlabel("Source nodes (count)")
ax.tick_params(axis="both", length=3, width=.6)
ax.tick_params(axis="y", length=0, pad=8)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_linewidth(.6)
for y, value in zip(ys, counts):
    ax.text(value + 48, y, f"{value:,}", va="center", ha="left", fontsize=9,
            color="#171717")
fig.text(.31, .92, "The C10 graph has an open boundary", fontsize=10,
         weight="semibold", color="#171717")
fig.text(.31, .845, "L = 24, depth 9 · 3,925 source nodes", fontsize=8.4,
         color="#5E6462")
out = ROOT / "paper/figures/c10_boundary_source_audit"
save_figure_atomic(fig, out.with_suffix(".pdf"), facecolor="white")
save_figure_atomic(fig, out.with_suffix(".png"), dpi=220, facecolor="white")
plt.close(fig)
print(str(out.with_suffix(".pdf")))
