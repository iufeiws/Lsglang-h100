#!/usr/bin/env python3
"""Figures for the 6-row pool-contract report (English labels, Morandi palette).

Reads logs/pool6row_<run-ts>/analyze-*.json produced by analyze_pool6row.py.

Usage:
  plot_pool6row.py --run-ts 20261023_1657 --tags Aprobe,DETSPLIT [--out docs/figs/...png]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/mnt/SSD01/models/.llm-env/Lsglang"
# Morandi palette
C = {
    "pool":      "#8C9A93",
    "pool2":     "#B0A8A0",
    "step":      "#6E7C7C",
    "acc":       "#A38C7E",
    "launch":    "#9AA3AE",
    "target":    "#7D8A93",
    "grid":      "#D6D2CC",
    "text":      "#4A4A4A",
    "hl":        "#B08A6E",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": C["grid"],
    "axes.labelcolor": C["text"],
    "text.color": C["text"],
    "xtick.color": C["text"],
    "ytick.color": C["text"],
    "axes.grid": True,
    "grid.color": C["grid"],
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
})


def load(run_ts, tag):
    p = f"{ROOT}/logs/pool6row_{run_ts}/analyze-{tag}.json"
    with open(p) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ts", required=True)
    ap.add_argument("--tags", required=True, help="comma list: probeA,detSplit")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    data = {t: load(a.run_ts, t) for t in tags}
    ts = a.run_ts
    out = a.out or f"{ROOT}/docs/figs/pool-6row-verify-{ts}.png"

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ax = axes[0]
    for tag, col in zip(tags, (C["pool"], C["hl"])):
        d = data[tag]
        ks = sorted(int(k) for k in d["by_streams"])
        ys = [d["by_streams"][str(k)]["seg_pool_sum_p50"] for k in ks]
        ax.plot(ks, ys, "o-", color=col, label=f"{tag}  pool-layers sum")
        if d.get("row_slope", {}).get("slope") is not None:
            sl = d["row_slope"]
            xs = np.array(ks)
            ax.plot(xs, sl["intercept"] + sl["slope"] * xs, "--", color=col, lw=0.9,
                    label=f"{tag}  fit {sl['slope']:.3f} ms/K-stream")
    ax.set_xlabel("concurrent streams K  (6 verify rows each)")
    ax.set_ylabel("pool-layer staircase, p50 per step (ms)")
    ax.set_title("(a) 20 pool layers: cost vs K", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    w = 0.34
    for i, (tag, col) in enumerate(zip(tags, (C["pool"], C["hl"]))):
        d = data[tag]
        ks = sorted(int(k) for k in d["by_streams"])
        per_layer = [d["by_streams"][str(k)]["seg_pool_sum_p50"] / 20.0 for k in ks]
        ax.bar(np.arange(len(ks)) + (i - 0.5) * w, per_layer, w, color=col, label=tag)
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("per pool layer, p50 (ms)")
    ax.set_title("(b) per-layer cost", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    for tag, col in zip(tags, (C["pool"], C["hl"])):
        d = data[tag]
        ks = sorted(int(k) for k in d["by_streams"])
        ms = [d["by_streams"][str(k)]["ms_step_p50"] for k in ks]
        ax.plot(ks, ms, "s-", color=col, label=f"{tag}  client ms/step p50")
        pv = [d["by_streams"][str(k)]["seg_pool_sum_p50"] for k in ks]
        ax.plot(ks, pv, "o--", color=col, alpha=0.6, lw=0.9,
                label=f"{tag}  pool sum p50")
    ax.set_xlabel("concurrent streams K")
    ax.set_ylabel("ms per decode step (p50)")
    ax.set_title("(c) step vs pool share", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"DSV4.1 LK_VERIFY_RESIDENT=1 pool verify: 6 rows x K streams  ({ts})",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
