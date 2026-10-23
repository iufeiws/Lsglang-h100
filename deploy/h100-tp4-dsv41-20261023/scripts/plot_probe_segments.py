#!/usr/bin/env python3
"""Figure: measured decode-step segments (stacked bar + step-total cross-check).

usage: plot_probe_segments.py <agg.json> <out.png>
Morandi palette, English labels only.
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MORANDI = {
    "draft": "#A8B5A2",      # sage
    "verify": "#9DB2BF",     # dusty blue
    "accept": "#C9ADA7",     # dusty rose
    "other": "#D6CFC4",      # warm grey
    "markov": "#B7A9C0",     # mauve
    "bcast": "#C3B091",      # sand
    "gather": "#8FA3A8",     # blue-grey
    "pool": "#B5BFA8",       # grey-green
    "kernel": "#C2A5A5",     # dusty rose dark
    "total": "#7F8C8D",      # slate
}


def fmt(v):
    return "n/a" if v is None or v != v else f"{v:.2f}"


def main():
    agg = json.load(open(sys.argv[1], encoding="utf-8"))
    out = sys.argv[2]
    seg = agg["segments"]
    rec = agg["reconciliation"]
    srv = agg.get("server") or {}

    def mean(key, fallback=None):
        d = seg.get(key)
        if d and d.get("mean") is not None:
            return d["mean"]
        return fallback

    step_total = rec.get("step_total")
    draft = rec.get("draft_propose")
    verify = rec.get("target_verify")
    accept = rec.get("accept_all")
    other = None
    if None not in (step_total, draft, verify, accept):
        other = step_total - (draft + verify + accept)

    fig = plt.figure(figsize=(13.5, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # ------------------------------------------------ left: stacked segments
    parts = [
        ("draft bracket\n(DSpark proposal)", draft, MORANDI["draft"]),
        ("target verify bracket", verify, MORANDI["verify"]),
        ("accept / finalize bracket", accept, MORANDI["accept"]),
        ("unattributed\n(launch + CPU + queue)", other, MORANDI["other"]),
    ]
    xmax = max(step_total or 1.0, 1.0) * 1.28
    left = 0.0
    tiny = 0
    for label, value, color in parts:
        if not value:
            continue
        ax0.barh(0, value, left=left, height=0.42, color=color,
                 edgecolor="#5c5c5c", linewidth=0.6)
        if value > 0.07 * xmax:
            ax0.text(left + value / 2, 0, fmt(value), ha="center", va="center",
                     fontsize=9.5, color="#2f2f2f")
        else:
            tiny += 1
            ax0.text(left + value / 2, 0.24 + 0.17 * (tiny % 2), fmt(value),
                     ha="center", va="bottom", fontsize=9, color="#2f2f2f")
        left += value
    if step_total:
        ax0.axvline(step_total, color="#4a4a4a", linestyle="--", linewidth=1.2)
        ax0.text(step_total * 0.99, 0.80, f"measured step total {fmt(step_total)} ms",
                 ha="right", va="center", fontsize=9, color="#3a3a3a")
    # nested sub-segments drawn as a second row (they are contained in row 0)
    nested = [
        ("markov 5-step block  (in draft graph)", rec.get("draft5_total"),
         MORANDI["markov"]),
        ("20 pool layers  (in target verify graph)", rec.get("pool_layer_sum"),
         MORANDI["pool"]),
        ("rejection-sampling kernel (eager)", rec.get("accept_kernel"),
         MORANDI["kernel"]),
    ]
    y = -0.62
    for label, value, color in nested:
        if value is None:
            continue
        ax0.barh(y, value, height=0.24, color=color, edgecolor="#5c5c5c",
                 linewidth=0.6)
        ax0.text(value, y, f" {fmt(value)}", ha="left", va="center", fontsize=9,
                 color="#333333")
        y -= 0.30
    ax0.set_yticks([0] + [-0.62 - 0.30 * i for i in range(len(nested))])
    ax0.set_yticklabels(["step segments\n(additive)"] +
                        [lbl for lbl, v, _ in nested if v is not None], fontsize=9)
    ax0.set_xlabel("time per decode iteration (ms, window mean)", fontsize=10)
    ax0.set_title("One decode iteration: additive segments and nested spans\n"
                  "(CUDA-event brackets where Python runs per step; device-tick "
                  "spans inside the CUDA graphs)", fontsize=10)
    ax0.set_xlim(0, max(step_total or 1, left) * 1.28)
    ax0.set_ylim(-1.55, 0.95)
    ax0.grid(axis="x", color="#e3e0da", linewidth=0.7)
    ax0.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax0.spines[sp].set_visible(False)

    # ------------------------------------------------ right: cross-checks
    names, vals, cols = [], [], []
    checks = [
        ("measured\nstep total", step_total, MORANDI["total"]),
        ("Σ of the three\neager brackets", (draft + verify + accept)
         if None not in (draft, verify, accept) else None, MORANDI["other"]),
        ("server-derived step\n(accept len ÷ tok/s)", srv.get("derived_step_ms"),
         MORANDI["verify"]),
        ("markov 5-step\n(in graph)", rec.get("draft5_total"), MORANDI["markov"]),
        ("Σ 5× DSPARK_DRAFT\nbroadcast (in graph)", rec.get("draft_bcast_sum"),
         MORANDI["bcast"]),
        ("Σ 5× vocab\nall-gather (in graph)", rec.get("vocab_ag_sum"),
         MORANDI["gather"]),
        ("Σ 20 pool layers\n(in verify graph)", rec.get("pool_layer_sum"),
         MORANDI["pool"]),
    ]
    for label, value, color in checks:
        if value is None:
            continue
        names.append(label)
        vals.append(value)
        cols.append(color)
    ypos = range(len(names))
    ax1.barh(list(ypos), vals, height=0.6, color=cols, edgecolor="#5c5c5c",
             linewidth=0.6)
    for i, v in enumerate(vals):
        ax1.text(v, i, f" {fmt(v)}", va="center", fontsize=9, color="#333333")
    ax1.set_yticks(list(ypos))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("ms", fontsize=10)
    ax1.set_title("Segment totals vs the measured iteration total", fontsize=10)
    ax1.grid(axis="x", color="#e3e0da", linewidth=0.7)
    ax1.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax1.spines[sp].set_visible(False)
    if step_total:
        ax1.axvline(step_total, color="#4a4a4a", linestyle="--", linewidth=1.1)

    handles = [
        Patch(facecolor=MORANDI["draft"], edgecolor="#5c5c5c", label="draft proposal"),
        Patch(facecolor=MORANDI["verify"], edgecolor="#5c5c5c", label="target verify"),
        Patch(facecolor=MORANDI["accept"], edgecolor="#5c5c5c", label="accept / finalize"),
        Patch(facecolor=MORANDI["other"], edgecolor="#5c5c5c", label="unattributed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.015))
    n_win = agg.get("windows_in_window")
    fig.suptitle(f"DSV4.1 segment probe — probe run, single stream "
                 f"({n_win} read-out windows inside the request window)",
                 fontsize=12, y=1.02)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
