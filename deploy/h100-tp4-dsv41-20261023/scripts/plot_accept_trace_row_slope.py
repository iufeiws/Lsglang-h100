#!/usr/bin/env python3
"""Figure for the accept-trace + pool row-slope report.

usage: plot_accept_trace_row_slope.py --accept-json <a.json> --slope-json <b.json>
                                      --out docs/figs/probe-accept-row-slope-<ts>.png

Two panels, English labels, Morandi palette:
  A: first-mismatch slot distribution (bars, per temperature) + per-slot
     conditional match rate (lines, right axis)
  B: pool-path span and step total (p50 / p95) vs verify rows per step, with
     the least-squares fits annotated.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_DARK = "#4A4E52"
C_T0 = "#8A9A8B"   # morandi sage   -> temp 0.0
C_T7 = "#B08A78"   # morandi clay   -> temp 0.7
C_POOL = "#7E8CA0"  # morandi blue  -> pool path
C_STEP = "#B9A489"  # morandi sand  -> step total
C_GRID = "#D8D5CE"


def panel_a(ax, groups):
    labels = ["1", "2", "3", "4", "5", "none\n(5/5)"]
    keys = ["1", "2", "3", "4", "5", "none(5/5)"]
    colors = [C_T0, C_T7]
    n = len(groups)
    w = 0.36 / max(1, n / 2)
    xs = np.arange(len(keys))
    for gi, g in enumerate(groups):
        hist = g["first_mismatch_hist"]
        tot = float(sum(hist.values())) or 1.0
        vals = [100.0 * hist[k] / tot for k in keys]
        off = (gi - (n - 1) / 2.0) * w
        ax.bar(xs + off, vals, width=w, color=colors[gi % len(colors)],
               edgecolor="white", linewidth=0.6,
               label=f"temp={g['label'].split('=')[-1]}  "
                     f"(n={g['n_records']}, accept len p50={g['commit_p50']})")
        for x, v in zip(xs + off, vals):
            if v > 0.4:
                ax.text(x, v + 0.6, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7, color=C_DARK)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("slot of first greedy mismatch  (1 = first draft slot)")
    ax.set_ylabel("% of recorded verify steps (requests)")
    ax.set_title("A. Where the draft first disagrees with the target argmax",
                 fontsize=10, color=C_DARK, loc="left")
    ax.grid(axis="y", color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.32)

    ax2 = ax.twinx()
    for gi, g in enumerate(groups):
        sc = g["slot_conditional"]
        ys = [100.0 * sc[str(j)]["match_rate"] if sc[str(j)]["match_rate"] is not None
              else np.nan for j in range(1, 6)]
        ax2.plot(range(1, 6), ys, marker="o", markersize=4, linewidth=1.2,
                 linestyle="--" if gi else "-",
                 color=colors[gi % len(colors)], alpha=0.95,
                 label="per-slot match rate (right axis)")
    ax2.set_ylabel("per-slot match rate given intact prefix  (%)", fontsize=8)
    ax2.set_ylim(0, 105)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h1 + h2, l1 + l2):
        if ll not in seen:
            seen[ll] = hh
    ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper right",
              framealpha=0.9)
    for s in ("top",):
        ax.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)


def panel_b(ax, res):
    stages = sorted(res["stages"], key=lambda r: r["rows_per_step"])
    x = np.array([r["rows_per_step"] for r in stages], dtype=float)
    pool_p50 = np.array([r["pool"]["p50"] for r in stages], dtype=float)
    pool_p95 = np.array([r["pool"]["p95"] for r in stages], dtype=float)
    step_p50 = np.array([r["step_total"]["p50"] for r in stages], dtype=float)
    step_p95 = np.array([r["step_total"]["p95"] for r in stages], dtype=float)

    ax.plot(x, pool_p50, "-o", color=C_POOL, markersize=6, linewidth=1.6,
            label="pool path $\\Sigma$  p50")
    ax.plot(x, pool_p95, "--o", color=C_POOL, markersize=5, linewidth=1.0,
            alpha=0.55, markerfacecolor="white", label="pool path $\\Sigma$  p95")
    ax.plot(x, step_p50, "-s", color=C_STEP, markersize=6, linewidth=1.6,
            label="step total  p50")
    ax.plot(x, step_p95, "--s", color=C_STEP, markersize=5, linewidth=1.0,
            alpha=0.55, markerfacecolor="white", label="step total  p95")

    fits = res.get("fits", {})
    for name, col, lab in (("pool_p50", C_POOL, "pool"), ("step_total_p50", C_STEP, "step")):
        f = fits.get(name)
        if not f or f.get("slope_ms_per_row") != f.get("slope_ms_per_row"):
            continue
        xs = np.linspace(0, max(x) * 1.08, 50)
        ax.plot(xs, f["intercept_ms"] + f["slope_ms_per_row"] * xs, ":",
                color=col, linewidth=1.1, alpha=0.9)
    txt = []
    for name, lab in (("pool_p50", "pool $\\Sigma$"), ("step_total_p50", "step total")):
        f = fits.get(name)
        if not f or f.get("slope_ms_per_row") != f.get("slope_ms_per_row"):
            continue
        txt.append(f"{lab}: y = {f['intercept_ms']:.2f} + "
                   f"{f['slope_ms_per_row']:.4f}·rows   $R^2$={f['r2']:.4f}")
    if txt:
        ax.text(0.02, 0.97, "\n".join(txt), transform=ax.transAxes, fontsize=7.5,
                va="top", ha="left", color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_GRID, alpha=0.9))
    for i, r in enumerate(stages):
        ha = "center" if 0 < i < len(stages) - 1 else ("left" if i == 0 else "right")
        ax.annotate(f"{r['stage']}\n{r['streams']} stream(s)",
                    (r["rows_per_step"], r["pool"]["p50"]),
                    textcoords="offset points",
                    xytext=(0, -28) if 0 < i < len(stages) - 1 else (6 if i == 0 else -6, -28),
                    ha=ha, fontsize=7, color=C_DARK)
    ax.set_xlabel("verify rows per decode step  (streams x 6)")
    ax.set_ylabel("time per decode step  (ms)")
    ax.set_title("B. Pool path span and step total vs verify rows per step",
                 fontsize=10, color=C_DARK, loc="left")
    ax.grid(color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(sorted(set(x)))
    ax.legend(fontsize=7.5, loc="center right", framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-json", required=True)
    ap.add_argument("--slope-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()

    acc = json.load(open(args.accept_json))
    slope = json.load(open(args.slope_json))
    groups = [g for g in acc["groups"] if g.get("n_records")]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    panel_a(axes[0], groups)
    panel_b(axes[1], slope)
    fig.suptitle(
        "DSV4.1 dspark verify step: draft-vs-target mismatch and pool-path row slope "
        "(gated probes, one process)",
        fontsize=10.5, color=C_DARK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(args.out, dpi=args.dpi)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
