#!/usr/bin/env python3
"""run-256-slots 吞吐结果图 (英文标注, 莫兰迪配色). 只画数字, 不作推断."""
import csv
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TS = open("/mnt/SSD01/models/.llm-env/Lsglang/logs/.bench256_ts").read().strip()
BASE = "/mnt/SSD01/models/.llm-env/Lsglang"
CSVP = f"{BASE}/logs/bench-256-{TS}.csv"
SUMP = f"{BASE}/logs/bench-256-{TS}.summary.json"
OUT = f"{BASE}/docs/bench-256-throughput-{TS}.png"

# 莫兰迪
M_SAGE = "#A8B5A0"
M_ROSE = "#C9ADA7"
M_BLUE = "#8E9AAF"
M_SAND = "#D8C3A5"
M_CLAY = "#B5838D"
M_SLATE = "#6D6875"
M_GREY = "#B0AFA8"

rows = [r for r in csv.DictReader(open(CSVP)) if r.get("round")]
sums = json.load(open(SUMP))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


BANDS = [("<=8K", 0, 8192), ("8-32K", 8192, 32768),
         ("32-64K", 32768, 65536), ("64-100K", 65536, 100001)]
# pristine-run-20261023_0324.md 第 14.3 节 (20 驻留层对照)
PRIS = {"<=8K": 66.792, "8-32K": 80.336, "32-64K": 71.078, "64-100K": None}

single = [r for r in rows if r["stage"] == "single-K1-ctx100000" and r["status"] == "ok"]
multi = sorted([s for s in sums if s["mode"] == "multi"], key=lambda s: s["K"])

fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.2))
fig.suptitle(
    "run-256-slots  TP4 / W4A16 / LK_POOL_SLOTS=256 / 40 pooled MoE layers\n"
    f"DeepSeek-V4.1-Flash-1017  @127.0.0.1:18010   pid=1630 (not restarted)   "
    f"2026-10-23 05:12-06:37", fontsize=11.5)

# ---- A: per-band decode
a = ax[0][0]
mine, base = [], []
for name, lo, hi in BANDS:
    b = [f(r["decode_tok_s"]) for r in single
         if lo < (f(r["ctx_after"]) or 0) <= hi and f(r["decode_tok_s"])]
    mine.append(statistics.mean(b) if b else 0)
    base.append(PRIS[name] or 0)
x = range(len(BANDS))
a.bar([i - 0.19 for i in x], mine, 0.38, label="this run: 40 pooled layers",
      color=M_BLUE, edgecolor="#4a5568", linewidth=0.6)
a.bar([i + 0.19 for i in x], base, 0.38, label="pristine: 20 resident layers",
      color=M_SAND, edgecolor="#4a5568", linewidth=0.6)
for i, (m, bb) in enumerate(zip(mine, base)):
    if m:
        a.text(i - 0.19, m + 1, f"{m:.1f}", ha="center", fontsize=8.5)
    if bb:
        a.text(i + 0.19, bb + 1, f"{bb:.1f}", ha="center", fontsize=8.5)
    else:
        a.text(i + 0.19, 1.5, "n/a", ha="center", fontsize=8, color="#888")
a.set_xticks(list(x)); a.set_xticklabels([b[0] for b in BANDS])
a.set_xlabel("context band (ctx_after)"); a.set_ylabel("decode tok/s (mean)")
a.set_title("A. Single-stream decode by context band", fontsize=10.5)
a.legend(fontsize=8.5); a.grid(axis="y", alpha=0.3)

# ---- B: K sweep, aggregate + per-stream + ms/step
b1 = ax[0][1]
Ks = [s["K"] for s in multi]
agg = []
for s in multi:
    sr = [r for r in rows if r["stage"] == s["stage"] and r["status"] == "ok"]
    so = sum(f(r["completion_tokens"]) or 0 for r in sr)
    agg.append(so / s["wall_full_span_s"] if s["wall_full_span_s"] else 0)
per = [s["per_stream_out_tok_s_mean"] for s in multi]
b1.plot(Ks, agg, "-o", color=M_CLAY, label="aggregate output tok/s (wall span)",
        markersize=7, linewidth=1.8)
for k, v in zip(Ks, agg):
    b1.text(k, v + 2, f"{v:.1f}", ha="center", fontsize=8.5, color="#7a4a55")
b1.set_xscale("log", base=2); b1.set_xticks(Ks); b1.set_xticklabels(Ks)
b1.set_xlabel("concurrency K"); b1.set_ylabel("aggregate output tok/s")
b1.set_title("B. Aggregate output throughput vs K", fontsize=10.5)
b1.set_ylim(0, 100)
b1.grid(alpha=0.3)
b2 = b1.twinx()
b2.plot(Ks, per, "-s", color=M_SLATE, label="per-stream output tok/s",
        markersize=6, linewidth=1.8)
b2.set_ylabel("per-stream output tok/s", color=M_SLATE)
b2.tick_params(axis="y", colors=M_SLATE)
for k, v in zip(Ks, per):
    b2.text(k * 1.06, v, f"{v:.2f}", fontsize=8, color="#5a5462")
h1, l1 = b1.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
b1.legend(h1 + h2, l1 + l2, fontsize=8.2, loc="center left")

# ---- C: ms/step client vs server
c = ax[1][0]
cms = [s["per_stream_ms_per_step_mean"] for s in multi]
srv = {1: 12.903, 4: 73.066, 8: 105.499, 16: 166.823}
sms = [srv.get(k) for k in Ks]
c.plot(Ks, cms, "-o", color=M_CLAY, label="client ms/step (mean per stream)",
       markersize=7, linewidth=1.8)
c.plot(Ks, sms, "--D", color=M_BLUE,
       label="server ms/step = 1000/(X_p50/N), TP0 Decode batch",
       markersize=6, linewidth=1.8)
for k, v in zip(Ks, cms):
    c.text(k, v + 5, f"{v:.1f}", ha="center", fontsize=8.5, color="#7a4a55")
for k, v in zip(Ks, sms):
    c.text(k, v - 12, f"{v:.1f}", ha="center", fontsize=8.5, color="#4a5568")
c.set_xscale("log", base=2); c.set_xticks(Ks); c.set_xticklabels(Ks)
c.set_xlabel("concurrency K"); c.set_ylabel("ms/step")
c.set_title("C. ms/step: client vs server", fontsize=10.5)
c.legend(fontsize=8.5); c.grid(alpha=0.3)

# ---- D: decode vs context (single stream)
d = ax[1][1]
d.scatter([f(r["ctx_after"]) for r in single],
          [f(r["decode_tok_s"]) for r in single],
          s=18, color=M_SAGE, edgecolor="#5f6b58", linewidth=0.5, zorder=3,
          label="per-round decode tok/s")
d.axhline(78.36, color=M_ROSE, linestyle="--", linewidth=1.5,
          label="pristine p50 decode 78.36 tok/s")
d.axvline(65536, color=M_GREY, linestyle=":", linewidth=1.2)
d.text(66000, 20, "64K", fontsize=8, color="#777")
d.set_xlabel("ctx_after (tokens)"); d.set_ylabel("decode tok/s")
d.set_title("D. Single-stream decode vs context (82 rounds, 0 fail)", fontsize=10.5)
d.legend(fontsize=8.5); d.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.955])
plt.savefig(OUT, dpi=155)
print("WROTE", OUT)
