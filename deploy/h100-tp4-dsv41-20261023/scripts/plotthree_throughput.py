#!/usr/bin/env python3
"""三版对照图 (英文标注, 莫兰迪配色). 只画数字, 不作推断.

v1 pristine : LK_GPU_RESIDENT_MOE_LAYERS=0-19 + LK_POOL_SLOTS=128, 2.38M KV, 无 expseg
v2 full-pool: LK_GPU_RESIDENT_MOE_LAYERS=""   + LK_POOL_SLOTS=256, 4M KV,   无 expseg
v3 expseg   : LK_GPU_RESIDENT_MOE_LAYERS=0-19 + LK_POOL_SLOTS=128, 4M KV,   expandable_segments:True

用法: python3 plotthree_throughput.py --ts <TS> --prefix bench-expseg --out <png>
"""
import argparse
import csv
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/mnt/SSD01/models/.llm-env/Lsglang/logs"

M_SAND, M_BLUE, M_SAGE = "#D8C3A5", "#8E9AAF", "#A8B5A0"
M_CLAY, M_SLATE, M_GREY = "#B5838D", "#6D6875", "#B0AFA8"

BANDS = [("<=8K", 0, 8192), ("8-32K", 8192, 32768),
         ("32-64K", 32768, 65536), ("64-100K", 65536, 100001)]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path, stage_sel=None):
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if not r.get("round") or r.get("status") != "ok":
                continue
            if stage_sel and r.get("stage") != stage_sel:
                continue
            ctx = f(r.get("ctx_after"))
            if not ctx:
                continue
            out.append({"ctx": ctx, "ct": f(r.get("completion_tokens")),
                        "ttft": f(r.get("ttft_s")), "dec": f(r.get("decode_tok_s")),
                        "pre": f(r.get("prefill_tok_s")),
                        "e2e": f(r.get("e2e_s")) or f(r.get("total_s"))})
    return out


def band_means(rows, field):
    return [statistics.mean([r[field] for r in rows if lo < r["ctx"] <= hi and r[field]])
            if [r[field] for r in rows if lo < r["ctx"] <= hi and r[field]] else None
            for _, lo, hi in BANDS]


def band_n(rows):
    return [sum(1 for r in rows if lo < r["ctx"] <= hi) for _, lo, hi in BANDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True)
    ap.add_argument("--prefix", default="bench-expseg")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # 单流三版
    V = [
        ("v1 20res + 128slot, 2.38M KV",
         f"{BASE}/longsession-20261023_0406.csv", None, M_SAND),
        ("v2 40 pooled x 256slot, 4M KV",
         f"{BASE}/bench-256-20261023_0514.csv", "single-K1-ctx100000", M_BLUE),
        ("v3 20res(0-19) + 128slot + expseg, 4M KV",
         f"{BASE}/{a.prefix}-{a.ts}.csv", "single-K1-ctx100000", M_SAGE),
    ]
    data = []
    for lbl, path, sel, col in V:
        try:
            data.append((lbl, load(path, sel), col))
        except FileNotFoundError:
            print("MISSING", path)
    n = max(1, len(data))

    # K 扫描两版
    S = [
        ("v2 40 pooled x 256slot",
         f"{BASE}/bench-256-20261023_0514.csv",
         f"{BASE}/bench-256-20261023_0514.summary.json",
         f"{BASE}/bench-256-20261023_0514.serverN.json", M_BLUE),
        ("v3 20res + 128slot + expseg",
         f"{BASE}/{a.prefix}-{a.ts}.csv",
         f"{BASE}/{a.prefix}-{a.ts}.summary.json",
         f"{BASE}/{a.prefix}-{a.ts}.serverN.json", M_SAGE),
    ]
    sweeps = []
    for name, csvp, sump, srvp, col in S:
        try:
            ms = sorted([x for x in json.load(open(sump)) if x["mode"] == "multi"],
                        key=lambda x: x["K"])
        except FileNotFoundError:
            print("MISSING", sump); continue
        agg, per = [], []
        for x in ms:
            rows = load(csvp, x["stage"])
            so = sum(r["ct"] or 0 for r in rows)
            agg.append(so / x["wall_full_span_s"] if x["wall_full_span_s"] else 0)
            per.append(x["per_stream_out_tok_s_mean"])
        try:
            srv = {int(k): v["ms_step_p50"] for k, v in json.load(open(srvp))["by_N"].items()}
        except Exception:
            srv = {}
        sweeps.append({"name": name, "col": col, "K": [x["K"] for x in ms],
                       "agg": agg, "per": per, "srv": srv,
                       "cms": [x["per_stream_ms_per_step_mean"] for x in ms],
                       "ttft50": [x["ttft_s"]["p50"] for x in ms],
                       "ttft95": [x["ttft_s"]["p95"] for x in ms]})

    fig, ax = plt.subplots(2, 2, figsize=(14.6, 9.8))
    fig.suptitle(
        "Three-config comparison — DeepSeek-V4.1-Flash-1017, TP4, W4A16, ctx 1M, 127.0.0.1:18010\n"
        "same harness (max_tokens=1500, temp=0.7, one 1000-char-novel session + repeated "
        "'continue'), measurer never restarted the server", fontsize=11)

    # ---- A: per-band decode
    ax0 = ax[0][0]
    w = 0.8 / n
    xs = list(range(len(BANDS)))
    for i, (lbl, rows, col) in enumerate(data):
        m = band_means(rows, "dec")
        pos = [x - 0.4 + w * (i + 0.5) for x in xs]
        ax0.bar(pos, [v or 0 for v in m], w * 0.92, label=lbl, color=col,
                edgecolor="#4a5568", linewidth=0.6)
        for p, v in zip(pos, m):
            if v:
                ax0.text(p, v + 1.2, f"{v:.1f}", ha="center", fontsize=7.8)
    ax0.set_xticks(xs)
    ax0.set_xticklabels([f"{b[0]}\nn={band_n(data[0][1])[i]}/{band_n(data[1][1])[i]}/"
                         f"{band_n(data[-1][1])[i]}" for i, b in enumerate(BANDS)], fontsize=8)
    ax0.set_xlabel("context band (ctx_after)"); ax0.set_ylabel("decode tok/s (mean)")
    ax0.set_title("A. Single-stream decode by context band  (v1/v2/v3 n)", fontsize=10.5)
    ax0.legend(fontsize=7.5); ax0.grid(axis="y", alpha=0.3)

    # ---- B: per-band prefill (log)
    ax1 = ax[0][1]
    for i, (lbl, rows, col) in enumerate(data):
        m = band_means(rows, "pre")
        pos = [x - 0.4 + w * (i + 0.5) for x in xs]
        # 缺失档位不画柱(避免 log 轴上出现误导性的贴底柱)
        ax1.bar(pos, [v if v else 0.0 for v in m], w * 0.92, label=lbl, color=col,
                edgecolor="#4a5568", linewidth=0.6)
        for p, v in zip(pos, m):
            if v:
                ax1.text(p, v * 1.15, f"{v:.0f}", ha="center", fontsize=7.2, rotation=90)
            else:
                ax1.text(p, 1.6, "n/a", ha="center", fontsize=7.0, color="#888")
    ax1.set_yscale("log")
    ax1.set_ylim(1, 2.2e5)
    ax1.set_xticks(xs); ax1.set_xticklabels([b[0] for b in BANDS], fontsize=8.5)
    ax1.set_xlabel("context band (ctx_after)"); ax1.set_ylabel("prefill tok/s (mean, log)")
    ax1.set_title("B. Single-stream prefill by context band", fontsize=10.5)
    ax1.legend(fontsize=7.2, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, which="both")

    # ---- C: K sweep
    ax2 = ax[1][0]
    for s in sweeps:
        ka = [(k, v) for k, v in zip(s["K"], s["agg"]) if v is not None]
        kp = [(k, v) for k, v in zip(s["K"], s["per"]) if v is not None]
        ax2.plot([k for k, _ in ka], [v for _, v in ka], "-o", color=s["col"],
                 markersize=6, linewidth=1.9, label=f"aggregate out tok/s — {s['name']}")
        ax2.plot([k for k, _ in kp], [v for _, v in kp], "--s", color=s["col"],
                 markersize=5, linewidth=1.4, alpha=0.8,
                 label=f"per-stream out tok/s — {s['name']}")
        for k, v in ka:
            ax2.text(k * 1.02, v + 2.0, f"{v:.1f}", fontsize=7.6, color="#4a5568")
        for k, v in kp:
            ax2.text(k * 1.02, v - 5.5, f"{v:.2f}", fontsize=7.3, color=s["col"])
    ax2.set_xscale("log", base=2); ax2.set_xticks([1, 4, 8, 16]); ax2.set_xticklabels([1, 4, 8, 16])
    ax2.set_xlabel("concurrency K"); ax2.set_ylabel("output tok/s")
    ax2.set_title("C. Output throughput vs K  (aggregate = sum(ct) / wall span)", fontsize=10.5)
    ax2.legend(fontsize=7.2); ax2.grid(alpha=0.3)

    # ---- D: ms/step client vs server
    ax3 = ax[1][1]
    for s in sweeps:
        kc = [(k, v) for k, v in zip(s["K"], s["cms"]) if v is not None]
        ax3.plot([k for k, _ in kc], [v for _, v in kc], "-o", color=s["col"],
                 markersize=6, linewidth=1.9, label=f"client ms/step — {s['name']}")
        ks = [k for k in s["K"] if k in s["srv"]]
        ax3.plot(ks, [s["srv"][k] for k in ks], "--D", color=s["col"], markersize=5,
                 linewidth=1.4, alpha=0.85,
                 label=f"server ms/step = 1000/(X_p50/N) — {s['name'].split()[0]}")
        for k in ks:
            ax3.text(k * 1.03, s["srv"][k] - 11, f"{s['srv'][k]:.1f}", fontsize=7.2,
                     color="#4a5568")
    ax3.set_xscale("log", base=2); ax3.set_xticks([1, 4, 8, 16]); ax3.set_xticklabels([1, 4, 8, 16])
    ax3.set_xlabel("concurrency K"); ax3.set_ylabel("ms/step")
    ax3.set_title("D. ms/step: client vs server", fontsize=10.5)
    ax3.legend(fontsize=7.2); ax3.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(a.out, dpi=150)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
