#!/usr/bin/env python3
"""把 longsession CSV 汇总成 prefill / decode 分离的数字表（Markdown）。

口径（分母写清）：
  A. prefill 每请求 : prompt_tokens / TTFT
     prefill 聚合   : Σ prompt_tokens / 墙钟（= 全服务聚合，所有流合并）
  B. decode  每请求 : (completion_tokens − 1) / (e2e − TTFT)
     decode  聚合   : Σ completion_tokens / 墙钟
  C. TPOT/ITL(ms)   : (e2e − TTFT) * 1000 / (completion_tokens − 1)   [decode tok/s ≈ 1000 / TPOT_ms]
长会话按上下文区间分段：≤8K / 8–32K / 32–64K / 64–100K
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(vals, q):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    lo = int(pos); hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def r(x, n=3):
    return "" if x is None else round(x, n)


BANDS = [("≤8K", 0, 8192), ("8–32K", 8192, 32768),
         ("32–64K", 32768, 65536), ("64–100K", 65536, 100001)]


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("round"):
                continue
            rows.append(row)
    return rows


def summarize(rows, label, out):
    ok = [x for x in rows if x.get("status") == "ok" and f(x.get("completion_tokens"))]
    bad = [x for x in rows if x.get("status") != "ok"]
    if not rows:
        return
    ends, starts = [], []
    for x in ok:
        try:
            te = datetime.strptime(x["ts"], "%Y-%m-%dT%H:%M:%S%z").timestamp()
            ends.append(te)
            starts.append(te - float(x["total_s"]))
        except Exception:
            pass
    wall = (max(ends) - min(starts)) if (ends and starts) else None

    pt = [f(x["prompt_tokens"]) for x in ok]
    ct = [f(x["completion_tokens"]) for x in ok]
    ttft = [f(x["ttft_s"]) for x in ok]
    dec = [f(x["decode_s"]) for x in ok]
    pre_r = [p / t for p, t in zip(pt, ttft) if p and t]
    dec_r = [(c - 1) / d for c, d in zip(ct, dec) if c and c > 1 and d and d > 0]
    tpot = [1000.0 / v for v in dec_r if v]

    out.append(f"\n#### {label}")
    out.append(f"- 轮数 {len(rows)}（成功 {len(ok)}，失败 {len(bad)}"
               + (f"：{sorted({x['status'] for x in bad})}" if bad else "") + "）")
    if wall:
        out.append(f"- 墙钟（首轮开始→末轮结束）: **{wall:.1f} s**")
    out.append(f"- 每请求 prefill tok/s      : "
               f"n={len(pre_r)} mean={r(statistics.mean(pre_r)) if pre_r else ''} "
               f"p50={r(pct(pre_r,50))} p95={r(pct(pre_r,95))}")
    out.append(f"- 每请求 decode  tok/s      : "
               f"n={len(dec_r)} mean={r(statistics.mean(dec_r)) if dec_r else ''} "
               f"p50={r(pct(dec_r,50))} p95={r(pct(dec_r,95))}")
    if wall:
        out.append(f"- 聚合 prefill tok/s (Σpt/墙钟, 全服务) : "
                   f"**{r(sum(v for v in pt if v)/wall)}**  (Σpt={int(sum(v for v in pt if v))})")
        out.append(f"- 聚合 decode  tok/s (Σct/墙钟, 全服务) : "
                   f"**{r(sum(v for v in ct if v)/wall)}**  (Σct={int(sum(v for v in ct if v))})")
        # active-time aggregate: per-stream serial execution time, so gate idle does not deflate it
        by_s = {}
        for x in ok:
            by_s.setdefault(x.get("stream"), []).append(x)
        act_dec, act_pre = [], []
        for s, xs in by_s.items():
            tot = sum(f(x["total_s"]) or 0 for x in xs)
            if tot <= 0:
                continue
            act_dec.append(sum(f(x["completion_tokens"]) or 0 for x in xs) / tot)
            act_pre.append(sum(f(x["prompt_tokens"]) or 0 for x in xs) / tot)
        if act_dec:
            out.append(f"- **每流平均活跃 decode tok/s**（Σct_s/Σe2e_s，n_stream={len(act_dec)}）: "
                       f"**{r(statistics.mean(act_dec))}**")
            out.append(f"- 每流平均活跃 prefill tok/s（Σpt_s/Σe2e_s）: {r(statistics.mean(act_pre))}")
            nstr = len(act_dec)
            out.append(f"- 全服务活跃 decode tok/s（ΣΣct/Σ(Σe2e)/n_stream，{nstr} 流连续跑）: "
                       f"**{r(statistics.mean(act_dec)*nstr)}**")
            idle = wall - sum((f(x['total_s']) or 0) for x in ok) / nstr
            out.append(f"- 客户端空档（墙钟 − 每流 Σe2e）= **{idle:.1f} s**"
                       f"（单流模式的 gate/间隔空档；墙钟口径因此被低估）")
    out.append(f"- TTFT s     : p50={r(pct(ttft,50))} p95={r(pct(ttft,95))} "
               f"p99={r(pct(ttft,99))} min={r(min(v for v in ttft if v is not None)) if ttft else ''} "
               f"max={r(max(v for v in ttft if v is not None)) if ttft else ''}")
    out.append(f"- TPOT/ITL ms: p50={r(pct(tpot,50))} p95={r(pct(tpot,95))} "
               f"(decode tok/s ≈ 1000/TPOT_ms)")

    out.append("")
    out.append("| 上下文区间 | n | avg prefill tok/s | avg decode tok/s | avg TTFT s | avg ctx |")
    out.append("|---|---|---|---|---|---|")
    for name, lo, hi in BANDS:
        b = [x for x in ok if (f(x.get("ctx_after")) or 0) > lo
             and (f(x.get("ctx_after")) or 0) <= hi]
        if not b:
            out.append(f"| {name} | 0 | | | | |")
            continue
        bpre = [p / t for p, t in ((f(x["prompt_tokens"]), f(x["ttft_s"])) for x in b) if p and t]
        bdec = [(c - 1) / d for c, d in ((f(x["completion_tokens"]), f(x["decode_s"])) for x in b)
                if c and c > 1 and d and d > 0]
        bttft = [f(x["ttft_s"]) for x in b if f(x["ttft_s"])]
        bctx = [f(x["ctx_after"]) for x in b if f(x["ctx_after"])]
        out.append(f"| {name} | {len(b)} | {r(statistics.mean(bpre)) if bpre else ''} | "
                   f"{r(statistics.mean(bdec)) if bdec else ''} | "
                   f"{r(statistics.mean(bttft)) if bttft else ''} | "
                   f"{int(statistics.mean(bctx)) if bctx else ''} |")

    out.append("")
    out.append("逐轮原始行：")
    out.append("")
    out.append("| # | ctx_after | prompt_tokens | completion_tokens | TTFT s | e2e s | prefill tok/s | decode tok/s | TPOT ms | win_hit% | cum_hit% |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, x in enumerate(ok, 1):
        p, c = f(x["prompt_tokens"]), f(x["completion_tokens"])
        t, d = f(x["ttft_s"]), f(x["decode_s"])
        pre = p / t if p and t else None
        de = (c - 1) / d if c and c > 1 and d and d > 0 else None
        out.append(f"| {i} | {x.get('ctx_after','')} | {x.get('prompt_tokens','')} | "
                   f"{x.get('completion_tokens','')} | {r(t)} | {x.get('total_s','')} | "
                   f"{r(pre)} | {r(de)} | {r(1000.0/de) if de else ''} | "
                   f"{x.get('win_hit_pct','')} | {x.get('cum_hit_pct','')} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = load(a.csv)
    tags = []
    for x in rows:
        if x["tag"] not in tags:
            tags.append(x["tag"])
    out = [f"### 明细来源 `{a.csv}`", f"（共 {len(rows)} 轮；tag: {tags}）"]
    for t in tags:
        summarize([x for x in rows if x["tag"] == t], f"配置 `{t}`", out)
    txt = "\n".join(out)
    print(txt)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write("\n" + txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
