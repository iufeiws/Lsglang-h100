#!/usr/bin/env python3
"""run-256-slots 结果汇总 — 只列数字与原始行, 不写结论.

输入:
  --csv    bench-256-<ts>.csv   (逐轮明细)
  --sum    bench-256-<ts>.summary.json
  --log    服务日志
输出 (stdout):
  B. 单流分档 prefill/decode/ms-step + 逐轮原始行
  C. K=1/4/8/16 客户端聚合与每流 + TTFT 分位
  D. 服务端 Decode batch 按 #running-req 分档 -> gen throughput 与 ms/step
  E. [pool] dstep 命中率/oom/stalls 首末与分布
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime

TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
DEC_RE = re.compile(
    r"#running-req: (\d+).*?cuda graph: (\w+), gen throughput \(token/s\): ([\d.]+)")
POOL_KV = re.compile(r"([A-Za-z_]+)=([-\d.]+%?)")
POOL_WIN = re.compile(
    r"dstep=(\d+) win hit=([\d.]+)% nr=([\d.]+)% psl=([\d.]+) link=([\d.]+)GB/s "
    r"gather=([\d.]+)us copyiss=([\d.]+)us que=(\d+)")
POOL_CUM = re.compile(    r"cum hit=([\d.]+)% h=(\d+) m=(\d+) p=(\d+) B=(\d+) resident=(\d+) nr=(\d+) "
    r"vf=(\d+) oom=(\d+) stalls=(\d+) sk=(\d+) stale=(\d+) tag=(\d+) pubbad=(\d+) "
    r"publag=(\d+) cmiss=(\d+) cdup=(\d+)")

BANDS = [("≤8K", 0, 8192), ("8–32K", 8192, 32768),
         ("32–64K", 32768, 65536), ("64–100K", 65536, 100001)]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(vals, q):
    s = sorted(v for v in vals if v is not None and v != "")
    if not s:
        return None
    if len(s) == 1:
        return round(float(s[0]), 3)
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return round(float(s[lo]) + (float(s[hi]) - float(s[lo])) * (pos - lo), 3)


def mean(vals):
    s = [float(v) for v in vals if v is not None and v != ""]
    return round(statistics.mean(s), 3) if s else None


def r3(x):
    return "" if x is None else (round(x, 3) if isinstance(x, float) else x)


def to_epoch(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


# ------------------------------------------------------------------ loaders
def load_rows(path):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("round"):
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sum", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--srvjson", default="",
                    help="把服务端 per-N 统计导出为 JSON（供绘图/三版对照读取）")
    a = ap.parse_args()

    rows = load_rows(a.csv)
    sums = json.load(open(a.sum))
    out = []
    P = out.append

    out.append(f"### 明细来源 `{a.csv}`")
    out.append("")
    out.append(f"- 逐轮明细行数（含失败）: {len(rows)}")
    out.append(f"- 阶段数: {len(sums)}; stage 名: {[s['stage'] for s in sums]}")
    out.append("")

    # ---------------------------------------------------------- B. single
    single = [s for s in sums if s["mode"] == "single"]
    for st in single:
        sr = [r for r in rows if r["stage"] == st["stage"]]
        ok = [r for r in sr if r["status"] == "ok" and f(r["completion_tokens"])]
        bad = [r for r in sr if r["status"] != "ok"]
        P(f"### B. 单流长会话 `{st['stage']}`")
        P("")
        P(f"- 轮数 {len(sr)}（成功 {len(ok)}，失败 {len(bad)}"
          + (f"：{sorted({r['status'] for r in bad})}" if bad else "") + "）")
        P(f"- 停止时上下文 `ctx_after` 最大: **{st['ctx_max']}** token "
          f"(min {st['ctx_min']} / mean {st['ctx_mean']})")
        P(f"- 测量全程墙钟（首轮起点→末轮终点）: **{st['wall_full_span_s']:.1f} s**")
        P(f"- 外部客户端连接数（每轮测前测后取大）最大值: **{st['foreign_sock_max']}**; "
          f"干净轮占比 {st['clean_round_share_pct']}%")
        pre = [f(r["prefill_tok_s"]) for r in ok if f(r["prefill_tok_s"])]
        dec = [f(r["decode_tok_s"]) for r in ok if f(r["decode_tok_s"])]
        mss = [f(r["ms_per_step"]) for r in ok if f(r["ms_per_step"])]
        ttft = [f(r["ttft_s"]) for r in ok if f(r["ttft_s"])]
        dec_s = [f(r["decode_s"]) for r in ok if f(r["decode_s"])]
        e2e = [f(r["e2e_s"]) for r in ok if f(r["e2e_s"])]
        ct = [f(r["completion_tokens"]) for r in ok if f(r["completion_tokens"])]
        P("")
        P("| 指标 | n | mean | p50 | p95 | min | max |")
        P("|---|---|---|---|---|---|---|")
        P(f"| prefill_tok_s = pt/TTFT | {len(pre)} | {mean(pre)} | {pct(pre,50)} | "
          f"{pct(pre,95)} | {pct(pre,0)} | {pct(pre,100)} |")
        P(f"| decode_tok_s = (ct−1)/(e2e−TTFT) | {len(dec)} | {mean(dec)} | {pct(dec,50)} | "
          f"{pct(dec,95)} | {pct(dec,0)} | {pct(dec,100)} |")
        P(f"| ms_per_step = 1000/decode_tok_s | {len(mss)} | {mean(mss)} | {pct(mss,50)} | "
          f"{pct(mss,95)} | {pct(mss,0)} | {pct(mss,100)} |")
        P(f"| TTFT s | {len(ttft)} | {mean(ttft)} | {pct(ttft,50)} | {pct(ttft,95)} | "
          f"{pct(ttft,0)} | {pct(ttft,100)} |")
        P(f"| decode_s (e2e−TTFT) | {len(dec_s)} | {mean(dec_s)} | {pct(dec_s,50)} | "
          f"{pct(dec_s,95)} | {pct(dec_s,0)} | {pct(dec_s,100)} |")
        P(f"| e2e s | {len(e2e)} | {mean(e2e)} | {pct(e2e,50)} | {pct(e2e,95)} | "
          f"{pct(e2e,0)} | {pct(e2e,100)} |")
        P("")
        P("按上下文分档（ctx_after 落档）:")
        P("")
        P("| 上下文区间 | n | avg prompt | avg completion | avg prefill tok/s | "
          "avg decode tok/s | avg ms/step | avg TTFT s | avg ctx |")
        P("|---|---|---|---|---|---|---|---|---|")
        for name, lo, hi in BANDS:
            b = [r for r in ok if lo < (f(r["ctx_after"]) or 0) <= hi]
            if not b:
                P(f"| {name} | 0 | | | | | | | |")
                continue
            P(f"| {name} | {len(b)} | {mean([r['prompt_tokens'] for r in b])} | "
              f"{mean([r['completion_tokens'] for r in b])} | "
              f"{mean([r['prefill_tok_s'] for r in b])} | "
              f"{mean([r['decode_tok_s'] for r in b])} | "
              f"{mean([r['ms_per_step'] for r in b])} | "
              f"{mean([r['ttft_s'] for r in b])} | "
              f"{int(mean([r['ctx_after'] for r in b]) or 0)} |")
        P("")
        P("逐轮原始行:")
        P("")
        P("| # | ctx_after | prompt_tokens | completion_tokens | TTFT s | e2e s | decode_s | "
          "prefill tok/s | decode tok/s | ms/step | foreign_sock | finish | status |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(ok, 1):
            P(f"| {i} | {r['ctx_after']} | {r['prompt_tokens']} | {r['completion_tokens']} | "
              f"{r['ttft_s']} | {r['e2e_s']} | {r['decode_s']} | {r['prefill_tok_s']} | "
              f"{r['decode_tok_s']} | {r['ms_per_step']} | {r['foreign_sock']} | "
              f"{r['finish']} | {r['status']} |")
        P("")

    # ---------------------------------------------------------- C. K table
    multi = [s for s in sums if s["mode"] == "multi"]
    if multi:
        P("### C. 多流 K 档对照（客户端口径）")
        P("")
        P("聚合分母 = **墙钟跨度** = 该档首轮起点 → 末轮终点（含收尾空档），"
          "分子 = 该档全部成功轮次的 Σ token（口径 A）。")
        P("另列 **公共重叠窗口** 口径（口径 B）= [所有流首轮起点的最大值, "
          "所有流末轮终点的最小值]，只统计完全落在窗口内的轮次；"
          "该口径在高 K 下会剔掉较多跨越窗口边界的轮次，仅作对照。")
        P("")
        P("| K | 墙钟跨度 s | 轮数 | Σ input tok | Σ output tok | **聚合 input tok/s (A)** | "
          "**聚合 output tok/s (A)** | 聚合 output tok/s (B,窗口) | 每流平均 output tok/s | "
          "每流 ms/step (mean) | 每流 ms/step (p50) | TTFT p50 s | TTFT p95 s | 失败数 | "
          "平均 ctx | 最大 ctx |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for st in sorted(multi, key=lambda s: s["K"]):
            sr = [r for r in rows if r["stage"] == st["stage"]]
            sok = [r for r in sr if r["status"] == "ok" and f(r["completion_tokens"])]
            sum_in = int(sum(f(r["prompt_tokens"]) or 0 for r in sok))
            sum_out = int(sum(f(r["completion_tokens"]) or 0 for r in sok))
            span = st["wall_full_span_s"]
            P(f"| {st['K']} | {span:.1f} | {len(sok)} | {sum_in} | {sum_out} | "
              f"**{round(sum_in/span, 3) if span else ''}** | "
              f"**{round(sum_out/span, 3) if span else ''}** | "
              f"{r3(st['agg_out_tok_s_commonwindow'])} | "
              f"{r3(st['per_stream_out_tok_s_mean'])} | "
              f"{r3(st['per_stream_ms_per_step_mean'])} | "
              f"{r3(st['per_stream_ms_per_step_p50'])} | "
              f"{r3(st['ttft_s']['p50'])} | {r3(st['ttft_s']['p95'])} | "
              f"{st['failures']} | {st['ctx_mean']} | {st['ctx_max']} |")
        P("")
        P("每流活跃口径与窗口明细:")
        P("")
        P("| K | 每流活跃 output tok/s (Σct/Σe2e) | 每流活跃 × K | 窗口口径 s | 窗口内轮数 / 总轮数 | "
          "外部连接最大值 | 干净轮占比 % |")
        P("|---|---|---|---|---|---|---|")
        for st in sorted(multi, key=lambda s: s["K"]):
            psa = st["per_stream_active_out_tok_s_mean"]
            P(f"| {st['K']} | {r3(psa)} | {round(psa*st['K'], 3) if psa else ''} | "
              f"{st['wall_common_window_s']:.1f} | "
              f"{st['rounds_in_common_window']} / {st['rounds_ok']} | "
              f"{st['foreign_sock_max']} | {st['clean_round_share_pct']} |")
        P("")
        P("各档失败明细（空=0 失败）:")
        P("")
        for st in sorted(multi, key=lambda s: s["K"]):
            if st["failures"]:
                P(f"- K={st['K']}: {st['failure_kinds']} :: {st['failure_examples']}")
        P("")

    # ---------------------------------------------------------- D. server per-step
    t0 = min(to_epoch(r["ts_start"].replace("T", " ")[:19]) for r in rows)
    t1 = max(to_epoch(r["ts_end"].replace("T", " ")[:19]) for r in rows)
    buckets = {}
    n_lines = 0
    with open(a.log, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if "Decode batch" not in ln:
                continue
            m = TS_RE.search(ln)
            if not m:
                continue
            t = to_epoch(m.group(1))
            if t < t0 - 60 or t > t1 + 60:
                continue
            d = DEC_RE.search(ln)
            if not d:
                continue
            n_lines += 1
            N = int(d.group(1))
            am = ACCEPT_RE.search(ln)
            buckets.setdefault(N, []).append(
                {"X": float(d.group(3)), "graph": d.group(2) == "True", "ts": m.group(1),
                 "alen": float(am.group(1)) if am else None,
                 "arate": float(am.group(2)) if am else None})

    P("### D. 服务端 per-step（`Decode batch` 行, 口径说明见下）")
    P("")
    P(f"- 测量窗口: `{datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')}` – "
      f"`{datetime.fromtimestamp(t1).strftime('%Y-%m-%d %H:%M:%S')}`"
      f"（前后各放宽 60 s）")
    P(f"- 窗口内 `Decode batch` 行数: **{n_lines}**（全部来自 TP0）")
    P("- `gen throughput (token/s): X` 为该 decode batch 的生成吞吐；服务端每步耗时 "
      "`ms/step = 1000/(X/N)`，N = `#running-req`。")
    P("")
    P("| #running-req N | 行数 | cuda graph True 占比 % | X p50 | X p95 | "
      "**ms/step p50 = 1000/(X_p50/N)** | ms/step p95 | X min | X max |")
    P("|---|---|---|---|---|---|---|---|---|")
    for N in sorted(buckets):
        xs = [b["X"] for b in buckets[N]]
        g = 100.0 * sum(1 for b in buckets[N] if b["graph"]) / len(xs)
        x50 = pct(xs, 50)
        x95 = pct(xs, 95)
        P(f"| {N} | {len(xs)} | {g:.1f} | {x50} | {x95} | "
          f"**{round(1000.0/(x50/N), 3) if x50 else ''}** | "
          f"{round(1000.0/(x95/N), 3) if x95 else ''} | {pct(xs,0)} | {pct(xs,100)} |")
    if a.srvjson:
        srv = {}
        for N in sorted(buckets):
            xs = [b["X"] for b in buckets[N]]
            x50 = pct(xs, 50)
            x95 = pct(xs, 95)
            srv[N] = {"n": len(xs), "X_p50": x50, "X_p95": x95,
                      "ms_step_p50": round(1000.0 / (x50 / N), 4) if x50 else None,
                      "ms_step_p95": round(1000.0 / (x95 / N), 4) if x95 else None,
                      "graph_true_pct": round(100.0 * sum(1 for b in buckets[N] if b["graph"])
                                              / len(xs), 2)}
        json.dump({"serve_log": a.log, "window": [t0, t1], "by_N": srv},
                  open(a.srvjson, "w"), ensure_ascii=False, indent=2)
        print(f"[serverN json] {a.srvjson}")
    P("")
    P("窗口内 `Decode batch` 原始行抽样（每个 N 档首/末各 1 行）:")
    P("")
    P("```")
    for N in sorted(buckets):
        P(buckets[N][0]["ts"] + f"  (N={N}, 首行) X={buckets[N][0]['X']}")
        P(buckets[N][-1]["ts"] + f"  (N={N}, 末行) X={buckets[N][-1]['X']}")
    P("```")
    P("")

    # 原始行全文（每 N 档前 3 行）
    P("窗口内 `Decode batch` 全文（每 N 档前 3 行）:")
    P("")
    P("```")
    with open(a.log, encoding="utf-8", errors="replace") as fh:
        want = {}
        for ln in fh:
            if "Decode batch" not in ln:
                continue
            m = TS_RE.search(ln)
            dm = DEC_RE.search(ln)
            if not (m and dm):
                continue
            t = to_epoch(m.group(1))
            if t < t0 - 60 or t > t1 + 60:
                continue
            N = int(dm.group(1))
            if want.get(N, 0) < 3:
                P(ln.rstrip())
                want[N] = want.get(N, 0) + 1
    P("```")
    P("")

    # ---------------------------------------------------------- E. pool
    allp = []
    with open(a.log, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("[pool] dstep=") or " [pool] dstep=" in ln:
                allp.append(ln.rstrip())
    P("### E. 专家池 `[pool] dstep=`")
    P("")
    P(f"- 服务日志中 `[pool] dstep=` 行数（含启动以来全部）: **{len(allp)}**")
    if allp:
        P(f"- `dstep` 取值范围: {POOL_WIN.search(allp[0]).group(1)} … "
          f"{POOL_WIN.search(allp[-1]).group(1)}")
        P(f"- 不同 `dstep` 值个数: {len({POOL_WIN.search(x).group(1) for x in allp if POOL_WIN.search(x)})} "
          f"(每 dstep 约 {len(allp)/max(1,len({POOL_WIN.search(x).group(1) for x in allp if POOL_WIN.search(x)})):.2f} 行 = 每 rank 1 行)")
        P("")
        P("首个 `[pool] dstep=` 行原文:")
        P("")
        P("```")
        P(allp[0])
        P("```")
        P("")
        P("末个 `[pool] dstep=` 行原文:")
        P("")
        P("```")
        P(allp[-1])
        P("```")
        P("")
        fields = {}
        for x in allp:
            wm = POOL_WIN.search(x)
            cm = POOL_CUM.search(x)
            if wm:
                for k, i in [("dstep", 1), ("win_hit", 2), ("nr", 3), ("psl", 4),
                             ("link", 5), ("gather", 6), ("copyiss", 7), ("que", 8)]:
                    fields.setdefault(k, []).append(float(wm.group(i)))
            if cm:
                for k, i in [("cum_hit", 1), ("h", 2), ("m", 3), ("p", 4), ("B", 5),
                             ("resident", 6), ("nr2", 7), ("vf", 8), ("oom", 9),
                             ("stalls", 10), ("sk", 11), ("stale", 12), ("tag", 13),
                             ("pubbad", 14), ("publag", 15), ("cmiss", 16), ("cdup", 17)]:
                    fields.setdefault(k, []).append(float(cm.group(i)))
        P("| 字段 | n | min | p50 | mean | p95 | max | 去重值个数 |")
        P("|---|---|---|---|---|---|---|---|")
        for k in ["win_hit", "cum_hit", "psl", "nr", "link", "gather", "copyiss", "que",
                  "resident", "oom", "stalls", "vf", "tag", "pubbad", "cmiss", "cdup",
                  "stale", "sk", "h", "m", "p", "publag", "B"]:
            v = fields.get(k)
            if not v:
                continue
            P(f"| `{k}` | {len(v)} | {min(v)} | {pct(v,50)} | {mean(v)} | {pct(v,95)} | "
              f"{max(v)} | {len(set(v))} |")
        P("")
        P("`oom` / `stalls` 非零行数:")
        P("")
        for k in ["oom", "stalls", "vf", "pubbad", "cmiss", "cdup", "stale", "tag"]:
            nz = [x for x in allp
                  if (lambda m: m and float(m.group({"oom": 9, "stalls": 10, "vf": 8,
                                                      "pubbad": 14, "cmiss": 16, "cdup": 17,
                                                      "stale": 12, "tag": 13}[k])) != 0)
                      (POOL_CUM.search(x))]
            P(f"- `{k}` 非零行数: **{len(nz)}**"
              + (f"；示例: `{nz[0][:220]}`" if nz else ""))
        P("")
        P("命中率逐 `dstep` 窗口序列（`win hit` / `cum hit`，四 rank 中取该 dstep 第一行）:")
        P("")
        seen = {}
        for x in allp:
            m = POOL_WIN.search(x)
            if m and m.group(1) not in seen:
                seen[m.group(1)] = x
        P("| dstep | win hit % | cum hit % | psl | link GB/s | copyiss us | que | oom | stalls |")
        P("|---|---|---|---|---|---|---|---|---|")
        for d, x in seen.items():
            wm, cm = POOL_WIN.search(x), POOL_CUM.search(x)
            P(f"| {d} | {wm.group(2)} | {cm.group(1) if cm else ''} | {wm.group(4)} | "
              f"{wm.group(5)} | {wm.group(7)} | {wm.group(8)} | "
              f"{cm.group(9) if cm else ''} | {cm.group(10) if cm else ''} |")
        P("")

    txt = "\n".join(out)
    print(txt)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt + "\n")


if __name__ == "__main__":
    main()
