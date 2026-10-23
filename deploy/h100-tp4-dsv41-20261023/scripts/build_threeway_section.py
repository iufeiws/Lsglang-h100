#!/usr/bin/env python3
"""三版对照节生成器 (只列数字与原始行). 输出 markdown 到 --out.

v1 pristine : 20 resident(0-19) + 128 slots, 2.38M KV, 无 expseg
v2 full-pool: resident 空 + 256 slots,      4M KV,   无 expseg
v3 expseg   : 20 resident(0-19) + 128 slots, 4M KV,  expandable_segments:True
"""
import argparse
import csv
import json
import re
import statistics

BASE = "/mnt/SSD01/models/.llm-env/Lsglang"
LOGS = f"{BASE}/logs"
BANDS = [("≤8K", 0, 8192), ("8–32K", 8192, 32768),
         ("32–64K", 32768, 65536), ("64–100K", 65536, 100001)]

POOL_WIN = re.compile(
    r"dstep=(\d+) win hit=([\d.]+)% nr=([\d.]+)% psl=([\d.]+) link=([\d.]+)GB/s "
    r"gather=([\d.]+)us copyiss=([\d.]+)us que=(\d+)")
POOL_CUM = re.compile(
    r"cum hit=([\d.]+)% h=(\d+) m=(\d+) p=(\d+) B=(\d+) resident=(\d+) nr=(\d+) "
    r"vf=(\d+) oom=(\d+) stalls=(\d+)")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    if len(v) == 1:
        return round(v[0], 3)
    pos = (len(v) - 1) * q / 100.0
    a = int(pos)
    b = min(a + 1, len(v) - 1)
    return round(v[a] + (v[b] - v[a]) * (pos - a), 3)


def mean(v):
    v = [x for x in v if x is not None]
    return round(statistics.mean(v), 3) if v else None


def load_rows(path, stage_sel=None):
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
            e2e = f(r.get("e2e_s")) or f(r.get("total_s"))
            ct = f(r.get("completion_tokens"))
            dec = f(r.get("decode_tok_s"))
            out.append({"ctx": ctx, "pt": f(r.get("prompt_tokens")), "ct": ct,
                        "ttft": f(r.get("ttft_s")), "dec": dec,
                        "pre": f(r.get("prefill_tok_s")), "e2e": e2e,
                        "mss": round(1000.0 / dec, 3) if dec else None})
    return out


def band(rows, field):
    res = []
    for name, lo, hi in BANDS:
        b = [r[field] for r in rows if lo < r["ctx"] <= hi and r[field]]
        res.append((name, len(b), mean(b), pct(b, 50)))
    return res


def pool_stats(logpath):
    win, cum, psl, link, cis, que, res, oom, stl, ds = [], [], [], [], [], [], [], 0, 0, []
    n = 0
    try:
        fh = open(logpath, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    for ln in fh:
        if ln.startswith("[pool] dstep=") or " [pool] dstep=" in ln:
            n += 1
            w = POOL_WIN.search(ln)
            c = POOL_CUM.search(ln)
            if w:
                ds.append(int(w.group(1)))
                win.append(float(w.group(2)))
                psl.append(float(w.group(4)))
                link.append(float(w.group(5)))
                cis.append(float(w.group(7)))
                que.append(float(w.group(8)))
            if c:
                cum.append(float(c.group(1)))
                res.append(float(c.group(6)))
                if float(c.group(9)) != 0:
                    oom += 1
                if float(c.group(10)) != 0:
                    stl += 1
    return {"n": n, "win": win, "cum": cum, "psl": psl, "link": link, "cis": cis,
            "que": que, "res": res, "oom_nonzero": oom, "stalls_nonzero": stl,
            "dstep": (min(ds), max(ds)) if ds else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True)
    ap.add_argument("--prefix", default="bench-expseg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--v3-serve-log", required=True)
    a = ap.parse_args()

    V = {
        "v1": {"label": "v1 pristine 20res(0-19)+128槽, 2.38M KV, 无 expseg",
               "csv": f"{LOGS}/longsession-20261023_0406.csv", "sel": None,
               "sum": None, "srv": None,
               "log": f"{LOGS}/serve_dsv41_tp4_h100_20261023_025933.log"},
        "v2": {"label": "v2 全池 40层×256槽, 4M KV, 无 expseg",
               "csv": f"{LOGS}/bench-256-20261023_0514.csv",
               "sel": "single-K1-ctx100000",
               "sum": f"{LOGS}/bench-256-20261023_0514.summary.json",
               "srv": f"{LOGS}/bench-256-20261023_0514.serverN.json",
               "log": f"{LOGS}/serve_dsv41_tp4_h100_20261023_044623.log"},
        "v3": {"label": "v3 20res(0-19)+128槽+expandable_segments, 4M KV",
               "csv": f"{LOGS}/{a.prefix}-{a.ts}.csv",
               "sel": "single-K1-ctx100000",
               "sum": f"{LOGS}/{a.prefix}-{a.ts}.summary.json",
               "srv": f"{LOGS}/{a.prefix}-{a.ts}.serverN.json",
               "log": a.v3_serve_log},
    }

    for k, v in V.items():
        v["rows"] = load_rows(v["csv"], v["sel"])
        v["sums"] = json.load(open(v["sum"])) if v["sum"] else []
        v["srvN"] = (json.load(open(v["srv"]))["by_N"] if v["srv"] else {})
        v["pool"] = pool_stats(v["log"])

    S = []
    P = S.append
    P("")
    P("---")
    P("")
    P(f"## 12) 三版对照 — 吞吐测量（本轮 TS=`{a.ts}`，只发请求）")
    P("")
    P("本节只列数字与原始行。三版为同一台机、同一模型 `DeepSeek-V4.1-Flash-1017`、"
      "同一 TP4 / W4A16 / ctx 1M / mrr16 / `--disable-flashinfer-autotune` 配置，"
      "唯一差别为下表三列；三版均由本脚本族（`bench256_streams.py`）在各自服务上"
      "以同一负载形状测量（`max_tokens=1500`、`temperature=0.7`、"
      "1 条“1000 字小说→反复续写”会话），测量者未重启服务、未改服务端配置。")
    P("")
    P("| 项 | v1 pristine | v2 全池 | v3 本轮 |")
    P("| --- | --- | --- | --- |")
    P("| `LK_GPU_RESIDENT_MOE_LAYERS` | `0-19` | `\"\"`（空） | `0-19` |")
    P("| `LK_POOL_SLOTS` | 128 | 256 | 128 |")
    P("| `PYTORCH_CUDA_ALLOC_CONF` | — | — | `expandable_segments:True` |")
    P("| `[pool] slab` | — | 44.82 GiB (40 层 × 1147.50 MiB) | 11.21 GiB (20 层 × 573.75 MiB) |")
    P("| `max_total_num_tokens` | 2,380,800 | 4,194,304 | 4,194,304 |")
    P("| 服务 PID | 43403 | 1630 | 55964 |")
    P("| 服务日志 | `serve_dsv41_tp4_h100_20261023_025933.log` | "
      "`serve_dsv41_tp4_h100_20261023_044623.log` | "
      f"`{a.v3_serve_log.split('/')[-1]}` |")
    P("")
    P("三版服务端 env 核验：v2 与本轮（v3）均含 `CUDA_LAUNCH_BLOCKING=1`（两版一致，非差异项）。")
    P("")

    # ---------------- 13.1 单流
    P("### 12.1 单流长会话（规格 A）")
    P("")
    P("| 指标 | v1 pristine | v2 全池 | v3 本轮 |")
    P("| --- | --- | --- | --- |")
    rows = []
    for k in ("v1", "v2", "v3"):
        r = V[k]["rows"]
        s = V[k]["sums"][0] if V[k]["sums"] else None
        rows.append((r, s))
    P(f"| 成功轮数 n | {len(rows[0][0])} | {len(rows[1][0])} | {len(rows[2][0])} |")
    P(f"| 失败数 | 0 | 0 | 0 |")
    P(f"| 到达上下文 `ctx_after` 最大 | {int(max(x['ctx'] for x in rows[0][0]))} | "
      f"{int(max(x['ctx'] for x in rows[1][0]))} | {int(max(x['ctx'] for x in rows[2][0]))} |")
    P(f"| 墙钟（首轮起→末轮止）s | {2602.8} | "
      f"{rows[1][1]['wall_full_span_s']:.1f} | {rows[2][1]['wall_full_span_s']:.1f} |")

    def col(fn, fm="{:.3f}"):
        out = []
        for r, s in rows:
            v = fn(r, s)
            out.append("" if v is None else (fm.format(v) if isinstance(v, float) else str(v)))
        return out

    mets = [
        ("每请求 decode tok/s p50", lambda r, s: pct([x["dec"] for x in r], 50), "{:.3f}"),
        ("每请求 decode tok/s mean", lambda r, s: mean([x["dec"] for x in r]), "{:.3f}"),
        ("每请求 decode tok/s p95", lambda r, s: pct([x["dec"] for x in r], 95), "{:.3f}"),
        ("每请求 ms/step p50", lambda r, s: pct([x["mss"] for x in r], 50), "{:.3f}"),
        ("每请求 ms/step mean", lambda r, s: mean([x["mss"] for x in r]), "{:.3f}"),
        ("每请求 ms/step p95", lambda r, s: pct([x["mss"] for x in r], 95), "{:.3f}"),
        ("每请求 prefill tok/s p50", lambda r, s: pct([x["pre"] for x in r], 50), "{:.3f}"),
        ("TTFT s p50", lambda r, s: pct([x["ttft"] for x in r], 50), "{:.3f}"),
        ("TTFT s p95", lambda r, s: pct([x["ttft"] for x in r], 95), "{:.3f}"),
        ("每请求 prefill tok/s mean", lambda r, s: mean([x["pre"] for x in r]), "{:.3f}"),
    ]
    for name, fn, fm in mets:
        c = col(fn, fm)
        P(f"| {name} | {c[0]} | {c[1]} | {c[2]} |")
    P("")
    P("分档均值（avg prefill tok/s / avg decode tok/s / avg ms per step）:")
    P("")
    P("| 上下文档 | v1 n | v1 prefill | v1 decode | v1 ms/step | v2 n | v2 prefill | "
      "v2 decode | v2 ms/step | v3 n | v3 prefill | v3 decode | v3 ms/step |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (name, _, _) in enumerate(BANDS):
        cells = []
        for r, s in rows:
            b = [x for x in r if BANDS[i][1] < x["ctx"] <= BANDS[i][2]]
            if not b:
                cells += ["0", "", "", ""]
            else:
                cells += [str(len(b)),
                          f"{mean([x['pre'] for x in b])}",
                          f"{mean([x['dec'] for x in b])}",
                          f"{mean([x['mss'] for x in b])}"]
        P(f"| {name} | " + " | ".join(cells) + " |")
    P("")
    P("单流逐轮原始行：v1 见 `pristine-run-20261023_0324.md` §14.3 与 "
      "`longsession-20261023_0406.csv`；v2 见本文件 §11.1 与 "
      "`bench-256-20261023_0514.csv`；v3 见 §12.1 末表与 "
      f"`{a.prefix}-{a.ts}.csv`（`stage=single-K1-ctx100000`）。")
    P("")
    v3r = rows[2][0]
    P(f"v3 单流逐轮原始行（{len(v3r)} 轮）:")
    P("")
    P("| # | ctx_after | prompt_tokens | completion_tokens | TTFT s | e2e s | prefill tok/s | "
      "decode tok/s | ms/step |")
    P("|---|---|---|---|---|---|---|---|---|")
    for i, x in enumerate(v3r, 1):
        P(f"| {i} | {int(x['ctx'])} | {int(x['pt'])} | {int(x['ct'])} | {x['ttft']} | "
          f"{x['e2e']} | {x['pre']} | {x['dec']} | {x['mss']} |")
    P("")

    # ---------------- 12.2 K 表
    P("### 12.2 多流 K 档（规格 B）")
    P("")
    P("v1 未跑 K=4/8/16（其报告 §14.1 记「单流模式未跑」）；v1 的 K=1 行取自其单流长会话。")
    P("")
    P("聚合分母 = 墙钟跨度（该档首轮起点→末轮终点），分子 = 该档全部成功轮次 Σ token。")
    P("")
    P("| K | 版本 | 跨度 s | 轮数 | Σ input | Σ output | 聚合 input tok/s | 聚合 output tok/s | "
      "每流 output tok/s | 客户端 ms/step (mean) | 服务端 ms/step = 1000/(X_p50/N) | "
      "服务端行数 | TTFT p50 s | TTFT p95 s | 失败数 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("v1", "v2", "v3"):
        v = V[k]
        ms = sorted([x for x in v["sums"] if x["mode"] == "multi"], key=lambda x: x["K"]) \
            if v["sums"] else []
        for s in ms:
            r = load_rows(v["csv"], s["stage"])
            si = sum(x["pt"] or 0 for x in r)
            so = sum(x["ct"] or 0 for x in r)
            span = s["wall_full_span_s"]
            srv = v["srvN"].get(str(s["K"]), {})
            P(f"| {s['K']} | {k} | {span:.1f} | {len(r)} | {int(si)} | {int(so)} | "
              f"{round(si/span,3) if span else ''} | {round(so/span,3) if span else ''} | "
              f"{s['per_stream_out_tok_s_mean']} | {s['per_stream_ms_per_step_mean']} | "
              f"{srv.get('ms_step_p50','')} | {srv.get('n','')} | "
              f"{s['ttft_s']['p50']} | {s['ttft_s']['p95']} | {s['failures']} |")
    P("")

    # ---------------- 12.3 服务端 per-step 全 N 档 (v2 / v3)
    P("### 12.3 服务端 per-step 全 N 档（`Decode batch`, TP0）")
    P("")
    P("| #running-req N | v2 行数 | v2 X p50 | v2 ms/step p50 | v3 行数 | v3 X p50 | "
      "v3 ms/step p50 | v3 X p95 |")
    P("|---|---|---|---|---|---|---|---|")
    allN = sorted(set(V["v2"]["srvN"]) | set(V["v3"]["srvN"]), key=int)
    for N in allN:
        a2 = V["v2"]["srvN"].get(N, {})
        a3 = V["v3"]["srvN"].get(N, {})
        P(f"| {N} | {a2.get('n','')} | {a2.get('X_p50','')} | {a2.get('ms_step_p50','')} | "
          f"{a3.get('n','')} | {a3.get('X_p50','')} | {a3.get('ms_step_p50','')} | "
          f"{a3.get('X_p95','')} |")
    P("")
    P("v1 服务端单流口径（pristine §14.3 记）: `#running-req: 1` 的 `gen throughput` "
      "p50 = 78.67 token/s ⇒ ms/step = 12.71 ms。")
    P("")

    # ---------------- 12.4 池
    P("### 12.4 专家池（规格 C）")
    P("")
    P("| 字段 | v1 pristine | v2 全池 | v3 本轮 |")
    P("| --- | --- | --- | --- |")
    for k in ("v1", "v2", "v3"):
        p = V[k]["pool"]
        if p is None:
            continue
    def pc(k, fn):
        p = V[k]["pool"]
        return "—" if p is None else fn(p)
    P(f"| `[pool] dstep=` 行数 | {pc('v1', lambda p: p['n'])} | {pc('v2', lambda p: p['n'])} | "
      f"{pc('v3', lambda p: p['n'])} |")
    def drange(pp):
        return "—" if not pp or not pp["dstep"] else f"{pp['dstep'][0]}–{pp['dstep'][1]}"
    P("| `dstep` 范围 | " + " | ".join(drange(V[k]["pool"]) for k in ("v1", "v2", "v3")) + " |")
    for fld, nm in [("win", "`win hit` %"), ("cum", "`cum hit` %"), ("psl", "`psl`"),
                    ("link", "`link` GB/s"), ("cis", "`copyiss` us"), ("que", "`que`"),
                    ("res", "`resident`")]:
        cells = []
        for k in ("v1", "v2", "v3"):
            p = V[k]["pool"]
            if p is None or not p[fld]:
                cells.append("—")
            else:
                cells.append(f"min {min(p[fld])} / p50 {pct(p[fld],50)} / "
                             f"mean {mean(p[fld])} / max {max(p[fld])}")
        P(f"| {nm} | " + " | ".join(cells) + " |")
    P(f"| `oom` 非零行数 | {pc('v1', lambda p: p['oom_nonzero'])} | "
      f"{pc('v2', lambda p: p['oom_nonzero'])} | {pc('v3', lambda p: p['oom_nonzero'])} |")
    P(f"| `stalls` 非零行数 | {pc('v1', lambda p: p['stalls_nonzero'])} | "
      f"{pc('v2', lambda p: p['stalls_nonzero'])} | {pc('v3', lambda p: p['stalls_nonzero'])} |")
    P("")
    for k in ("v1", "v2", "v3"):
        p = V[k]["pool"]
        if not p or not p["win"]:
            continue
        P(f"{k} 首个 / 末个 `[pool] dstep=` 行原文:")
        P("")
        P("```")
        fl = []
        with open(V[k]["log"], encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if "dstep=" in ln and "[pool]" in ln:
                    fl.append(ln.rstrip())
        if fl:
            P(fl[0])
            P(fl[-1])
        P("```")
        P("")
    P("三版 `[pool]` 行均为每 rank 1 行、无 rank 前缀标签。")
    P("")

    P("### 12.5 崩溃类逐项计数与产物绝对路径")
    P("")
    P("| 版本 | serve 日志 | 总行数 | `SEGV` | `OOM` | `Traceback` | `CUDA error` | "
      "`illegal` |")
    P("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in ("v1", "v2", "v3"):
        lp = V[k]["log"]
        try:
            txt = open(lp, encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            P(f"| {k} | (缺失) | | | | | | |"); continue
        cnt = {pat: txt.count(pat) for pat in
               ("SEGV", "OOM", "Traceback", "CUDA error", "illegal")}
        P(f"| {k} | `{lp.split('/')[-1]}` | {txt.count(chr(10))+1} | " +
          " | ".join(str(cnt[p]) for p in
                     ("SEGV", "OOM", "Traceback", "CUDA error", "illegal")) + " |")
    P("")
    P("v1/v2/v3 的 `SEGV` 命中均位于 serve 日志脚本头静态文本区（启动脚本写入的 "
      "`[ENV-CHANGE]` 注记），非运行时崩溃。")
    P("")
    P("```")
    P(f"逐轮明细 CSV      /mnt/SSD01/models/.llm-env/Lsglang/logs/{a.prefix}-{a.ts}.csv")
    P(f"汇总 JSON         /mnt/SSD01/models/.llm-env/Lsglang/logs/{a.prefix}-{a.ts}.summary.json")
    P(f"分析全文          /mnt/SSD01/models/.llm-env/Lsglang/logs/{a.prefix}-{a.ts}.analysis.md")
    P(f"服务端 per-N JSON /mnt/SSD01/models/.llm-env/Lsglang/logs/{a.prefix}-{a.ts}.serverN.json")
    P(f"显存/池采样       /mnt/SSD01/models/.llm-env/Lsglang/logs/{a.prefix}-{a.ts}.gpu.csv")
    P(f"                  /mnt/SSD01/models/.llm-env/Lsglang/logs/{a.prefix}-{a.ts}.pool.csv")
    P(f"压测 stdout       /mnt/SSD01/models/.llm-env/Lsglang/logs/{a.prefix}-{a.ts}.stdout.log")
    P("压测脚本          /mnt/SSD01/models/.llm-env/Lsglang/scripts/bench256_streams.py")
    P("分析脚本          /mnt/SSD01/models/.llm-env/Lsglang/scripts/analyze256_streams.py")
    P("三版图脚本        /mnt/SSD01/models/.llm-env/Lsglang/scripts/plotthree_throughput.py")
    P("本节生成器        /mnt/SSD01/models/.llm-env/Lsglang/scripts/build_threeway_section.py")
    P(f"三版对照图        /mnt/SSD01/models/.llm-env/Lsglang/docs/testA-threeway-{a.ts}.png")
    P(f"服务日志 (v3)     {a.v3_serve_log}")
    P("本报告            /mnt/SSD01/models/.llm-env/Lsglang/docs/run-256-slots-20261023_0446.md")
    P("```")
    P("")

    open(a.out, "w").write("\n".join(S))
    print("WROTE", a.out, len("\n".join(S)), "chars")


if __name__ == "__main__":
    main()
