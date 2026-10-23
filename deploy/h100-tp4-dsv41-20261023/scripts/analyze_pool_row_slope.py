#!/usr/bin/env python3
"""Pool-path row-count slope + per-tier step cost from the segment probe.

usage: analyze_pool_row_slope.py --probe <jsonl|dir> --bench-csv <csv>
                                 [--serve-log <log>] [--accept-dir <dir>]
                                 [--accept-rank 0] [--json out.json]

Calibre
-------
* one sample = one decode step.  ``pool_layer.sum`` (graph tick) = sum over the
  20 pooled layers (20...39) of the main-stream bracket that wraps the whole
  pool path (fork + launch + ``lk_moe.cpu_decode`` + ``wait_stream(pool)`` +
  ``output_gpu.add_`` + ``.to(dtype)``); it therefore **contains the cross-stream
  wait for the pool stream** (documented in docs/probe-verify-step-segments.md).
* ``step_total`` (eager CUDA event) = one decode iteration
  (draft proposal + target verify + accept + host side).
* rows per step = ``streams x 6`` (verify_num_draft_tokens = 6; static ragged
  verify mode => no ragged trimming).
* per-step values come from ``raw_sum`` / ``raw`` (SGLANG_DSV41_PROBE_RAW=1),
  so per-tier p50/p95 are **per-step** quantiles, not window means.
* a probe window is assigned to a tier when its flush timestamp falls inside the
  tier's bench window (stage start + 1 s .. stage end); windows that straddle a
  tier boundary or fall in the 20 s inter-tier gap are dropped.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone

CST = timezone(timedelta(hours=8))


def pct(vals, q):
    v = sorted(vals)
    if not v:
        return None
    if len(v) == 1:
        return float(v[0])
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return float(v[lo]) * (1 - (pos - lo)) + float(v[hi]) * (pos - lo)


def lsfit(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    slope = sxy / sxx if sxx else float("nan")
    icept = my - slope * mx
    ss_res = sum((b - (icept + slope * a)) ** 2 for a, b in zip(x, y))
    ss_tot = sum((b - my) ** 2 for b in y)
    return {"slope_ms_per_row": slope, "intercept_ms": icept,
            "r2": (1 - ss_res / ss_tot) if ss_tot else float("nan")}


def parse_ts(s):
    s = s.strip()
    fmt = "%Y-%m-%dT%H:%M:%S" if "T" in s else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(s, fmt).replace(tzinfo=CST).timestamp()


def stage_windows(csv_path):
    stages = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "ok":
                continue
            st = row.get("stage") or f"{row['mode']}-K{row['K']}"
            e0 = float(row["epoch_start"])
            e1 = float(row["epoch_end"])
            d = stages.setdefault(st, {"start": e0, "end": e1, "rows": [],
                                       "tok_s": [], "ms_step": [], "ttft": [],
                                       "ct": [], "ctx": []})
            d["start"] = min(d["start"], e0)
            d["end"] = max(d["end"], e1)
            d["rows"].append(row)
            if row.get("decode_tok_s"):
                d["tok_s"].append(float(row["decode_tok_s"]))
            if row.get("ms_per_step"):
                d["ms_step"].append(float(row["ms_per_step"]))
            if row.get("ttft_s"):
                d["ttft"].append(float(row["ttft_s"]))
            if row.get("completion_tokens"):
                d["ct"].append(int(row["completion_tokens"]))
            if row.get("ctx_after"):
                d["ctx"].append(int(row["ctx_after"]))
    return stages


def load_windows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if "win" in obj:
            rows.append(obj)
    return rows


def pick_probe(arg):
    if os.path.isdir(arg):
        cands = sorted(glob.glob(os.path.join(arg, "dsv41-segprobe.*.jsonl")))
        best, best_n = None, -1
        for c in cands:
            n = sum(1 for _ in open(c, encoding="utf-8"))
            if n > best_n:
                best, best_n = c, n
        return best
    return arg


def probe_per_step(wins):
    """[(flush_epoch, [pool_sum per step], [step_total per step], [accept_all], [target_verify])]"""
    out = []
    for w in wins:
        ts = parse_ts(w["ts"])
        g = (w.get("graph") or {}).get("pool") or {}
        raw = (g.get("raw_sum") or {}).get("pool_layer") or []
        eg = w.get("eager") or {}
        st_raw = (eg.get("step_total") or {}).get("raw") or []
        tv_raw = (eg.get("target_verify") or {}).get("raw") or []
        aa_raw = (eg.get("accept_all") or {}).get("raw") or []
        out.append({"ts": ts, "win": w.get("win"), "step": w.get("step"),
                    "pool": raw, "step_total": st_raw, "target_verify": tv_raw,
                    "accept_all": aa_raw,
                    "pool_mean": (g.get("seg") or {}).get("pool_layer", {}).get("sum_mean"),
                    "step_total_mean": (eg.get("step_total") or {}).get("mean")})
    return out


def accept_trace_series(arg, rank):
    if not arg:
        return []
    paths = sorted(glob.glob(os.path.join(arg, "dsv41-acctrace.*.jsonl")))
    out = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("kind") != "flush" or obj.get("tp_rank") != rank:
                continue
            recs = [r for st in obj.get("steps", []) for r in st["records"]]
            if recs:
                out.append({"ts": obj["ts"], "commit": [r["commit"] for r in recs],
                            "glen": [r["glen"] for r in recs], "n": len(recs)})
    return out


def server_log_windows(log_path, stages):
    """Decode-batch (accept len / gen throughput / running req) and [pool] lines per tier."""
    out = {k: {"accept_len": [], "gen_tps": [], "running": [], "pool_hit": [],
               "pool_copyiss": [], "pool_link": [], "n_decode_lines": 0}
           for k in stages}
    if not log_path or not os.path.exists(log_path):
        return out
    dl = re.compile(r"\[(?P<ts>[0-9\-]+ [0-9:]+) TP0\] Decode batch, #running-req: (?P<nr>\d+).*accept len: (?P<al>[0-9.]+), accept rate: (?P<ar>[0-9.]+).*gen throughput \(token/s\): (?P<gt>[0-9.]+)")
    pl = re.compile(r"\[pool\] dstep=\d+ win hit=(?P<hit>[0-9.]+)%.*link=(?P<link>[0-9.]+)GB/s gather=(?P<g>\d+)us copyiss=(?P<ci>[0-9.]+)us")
    last_t = None
    for line in open(log_path, encoding="utf-8", errors="replace"):
        m = dl.search(line)
        if m:
            last_t = parse_ts(m.group("ts"))
            for k, d in out.items():
                if stages[k]["start"] <= last_t <= stages[k]["end"]:
                    d["accept_len"].append(float(m.group("al")))
                    d["gen_tps"].append(float(m.group("gt")))
                    d["running"].append(int(m.group("nr")))
                    d["n_decode_lines"] += 1
            continue
        m2 = pl.search(line)
        if m2 and last_t is not None:
            # [pool] lines carry no timestamp: they are attributed to the most
            # recent Decode-batch timestamp (<= one log interval, ~2 s, earlier).
            for k, d in out.items():
                if stages[k]["start"] <= last_t <= stages[k]["end"]:
                    d["pool_hit"].append(float(m2.group("hit")))
                    d["pool_link"].append(float(m2.group("link")))
                    d["pool_copyiss"].append(float(m2.group("ci")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--bench-csv", required=True)
    ap.add_argument("--serve-log", default="")
    ap.add_argument("--accept-dir", default="")
    ap.add_argument("--accept-rank", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    pf = pick_probe(args.probe)
    wins = load_windows(pf)
    if not wins:
        print(f"no windows in {pf}", file=sys.stderr)
        return 2
    per_step = probe_per_step(wins)
    stages = stage_windows(args.bench_csv)
    atr = accept_trace_series(args.accept_dir, args.accept_rank)
    slog = server_log_windows(args.serve_log, stages)

    print(f"probe jsonl : {pf}  (windows={len(wins)}, pid={wins[0].get('pid')})")
    print(f"window raw per-step rows available: "
          f"{sum(1 for w in per_step if w['pool'])} windows with pool raw, "
          f"{sum(1 for w in per_step if w['step_total'])} with step_total raw")
    print(f"stages (bench): {sorted(stages)}")

    res = {"probe": pf, "n_windows": len(wins), "stages": [], "fits": {}}
    tiers = []
    for st in sorted(stages, key=lambda k: stages[k]["start"]):
        S = stages[st]
        keep = [w for w in per_step if S["start"] + 1.0 <= w["ts"] <= S["end"]]
        drop = [w for w in per_step if S["start"] - 30 <= w["ts"] < S["start"] + 1.0]
        pool = [v for w in keep for v in w["pool"]]
        stept = [v for w in keep for v in w["step_total"]]
        tv = [v for w in keep for v in w["target_verify"]]
        aa = [v for w in keep for v in w["accept_all"]]
        nstreams = len({r["stream"] for r in S["rows"]})
        rec = {
            "stage": st, "streams": nstreams, "rows_per_step": 6 * nstreams,
            "window_start": S["start"], "window_end": S["end"],
            "n_windows_used": len(keep), "n_windows_dropped": len(drop),
            "pool": {"n": len(pool), "mean": statistics.fmean(pool) if pool else None,
                     "p50": pct(pool, 0.5), "p95": pct(pool, 0.95),
                     "max": max(pool) if pool else None,
                     "min": min(pool) if pool else None},
            "step_total": {"n": len(stept),
                           "mean": statistics.fmean(stept) if stept else None,
                           "p50": pct(stept, 0.5), "p95": pct(stept, 0.95),
                           "max": max(stept) if stept else None},
            "target_verify": {"n": len(tv), "mean": statistics.fmean(tv) if tv else None,
                              "p50": pct(tv, 0.5), "p95": pct(tv, 0.95)},
            "accept_all": {"n": len(aa), "mean": statistics.fmean(aa) if aa else None,
                           "p50": pct(aa, 0.5), "p95": pct(aa, 0.95)},
            "client": {
                "decode_tok_s_mean": statistics.fmean(S["tok_s"]) if S["tok_s"] else None,
                "decode_tok_s_p50": pct(S["tok_s"], 0.5),
                "ms_per_step_mean": statistics.fmean(S["ms_step"]) if S["ms_step"] else None,
                "ms_per_step_p50": pct(S["ms_step"], 0.5),
                "ttft_p50": pct(S["ttft"], 0.5),
                "completion_tokens_mean": statistics.fmean(S["ct"]) if S["ct"] else None,
                "rounds": len(S["rows"]), "ctx_max": max(S["ctx"]) if S["ctx"] else None,
            },
            "server_log": {
                "n_decode_lines": slog[st]["n_decode_lines"],
                "accept_len_mean": (statistics.fmean(slog[st]["accept_len"])
                                    if slog[st]["accept_len"] else None),
                "gen_tps_mean": (statistics.fmean(slog[st]["gen_tps"])
                                 if slog[st]["gen_tps"] else None),
                "running_req_mean": (statistics.fmean(slog[st]["running"])
                                     if slog[st]["running"] else None),
                "pool_hit_mean": (statistics.fmean(slog[st]["pool_hit"])
                                  if slog[st]["pool_hit"] else None),
                "pool_copyiss_mean_us": (statistics.fmean(slog[st]["pool_copyiss"])
                                         if slog[st]["pool_copyiss"] else None),
                "pool_link_mean_gbs": (statistics.fmean(slog[st]["pool_link"])
                                       if slog[st]["pool_link"] else None),
            },
        }
        a = [r for w in atr if S["start"] + 1.0 <= w["ts"] <= S["end"] for r in w["commit"]]
        g = [r for w in atr if S["start"] + 1.0 <= w["ts"] <= S["end"] for r in w["glen"]]
        rec["accept_trace"] = {
            "n_records": len(a),
            "commit_mean": statistics.fmean(a) if a else None,
            "commit_p50": pct(a, 0.5),
            "glen_mean": statistics.fmean(g) if g else None,
            "commit_dist": {str(i): sum(1 for x in a if x == i) for i in range(1, 7)},
        }
        res["stages"].append(rec)
        tiers.append(rec)

    # ---- table ----
    hdr = ("stage", "streams", "rows", "n_pool", "pool_p50", "pool_p95", "pool_mean",
           "step_p50", "step_p95", "step_mean", "acc_len_srv", "tok_s", "ms_step",
           "gen_tps", "pool_hit%")
    print("\n| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in tiers:
        def f(x, nd=3):
            return "" if x is None else round(x, nd)
        row = (r["stage"], r["streams"], r["rows_per_step"], r["pool"]["n"],
               f(r["pool"]["p50"]), f(r["pool"]["p95"]), f(r["pool"]["mean"]),
               f(r["step_total"]["p50"]), f(r["step_total"]["p95"]), f(r["step_total"]["mean"]),
               f(r["accept_trace"]["commit_mean"], 4), f(r["client"]["decode_tok_s_p50"]),
               f(r["client"]["ms_per_step_p50"]), f(r["server_log"]["gen_tps_mean"], 2),
               f(r["server_log"]["pool_hit_mean"], 2))
        print("| " + " | ".join(str(x) for x in row) + " |")

    # ---- fits ----
    xs = [r["rows_per_step"] for r in tiers]
    if len(xs) < 2:
        print("\n(only one concurrency tier: fit needs >= 2 distinct row counts)")
    for key, sub in (("p50", "p50"), ("mean", "mean"), ("p95", "p95")):
        if len(xs) < 2:
            continue
        yp = [r["pool"][sub] for r in tiers]
        ys = [r["step_total"][sub] for r in tiers]
        if all(v is not None for v in yp):
            res["fits"][f"pool_{key}"] = lsfit(xs, yp)
        if all(v is not None for v in ys):
            res["fits"][f"step_total_{key}"] = lsfit(xs, ys)
    print("\nfits over rows = " + ", ".join(str(x) for x in xs))
    for k, v in res["fits"].items():
        print(f"  {k}: slope={v['slope_ms_per_row']:.5f} ms/row "
              f"intercept={v['intercept_ms']:.4f} ms r2={v['r2']:.5f}")
    for r in tiers:
        if r["pool"]["p50"] and r["step_total"]["p50"]:
            res.setdefault("per_row", []).append(
                {"stage": r["stage"], "rows": r["rows_per_step"],
                 "pool_p50_over_rows": r["pool"]["p50"] / r["rows_per_step"],
                 "step_p50_over_rows": r["step_total"]["p50"] / r["rows_per_step"]})
    if "per_row" in res:
        print("\nper-row p50 (ms/row):")
        for x in res["per_row"]:
            print(f"  {x['stage']}: rows={x['rows']} pool={x['pool_p50_over_rows']:.4f} "
                  f"step_total={x['step_p50_over_rows']:.4f}")

    if args.json:
        json.dump(res, open(args.json, "w"), ensure_ascii=False, indent=1)
        print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
