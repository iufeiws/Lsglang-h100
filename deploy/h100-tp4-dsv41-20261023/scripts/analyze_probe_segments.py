#!/usr/bin/env python3
"""Aggregate the DSV4.1 segment-probe output into the report's number tables.

usage: analyze_probe_segments.py <probe_jsonl_or_dir> <serve_log> [bench_csv]

* aligns probe windows with the bench request window (bench CSV epoch_start/end)
* per segment: mean / p50 / p95 / max, sample counts, per-window row counts
* reconciliation: sum of segments vs measured step total vs server-reported
  accept len / gen throughput
"""
from __future__ import annotations

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
    if not vals:
        return float("nan")
    v = sorted(vals)
    if len(v) == 1:
        return v[0]
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] * (1 - (pos - lo)) + v[hi] * (pos - lo)


def load_windows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
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


def pick_file(arg):
    if os.path.isdir(arg):
        cands = sorted(glob.glob(os.path.join(arg, "dsv41-segprobe.*.jsonl")))
        best, best_n = None, -1
        for c in cands:
            n = sum(1 for _ in open(c, encoding="utf-8"))
            if n > best_n:
                best, best_n = c, n
        return best, cands
    return arg, [arg]


def bench_window(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    t0, t1 = [], []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for key, acc in (("epoch_start", t0), ("epoch_end", t1)):
                try:
                    acc.append(float(row[key]))
                except Exception:
                    pass
    if not t0 or not t1:
        return None
    return min(t0), max(t1)


def serve_stats(serve_log, w0, w1):
    """accept len / gen throughput / [pool] fields inside the request window."""
    dl = re.compile(r"Decode batch.*accept len: ([0-9.]+), accept rate: ([0-9.]+).*"
                    r"gen throughput \(token/s\): ([0-9.]+)")
    pl = re.compile(r"\[pool\] dstep=(\d+) win hit=([0-9.]+)%.*gather=(\d+)us "
                    r"copyiss=([0-9.]+)us")
    ts_re = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    acc, rate, thr, pool = [], [], [], {}
    last_e = None
    with open(serve_log, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = ts_re.match(line)
            if m:
                try:
                    last_e = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
                        tzinfo=CST).timestamp()
                except Exception:
                    last_e = None
            e = last_e  # [pool] lines carry no timestamp: inherit the previous one
            if e is None or e < w0 or e > w1:
                continue
            d = dl.search(line)
            if d:
                acc.append(float(d.group(1)))
                rate.append(float(d.group(2)))
                thr.append(float(d.group(3)))
            p = pl.search(line)
            if p:
                for name, key in (("hit_pct", 2), ("gather_us", 3), ("copyiss_us", 4)):
                    pool.setdefault(name, []).append(float(p.group(key)))
    return {"accept_len": acc, "accept_rate": rate, "gen_tok_s": thr, "pool": pool}


def main():
    probe_arg, serve_log = sys.argv[1], sys.argv[2]
    bench_csv = sys.argv[3] if len(sys.argv) > 3 else None
    path, all_files = pick_file(probe_arg)
    print(f"# probe file: {path}")
    print(f"# candidate files ({len(all_files)}): "
          f"{', '.join(os.path.basename(f) for f in all_files)}")
    rows = load_windows(path)
    if not rows:
        print("NO PROBE WINDOWS")
        return 1
    bw = bench_window(bench_csv)
    sel = rows
    if bw:
        w0, w1 = bw
        sel = [r for r in rows
               if w0 - 3 <= datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%S")
               .replace(tzinfo=CST).timestamp() <= w1 + 3]
    print(f"# windows: total={len(rows)} in-request-window={len(sel)}")
    print(f"# window span: {sel[0]['ts']} .. {sel[-1]['ts']}" if sel else "none")

    segs = {}
    for r in sel:
        for cname, blk in (r.get("graph") or {}).items():
            for name, st in (blk.get("seg") or {}).items():
                key = f"{name}.sum@graph[{cname}]"
                segs.setdefault(key, {"vals": [], "rows": []})
                segs[key]["vals"].append(st["sum_mean"])
                segs[key]["rows"].append(st["n_rows"])
        for name, st in (r.get("eager") or {}).items():
            key = f"{name}@eager"
            segs.setdefault(key, {"vals": [], "rows": []})
            segs[key]["vals"].append(st["mean"])
            segs[key]["rows"].append(st["n"])

    print("\n## segment table (per-window aggregates inside the request window)")
    print(f"{'segment':34s} {'n_win':>5s} {'mean_ms':>9s} {'p50_ms':>9s} "
          f"{'p95_ms':>9s} {'max_ms':>9s} {'rows/win':>9s}")
    for key in sorted(segs):
        v = segs[key]["vals"]
        if not v:
            continue
        print(f"{key:34s} {len(v):5d} {statistics.fmean(v):9.4f} {pct(v, .5):9.4f} "
              f"{pct(v, .95):9.4f} {max(v):9.4f} "
              f"{statistics.fmean(segs[key]['rows']):9.2f}")

    # per-index detail (single-shot segments)
    print("\n## per-index detail (mean ms over the window)")
    for r in sel[:1]:
        for cname, blk in (r.get("graph") or {}).items():
            for name, st in (blk.get("seg") or {}).items():
                pi = st.get("per_idx_mean") or {}
                if len(pi) > 1:
                    print(f"  {name}@graph[{cname}] "
                          f"{ {k: round(v, 4) for k, v in sorted(pi.items(), key=lambda x: int(x[0]))} }")

    # ---------------------------------------------------------- reconciliation
    def m(key):
        v = segs.get(key, {}).get("vals")
        return statistics.fmean(v) if v else float("nan")

    step = m("step_total@eager")
    draft = m("draft_propose@eager")
    tverify = m("target_verify@eager")
    accall = m("accept_all@eager")
    draft5 = m("draft5_total.sum@graph[draft]")
    pool = m("pool_layer.sum@graph[pool]")
    print("\n## reconciliation (window means, ms)")
    print(f"step_total (eager bracket)         = {step:9.4f}")
    print(f"  draft_propose (eager bracket)    = {draft:9.4f}")
    print(f"    draft5_total (in draft graph)  = {draft5:9.4f}")
    print(f"  target_verify (eager bracket)    = {tverify:9.4f}")
    print(f"    pool_layer.sum (in verify graph)= {pool:9.4f}")
    print(f"  accept_all (eager bracket)       = {accall:9.4f}")
    print(f"  Σ(draft+verify+accept)           = {draft + tverify + accall:9.4f}")
    print(f"  step - Σ                         = {step - (draft + tverify + accall):9.4f}")
    bcast = m("draft_bcast.sum@graph[draft]")
    ag = m("vocab_ag.sum@graph[draft]")
    dstep = m("draft_step.sum@graph[draft]")
    if draft5 == draft5:
        print(f"    draft5_total vs Σ draft_step    = {draft5:9.4f} vs {dstep:9.4f} "
              f"(delta {draft5 - dstep:9.4f})")
    print(f"    draft_bcast.sum / vocab_ag.sum  = {bcast:9.4f} / {ag:9.4f}")
    aker = m("accept_kernel@eager")
    print(f"  accept_kernel (eager, rejection)  = {aker:9.4f}")

    # ------------------------------------------------- server-side numbers
    agg = {"probe_file": path, "windows_total": len(rows), "windows_in_window": len(sel),
           "segments": {k: {"n": len(v["vals"]), "mean": statistics.fmean(v["vals"]),
                            "p50": pct(v["vals"], .5), "p95": pct(v["vals"], .95),
                            "max": max(v["vals"]),
                            "rows_per_win": statistics.fmean(v["rows"])}
                        for k, v in segs.items() if v["vals"]},
           "reconciliation": {"step_total": step, "draft_propose": draft,
                              "target_verify": tverify, "accept_all": accall,
                              "draft5_total": draft5, "pool_layer_sum": pool,
                              "draft_bcast_sum": bcast, "vocab_ag_sum": ag,
                              "draft_step_sum": dstep, "accept_kernel": aker,
                              "sum_three": draft + tverify + accall}}
    if bw:
        st = serve_stats(serve_log, *bw)
        agg["server"] = {k: ({"n": len(v), "mean": statistics.fmean(v), "p50": pct(v, .5),
                              "p95": pct(v, .95)} if v else None)
                         for k, v in st.items() if k != "pool"}
        agg["server"]["pool"] = {k: {"n": len(v), "mean": statistics.fmean(v),
                                     "p50": pct(v, .5)} for k, v in st["pool"].items() if v}
        if st["accept_len"] and st["gen_tok_s"]:
            agg["server"]["derived_step_ms"] = (statistics.fmean(st["accept_len"])
                                                / statistics.fmean(st["gen_tok_s"]) * 1000)
            agg["server"]["tokens_per_step"] = statistics.fmean(st["accept_len"])
        print("\n## server-side within the request window")
        for k in ("accept_len", "accept_rate", "gen_tok_s"):
            v = st[k]
            if v:
                print(f"  {k:12s} n={len(v):4d} mean={statistics.fmean(v):8.4f} "
                      f"p50={pct(v, .5):8.4f} p95={pct(v, .95):8.4f}")
        if st["accept_len"] and st["gen_tok_s"]:
            al = statistics.fmean(st["accept_len"])
            th = statistics.fmean(st["gen_tok_s"])
            print(f"  derived per-step ms = accept_len/gen_tok_s*1000 = {al / th * 1000:.4f}"
                  f"   (tokens/step = {al:.4f})")
        for k, v in st["pool"].items():
            if v:
                print(f"  pool.{k:12s} n={len(v):4d} mean={statistics.fmean(v):9.3f} "
                      f"p50={pct(v, .5):9.3f}")

    # whole-process per-step SUMMARY rows (written when the probe closes)
    summ = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if '"summary"' not in line:
                continue
            try:
                t = json.loads(line)["summary"]
            except Exception:
                continue
            mm = re.search(
                r"SUMMARY segment=(\S+) mode=(\S+) n=(\d+) mean_ms=([0-9.]+) "
                r"p50_ms=([0-9.]+) p95_ms=([0-9.]+) max_ms=([0-9.]+)", t)
            if mm:
                summ[mm.group(1)] = {
                    "mode": mm.group(2), "n": int(mm.group(3)),
                    "mean": float(mm.group(4)), "p50": float(mm.group(5)),
                    "p95": float(mm.group(6)), "max": float(mm.group(7)),
                }
    agg["summary"] = summ

    # extras used by make_probe_report.py
    ca = set()
    for r in sel:
        for cname, blk in (r.get("graph") or {}).items():
            if blk.get("capture_alloc") is not None:
                ca.add(bool(blk["capture_alloc"]))
    agg["capture_alloc_note"] = ("; ".join(
        f"{cname}: capture_alloc={v}" for cname in
        sorted({c for r in sel for c in (r.get("graph") or {})})
        for v in [any((r.get("graph") or {}).get(cname, {}).get("capture_alloc")
                      for r in sel if cname in (r.get("graph") or {}))]) or "无通道数据")
    if bw:
        agg["bench_window_str"] = (f"{datetime.fromtimestamp(bw[0], CST):%Y-%m-%dT%H:%M:%S}"
                                   f" .. {datetime.fromtimestamp(bw[1], CST):%Y-%m-%dT%H:%M:%S}")
    if bench_csv:
        base = os.path.basename(bench_csv)
        agg["bench_prefix"] = base.rsplit("-", 1)[0]
        agg["bench_ts"] = base.rsplit("-", 1)[1].replace(".csv", "")
    first = sel[0] if sel else None
    if first:
        agg["per_idx_first_window"] = {
            name: st.get("per_idx_mean") or {}
            for cname, blk in (first.get("graph") or {}).items()
            for name, st in (blk.get("seg") or {}).items()
            if len(st.get("per_idx_mean") or {}) > 1
        }
    out_json = sys.argv[4] if len(sys.argv) > 4 else None
    if out_json:
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(agg, fh, indent=1, sort_keys=True)
        print(f"\n# agg json: {out_json}")

    # ----------------------------------------------------------- whole-process
    print("\n## whole-process SUMMARY lines (exact per-step p50/p95, all phases)")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if '"summary"' in line:
                try:
                    s = json.loads(line)["summary"]
                except Exception:
                    continue
                if any(k in s for k in (
                        "segment=step_total", "segment=target_verify",
                        "segment=draft_propose", "segment=accept_all",
                        "segment=accept_kernel", "segment=draft5_total.sum",
                        "segment=pool_layer.sum", "segment=draft_bcast.sum",
                        "segment=vocab_ag.sum", "segment=draft_step.sum")):
                    print("  " + s.split("] ")[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
