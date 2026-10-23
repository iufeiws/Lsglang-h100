#!/usr/bin/env python3
"""Aggregate one pool6row measurement round: bench CSV + segment-probe jsonl +
pool-counter-probe jsonl + native `[pool]` lines -> one JSON + one CSV table.

Usage (venv python):
  analyze_pool6row.py --run-ts 20261023_1657 --tag Aprobe [--root ...]
Outputs (next to the inputs, under logs/pool6row_<run-ts>/):
  analyze-<tag>.json / analyze-<tag>.csv
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

ROOT = "/mnt/SSD01/models/.llm-env/Lsglang"
LOGS = f"{ROOT}/logs"


def q(vals, p):
    if not vals:
        return float("nan")
    xs = sorted(vals)
    i = int(round(p * (len(xs) - 1)))
    return float(xs[max(0, min(len(xs) - 1, i))])


def load_bench(csv_path):
    rounds = []
    if not os.path.exists(csv_path):
        return rounds
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            try:
                rounds.append({
                    "K": int(row["K"]),
                    "stream": int(row["stream"]),
                    "round": int(row["round"]),
                    "e0": float(row["epoch_start"]),
                    "e1": float(row["epoch_end"]),
                    "tok_s": float(row["decode_tok_s"]),
                    "ms_step": float(row["ms_per_step"]),
                    "acc_len": float(row["decode_tok_s"]) * float(row["ms_per_step"]) / 1000.0,
                    "ttft": float(row["ttft_s"]),
                    "out_tokens": int(row["completion_tokens"]),
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "foreign": int(row.get("foreign_sock") or 0),
                })
            except Exception:
                continue
    return rounds


def _seg_pool(v):
    """per-step ms of the pool_layer segment from a segment-probe window dict."""
    if not isinstance(v, dict):
        return None
    if v.get("n"):
        return v["mean"]
    return None


def load_segprobe(paths, logs=()):
    """Segment-probe windows from jsonl files and/or serve-log lines.

    The probe mirrors every window into the serve log as ``[seg-probe] {json}``,
    which is the only channel that survives when the jsonl cannot be opened.
    """
    wins = []
    lines = []
    for path in paths:
        with open(path, errors="replace") as fh:
            lines += [(None, ln) for ln in fh.read().splitlines()]
    for path in logs:
        try:
            with open(path, errors="replace") as fh:
                for ln in fh:
                    i = ln.find("[seg-probe] {")
                    if i >= 0:
                        ts = ln[1:20].replace(" ", "T") if ln.startswith("[") else None
                        lines.append((ts, ln[i + len("[seg-probe] "):]))
        except Exception:
            continue
    for log_ts, line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "win" not in d or "eager" not in d:
            continue
        eager = d.get("eager") or {}
        graph = d.get("graph") or {}
        split = (graph.get("pool") or {}).get("seg") or {}
        layers = split.get("pool_layer") or {}
        draft = (graph.get("draft") or {}).get("seg") or {}
        row = {
            "pid": d.get("pid"),
            "win": d.get("win"),
            "t": log_ts or d.get("ts"),
            "mono": d.get("mono"),
            "steps": d.get("step"),
            "n_rows": layers.get("n_rows"),
            "draft_rows": (graph.get("draft") or {}).get("rows"),
            "step_total": (eager.get("step_total") or {}).get("mean"),
            "target_verify": (eager.get("target_verify") or {}).get("mean"),
            "accept_all": (eager.get("accept_all") or {}).get("mean"),
            "accept_kernel": (eager.get("accept_kernel") or {}).get("mean"),
            "draft_propose": (eager.get("draft_propose") or {}).get("mean"),
            "pool_sum": layers.get("sum_mean"),
            "draft5_total": (draft.get("draft5_total") or {}).get("sum_mean"),
            "pool_layers": {str(k): v for k, v in (layers.get("per_idx_mean") or {}).items()},
        }
        wins.append(row)
    return wins


def load_counterprobe(paths):
    wins = []
    for path in paths:
        with open(path, errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("kind") != "window":
                    continue
                wins.append(d)
    return wins


POOL_RE = re.compile(
    r"\[pool\] dstep=(?P<dstep>\d+) win hit=(?P<hit>[\d.]+)% nr=(?P<nr>[\d.]+)% "
    r"psl=(?P<psl>[\d.]+) link=(?P<link>[\d.]+)GB/s gather=(?P<gather>\d+)us "
    r"copyiss=(?P<copyiss>[\d.]+)us que=(?P<que>\d+) hf_hook=(?P<hf_hook>\d+)us "
    r"hf_dec=(?P<hf_dec>\d+)us hf_hook_max=(?P<hf_hook_max>\d+)us "
    r"hf_dec_max=(?P<hf_dec_max>\d+)us \| cum hit=(?P<chit>[\d.]+)% h=(?P<h>\d+) "
    r"m=(?P<m>\d+) p=(?P<p>\d+) B=(?P<B>\d+) resident=(?P<resident>\d+) nr=(?P<cnr>\d+) "
    r"vf=(?P<vf>\d+) oom=(?P<oom>\d+) stalls=(?P<stalls>\d+) sk=(?P<sk>\d+) "
    r"stale=(?P<stale>\d+) tag=(?P<tag>\d+) pubbad=(?P<pubbad>\d+) "
    r"publag=(?P<publag>\d+) cmiss=(?P<cmiss>\d+) cdup=(?P<cdup>\d+)")


def load_pool_lines(paths):
    out = []
    for path in paths:
        with open(path, errors="replace") as fh:
            for line in fh:
                m = POOL_RE.search(line)
                if not m:
                    continue
                d = {k: float(v) for k, v in m.groupdict().items()}
                out.append(d)
    return out


def summarize_run(run_ts, tag, seg_rank=0, root=ROOT):
    outdir = f"{root}/logs/pool6row_{run_ts}"
    bench = sorted(glob.glob(f"{outdir}/*{tag}*.csv"))
    bench = [b for b in bench if not b.endswith((".gpu.csv", ".pool.csv"))]
    rounds = []
    for b in bench:
        rounds += load_bench(b)
    # segment-probe jsonl: pick the run by PID (one file per TP rank), newest set
    all_seg = sorted(glob.glob(f"{root}/logs/dsv41-segprobe.*.jsonl"), key=os.path.getmtime)
    live_pids = []
    if all_seg:
        pids = []
        for p in all_seg[-8:]:
            for line in open(p, errors="replace"):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("hello"):
                    pids.append((os.path.getmtime(p), d.get("pid")))
                break
        if pids:
            newest_pid = max(pids)[1]
            live_pids = [pid for _t, pid in pids if pid is not None and
                         abs(pid - newest_pid) <= 8]
    if not live_pids:
        live_pids = []
    seg_paths = [p for p in all_seg
                 if any((".%d.jsonl" % pid) in p for pid in live_pids)] or all_seg[-4:]
    serve_logs = sorted(glob.glob(f"{root}/logs/serve_dsv41_tp4_h100_*.log"),
                        key=os.path.getmtime)[-1:]
    wins = load_segprobe(seg_paths, serve_logs)
    cp_paths = sorted(glob.glob(f"{root}/logs/dsv41-pool6row.*.jsonl"),
                      key=os.path.getmtime)
    cp_live = []
    if cp_paths:
        first_pid = None
        for p in reversed(cp_paths):
            try:
                with open(p, errors="replace") as fh:
                    for line in fh:
                        d = json.loads(line)
                        if d.get("kind") == "window":
                            first_pid = d.get("pid")
                            break
            except Exception:
                pass
            if first_pid:
                break
        if first_pid:
            cp_live = [p for p in cp_paths if (".%d.jsonl" % first_pid) in p]
    cp = load_counterprobe(cp_live or cp_paths[-1:])
    pool_lines = load_pool_lines(sorted(
        glob.glob(f"{outdir}/*.pool.csv")) + sorted(glob.glob(f"{LOGS}/dsv41-pool6row.*.log")))

    res = {
        "run_ts": run_ts, "tag": tag,
        "files": {"bench": bench, "segprobe": seg_paths[-1:] or serve_logs, "counterprobe": cp_paths[-1:]},
        "bench_rounds": len(rounds),
        "seg_windows": len(wins),
        "counter_windows": len(cp),
    }
    by_k = {}
    for r in rounds:
        by_k.setdefault(r["K"], []).append(r)
    cont = {}
    for k in sorted(by_k):
        rs = by_k[k]
        # round windows fully inside the common overlap of this K's rounds
        e0, e1 = max(x["e0"] for x in rs), min(x["e1"] for x in rs)
        w_in = [w for w in wins if _seg_ok(w, e0, e1)]
        cp_in = [w for w in cp if _cp_ok(w, e0, e1)]
        cont[str(k)] = {
            "rounds": len(rs),
            "window_s": round(e1 - e0, 1),
            "ms_step_p50": q([x["ms_step"] for x in rs], .50),
            "ms_step_p95": q([x["ms_step"] for x in rs], .95),
            "tok_s_p50": q([x["tok_s"] for x in rs], .50),
            "acc_len_p50": q([x["acc_len"] for x in rs], .50),
            "acc_len_mean": round(statistics.fmean([x["acc_len"] for x in rs]), 3),
            "ttft_p50": q([x["ttft"] for x in rs], .50),
            "windows": len(w_in),
            "seg_step_total_p50": q([w["step_total"] for w in w_in if w["step_total"]], .50),
            "seg_target_verify_p50": q([w["target_verify"] for w in w_in if w["target_verify"]], .50),
            "seg_pool_sum_p50": q([w["pool_sum"] for w in w_in if w["pool_sum"]], .50),
            "seg_pool_sum_p95": q([w["pool_sum"] for w in w_in if w["pool_sum"]], .95),
            "seg_pool_sum_mean": round(statistics.fmean([w["pool_sum"] for w in w_in if w["pool_sum"]]), 3) if w_in else float("nan"),
            "seg_accept_all_p50": q([w["accept_all"] for w in w_in if w["accept_all"]], .50),
            "seg_rows_per_win_p50": q([w["n_rows"] for w in w_in if w["n_rows"]], .50),
            "cp_launch_ms_per_step_p50": q([w.get("launch_ms_sum_per_step") for w in cp_in
                                            if w.get("launch_ms_sum_per_step")], .50),
            "cp_launch_ms_per_step_mean": round(statistics.fmean(
                [w["launch_ms_sum_per_step"] for w in cp_in if w.get("launch_ms_sum_per_step")]), 3) if cp_in else float("nan"),
            "cp_windows": len(cp_in),
            "cp_snap_mismatch": sum(w.get("snap_mismatch", 0) for w in cp_in),
            "cp_slots_probed": sum(w.get("slots_probed", 0) for w in cp_in),
            "cp_snap_claim": sum(w.get("snap_claim", 0) for w in cp_in),
            "cp_snap_absent": sum(w.get("snap_absent", 0) for w in cp_in),
            "cp_id_dup_rows": sum(w.get("id_dup_rows", 0) for w in cp_in),
            "cp_id_nodup_rows": sum(w.get("id_nodup_rows", 0) for w in cp_in),
            "cp_id_missing": sum(w.get("id_missing", 0) for w in cp_in),
            "cp_id_gt_pool": sum(w.get("id_gt_pool", 0) for w in cp_in),
            "cp_rows_mismatch": sum(w.get("rows_mismatch", 0) for w in cp_in),
            "cp_gpu_tok_ids": sum(w.get("gpu_tok_ids", 0) for w in cp_in),
            "cp_rows_per_token": cp_in[0].get("rows_per_token") if cp_in else None,
        }
    # per-layer p50 across all K (from the K=1 slice and the union)
    per_layer = {}
    for w in wins:
        for k, v in (w["pool_layers"] or {}).items():
            per_layer.setdefault(k, []).append(v)
    res["pool_layer_p50_ms"] = {k: round(q(v, .50), 4) for k, v in sorted(per_layer.items(), key=lambda kv: int(kv[0]))}
    res["pool_layer_mean_ms"] = {k: round(statistics.fmean(v), 4) for k, v in sorted(per_layer.items(), key=lambda kv: int(kv[0]))}
    res["by_streams"] = cont
    res["native_pool"] = _summ_pool(pool_lines)
    # slope fit: seg_pool_sum vs K
    ks = sorted(int(k) for k in cont)
    ys = [cont[str(k)]["seg_pool_sum_p50"] for k in ks]
    res["row_slope"] = _fit(ks, ys, "seg_pool_sum_p50_ms_per_K")
    ys2 = [cont[str(k)]["seg_step_total_p50"] for k in ks]
    res["step_slope"] = _fit(ks, ys2, "seg_step_total_p50_ms_per_K")
    ys3 = [cont[str(k)]["cp_launch_ms_per_step_p50"] for k in ks]
    res["launch_slope"] = _fit(ks, ys3, "cp_launch_ms_per_step_p50_per_K")
    return res


def _t_epoch(ts):
    """'2026-10-23T18:44:12' (CST, no offset) -> epoch seconds; None on failure."""
    import calendar, time as _time
    try:
        tm = _time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
        return calendar.timegm(tm) - 8 * 3600
    except Exception:
        return None


def _seg_ok(w, e0, e1):
    """True when the window belongs to the measured common clock window."""
    if w.get("step_total") is None and w.get("pool_sum") is None:
        return False
    t = _t_epoch(w.get("t") or "")
    if t is None:
        return True
    return e0 <= t <= e1


def _cp_ok(w, e0, e1):
    t = w.get("t_s")
    return True  # host-side probe fires at capture time only; see report


def _fit(xs, ys, label):
    pairs = [(x, y) for x, y in zip(xs, ys) if y == y]
    if len(pairs) < 2:
        return {"label": label, "n": len(pairs), "slope": None, "intercept": None}
    n = len(pairs)
    mx = statistics.fmean(p[0] for p in pairs)
    my = statistics.fmean(p[1] for p in pairs)
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    slope = sxy / sxx if sxx else float("nan")
    inter = my - slope * mx
    return {"label": label, "n": n, "slope": round(slope, 6), "intercept": round(inter, 5),
            "x": [p[0] for p in pairs], "y": [round(p[1], 5) for p in pairs]}


def _summ_pool(lines):
    if not lines:
        return {}
    keys = ["cmiss", "cdup", "pubbad", "publag", "stale", "tag", "oom", "vf", "stalls", "sk"]
    out = {"n_lines": len(lines)}
    for k in keys:
        vals = [x[k] for x in lines if k in x]
        out[k + "_min"] = min(vals) if vals else None
        out[k + "_max"] = max(vals) if vals else None
    out["hit_cum_min"] = min(x["chit"] for x in lines)
    out["hit_cum_max"] = max(x["chit"] for x in lines)
    out["resident_first"] = lines[0].get("resident")
    out["resident_last"] = lines[-1].get("resident")
    out["copyiss_mean"] = round(statistics.fmean([x["copyiss"] for x in lines]), 2)
    out["copyiss_max"] = max(x["copyiss"] for x in lines)
    out["gather_mean"] = round(statistics.fmean([x["gather"] for x in lines]), 2)
    out["que_max"] = max(x["que"] for x in lines)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seg-rank", type=int, default=0)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--glob-seg", default="")
    a = ap.parse_args()
    res = summarize_run(a.run_ts, a.tag, a.seg_rank, a.root)
    outdir = f"{a.root}/logs/pool6row_{a.run_ts}"
    os.makedirs(outdir, exist_ok=True)
    jp = f"{outdir}/analyze-{a.tag}.json"
    with open(jp, "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    cp = f"{outdir}/analyze-{a.tag}.csv"
    with open(cp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["K", "rounds", "ms_step_p50", "tok_s_p50", "acc_len_p50",
                    "seg_step_total_p50", "seg_target_verify_p50", "seg_pool_sum_p50",
                    "seg_pool_sum_p95", "cp_launch_ms_per_step_p50",
                    "cp_snap_mismatch", "cp_slots_probed", "cp_id_dup_rows",
                    "cp_id_nodup_rows", "cp_id_missing", "cp_rows_mismatch"])
        for k, v in sorted(res["by_streams"].items(), key=lambda kv: int(kv[0])):
            w.writerow([k, v["rounds"], v["ms_step_p50"], v["tok_s_p50"], v["acc_len_p50"],
                        v["seg_step_total_p50"], v["seg_target_verify_p50"],
                        v["seg_pool_sum_p50"], v["seg_pool_sum_p95"],
                        v["cp_launch_ms_per_step_p50"], v["cp_snap_mismatch"],
                        v["cp_slots_probed"], v["cp_id_dup_rows"], v["cp_id_nodup_rows"],
                        v["cp_id_missing"], v["cp_rows_mismatch"]])
    print(json.dumps({k: res[k] for k in ("run_ts", "tag", "bench_rounds", "seg_windows",
                                          "counter_windows", "by_streams", "row_slope",
                                          "step_slope", "launch_slope", "native_pool")},
                     indent=2, ensure_ascii=False))
    print(f"\nwrote {jp}\nwrote {cp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
