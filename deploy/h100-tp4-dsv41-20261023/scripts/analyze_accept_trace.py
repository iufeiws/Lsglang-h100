#!/usr/bin/env python3
"""Aggregate the DSV4.1 accept-trace jsonl into the report's number tables.

usage: analyze_accept_trace.py <jsonl-or-dir> [--rank 0] [--json out.json]

Only numbers + their calibre are produced; no interpretation.

Calibre
-------
* one record = one (verify step, request) pair.  A request emits one record per
  decode step, so a 60 s single-stream arm contributes ~1 record/step.
* ``cand``  = ``verify_ids_2d`` row = [current token, gamma draft tokens]
  (slot 0 is the current token and is never "proposed").
* ``tgt``   = ``argmax`` over the full vocab of ``logits_output.next_token_logits``
  reshaped to (bs, 6); ``tgt[r]`` is the target's top-1 for verify row r.
* greedy rule (documented DFlash rule): ``cand[s] == tgt[s-1]`` accepts draft
  slot s; ``glen`` = length of the leading matching run (0..5).
* ``acc``   = engine ``correct_len`` (0..5): the accept path actually used
  (AcceptGreedy when the batch is all-greedy, else AcceptSampling).
* ``commit``= engine ``commit_lens`` = tokens committed this step; the server's
  ``accept len`` log line is this value.
* first mismatch slot (1-based) = ``glen + 1`` when ``glen < 5``.
* temperatures: ``ttmp`` = target sampling temperature of the request,
  ``dtmp`` = draft temperature of the request.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

NSLOT = 6


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


def pick_files(arg):
    if os.path.isdir(arg):
        fs = sorted(glob.glob(os.path.join(arg, "dsv41-acctrace.*.jsonl")))
    else:
        fs = [arg]
    return fs


def load(paths, rank=0):
    recs, steps = [], []
    per_rank = {}
    counters = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            k = obj.get("kind")
            if k == "summary":
                counters[str(obj.get("tp_rank"))] = obj.get("cumulative", {})
                continue
            if k != "flush":
                continue
            r = obj.get("tp_rank")
            per_rank.setdefault(r, 0)
            per_rank[r] += obj.get("n_records", 0)
            for st in obj.get("steps", []):
                rows = [x for x in st["records"]]
                per_rank.setdefault(f"steps_rank{r}", 0)
                per_rank[f"steps_rank{r}"] += 1
                if r != rank:
                    continue
                for x in rows:
                    recs.append({"step": st["step"], "bs": st["bs"], **x})
    return recs, per_rank, counters


def rank_agreement(paths, rank0=0, rank1=1):
    """Compare commit_lens per (step, row index) between two ranks."""
    out = {}
    for r in (rank0, rank1):
        m = {}
        for p in paths:
            for line in open(p, encoding="utf-8"):
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("kind") != "flush" or obj.get("tp_rank") != r:
                    continue
                for st in obj.get("steps", []):
                    for x in st["records"]:
                        m[(st["step"], x["i"])] = (x["commit"], x["glen"], x["acc"])
        out[r] = m
    a, b = out[rank0], out[rank1]
    keys = set(a) & set(b)
    same = sum(1 for k in keys if a[k] == b[k])
    return {"n_common": len(keys), "identical": same,
            "differ": len(keys) - same,
            "example_diffs": [[k, a[k], b[k]] for k in sorted(keys) if a[k] != b[k]][:5]}


def dist(vals, lo, hi):
    n = len(vals)
    d = {str(i): 0 for i in range(lo, hi + 1)}
    for v in vals:
        if str(v) in d:
            d[str(v)] += 1
    return {"n": n, **{k: v for k, v in d.items()}}


def group_tables(recs, label):
    n = len(recs)
    if not n:
        return {"label": label, "n": 0}
    commit = [r["commit"] for r in recs]
    glen = [r["glen"] for r in recs]
    acc = [r["acc"] for r in recs]
    # per-slot conditional match rate
    slot_cond = {}
    first_mismatch = {"none(5/5)": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    for r in recs:
        g = r["glen"]
        first_mismatch["none(5/5)" if g >= 5 else str(g + 1)] += 1
    for j in range(1, NSLOT):
        intact = 0
        match = 0
        for r in recs:
            if r["glen"] >= j - 1:
                intact += 1
                if r["cand"][j] == r["tgt"][j - 1]:
                    match += 1
        slot_cond[j] = {"prefix_intact": intact, "match": match,
                        "match_rate": (match / intact) if intact else None}
    # unconditional mismatch + shifted diagnostics
    slot_uncond = {}
    for j in range(1, NSLOT):
        mis = sum(1 for r in recs if r["cand"][j] != r["tgt"][j - 1])
        shift_same = sum(1 for r in recs if j < NSLOT and r["cand"][j] == r["tgt"][j])
        shift_prev = sum(1 for r in recs if j >= 2 and r["cand"][j] == r["tgt"][j - 2])
        slot_uncond[j] = {"mismatch": mis, "mismatch_rate": mis / n,
                          "eq_own_row": shift_same, "eq_row_minus2": shift_prev}
    # engine vs greedy
    eq = sum(1 for r in recs if r["acc"] == r["glen"])
    above = sum(1 for r in recs if r["acc"] > r["glen"])
    below = sum(1 for r in recs if r["acc"] < r["glen"])
    # top-k evidence on first-mismatch records
    tk = [r for r in recs if r["glen"] < 5 and r["drank"] >= 0 and r["tk_ids"][0] >= 0]
    tk_stats = None
    if tk:
        ranks = [r["drank"] for r in tk]
        gaps = [r["tk_vals"][0] - r["dval"] for r in tk]
        k = len(tk[0]["tk_ids"])
        in_topk = sum(1 for r in tk if r["drank"] < k)
        top1_in_draft = 0
        ov = []
        has_draft = 0
        for r in tk:
            if r["dt_ids"][0] >= 0:
                has_draft += 1
                if r["dt_ids"][0] == r["tk_ids"][0]:
                    top1_in_draft += 1
                a = set(r["tk_ids"])
                b = set(r["dt_ids"])
                ov.append(len(a & b) / float(k))
        tk_stats = {
            "n": len(tk), "k": k,
            "draft_rank_median": pct(ranks, 0.5), "draft_rank_mean": statistics.fmean(ranks),
            "draft_rank_p90": pct(ranks, 0.9), "draft_rank_max": max(ranks),
            "draft_in_target_topk": in_topk, "draft_in_target_topk_pct": 100.0 * in_topk / len(tk),
            "rank_hist_1_8": {str(i): sum(1 for x in ranks if x == i) for i in range(1, k + 1)},
            "logit_gap_top1_minus_draft_median": pct(gaps, 0.5),
            "logit_gap_top1_minus_draft_mean": statistics.fmean(gaps),
            "n_with_draft_topk": has_draft,
            "target_top1_equals_draft_top1": top1_in_draft,
            "target_top1_equals_draft_top1_pct": (100.0 * top1_in_draft / has_draft) if has_draft else None,
            "topk_set_overlap_mean": statistics.fmean(ov) if ov else None,
            "sample_rows": [{kk: r[kk] for kk in ("step", "rid", "glen", "acc", "commit",
                                                 "cand", "tgt", "tk_row", "tk_ids", "tk_vals",
                                                 "dval", "drank", "dt_ids", "dt_vals")}
                            for r in tk[:6]],
        }
    return {
        "label": label, "n_records": n,
        "n_steps": len({r["step"] for r in recs}),
        "ttmp_set": sorted({round(r["ttmp"], 4) for r in recs}),
        "dtmp_set": sorted({round(r["dtmp"], 4) for r in recs}),
        "commit_dist": dist(commit, 1, 6),
        "commit_mean": statistics.fmean(commit), "commit_p50": pct(commit, 0.5),
        "commit_p95": pct(commit, 0.95),
        "acc_dist": dist(acc, 0, 5), "acc_mean": statistics.fmean(acc),
        "glen_dist": dist(glen, 0, 5), "glen_mean": statistics.fmean(glen),
        "glen_p50": pct(glen, 0.5),
        "first_mismatch_hist": first_mismatch,
        "slot_conditional": {str(k): v for k, v in slot_cond.items()},
        "slot_unconditional": {str(k): v for k, v in slot_uncond.items()},
        "engine_vs_greedy": {"equal": eq, "engine_above_greedy": above,
                             "engine_below_greedy": below},
        "topk": tk_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--json", default="")
    ap.add_argument("--group-tmp-eps", type=float, default=0.05)
    args = ap.parse_args()

    paths = pick_files(args.path)
    if not paths:
        print("no jsonl found", file=sys.stderr)
        return 2
    recs, per_rank, counters = load(paths, args.rank)
    print(f"files: {paths}")
    print(f"rank filter: {args.rank}; records: {len(recs)}; per_rank_records: "
          f"{ {k: v for k, v in per_rank.items() if not str(k).startswith('steps_')} }")

    ra = rank_agreement(paths, args.rank, (args.rank + 1) % 4)
    print(f"rank agreement (0 vs 1) on (commit, glen, acc): {ra['identical']}/"
          f"{ra['n_common']} identical, {ra['differ']} differ")
    if ra["differ"]:
        print("  diffs:", ra["example_diffs"])

    # split by target temperature
    temps = sorted({round(r["ttmp"], 3) for r in recs})
    print(f"target temperature values present: {temps}")
    out = {"files": paths, "rank": args.rank, "n_records": len(recs),
           "per_rank_records": {str(k): v for k, v in per_rank.items()},
           "rank_agreement": ra, "counters": counters, "groups": []}
    groups = []
    if len(temps) > 1:
        for t in temps:
            sub = [r for r in recs if abs(round(r["ttmp"], 3) - t) <= args.group_tmp_eps]
            g = group_tables(sub, f"temp={t}")
            groups.append(g)
            print(f"\n===== GROUP {g['label']}  (n={g['n_records']}) =====")
            show(g)
    else:
        g = group_tables(recs, f"temp={temps[0] if temps else '?'}")
        groups.append(g)
        show(g)
    out["groups"] = groups
    if args.json:
        json.dump(out, open(args.json, "w"), ensure_ascii=False, indent=1)
        print(f"\njson -> {args.json}")
    return 0


def _f(x, nd=3):
    return "None" if x is None else f"{x:.{nd}f}"


def show(g):
    if not g.get("n_records"):
        print("  (empty)")
        return
    print(f"  n_records={g['n_records']} n_steps={g['n_steps']} "
          f"ttmp={g['ttmp_set']} dtmp={g['dtmp_set']}")
    print(f"  accept len (commit): mean={g['commit_mean']:.4f} p50={g['commit_p50']} "
          f"p95={g['commit_p95']}")
    print(f"  commit dist: {g['commit_dist']}")
    print(f"  engine correct_len acc: mean={g['acc_mean']:.4f} dist={g['acc_dist']}")
    print(f"  greedy glen: mean={g['glen_mean']:.4f} p50={g['glen_p50']} dist={g['glen_dist']}")
    print(f"  first mismatch slot hist: {g['first_mismatch_hist']}")
    print("  per-slot conditional match (given prefix intact):")
    for j, v in g["slot_conditional"].items():
        r = v["match_rate"]
        print(f"    slot {j}: intact={v['prefix_intact']:6d} match={v['match']:6d} "
              f"rate={r if r is None else round(r, 4)}")
    print("  per-slot unconditional mismatch:")
    for j, v in g["slot_unconditional"].items():
        print(f"    slot {j}: mismatch={v['mismatch']:6d} ({100*v['mismatch_rate']:.2f}%) "
              f"cand==tgt[same row]={v['eq_own_row']} cand==tgt[row-2]={v['eq_row_minus2']}")
    print(f"  engine vs greedy: equal={g['engine_vs_greedy']}")
    t = g["topk"]
    if t:
        print(f"  top-k evidence (first-mismatch rows, k={t['k']}, n={t['n']}):")
        print(f"    draft-token rank in target row: median={t['draft_rank_median']} "
              f"mean={_f(t['draft_rank_mean'],2)} p90={t['draft_rank_p90']} max={t['draft_rank_max']}")
        print(f"    rank hist 1..k: {t['rank_hist_1_8']}")
        print(f"    draft token inside target top-{t['k']}: {t['draft_in_target_topk']} "
              f"({t['draft_in_target_topk_pct']:.1f}%)")
        print(f"    logit gap (target top1 - draft token): median="
              f"{_f(t['logit_gap_top1_minus_draft_median'])} mean="
              f"{_f(t['logit_gap_top1_minus_draft_mean'])}")
        print(f"    target top1 == draft top1: {t['target_top1_equals_draft_top1']}/"
              f"{t['n_with_draft_topk']} ({t['target_top1_equals_draft_top1_pct']})")
        print(f"    mean top-k set overlap |T∩D|/k: {_f(t['topk_set_overlap_mean'],4)}")


if __name__ == "__main__":
    sys.exit(main())
