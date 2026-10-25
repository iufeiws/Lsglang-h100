#!/usr/bin/env python3
"""按 kernel family 汇总 torch profiler chrome trace (GPU 事件), 两个 trace 用同一套规则, 便于逐轮对照。
用法: python trace_families.py <trace.json.gz> [更多 trace ...]
"""
import gzip
import json
import re
import sys
from collections import defaultdict

FAMILIES = [
    ("allreduce_fusion", lambda n: "allreduce_fusion" in n),
    ("UVA miss kernel gate_up", lambda n: n.startswith("_gate_up_kernel")),
    ("UVA miss kernel down", lambda n: n.startswith("_down_kernel")),
    ("cutlass_moe (resident+cache)", lambda n: "cutlass13device_kernel" in n),
    ("deep_gemm fp8", lambda n: "deep_gemm::sm90_fp8_gemm_1d2d_impl" in n),
    ("attention (sparse MLA)", lambda n: "flash_fwd_splitkv_mla_fp8_sparse_kernel" in n
        or "flash_fwd_mla_combine_kernel" in n),
    ("cutlass work-tile map/finalize/activation", lambda n: "build_precomputed_work_tile_map_kernel" in n
        or "finalizeMoeRoutingKernel" in n or "doActivationKernel" in n),
    ("route_to_slots / assign_slots", lambda n: "_route_to_slots_kernel" in n or "_assign_slots_kernel" in n),
    ("staging copy kernel (61d60cfa)", lambda n: "_stage_copy_kernel" in n),
    ("plan kernel (61d60cfa)", lambda n: "_plan_kernel" in n),
    ("memcpy HtoD", lambda n: n.startswith("Memcpy HtoD")),
]


def load(path):
    with gzip.open(path, "rt") as f:
        tr = json.load(f)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    return [e for e in ev if e.get("ph") == "X" and ("kernel" in e.get("cat", "") or "gpu_" in e.get("cat", ""))]


for path in sys.argv[1:]:
    kern = load(path)
    steps = sum(1 for e in kern if str(e.get("name", "")).startswith("_gate_up_kernel")) / 20.0
    if steps < 1:
        steps = float("nan")
    total = sum(float(e.get("dur", 0.0)) for e in kern) / 1000.0
    agg = defaultdict(lambda: [0, 0.0])
    for e in kern:
        n = re.sub(r"\s+", " ", str(e.get("name", "?")))
        dur = float(e.get("dur", 0.0))
        hit = "other"
        for fam, pred in FAMILIES:
            if pred(n):
                hit = fam
                break
        agg[hit][0] += 1
        agg[hit][1] += dur
    print("=" * 100)
    print(f"trace={path}")
    print(f"步数(=_gate_up_kernel 调用数/20)={steps:.1f}   GPU 事件={len(kern)}   累计={total:.1f} ms   {total/steps:.2f} ms/步")
    print(f"{'family':<46}{'calls/步':>10}{'ms/步':>10}{'%':>8}")
    for fam, _ in FAMILIES + [("other", None)]:
        c, us = agg.get(fam, [0, 0.0])
        print(f"{fam:<46}{c/steps:>10.1f}{us/1000/steps:>10.3f}{100*us/1000/max(total,1e-9):>8.1f}")
