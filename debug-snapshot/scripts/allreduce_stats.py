#!/usr/bin/env python3
"""allreduce_fusion kernel 的时长分布与时间重叠统计 (GPU 事件)。
重叠判据: 两个 allreduce 事件的时间区间 [ts, ts+dur) 相交 (同一 trace 内两两比较, O(n^2) 用排序实现)。
用法: python allreduce_stats.py <trace.json.gz> [更多 trace ...]
"""
import gzip
import json
import sys


def load(path):
    with gzip.open(path, "rt") as f:
        tr = json.load(f)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    return [e for e in ev if e.get("ph") == "X" and ("kernel" in e.get("cat", "") or "gpu_" in e.get("cat", ""))]


for path in sys.argv[1:]:
    kern = load(path)
    steps = sum(1 for e in kern if str(e.get("name", "")).startswith("_gate_up_kernel")) / 20.0
    ar = sorted(((float(e["ts"]), float(e["dur"])) for e in kern
                 if "allreduce_fusion" in str(e.get("name", ""))), key=lambda t: t[0])
    n = len(ar)
    total = sum(d for _, d in ar) / 1000.0
    span = (ar[-1][0] + ar[-1][1] - ar[0][0]) / 1000.0 if n else 0.0
    durs = sorted(d for _, d in ar)
    q = lambda p: durs[min(n - 1, int(p * n))] if n else float("nan")
    part = set()
    pairs = 0
    for i in range(n):
        ts_i, dur_i = ar[i]
        end_i = ts_i + dur_i
        j = i + 1
        while j < n and ar[j][0] < end_i:
            pairs += 1
            part.add(i); part.add(j)
            j += 1
    print("=" * 100)
    print(f"trace={path}")
    print(f"步数={steps:.1f}  allreduce 事件={n}  ({n/steps:.1f}/步)   累计={total:.1f} ms ({total/steps:.3f} ms/步)")
    mx = durs[-1] if n else 0.0
    print(f"时长 us: p10={q(0.10):.1f} p50={q(0.50):.1f} p90={q(0.90):.1f} max={mx:.1f} "
          f"mean={total*1000/max(n,1):.1f}")
    print(f"全部 allreduce 覆盖的时间跨度={span:.1f} ms ({span/steps:.3f} ms/步)  "
          f"⇒ 串行占比={100*total/max(span,1e-9):.1f}%")
    print(f"时间区间相交的 allreduce 事件数={len(part)}/{n}  相交对数={pairs}")
