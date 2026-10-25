#!/usr/bin/env python3
"""汇总 torch profiler chrome trace 里的 GPU kernel 时间 (按名字)。
用法: python analyze_trace.py <trace.json.gz> [topN]
"""
import gzip
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
topn = int(sys.argv[2]) if len(sys.argv) > 2 else 30

with gzip.open(path, "rt") as f:
    tr = json.load(f)
ev = tr["traceEvents"] if isinstance(tr, dict) else tr

kern = defaultdict(lambda: [0, 0.0])          # name -> [count, us]
gpu_all = 0.0
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    if "kernel" not in cat and "gpu_" not in cat:
        continue
    dur = float(e.get("dur", 0.0))
    gpu_all += dur
    name = re.sub(r"\s+", " ", str(e.get("name", "?")))[:110]
    kern[name][0] += 1
    kern[name][1] += dur

print(f"trace={path}")
print(f"GPU 事件总数={sum(v[0] for v in kern.values())}  累计时长={gpu_all/1000:.1f} ms")
print(f"{'kernel':<70} {'count':>7} {'total ms':>10} {'%':>6}")
for name, (n, us) in sorted(kern.items(), key=lambda kv: -kv[1][1])[:topn]:
    print(f"{name:<70} {n:>7} {us/1000:>10.2f} {100*us/max(gpu_all,1e-9):>6.1f}")
