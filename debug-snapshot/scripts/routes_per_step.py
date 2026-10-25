#!/usr/bin/env python3
"""从 serve 日志算 [gpu-expert-cache] routes 与 Decode 步数的比值 (每步每层的路由数)。"""
import re
import sys
from datetime import datetime

log = sys.argv[1]
rank = sys.argv[2] if len(sys.argv) > 2 else "TP0 EP0"
lo = sys.argv[3] if len(sys.argv) > 3 else None
hi = sys.argv[4] if len(sys.argv) > 4 else None

ts_re = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) " + re.escape(rank) + r"\]")
routes_re = re.compile(r"\[gpu-expert-cache\] 20 layers, last (\d+)s: routes=(\d+) hit=([\d.]+)%")
steps = []   # (ts, 40)  # 每行 Decode batch = 40 步 (服务端日志口径)
caches = []  # (ts, window_s, routes, hit)

def in_win(t):
    if lo and t < datetime.strptime(lo, "%Y-%m-%d %H:%M:%S"):
        return False
    if hi and t > datetime.strptime(hi, "%Y-%m-%d %H:%M:%S"):
        return False
    return True

with open(log, errors="replace") as f:
    for line in f:
        m = ts_re.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if not in_win(t):
            continue
        mr = routes_re.search(line)
        if mr:
            caches.append((t, int(mr.group(1)), int(mr.group(2)), float(mr.group(3))))
        elif "Decode batch" in line:
            steps.append(t)

if not steps:
    print("no decode lines"); sys.exit(1)
span = (steps[-1] - steps[0]).total_seconds()
print(f"rank={rank} 窗口 {lo}..{hi}: Decode 行 {len(steps)} × 40 步 = {40*len(steps)} 步, "
      f"跨度 {span:.0f}s ⇒ {40*len(steps)/span:.2f} 步/s, {(span/ (40*len(steps)))*1000:.1f} ms/步")
tot = sum(c[2] for c in caches)
print(f"cache 行 {len(caches)} 条, routes 合计 {tot}, 对应墙钟跨度 "
      f"{(caches[-1][0]-caches[0][0]).total_seconds():.0f}s ⇒ {tot/max(1,(caches[-1][0]-caches[0][0]).total_seconds()):.0f} routes/s")
if steps:
    print(f"⇒ 每步路由 = {tot / (40*len(steps)):.0f}  (20 层 ⇒ {tot/(40*len(steps))/20:.1f} 路由/层/步)")
print(f"hit 均值 (按 routes 加权) = "
      f"{sum(c[2]*c[3] for c in caches)/max(1,tot):.1f}%")
