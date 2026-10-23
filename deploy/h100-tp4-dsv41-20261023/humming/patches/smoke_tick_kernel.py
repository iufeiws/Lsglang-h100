#!/usr/bin/env python3
"""Smoke test: Triton globaltimer 'tick' kernel inside CUDA graph capture.

Checks (a) first-Triton-compile happens *inside* capture without breaking it,
(b) ticks are captured as graph nodes and report the replayed span,
(c) per-`step` ring rows give per-replay resolution.
"""
import os
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _probe_tick_kernel(buf_ptr, ctr_ptr, bid, nb, w):
    """1-program kernel: stamp %globaltimer (ns) into row = step % w, col = bid."""
    if bid == 0:
        tl.atomic_add(ctr_ptr, 1, sem="release")
    step = tl.atomic_add(ctr_ptr, 0, sem="acquire")
    row = step % w
    t = tl.extra.cuda.globaltimer()
    tl.store(buf_ptr + row * nb + bid, t, cache_modifier=".cg")


W = 8
NB = 16
buf = torch.zeros((W, NB), dtype=torch.int64, device="cuda")
ctr = torch.zeros((1,), dtype=torch.int32, device="cuda")


def tick(bid):
    _probe_tick_kernel[(1,)](buf, ctr, bid, NB, W, num_warps=1)


x = torch.zeros(4096, device="cuda")
fails = []

print("first launch (compiles) inside a graph capture ...")
t0 = time.time()
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        tick(0)          # first compile happens here
        for _ in range(100):
            x.add_(1.0)
        tick(1)
        for _ in range(400):
            x.add_(1.0)
        tick(2)
    print(f"capture with first-time Triton compile OK in {time.time()-t0:.1f}s")
except Exception as exc:
    fails.append(f"capture failed: {exc!r}")
    print("CAPTURE FAIL", repr(exc))

torch.cuda.synchronize()
print("after capture: ctr =", ctr.item(), "(no ticks executed yet)")
buf.zero_(); ctr.zero_()
torch.cuda.synchronize()

for i in range(3):
    g.replay()
torch.cuda.synchronize()
host = buf.cpu().numpy()
print("counter after 3 replays:", ctr.item())
for r in range(W):
    row = host[r]
    if (row != 0).any():
        t0n, t1n, t2n = row[0], row[1], row[2]
        print(f"row{r}: start={t0n} seg1={(t1n-t0n)/1e6:.4f} ms  seg2={(t2n-t1n)/1e6:.4f} ms")

# per-replay resolution check: rows must be distinct steps
starts = [host[r][0] for r in range(W) if host[r][0] != 0]
print("rows with data:", len(starts), "monotonic:", starts == sorted(starts))
if len(starts) != 3:
    fails.append(f"expected 3 rows, got {len(starts)}")

# eager ticks after a flush-zero cycle
buf.zero_(); ctr.zero_()
torch.cuda.synchronize()
t_a = time.perf_counter()
for _ in range(20):
    tick(0)
    for _ in range(100):
        x.add_(1.0)
    tick(1)
torch.cuda.synchronize()
t_b = time.perf_counter()
host = buf.cpu().numpy()
seg = (host[0][1] - host[0][0]) / 1e6 if host[0][1] else float("nan")
print(f"eager: 20 iterations in {(t_b-t_a)*1e3:.2f} ms; last-row seg={seg:.4f} ms")

print("FAILS:", fails if fails else "none")
raise SystemExit(1 if fails else 0)
