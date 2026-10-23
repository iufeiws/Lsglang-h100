#!/usr/bin/env python3
"""Probe the CUDA event semantics around graph capture (query vs elapsed_time)."""
import os
import sys

import torch

torch.cuda.init()
dev = torch.device("cuda:0")
x = torch.zeros(4096, device=dev)


def try_(label, fn):
    try:
        print(f"{label}: {fn()}")
    except Exception as exc:
        print(f"{label}: EXC {type(exc).__name__}: {str(exc).splitlines()[0]}")


e0 = torch.cuda.Event(enable_timing=True)
e1 = torch.cuda.Event(enable_timing=True)
print("== clean, unrecorded ==")
try_("query", lambda: e0.query())

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    e0.record()
    for _ in range(100):
        x.add_(1.0)
    e1.record()
for _ in range(20):
    x.add_(1.0)

print("== right after capture, before replay ==")
try_("e0.query", lambda: e0.query())
try_("e1.query", lambda: e1.query())
try_("e0.elapsed_time(e1)", lambda: e0.elapsed_time(e1))

print("== after replay + sync ==")
g.replay()
torch.cuda.synchronize()
try_("e0.query", lambda: e0.query())
try_("e1.query", lambda: e1.query())
try_("e0.elapsed_time(e1)", lambda: e0.elapsed_time(e1))
try_("e1.elapsed_time(e0)", lambda: e1.elapsed_time(e0))

print("== replay again, read without query ==")
for _ in range(200):
    x.add_(1.0)
g.replay()
torch.cuda.synchronize()
try_("elapsed", lambda: e0.elapsed_time(e1))

print("== re-record OUTSIDE capture on the same handles (does query recover?) ==")
e0.record()
for _ in range(50):
    x.add_(1.0)
e1.record()
torch.cuda.synchronize()
try_("e0.query", lambda: e0.query())
try_("elapsed", lambda: e0.elapsed_time(e1))
print("== replay the graph after the eager re-record ==")
try_("g.replay", lambda: g.replay())
torch.cuda.synchronize()
try_("elapsed after replay", lambda: e0.elapsed_time(e1))
try_("e0.query", lambda: e0.query())

print("== disable-timing events inside capture ==")
d0 = torch.cuda.Event(enable_timing=False)
d1 = torch.cuda.Event(enable_timing=False)
g2 = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g2):
        d0.record()
        for _ in range(100):
            x.add_(1.0)
        d1.record()
    print("capture with enable_timing=False OK")
except Exception as exc:
    print("capture with enable_timing=False FAIL:", repr(exc))
g2.replay()
torch.cuda.synchronize()
try_("d0.query", lambda: d0.query())
try_("d1.query", lambda: d1.query())
try_("d0.elapsed_time(d1)", lambda: d0.elapsed_time(d1))
