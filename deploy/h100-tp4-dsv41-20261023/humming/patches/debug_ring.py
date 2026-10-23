#!/usr/bin/env python3
"""Debug: why does the ring row not advance per replay inside the probe module?"""
import sys

sys.path.insert(0, "/mnt/SSD01/models/.llm-env/humming/patches")
import torch
import dsv41_segment_probe as P  # noqa: E402

kern = P._probe_tick_kernel
W, NB = 8, 8
buf = torch.zeros((W, NB), dtype=torch.int64, device="cuda")
ctr = torch.zeros((1,), dtype=torch.int32, device="cuda")


def tick(col, row_mark, label):
    kern[(1,)](buf, ctr, row_mark, col, NB, W, num_warps=1)


x = torch.zeros(64, device="cuda")
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    tick(0, 1, "stamp")
    x.add_(1.0)
    tick(1, 0, "end")
print("counter after capture:", ctr.item())
for i in range(4):
    g.replay()
    torch.cuda.synchronize()
    print(f"after replay {i}: ctr={ctr.item()}")
    h = buf.cpu().numpy()
    nz = [(r, h[r][0], h[r][1]) for r in range(W) if h[r][0] != 0 or h[r][1] != 0]
    print("   non-zero rows:", nz)
