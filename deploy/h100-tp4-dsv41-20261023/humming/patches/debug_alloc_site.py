#!/usr/bin/env python3
"""Is the loss of the counter increment caused by allocating the ring buffer
INSIDE the capture (graph private pool) vs outside?"""
import sys

sys.path.insert(0, "/mnt/SSD01/models/.llm-env/humming/patches")
import torch
import dsv41_segment_probe as P  # noqa: E402

kern = P._probe_tick_kernel
W, NB = 8, 8
x = torch.zeros(64, device="cuda")


def run(label, alloc_inside):
    box = {}

    def tick(col, row_mark):
        kern[(1,)](box["buf"], box["ctr"], row_mark, col, NB, W, num_warps=1)

    if not alloc_inside:
        box["buf"] = torch.zeros((W, NB), dtype=torch.int64, device="cuda")
        box["ctr"] = torch.zeros((1,), dtype=torch.int32, device="cuda")

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        if alloc_inside:
            box["buf"] = torch.zeros((W, NB), dtype=torch.int64, device="cuda")
            box["ctr"] = torch.zeros((1,), dtype=torch.int32, device="cuda")
        tick(0, 1)
        x.add_(1.0)
        tick(1, 0)

    trace = []
    for i in range(3):
        g.replay()
        torch.cuda.synchronize()
        h = box["buf"].cpu().numpy()
        rows = [r for r in range(W) if h[r][0] or h[r][1]]
        trace.append((int(box["ctr"].item()), rows))
    print(f"{label}: (ctr, rows) after each replay = {trace}")


run("allocated OUTSIDE capture", False)
run("allocated INSIDE  capture", True)
