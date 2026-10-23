#!/usr/bin/env python3
"""Repro: probe-module ticks inside capture — do rows advance per replay?"""
import os
import sys

os.environ["SGLANG_DSV41_PROBE_SEGMENTS"] = "1"
os.environ["SGLANG_DSV41_PROBE_WINDOW"] = "4"
os.environ["SGLANG_DSV41_PROBE_OUT"] = "/mnt/SSD01/models/.llm-env/humming/patches/_dbg.jsonl"
sys.path.insert(0, "/mnt/SSD01/models/.llm-env/humming/patches")
import torch  # noqa: E402
import dsv41_segment_probe as P  # noqa: E402

x = torch.zeros(4096, device="cuda")
gA = torch.cuda.CUDAGraph()
with torch.cuda.graph(gA):
    for i in range(3):
        with P.span("pool_layer", idx=i, stamp=(i == 0)):
            for _ in range(40):
                x.add_(1.0)
print("captured")
for r in range(4):
    gA.replay()
    torch.cuda.synchronize()
    chan = P.PROBE._channels["pool"]
    h = chan.buf.cpu().numpy()
    nz = [(i, int(h[i][0]), int(h[i][1])) for i in range(chan.w) if h[i][0] or h[i][1]]
    print(f"replay {r}: ctr={chan.ctr.item()} rows={nz}")
