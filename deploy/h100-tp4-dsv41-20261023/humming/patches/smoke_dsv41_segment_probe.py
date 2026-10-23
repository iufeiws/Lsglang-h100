#!/usr/bin/env python3
"""Smoke test for the full segment-probe module (tick mode inside capture +
eager event mode), on ONE physical GPU given by UUID.
"""
import json
import os
import sys

os.environ["SGLANG_DSV41_PROBE_SEGMENTS"] = "1"
os.environ["SGLANG_DSV41_PROBE_WINDOW"] = "4"
os.environ["SGLANG_DSV41_PROBE_OUT"] = "/mnt/SSD01/models/.llm-env/humming/patches/_smoke_segprobe.jsonl"
open(os.environ["SGLANG_DSV41_PROBE_OUT"], "w").close()

sys.path.insert(0, "/mnt/SSD01/models/.llm-env/humming/patches")
import torch  # noqa: E402
import dsv41_segment_probe as P  # noqa: E402

print("probe enabled:", P.PROBE_ENABLED)
assert P.PROBE_ENABLED

fails = []
x = torch.zeros(4096, device="cuda")
side = torch.cuda.Stream()


def fake_target_verify_replay(graph):
    """mimics one target-verify replay: 20 pool layers + 1 eager bracket"""
    with graph:
        pass
    for _ in range(60):
        x.add_(1.0)


# eager pre-allocation, exactly like the patched _init_pool site does
P.ensure_channels()
print("channels:", sorted(P.PROBE._channels),
      {k: v.capture_alloc for k, v in P.PROBE._channels.items()})

# ---- graph A: 20 pool-layer spans (stamp at idx 0) + a draft-like group
gA = torch.cuda.CUDAGraph()
with torch.cuda.graph(gA):
    for i in range(20):
        with P.span("pool_layer", idx=i):
            for _ in range(40):
                x.add_(1.0)
    with P.span("draft5_total"):
        for step in range(5):
            with P.span("draft_step", idx=step):
                for _ in range(30):
                    x.add_(1.0)
                with P.span("vocab_ag", idx=step):
                    for _ in range(10):
                        x.add_(1.0)
            with P.span("draft_bcast", idx=step):
                for _ in range(5):
                    x.add_(1.0)
print("captured gA")

# ---- 8 simulated steps: replay gA, bracket eagerly with events
last_row = {}
for step in range(8):
    P.on_step()   # eager row mark, exactly like the patched decode entry
    with P.span("step_total"):
        with P.span("target_verify"):
            gA.replay()
        with P.span("accept_all"):
            for _ in range(20):
                x.add_(1.0)
    last_row = P.PROBE._last_row

row = last_row
print("\n--- flush row ---")
print(json.dumps(row, indent=1, sort_keys=True)[:3000])

pool = None
draft5 = None
eager = row["eager"]
if "pool" in row["graph"] and "pool_layer" in row["graph"]["pool"]["seg"]:
    pool = row["graph"]["pool"]["seg"]["pool_layer"]
if "draft" in row["graph"] and "draft5_total" in row["graph"]["draft"]["seg"]:
    draft5 = row["graph"]["draft"]["seg"]["draft5_total"]

if pool is None or pool["n_rows"] != 4:
    fails.append(f"pool rows != 8: {pool}")
if draft5 is None or draft5["n_rows"] != 4:
    fails.append(f"draft rows != 8: {draft5}")
if "target_verify" not in eager or eager["target_verify"]["n"] != 4:
    fails.append(f"eager target_verify: {eager.get('target_verify')}")
if "accept_all" not in eager:
    fails.append("eager accept_all missing")

if pool:
    # 20 layers x 40 adds vs 5 steps x 30 adds: pool sum must be ~20x the step sum
    dstep = row["graph"]["draft"]["seg"]["draft_step"]["sum_mean"]
    print(f"\npool.sum mean = {pool['sum_mean']:.4f} ms ; draft_step.sum mean = {dstep:.4f} ms")
    if not pool["sum_mean"] > dstep:
        fails.append("pool sum smaller than one draft-step sum")
if draft5:
    bcast = row["graph"]["draft"]["seg"]["draft_bcast"]["sum_mean"]
    ag = row["graph"]["draft"]["seg"]["vocab_ag"]["sum_mean"]
    print(f"draft5_total={draft5['sum_mean']:.4f}  bcast.sum={bcast:.4f}  ag.sum={ag:.4f}")
    if not (draft5["sum_mean"] >= bcast):
        fails.append("draft5_total < bcast sum")

print("\n=== SUMMARY LINES ===")
for line in P.PROBE.summary_lines():
    print(line)
print("FAILS:", fails if fails else "none")
P.PROBE.close()
sys.exit(1 if fails else 0)
