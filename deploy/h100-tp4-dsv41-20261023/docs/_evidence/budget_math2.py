#!/usr/bin/env python
"""budget_math2.py — corrected Lsglang KV/VRAM arithmetic (post §8 revision).

Corrections vs budget_math.py:
  * KV is priced with the *cap-mode* model: 1630 B/full-token (compressed only,
    only 4 kv_source layers own compressed storage) + FIXED SWA bytes;
    ratio mode = 1630 + 23400*r (only when --swa-full-tokens-ratio is explicit).
  * resident layers are priced with the mxfp4 flashinfer-cutlass 128-padding
    (640/576 = 1.11111); pool layers take raw strides (no padding).
Pure stdlib.
"""
GI = 1024**3
MEM = 81559 * 1024**2 * 0.88          # bytes per rank at mem-fraction 0.88
W_FIX = 2.987                          # GiB/rank non-expert (TP4) + replicated
EXPERT_RANK = 4700160                  # bytes/expert/rank (raw, pool layout)
PAD = 640 / 576                        # mxfp4 cutlass intermediate padding factor
LAYERS = 40

def slab(S, layers=LAYERS):
    return S * EXPERT_RANK * layers / GI

def resident(k, pad=True):
    return k * EXPERT_RANK * 384 * (PAD if pad else 1.0) / GI

CAP = 1630.0
def bpt(r=None):
    return CAP if r is None else CAP + 23400 * r

def kv(N, C, r=None):
    return N * C * bpt(r) / GI

def swa_fixed(cps, mrr, page=128, evict=128, window=128, draft=1):
    per_req = window + evict * draft + page + page
    cap = per_req * mrr + 2 * cps + page
    head = 4 * mrr * (window + page)
    tokens = ((cap + head + page - 1) // page) * page
    return tokens, tokens * 23400 / GI

print("=== per-token ===")
for r in (None, 0.005, 0.1):
    lab = "cap" if r is None else f"r={r}"
    print(f"  {lab:8s} {bpt(r):8.0f} B/token  4x1M={kv(4,1048576,r):6.2f}  10x1M={kv(10,1048576,r):6.2f}  16x1M={kv(16,1048576,r):6.2f} GiB")

print("\n=== SWA cap mode (page=128, evict=128, no spec) ===")
for cps in (8192, 16384):
    for mrr in (4, 16):
        t, g = swa_fixed(cps, mrr)
        print(f"  cps={cps:6d} mrr={mrr:3d} -> swa_cap_tokens={t:7d} = {g:.3f} GiB")

print("\n=== variant P: S=288 pool, all 40 layers ===")
W = slab(288) + W_FIX
for r in (None, 0.005, 0.1):
    t4, s4 = swa_fixed(8192, 4)
    t16, s16 = swa_fixed(8192, 16)
    need4 = kv(4, 1048576, r) + (s4 if r is None else 0.0)
    need16 = kv(16, 1048576, r) + (s16 if r is None else 0.0)
    avail = MEM / GI - W
    print(f"  {'cap' if r is None else 'r=%s'%r:8s} W={W:.2f} KV_budget={avail:.2f} | 4x1M {need4:6.2f} {'OK' if need4<=avail else 'FAIL'} | 16x1M {need16:6.2f} {'OK' if need16<=avail else 'FAIL'}")

print("\n=== S_max ===")
for label, N in (("4x1M", 4), ("16x1M", 16)):
    s = swa_fixed(8192, N)[1]
    need = kv(N, 1048576, None) + s
    print(f"  {label:6s} cap: need={need:6.2f} -> S_max={(MEM/GI-W_FIX-need)/(EXPERT_RANK*LAYERS/GI):7.1f} slots/layer")

print("\n=== variant R(k): k resident(padded) + (40-k) pooled at 288 ===")
for k in (0, 4, 8, 12, 16, 20):
    W = resident(k) + slab(288, LAYERS - k) + W_FIX
    avail = MEM / GI - W
    need4 = kv(4, 1048576, None) + swa_fixed(8192, 4)[1]
    print(f"  k={k:3d} W={W:6.2f} KV={avail:6.2f} 4x1M {'OK' if need4<=avail else 'FAIL'}")

print("\n=== variant S(k): no pool, k resident(padded) ===")
for k in (0, 16, 21, 30, 32, 35, 36):
    W = resident(k) + W_FIX
    avail = MEM / GI - W
    n4 = kv(4, 1048576, None) + swa_fixed(8192, 4)[1]
    n16 = kv(16, 1048576, None) + swa_fixed(8192, 16)[1]
    print(f"  k={k:3d} W={W:6.2f} KV={avail:6.2f} | 4x1M {'OK' if n4<=avail else 'FAIL'} | 16x1M {'OK' if n16<=avail else 'FAIL'}")
for label, need in (("4x1M", kv(4,1048576,None)+swa_fixed(8192,4)[1]), ("16x1M", kv(16,1048576,None)+swa_fixed(8192,16)[1])):
    print(f"  max k for {label}: {(MEM/GI-W_FIX-need)/(EXPERT_RANK*384*PAD/GI):.1f}")

print("\n=== host ===")
print(f"  experts(all)+engram = {275.77+188.83:.2f} GiB = {(275.77+188.83)*GI/1e9:.2f} GB ; node1 free = 411410 MB = {411410/1024:.2f} GiB")
