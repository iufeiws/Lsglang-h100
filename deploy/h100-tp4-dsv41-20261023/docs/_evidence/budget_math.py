#!/usr/bin/env python
"""Lsglang budget arithmetic for DeepSeek-V4.1-Flash-1017 on 4xH100 80GB (TP4).

Inputs are checkpoint-header measurements + Lsglang source constants:
  - expert bytes per expert (all ranks): 18,800,640 B   (6 tensors, MXFP4 4-bit + E8M0)
  - per-rank (TP4, no EP): intermediate 2304//4 = 576 rows -> 4,700,160 B/expert
  - kv_bytes (V4 FlashMLA layout) = 448 + 64*2 + 8 = 584 B/token/layer
  - low-ratio indexer (fp4, force_fp4=True) = 128//2 + 128//32 = 68 B
  - SWA slot cost = 584 B * 40 layers = 23,360 B... (padded: 585 * 40)
Pure stdlib.
"""
MI = 1024**2
GI = 1024**3

GPU_MIB = 81559
GPU_TOTAL = GPU_MIB * MI
MEM_FRAC = 0.88
BUDGET = GPU_TOTAL * MEM_FRAC

# --- weights ---
CKPT = 510286023000
ENGRAM_ALL = 203073076240
ENGRAM_HOST = 202758032400
BB_EXPERTS = 40 * 7219445760          # 40 backbone layers x 384 experts
MTP_EXPERTS = 7325717760              # mtp.0/1/2 (3 x 128 experts + shared)
REST = CKPT - ENGRAM_ALL - BB_EXPERTS - MTP_EXPERTS
ONE_EXPERT_ALL = 18800640
TP = 4
ONE_EXPERT_RANK = ONE_EXPERT_ALL // TP
N_LAYERS = 40
N_EXPERTS = 384

# --- KV ---
KV_BYTES = 584
IDX_LOW = 68          # fp4 indexer for ratio-1/2 pools
PAGE = 128            # sglang page_size (multiple of 128, asserted in configurator)
KB = -(-PAGE * KV_BYTES // 576) * 576 // PAGE      # padded bytes/token for V4 layout
SWA_PER_SLOT = KB * N_LAYERS                        # bytes per SWA token slot
COMPRESSED = 3 * (KV_BYTES + IDX_LOW) // 2 + 1 * (KV_BYTES + IDX_LOW)   # ratio2 x3 + ratio1 x1

def gi(x): return x / GI
def mib(x): return x / MI

print("=== GPU budget (per rank) ===")
print("total              : %d MiB = %.3f GiB" % (GPU_MIB, gi(GPU_TOTAL)))
print("mem-fraction 0.88  : %.3f GiB/rank   (4 ranks: %.3f GiB)" % (gi(BUDGET), 4 * gi(BUDGET)))

print("\n=== weights ===")
print("checkpoint         : %.3f GiB" % gi(CKPT))
print("engram host tables : %.3f GiB  (host, pinned)" % gi(ENGRAM_HOST))
print("backbone experts   : %.3f GiB total  (%.3f GiB/rank)" % (gi(BB_EXPERTS), gi(BB_EXPERTS / TP)))
print("mtp/dspark experts : %.3f GiB total  (%.3f GiB/rank)" % (gi(MTP_EXPERTS), gi(MTP_EXPERTS / TP)))
print("rest (non-engram, non-expert): %.3f GiB total (%.3f GiB/rank TP-sharded)"
      % (gi(REST), gi(REST / TP)))
print("1 expert           : %.3f MiB all-ranks = %.3f MiB per rank" % (ONE_EXPERT_ALL / MI, ONE_EXPERT_RANK / MI))
SLOT_LAYER_RANK = ONE_EXPERT_RANK / GI          # GiB per slot per layer per rank
print("1 slot x 1 layer x 1 rank = %.6f GiB; x40 layers = %.4f GiB/rank"
      % (SLOT_LAYER_RANK, SLOT_LAYER_RANK * N_LAYERS))

W_FIX = gi(REST / TP) + 0.40     # + replicated small tensors (fused wqa_wkv etc.)
print("W_fix (non-expert, per rank, incl. 0.40 replicated) = %.3f GiB" % W_FIX)

def slab(S, layers=None):
    """pool slab GiB per rank: S slots x layers x per-rank expert bytes"""
    layers = N_LAYERS if layers is None else layers
    return S * ONE_EXPERT_RANK * layers / GI

def resident(k):
    """k GPU-resident layers (all 384 experts, sglang path) GiB/rank"""
    return k * ONE_EXPERT_RANK * N_EXPERTS / GI

print("\n=== pool slab per rank ===")
for S in (8, 16, 32, 64, 96, 128, 192, 224, 227, 256, 288, 384):
    print("  S=%-4d all-40-layers: %8.2f GiB/rank" % (S, slab(S)))

print("\n=== KV per token ===")
print("compressed (irreducible): ratio2 x3 + ratio1 x1 = %d B/full-token" % COMPRESSED)
for ratio in (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1):
    bpt = COMPRESSED + ratio * SWA_PER_SLOT
    print("  swa_full_tokens_ratio=%-6s -> %8.1f B/token ; 16x1M = %7.2f GiB ; 4x1M = %6.2f GiB ; 1x1M = %5.2f GiB"
          % (ratio, bpt, mib(16 * 1048576 * bpt) / 1024, mib(4 * 1048576 * bpt) / 1024, mib(1048576 * bpt) / 1024))

def tokens_for(kv_gib, ratio):
    bpt = COMPRESSED + ratio * SWA_PER_SLOT
    return kv_gib * GI / bpt

print("\n=== variant P: all 40 layers pooled, S=288 ===")
for ratio in (0.1, 0.005):
    W = slab(288) + W_FIX
    kvb = gi(BUDGET) - W
    print("  swa=%-5s W=%.2f GiB/rank -> KV budget %.2f GiB -> %.2f M tokens (%.1f x 1M)"
          % (ratio, W, kvb, tokens_for(kvb, ratio) / 1e6, tokens_for(kvb, ratio) / 1048576))

print("\n=== max S keeping 16 x 1M ===")
for ratio in (0.1, 0.02, 0.005, 0.0025):
    need = 16 * 1048576 * (COMPRESSED + ratio * SWA_PER_SLOT) / GI
    avail = gi(BUDGET) - W_FIX - need
    print("  swa=%-6s need KV=%.2f GiB -> slab budget %.2f GiB -> S_max=%.1f slots/layer"
          % (ratio, need, avail, avail / (SLOT_LAYER_RANK * N_LAYERS)))

print("\n=== variant R(k): k resident (384 exp) + (40-k) pooled at S=288 ===")
for ratio in (0.1, 0.005):
    print("  --- swa ratio = %s ---" % ratio)
    for k in (0, 4, 8, 12, 16, 20):
        W = resident(k) + slab(288, N_LAYERS - k) + W_FIX
        kvb = gi(BUDGET) - W
        n = tokens_for(kvb, ratio)
        print("    k=%-3d W=%.2f GiB/rank  KV=%.2f GiB -> %.2f M tokens (=%.1f x 1M, or %.0f x 100K)"
              % (k, W, kvb, n / 1e6, n / 1048576, n / 100e3))

print("\n=== variant S: no pool, k resident layers only ===")
for ratio in (0.1, 0.005):
    print("  --- swa ratio = %s ---" % ratio)
    for k in (0, 4, 8, 12, 16, 20, 23, 24, 32, 40):
        W = resident(k) + W_FIX
        kvb = gi(BUDGET) - W
        n = tokens_for(kvb, ratio)
        print("    k=%-3d W=%.2f GiB/rank  KV=%.2f GiB -> %.2f M tokens (=%.1f x 1M)"
              % (k, W, kvb, n / 1e6, n / 1048576))
    # max k for 16 x 1M
    need = 16 * 1048576 * (COMPRESSED + ratio * SWA_PER_SLOT) / GI
    kmax = (gi(BUDGET) - W_FIX - need) / (ONE_EXPERT_RANK * N_EXPERTS / GI)
    print("    max k for 16x1M: %.2f -> k=%d" % (kmax, int(kmax)))

print("\n=== host memory ===")
tot_experts = gi(BB_EXPERTS) + gi(MTP_EXPERTS)
print("all experts on host        : %.2f GiB" % tot_experts)
print("engram host tables         : %.2f GiB" % gi(ENGRAM_HOST))
print("sum (must-resident host)   : %.2f GiB = %.2f GB" % (tot_experts + gi(ENGRAM_HOST),
                                                             (tot_experts + gi(ENGRAM_HOST)) * GI / 1e9))
print("variant S k=23 host experts: %.2f GiB (17 layers) + engram %.2f = %.2f GiB"
      % (gi(BB_EXPERTS) * 17 / 40, gi(ENGRAM_HOST), gi(BB_EXPERTS) * 17 / 40 + gi(ENGRAM_HOST)))
print("NUMA node1 free 411410 MB = %.2f GiB ; node0 free 721858 MB = %.2f GiB ; total %.2f GiB"
      % (411410 / 1024, 721858 / 1024, (411410 + 721858) / 1024))
