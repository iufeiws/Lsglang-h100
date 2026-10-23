"""
copyright (c) 2025-2026 guqiong9696@gmail.com, All Rights Reserved.
GPU-side GEMV for the super-expert pool: computes the routed experts whose
slot is resident+landed this step; the CPU side skips exactly those (same
frozen snapshot word, see csrc/cuda/moe_v2_expert_pool.cuh). CUDA-graph safe:
every program reads the LIVE topk_ids and the mapped snapshot.
Quant formats / activations / slot layout: single source is
csrc/cuda/moe_v2_gpu_pool_kernels.cuh. Addresses arrive as i64 (UVA > 2 GiB).
"""
import triton
import triton.language as tl
try:
    from ._pool_kernel_common import _e2m1_bits, _e4m3, _e8m0, _act  # noqa: F401
except ImportError:
    # Run as a script: import via package, a 2nd module copy breaks @triton.jit.
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from lk_moe._pool_kernel_common import _e2m1_bits, _e4m3, _act  # noqa: F401


@triton.jit
def slot_table_fill(ids_ptr, snapshot_addr, slot_tab_ptr, k, BLOCK: tl.constexpr):
    """ONE program per launch: gather the mapped-snapshot word into a device
    table. The word is the SINGLE frozen split both halves read: C++ publishes
    it only after the slot's owner/claim and H2D landing were validated, and
    the step's pumps lease every routed resident slot, so no live owner/land
    re-check belongs here (that re-check at a different instant is exactly the
    CPU/GPU skew this contract removes). Zero -> -1 (CPU computes it)."""
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = col < k
    eid = tl.load(ids_ptr + col, mask=mask, other=0)
    snap_p = tl.cast(snapshot_addr.to(tl.int64), tl.pointer_type(tl.int64))
    word = tl.load(snap_p + eid, mask=mask & (eid >= 0), other=0, volatile=True)
    slot = tl.where(mask & (word != 0), tl.cast(word & 0xFFFFFFFF, tl.int32) - 1, -1)
    tl.store(slot_tab_ptr + col, slot, mask=mask)


@triton.jit
def gather_rows(ids_ptr, weights_ptr, gathered_ids_ptr, gathered_weights_ptr, k, rows_per_token, n_slots,
                BLOCK: tl.constexpr):
    """Pack the real tokens' ids/weights into a contiguous [real_tokens*k] view.
    Not needed when rows_per_token == 1 (plain decode)."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_slots
    tok = idx // k
    src_row = tok * (rows_per_token * k) + (idx - tok * k)
    tl.store(gathered_ids_ptr + idx, tl.load(ids_ptr + src_row, mask=mask, other=0), mask=mask)
    tl.store(gathered_weights_ptr + idx, tl.load(weights_ptr + src_row, mask=mask, other=0.0), mask=mask)


# gate/up phase, split-K: grid (n_slots, NB_GATEUP, NGK). Each program plain-
# stores its K-chunk partial to hpart[k_chunk, slot, col_blk*32 + {gate|up}];
# reduce_gate folds the NGK partials into hbuf (silu deferred to down phase).
# Split-K lifts per-layer parallelism ~100 -> ~800 programs to fill the SMs.
@triton.jit
def moe_gateup(
    ids_ptr, slot_tab_ptr,
    gate_w, gate_s, up_w, up_s, gateup_gds,
    hpart_ptr, x_ptr,
    n_slots, topk, rows_per_token,
    GDS_COLS: tl.constexpr,
    NCOL: tl.constexpr, K_PER_BLOCK: tl.constexpr, SCALE_GROUPS_BLK: tl.constexpr,
    SLOT_BYTES: tl.constexpr, SLOT_SCALE_BYTES: tl.constexpr, HIDDEN: tl.constexpr,
    ITERS: tl.constexpr, SCALE_GROUP: tl.constexpr, FORMAT: tl.constexpr, FORMAT_PACKED_K: tl.constexpr,
    HAS_GATE: tl.constexpr, ACTIVATION: tl.constexpr,
    ALPHA: tl.constexpr, LIMIT: tl.constexpr,
    KS: tl.constexpr, NGK: tl.constexpr,
    SCALE_SIZE: tl.constexpr = 1,
):
    slot = tl.program_id(0)
    col_blk = tl.program_id(1)
    k_chunk = tl.program_id(2)
    if slot >= n_slots:
        return
    eid = tl.load(ids_ptr + slot)
    if eid < 0:
        return
    pool_slot = tl.load(slot_tab_ptr + slot)
    if pool_slot < 0:
        return
    x_row_p = x_ptr + ((slot // topk) * rows_per_token).to(tl.int64) * HIDDEN

    gate_p = tl.cast(gate_w.to(tl.int64), tl.pointer_type(tl.uint8))
    gate_scale_p = tl.cast(gate_s.to(tl.int64), tl.pointer_type(tl.uint8))
    up_p = tl.cast(up_w.to(tl.int64), tl.pointer_type(tl.uint8))
    up_scale_p = tl.cast(up_s.to(tl.int64), tl.pointer_type(tl.uint8))

    gate_col_p = gate_p + pool_slot.to(tl.int64) * SLOT_BYTES + col_blk * (K_PER_BLOCK * 32)
    up_col_p = up_p + pool_slot.to(tl.int64) * SLOT_BYTES + col_blk * (K_PER_BLOCK * 32)
    gate_scol_p = gate_scale_p + pool_slot.to(tl.int64) * SLOT_SCALE_BYTES + col_blk * (SCALE_GROUPS_BLK * 32 * SCALE_SIZE)
    up_scol_p = up_scale_p + pool_slot.to(tl.int64) * SLOT_SCALE_BYTES + col_blk * (SCALE_GROUPS_BLK * 32 * SCALE_SIZE)

    if FORMAT == "nvfp4":
        gateup_gds_p = tl.cast(gateup_gds.to(tl.int64), tl.pointer_type(tl.float32))
        gate_gscale = tl.load(gateup_gds_p + eid * GDS_COLS + 0)
        up_gscale = tl.load(gateup_gds_p + eid * GDS_COLS + (GDS_COLS - 1))
    else:
        gate_gscale = 1.0
        up_gscale = 1.0

    col32 = tl.arange(0, 32)
    k_off = tl.arange(0, ITERS)
    acc_gate = tl.zeros((32,), dtype=tl.float32)
    acc_up = tl.zeros((32,), dtype=tl.float32)
    for it in range(KS):
        cols = (k_chunk * KS + it) * ITERS + k_off
        scale_grp = (FORMAT_PACKED_K * cols) // SCALE_GROUP
        up_byte = tl.load(up_col_p + cols[:, None] * 32 + col32[None, :])
        up_scale_raw = tl.load(up_scol_p + scale_grp[:, None] * 32 + col32[None, :])
        if FORMAT == "nvfp4":
            up_scale = _e4m3(up_scale_raw) * up_gscale
        else:
            up_scale = _e8m0(up_scale_raw)
        if FORMAT == "nvfp4" or FORMAT == "mxfp4":
            k_elem = FORMAT_PACKED_K * cols
            x_even = tl.load(x_row_p + k_elem[:, None]).to(tl.float32)
            x_odd = tl.load(x_row_p + k_elem[:, None] + 1).to(tl.float32)
            acc_up += tl.sum((_e2m1_bits(up_byte).to(tl.float32) * x_even
                             + _e2m1_bits(up_byte >> 4).to(tl.float32) * x_odd) * up_scale, axis=0)
            if HAS_GATE:
                gate_byte = tl.load(gate_col_p + cols[:, None] * 32 + col32[None, :])
                gate_scale_raw = tl.load(gate_scol_p + scale_grp[:, None] * 32 + col32[None, :])
                if FORMAT == "nvfp4":
                    gate_scale = _e4m3(gate_scale_raw) * gate_gscale
                else:
                    gate_scale = _e8m0(gate_scale_raw)
                acc_gate += tl.sum((_e2m1_bits(gate_byte).to(tl.float32) * x_even
                                 + _e2m1_bits(gate_byte >> 4).to(tl.float32) * x_odd) * gate_scale, axis=0)

    if FORMAT == "fp8":
        up_sc_f = tl.cast(up_scol_p, tl.pointer_type(tl.float32))
        gate_sc_f = tl.cast(gate_scol_p, tl.pointer_type(tl.float32))
        for it in range(KS):
            cols = (k_chunk * KS + it) * ITERS + k_off
            scale_grp = (FORMAT_PACKED_K * cols) // SCALE_GROUP
            up_byte = tl.load(up_col_p + cols[:, None] * 32 + col32[None, :])
            up_scale = tl.load(up_sc_f + scale_grp[:, None] * 32 + col32[None, :])
            xv = tl.load(x_row_p + (FORMAT_PACKED_K * cols)[:, None]).to(tl.float32)
            acc_up += tl.sum(_e4m3(up_byte).to(tl.float32) * xv * up_scale, axis=0)
            if HAS_GATE:
                gate_byte = tl.load(gate_col_p + cols[:, None] * 32 + col32[None, :])
                gate_scale = tl.load(gate_sc_f + scale_grp[:, None] * 32 + col32[None, :])
                acc_gate += tl.sum(_e4m3(gate_byte).to(tl.float32) * xv * gate_scale, axis=0)

    hpart_p = tl.cast(hpart_ptr, tl.pointer_type(tl.float32))
    base = k_chunk.to(tl.int64) * (n_slots * 2 * NCOL) + slot.to(tl.int64) * (2 * NCOL) + col_blk * 32 + col32
    if HAS_GATE:
        tl.store(hpart_p + base, acc_gate)
    tl.store(hpart_p + base + NCOL, acc_up)


@triton.jit
def reduce_gate(hpart_ptr, hbuf_ptr, total, NGK: tl.constexpr, BLOCK: tl.constexpr):
    # hbuf[j] = sum over k_chunks of hpart[k_chunk, j], j in [0, total=n_slots*2*NCOL)
    hpart_p = tl.cast(hpart_ptr, tl.pointer_type(tl.float32))
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k_chunk in range(NGK):
        acc += tl.load(hpart_p + k_chunk.to(tl.int64) * total + idx, mask=mask, other=0.0)
    tl.store(hbuf_ptr + idx, acc, mask=mask)


# down phase, no atomics: grid (n_slots, NB_DOWN). Each program folds the slot's
# activation (silu(gate)*up) with the down weight + routing weight and plain-
# stores into gout_part[slot, col_blk*32+col]; reduce_down sums each token's
# topk slots into gout. Plain stores beat the fp32 atomic add (faster, deterministic).
@triton.jit
def moe_down(
    ids_ptr, slot_tab_ptr, weights_ptr,
    down_w, down_s, down_gds,
    hbuf_ptr, gout_part_ptr,
    n_slots,
    NCOL: tl.constexpr, N_PARTIAL: tl.constexpr, K_PER_BLOCK: tl.constexpr,
    SCALE_GROUPS_BLK: tl.constexpr, SLOT_BYTES: tl.constexpr, SLOT_SCALE_BYTES: tl.constexpr,
    ITERS: tl.constexpr, SCALE_GROUP: tl.constexpr, FORMAT: tl.constexpr, FORMAT_PACKED_K: tl.constexpr,
    HAS_GATE: tl.constexpr, ACTIVATION: tl.constexpr,
    ALPHA: tl.constexpr, LIMIT: tl.constexpr,
    SCALE_SIZE: tl.constexpr = 1,
):
    slot = tl.program_id(0)
    col_blk = tl.program_id(1)
    if slot >= n_slots:
        return
    eid = tl.load(ids_ptr + slot)
    if eid < 0:
        return
    pool_slot = tl.load(slot_tab_ptr + slot)
    if pool_slot < 0:
        return

    down_p = tl.cast(down_w.to(tl.int64), tl.pointer_type(tl.uint8))
    down_scale_p = tl.cast(down_s.to(tl.int64), tl.pointer_type(tl.uint8))
    down_col_p = down_p + pool_slot.to(tl.int64) * SLOT_BYTES + col_blk * (K_PER_BLOCK * 32)
    down_scol_p = down_scale_p + pool_slot.to(tl.int64) * SLOT_SCALE_BYTES + col_blk * (SCALE_GROUPS_BLK * 32 * SCALE_SIZE)
    if FORMAT == "nvfp4":
        down_gds_p = tl.cast(down_gds.to(tl.int64), tl.pointer_type(tl.float32))
        down_gscale = tl.load(down_gds_p + eid)
    else:
        down_gscale = 1.0

    col32 = tl.arange(0, 32)
    k_off = tl.arange(0, ITERS)
    acc = tl.zeros((32,), dtype=tl.float32)
    hbuf_slot_p = hbuf_ptr + slot.to(tl.int64) * (2 * N_PARTIAL)
    route_w = tl.load(weights_ptr + slot)
    for it in range(K_PER_BLOCK // ITERS):
        cols = it * ITERS + k_off
        scale_grp = (FORMAT_PACKED_K * cols) // SCALE_GROUP
        down_byte = tl.load(down_col_p + cols[:, None] * 32 + col32[None, :])
        down_scale_raw = tl.load(down_scol_p + scale_grp[:, None] * 32 + col32[None, :])
        if FORMAT == "nvfp4":
            down_scale = _e4m3(down_scale_raw) * down_gscale
        else:
            down_scale = _e8m0(down_scale_raw)
        if FORMAT == "nvfp4" or FORMAT == "mxfp4":
            k_elem = FORMAT_PACKED_K * cols
            u_even = tl.load(hbuf_slot_p + N_PARTIAL + k_elem)
            u_odd = tl.load(hbuf_slot_p + N_PARTIAL + k_elem + 1)
            if HAS_GATE:
                g_even = tl.load(hbuf_slot_p + k_elem)
                g_odd = tl.load(hbuf_slot_p + k_elem + 1)
                act_even = _act(g_even, u_even, ACTIVATION, ALPHA, LIMIT)
                act_odd = _act(g_odd, u_odd, ACTIVATION, ALPHA, LIMIT)
            else:
                act_even = _act(u_even, u_even, ACTIVATION, ALPHA, LIMIT)
                act_odd = _act(u_odd, u_odd, ACTIVATION, ALPHA, LIMIT)
            prod = _e2m1_bits(down_byte) * act_even[:, None] + _e2m1_bits(down_byte >> 4) * act_odd[:, None]
            acc += tl.sum(prod.to(tl.float32) * down_scale, axis=0)
    if FORMAT == "fp8":
        down_sc_f = tl.cast(down_scol_p, tl.pointer_type(tl.float32))
        for it in range(K_PER_BLOCK // ITERS):
            cols = it * ITERS + k_off
            scale_grp = (FORMAT_PACKED_K * cols) // SCALE_GROUP
            down_byte = tl.load(down_col_p + cols[:, None] * 32 + col32[None, :])
            dsc = tl.load(down_sc_f + scale_grp[:, None] * 32 + col32[None, :])
            k_idx = (FORMAT_PACKED_K * cols)[:, None]
            u = tl.load(hbuf_slot_p + N_PARTIAL + k_idx)
            if HAS_GATE:
                g = tl.load(hbuf_slot_p + k_idx)
                act = _act(g, u, ACTIVATION, ALPHA, LIMIT)
            else:
                act = _act(u, u, ACTIVATION, ALPHA, LIMIT)
            acc += tl.sum(_e4m3(down_byte).to(tl.float32) * act * dsc, axis=0)

    gout_part_p = tl.cast(gout_part_ptr, tl.pointer_type(tl.float32))
    tl.store(gout_part_p + slot.to(tl.int64) * NCOL + col_blk * 32 + col32, acc * route_w)


@triton.jit
def reduce_down(gout_part_ptr, gout_ptr, slot_tab_ptr, HIDDEN: tl.constexpr,
                SLOTS_PER_TOKEN: tl.constexpr,
                BATCH: tl.constexpr, BLOCK: tl.constexpr,
                RESIDENT_ONLY: tl.constexpr = 0, TOKS: tl.constexpr = 0,
                NTOK=0):
    # RESIDENT_ONLY: skip slots the pool snapshot does not own (paired-skip with
    # the CPU side on the same rows). TOKS = tokens/program (grid axis 1) keeps
    # big-batch prefills off a single-program token loop. Slots are token-major.
    # On the TOKS (prefill bucket) path BATCH is a fixed unused constexpr and
    # NTOK carries the true token count as a RUNTIME scalar: the exact prefill
    # M must stay out of the constexpr key, or every new batch size recompiles
    # this kernel mid-serving (triton late-load = compile stall + OOM risk).
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < HIDDEN
    toks = BATCH if TOKS == 0 else TOKS
    bound = BATCH if TOKS == 0 else NTOK
    for i in range(toks):
        tok = tl.program_id(1) * toks + i
        if tok < bound:
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for slot_in_tok in range(SLOTS_PER_TOKEN):
                slot = tok * SLOTS_PER_TOKEN + slot_in_tok
                if RESIDENT_ONLY == 0:
                    live = True
                else:
                    live = tl.load(slot_tab_ptr + slot) >= 0
                if live:
                    acc += tl.load(gout_part_ptr + slot.to(tl.int64) * HIDDEN + idx,
                                   mask=mask, other=0.0)
            tl.store(gout_ptr + tok.to(tl.int64) * HIDDEN + idx, acc, mask=mask)
