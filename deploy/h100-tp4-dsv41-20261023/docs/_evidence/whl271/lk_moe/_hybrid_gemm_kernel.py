"""
copyright (c) 2025-2026 guqiong9696@gmail.com, All Rights Reserved.
GPU-side group-GEMM for the super-expert pool (prefill). Same pool slots as the
GEMV decode half; resident slots here, misses stay with the CPU side.
Segmentation: histogram -> exclusive scan -> stable gather. A tile never
crosses an expert segment, so its weight block loads once for M_TILE rows.

  slot layout: set[slot][col_blk][K/2][32], set_s[slot][col_blk][K/16][32]
  hbuf: [n_slots, 2*I] (gate half 0, up half 1); gout_part: [n_slots, H].
Addresses arrive as i64 (UVA allocations can sit above 2 GiB).
"""
import triton
import triton.language as tl
from ._pool_kernel_common import _act, _e2m1_bits, _e4m3, _e8m0  # noqa: F401


# Radix/bin sort over E experts; row_of is STABLE (slot order within a segment
# = token-major). The resident filter is NOT part of the order: the GEMM skips
# miss rows per-row so the CPU/GPU split stays identical.

@triton.jit
def moe_seg_count(ids_ptr, counts_ptr, n_slots, E: tl.constexpr, BLOCK: tl.constexpr):
    """One program per BLOCK slots: histogram of eids. counts pre-zeroed."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_slots
    eid = tl.load(ids_ptr + idx, mask=mask, other=-1)
    ok = mask & (eid >= 0)
    tl.atomic_add(counts_ptr + eid, 1, mask=ok)


@triton.jit
def moe_seg_scan(counts_ptr, counts_bak_ptr, offsets_ptr, tile_id_ptr, n_seg_ptr,
                 E: tl.constexpr, EP: tl.constexpr, M_TILE: tl.constexpr):
    """One program (mirrors bf16_compute_aligned_layout + cub::ExclusiveSum):
    segment start = PADDED prefix sum, so every segment is M_TILE-aligned and
    a tile NEVER crosses experts. tile_id[t] = owner of tile t (tiles split by
    padded offset), n_seg = total tiles. tile_id buffer needs n_slots/M_TILE+E.
    E need not be a power of two: EP (next power of two >= E) masks the arange;
    phantom lanes carry zero count and never claim a tile."""
    e = tl.arange(0, EP)
    ok = e < E
    cnt = tl.load(counts_ptr + e, mask=ok, other=0)
    tl.store(counts_bak_ptr + e, cnt, mask=ok)
    padded = ((cnt + M_TILE - 1) // M_TILE) * M_TILE
    off = tl.cumsum(padded, axis=0) - padded
    tl.store(offsets_ptr + e, off, mask=ok)
    nct = tl.where(ok, padded // M_TILE, 0)
    tstart = tl.cumsum(nct, axis=0) - nct
    ntiles = tl.sum(nct, axis=0)
    tl.store(n_seg_ptr, ntiles)
    t = tstart
    more = ntiles
    while more > 0:
        put = nct > 0
        tl.store(tile_id_ptr + t, e, mask=put & (t < ntiles) & ok)
        t = t + tl.where(put, 1, 0)
        nct = tl.where(put & ok, nct - 1, nct)
        more = tl.sum(nct, axis=0)


@triton.jit
def moe_seg_place(
    ids_ptr, offsets_ptr, cursor_ptr, row_of_ptr, n_slots, slot_base,
    BLOCK: tl.constexpr = 1024,
):
    """row_of[padded_off[eid] + rank] = slot + slot_base（pad 行保持 -1）。"""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(ids_ptr + idx, mask=idx < n_slots, other=-1)
    ok = (idx < n_slots) & (ids >= 0)
    offs = tl.load(offsets_ptr + ids, mask=ok, other=0)
    ranks = tl.atomic_add(cursor_ptr + ids, 1, mask=ok)
    tl.store(row_of_ptr + offs + ranks, idx + slot_base, mask=ok)



@triton.jit
def route_scan(snap_addr, out_ptr, c_slots, block_base,
               E: tl.constexpr, BLOCK: tl.constexpr):
    """ONE program: expert->position table from the frozen snapshot words.
    resident -> pool slot; non-resident -> C + block_base + e (staging row).
    rank = #non-resident experts below e (matches the copy's rank order)."""
    e = tl.arange(0, BLOCK)
    ok = e < E
    snap = tl.cast(snap_addr.to(tl.int64), tl.pointer_type(tl.int64))
    word = tl.load(snap + e, mask=ok, other=0, volatile=True)
    res = ok & (word != 0)
    slot = tl.where(res, tl.cast(word & 0xFFFFFFFF, tl.int32) - 1, 0)
    res = res & (slot < c_slots)
    # non-resident experts sit at their DIRECT row (C + base + e): a block
    # is a full-capacity E-row window, so staging can be moved with strided
    # 2D copies (no CPU rank packing).
    # MIRRORS PrefillStaging::staging_pos: pos = C + parity*E + eid; the
    # gemm kernels invert it with (pos - POOL_C) -> ring row.
    val = tl.where(~ok, -1, tl.where(res, slot, c_slots + block_base + e))
    tl.store(out_ptr + e, val, mask=ok)


@triton.jit
def moe_gateup_group_gemm(
    route_tab_ptr, row_of_ptr, tile_id_ptr,
    gate_w, gate_s, up_w, up_s, gateup_gds,
    st_gate_w, st_gate_s, st_up_w, st_up_s,
    hbuf_ptr, x_ptr,
    n_slots, topk, rows_per_token,
    GDS_COLS: tl.constexpr, N_TILE_ROWS_MAX: tl.constexpr,
    NCOL: tl.constexpr, K_PER_BLOCK: tl.constexpr, SCALE_GROUPS_BLK: tl.constexpr,
    SLOT_BYTES: tl.constexpr, SLOT_SCALE_BYTES: tl.constexpr, KCOL_STRIDE: tl.constexpr, HIDDEN: tl.constexpr,
    SCALE_GROUP: tl.constexpr, FORMAT: tl.constexpr, FORMAT_PACKED_K: tl.constexpr,
    HAS_GATE: tl.constexpr, ACTIVATION: tl.constexpr, ALPHA: tl.constexpr, LIMIT: tl.constexpr,
    M_TILE: tl.constexpr, SLOT_COUNT: tl.constexpr,
    POOL_C: tl.constexpr = 1 << 30,
    KB: tl.constexpr = 32, COLS: tl.constexpr = 1,
    SCALE_SIZE: tl.constexpr = 1,
):
    # K-outer, weights loaded once per (tile, col_blk): a fully-non-resident
    # tile returns before touching any weight byte. POOL_C splits the slot
    # space: [0,POOL_C) = this layer's pool, [POOL_C,..) = shared staging.
    tile_blk = tl.program_id(0)
    col_blk = tl.program_id(1)
    if col_blk >= NCOL // 32:
        return
    eid = tl.load(tile_id_ptr + tile_blk)
    if eid < 0:
        return
    pos = tl.load(route_tab_ptr + eid)
    if pos < 0:
        return
    m_ok = tile_blk * M_TILE + tl.arange(0, M_TILE)
    slots = tl.load(row_of_ptr + m_ok, mask=m_ok < N_TILE_ROWS_MAX, other=-1)
    live = (slots >= 0) & (slots < n_slots)
    if tl.sum(live.to(tl.int32)) == 0:
        return
    pool_slot = pos.to(tl.int64)
    slot64 = tl.where(live, slots, 0).to(tl.int64)
    tok_r = (slot64 // topk) * rows_per_token
    if FORMAT == "nvfp4":
        gds_p = tl.cast(gateup_gds.to(tl.int64), tl.pointer_type(tl.float32))
        gate_g = tl.load(gds_p + eid * GDS_COLS + 0)
        up_g = tl.load(gds_p + eid * GDS_COLS + (GDS_COLS - 1))
    else:
        gate_g = 1.0
        up_g = 1.0
    up_p = tl.cast(up_w.to(tl.int64), tl.pointer_type(tl.uint8))
    up_sp = tl.cast(up_s.to(tl.int64), tl.pointer_type(tl.uint8))
    gate_p = tl.cast(gate_w.to(tl.int64), tl.pointer_type(tl.uint8))
    gate_sp = tl.cast(gate_s.to(tl.int64), tl.pointer_type(tl.uint8))
    hb = tl.cast(hbuf_ptr, tl.pointer_type(tl.float32))
    col_blk = col_blk * COLS
    if POOL_C < (1 << 30):               # dual-domain: staging slots >= POOL_C
        in_st = pool_slot >= POOL_C
        off = tl.where(in_st, pool_slot - POOL_C, pool_slot) * SLOT_BYTES
        soff = tl.where(in_st, pool_slot - POOL_C, pool_slot) * SLOT_SCALE_BYTES
        up_col_p = tl.where(in_st,
            tl.cast(tl.cast(st_up_w.to(tl.int64), tl.pointer_type(tl.uint8)), tl.int64, bitcast=True),
            tl.cast(tl.cast(up_w.to(tl.int64), tl.pointer_type(tl.uint8)), tl.int64, bitcast=True)) + off
        up_col_p = up_col_p.to(tl.pointer_type(tl.uint8), bitcast=True) + \
                   col_blk * KCOL_STRIDE
        gate_col_p = tl.where(in_st,
            tl.cast(tl.cast(st_gate_w.to(tl.int64), tl.pointer_type(tl.uint8)), tl.int64, bitcast=True),
            tl.cast(tl.cast(gate_w.to(tl.int64), tl.pointer_type(tl.uint8)), tl.int64, bitcast=True)) + off
        gate_col_p = gate_col_p.to(tl.pointer_type(tl.uint8), bitcast=True) + \
                     col_blk * KCOL_STRIDE
        up_sc_p = tl.where(in_st,
            tl.cast(tl.cast(st_up_s.to(tl.int64), tl.pointer_type(tl.uint8)), tl.int64, bitcast=True),
            tl.cast(tl.cast(up_s.to(tl.int64), tl.pointer_type(tl.uint8)), tl.int64, bitcast=True)) + soff
        up_sc_p = up_sc_p.to(tl.pointer_type(tl.uint8), bitcast=True) + \
                  col_blk * (SCALE_GROUPS_BLK * 32 * SCALE_SIZE)
        gate_sc_p = tl.where(in_st,
            tl.cast(tl.cast(st_gate_s.to(tl.int64), tl.pointer_type(tl.uint8)), tl.int64, bitcast=True),
            tl.cast(tl.cast(gate_s.to(tl.int64), tl.pointer_type(tl.uint8)), tl.int64, bitcast=True)) + soff
        gate_sc_p = gate_sc_p.to(tl.pointer_type(tl.uint8), bitcast=True) + \
                    col_blk * (SCALE_GROUPS_BLK * 32 * SCALE_SIZE)
    else:
        up_col_p = up_p + pool_slot * SLOT_BYTES + col_blk * KCOL_STRIDE
        up_sc_p = up_sp + pool_slot * SLOT_SCALE_BYTES + col_blk * (SCALE_GROUPS_BLK * 32 * SCALE_SIZE)
        gate_col_p = gate_p + pool_slot * SLOT_BYTES + col_blk * KCOL_STRIDE
        gate_sc_p = gate_sp + pool_slot * SLOT_SCALE_BYTES + col_blk * (SCALE_GROUPS_BLK * 32 * SCALE_SIZE)
    col32 = tl.arange(0, 32)
    kb = tl.arange(0, KB)
    coff = tl.arange(0, COLS)
    NDIM: tl.constexpr = COLS * 32
    acc_gate = tl.zeros((M_TILE, NDIM), tl.float32)
    acc_up = tl.zeros((M_TILE, NDIM), tl.float32)
    for it in range(0 if FORMAT == "fp8" else (K_PER_BLOCK + KB - 1) // KB):
        kidx = it * KB + kb
        keep_k = kidx < K_PER_BLOCK
        sel = keep_k[:, None, None] & tl.full((1, COLS, 1), 1, tl.int1)
        sgrp = (FORMAT_PACKED_K * kidx) // SCALE_GROUP
        gsel = (sgrp < SCALE_GROUPS_BLK)[:, None, None] & tl.full((1, COLS, 1), 1, tl.int1)
        rmask = live[:, None] & keep_k[None, :]
        xe = tl.load(x_ptr + tok_r[:, None] * HIDDEN + FORMAT_PACKED_K * kidx[None, :],
                     mask=rmask, other=0.0).to(tl.bfloat16)
        xo = tl.load(x_ptr + tok_r[:, None] * HIDDEN + FORMAT_PACKED_K * kidx[None, :] + 1,
                     mask=rmask, other=0.0).to(tl.bfloat16)
        # fp16/bf16 MMA is exact here: e2m1*e4m3 products fit <=4 significant
        # bits, so the dot only reorders an exact fp32 sum. bf16 additionally
        # keeps |x| > 65504 finite (fp16 conversion would overflow to inf).
        xe = tl.where(live[:, None], xe, 0.0)
        xo = tl.where(live[:, None], xo, 0.0)
        up_sc_raw = tl.load(
            up_sc_p + coff[None, :, None] * (SCALE_GROUPS_BLK * 32)
            + sgrp[:, None, None] * 32 + col32[None, None, :],
            mask=gsel, other=0)
        if FORMAT == "nvfp4":
            up_sc16 = _e4m3(up_sc_raw)
        else:
            up_sc16 = _e8m0(up_sc_raw)
        up_sc16 = up_sc16.reshape(KB, NDIM).to(tl.bfloat16)
        up_byte = tl.load(up_col_p + coff[None, :, None] * KCOL_STRIDE
                          + kidx[:, None, None] * 32 + col32[None, None, :],
                          mask=sel, other=0).reshape(KB, NDIM)
        acc_up += tl.dot(xe, (_e2m1_bits(up_byte) * up_sc16).to(tl.bfloat16), out_dtype=tl.float32)
        acc_up += tl.dot(xo, (_e2m1_bits(up_byte >> 4) * up_sc16).to(tl.bfloat16), out_dtype=tl.float32)
        if HAS_GATE:
            g_sc_raw = tl.load(
                gate_sc_p + coff[None, :, None] * (SCALE_GROUPS_BLK * 32)
                + sgrp[:, None, None] * 32 + col32[None, None, :],
                mask=gsel, other=0)
            if FORMAT == "nvfp4":
                g_sc16 = _e4m3(g_sc_raw)
            else:
                g_sc16 = _e8m0(g_sc_raw)
            g_sc16 = g_sc16.reshape(KB, NDIM).to(tl.bfloat16)
            g_byte = tl.load(gate_col_p + coff[None, :, None] * KCOL_STRIDE
                             + kidx[:, None, None] * 32 + col32[None, None, :],
                             mask=sel, other=0).reshape(KB, NDIM)
            acc_gate += tl.dot(xe, (_e2m1_bits(g_byte) * g_sc16).to(tl.bfloat16), out_dtype=tl.float32)
            acc_gate += tl.dot(xo, (_e2m1_bits(g_byte >> 4) * g_sc16).to(tl.bfloat16), out_dtype=tl.float32)
    if FORMAT == "fp8":
        up_sc_f = tl.cast(up_sc_p, tl.pointer_type(tl.float32))
        gate_sc_f = tl.cast(gate_sc_p, tl.pointer_type(tl.float32))
        for it in range((K_PER_BLOCK + KB - 1) // KB):
            kidx = it * KB + kb
            keep_k = kidx < K_PER_BLOCK
            sel = keep_k[:, None, None] & tl.full((1, COLS, 1), 1, tl.int1)
            sgrp = kidx // SCALE_GROUP
            gsel = (sgrp < SCALE_GROUPS_BLK)[:, None, None] & tl.full((1, COLS, 1), 1, tl.int1)
            rmask = live[:, None] & keep_k[None, :]
            xv = tl.load(x_ptr + tok_r[:, None] * HIDDEN + kidx[None, :],
                         mask=rmask, other=0.0).to(tl.float32)
            xv = tl.where(live[:, None], xv, 0.0)
            up_sf = tl.load(up_sc_f + coff[None, :, None] * (SCALE_GROUPS_BLK * 32)
                            + sgrp[:, None, None] * 32 + col32[None, None, :],
                            mask=gsel, other=0).reshape(KB, NDIM)
            up_byte = tl.load(up_col_p + coff[None, :, None] * KCOL_STRIDE
                              + kidx[:, None, None] * 32 + col32[None, None, :],
                              mask=sel, other=0).reshape(KB, NDIM)
            u_val = _e4m3(up_byte).to(tl.float32) * up_sf
            u_hi = (u_val.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
            acc_up += tl.dot(xv, u_hi, out_dtype=tl.float32, input_precision="tf32")
            acc_up += tl.dot(xv, u_val - u_hi, out_dtype=tl.float32, input_precision="tf32")
            if HAS_GATE:
                gate_sf = tl.load(gate_sc_f + coff[None, :, None] * (SCALE_GROUPS_BLK * 32)
                                  + sgrp[:, None, None] * 32 + col32[None, None, :],
                                  mask=gsel, other=0).reshape(KB, NDIM)
                gate_byte = tl.load(gate_col_p + coff[None, :, None] * KCOL_STRIDE
                                    + kidx[:, None, None] * 32 + col32[None, None, :],
                                    mask=sel, other=0).reshape(KB, NDIM)
                g_val = _e4m3(gate_byte).to(tl.float32) * gate_sf
                g_hi = (g_val.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
                acc_gate += tl.dot(xv, g_hi, out_dtype=tl.float32, input_precision="tf32")
                acc_gate += tl.dot(xv, g_val - g_hi, out_dtype=tl.float32, input_precision="tf32")
    smask = live[:, None, None] & tl.full((1, COLS, 1), 1, tl.int1)
    srow = (hb + slot64[:, None, None] * (2 * NCOL)
            + (col_blk + coff)[None, :, None] * 32 + col32[None, None, :])
    if HAS_GATE:
        gv = acc_gate.reshape(M_TILE, COLS, 32) * gate_g
        uv = acc_up.reshape(M_TILE, COLS, 32) * up_g
        tl.store(srow, gv, mask=smask)
        tl.store(srow + NCOL, uv, mask=smask)
    else:
        uv = acc_up.reshape(M_TILE, COLS, 32) * up_g
        tl.store(srow + NCOL, uv, mask=smask)


@triton.jit
def moe_down_group_gemm(
    route_tab_ptr, weights_ptr, row_of_ptr, tile_id_ptr,
    down_w, down_s, down_gds,
    st_down_w, st_down_s,
    hbuf_ptr, gout_part_ptr,
    n_slots,
    GDS_COLS: tl.constexpr, N_TILE_ROWS_MAX: tl.constexpr,
    NCOL: tl.constexpr, N_PARTIAL: tl.constexpr, K_PER_BLOCK: tl.constexpr,
    SCALE_GROUPS_BLK: tl.constexpr,
    SLOT_BYTES: tl.constexpr, SLOT_SCALE_BYTES: tl.constexpr, KCOL_STRIDE: tl.constexpr,
    SCALE_GROUP: tl.constexpr, FORMAT: tl.constexpr, FORMAT_PACKED_K: tl.constexpr,
    HAS_GATE: tl.constexpr, ACTIVATION: tl.constexpr, ALPHA: tl.constexpr, LIMIT: tl.constexpr,
    M_TILE: tl.constexpr, SLOT_COUNT: tl.constexpr,
    POOL_C: tl.constexpr = 1 << 30,
    KB: tl.constexpr = 32, COLS: tl.constexpr = 1,
    SCALE_SIZE: tl.constexpr = 1,
):
    # K-outer, weights loaded ONCE per (tile, col_blk) (see gateup above).
    tile_blk = tl.program_id(0)
    col_blk = tl.program_id(1)
    if col_blk >= NCOL // 32:
        return
    eid = tl.load(tile_id_ptr + tile_blk)
    if eid < 0:
        return
    pos = tl.load(route_tab_ptr + eid)
    if pos < 0:
        return
    m_ok = tile_blk * M_TILE + tl.arange(0, M_TILE)
    slots = tl.load(row_of_ptr + m_ok, mask=m_ok < N_TILE_ROWS_MAX, other=-1)
    live = (slots >= 0) & (slots < n_slots)
    if tl.sum(live.to(tl.int32)) == 0:
        return
    pool_slot = pos.to(tl.int64)
    slot64 = tl.where(live, slots, 0).to(tl.int64)
    rw = tl.load(weights_ptr + slots, mask=live, other=0.0)
    if FORMAT == "nvfp4":
        gds_p = tl.cast(down_gds.to(tl.int64), tl.pointer_type(tl.float32))
        down_g = tl.load(gds_p + eid)
    else:
        down_g = 1.0
    down_p = tl.cast(down_w.to(tl.int64), tl.pointer_type(tl.uint8))
    down_sp = tl.cast(down_s.to(tl.int64), tl.pointer_type(tl.uint8))
    hb = tl.cast(hbuf_ptr, tl.pointer_type(tl.float32))
    gout_part_p = tl.cast(gout_part_ptr, tl.pointer_type(tl.float32))
    col_blk = col_blk * COLS
    if POOL_C < (1 << 30):               # dual-domain: staging slots >= POOL_C
        in_st = pool_slot >= POOL_C
        off = tl.where(in_st, pool_slot - POOL_C, pool_slot) * SLOT_BYTES
        soff = tl.where(in_st, pool_slot - POOL_C, pool_slot) * SLOT_SCALE_BYTES
        down_col_p = tl.where(in_st,
            tl.cast(tl.cast(st_down_w.to(tl.int64), tl.pointer_type(tl.uint8)), tl.int64, bitcast=True),
            tl.cast(tl.cast(down_w.to(tl.int64), tl.pointer_type(tl.uint8)), tl.int64, bitcast=True)) + off
        down_col_p = down_col_p.to(tl.pointer_type(tl.uint8), bitcast=True) + \
                     col_blk * KCOL_STRIDE
        down_sc_p = tl.where(in_st,
            tl.cast(tl.cast(st_down_s.to(tl.int64), tl.pointer_type(tl.uint8)), tl.int64, bitcast=True),
            tl.cast(tl.cast(down_s.to(tl.int64), tl.pointer_type(tl.uint8)), tl.int64, bitcast=True)) + soff
        down_sc_p = down_sc_p.to(tl.pointer_type(tl.uint8), bitcast=True) + \
                    col_blk * (SCALE_GROUPS_BLK * 32 * SCALE_SIZE)
    else:
        down_col_p = down_p + pool_slot * SLOT_BYTES + col_blk * KCOL_STRIDE
        down_sc_p = down_sp + pool_slot * SLOT_SCALE_BYTES + col_blk * (SCALE_GROUPS_BLK * 32 * SCALE_SIZE)
    col32 = tl.arange(0, 32)
    kb = tl.arange(0, KB)
    coff = tl.arange(0, COLS)
    NDIM: tl.constexpr = COLS * 32
    hrow = hb + slot64[:, None] * (2 * N_PARTIAL)
    acc = tl.zeros((M_TILE, NDIM), tl.float32)
    for it in range(0 if FORMAT == "fp8" else (K_PER_BLOCK + KB - 1) // KB):
        kidx = it * KB + kb
        keep_k = kidx < K_PER_BLOCK
        sel = keep_k[:, None, None] & tl.full((1, COLS, 1), 1, tl.int1)
        sgrp = (FORMAT_PACKED_K * kidx) // SCALE_GROUP
        gsel = (sgrp < SCALE_GROUPS_BLK)[:, None, None] & tl.full((1, COLS, 1), 1, tl.int1)
        kmask = live[:, None] & keep_k[None, :]
        ue = tl.load(hrow + N_PARTIAL + FORMAT_PACKED_K * kidx[None, :], mask=kmask, other=0).to(tl.float32)
        uo = tl.load(hrow + N_PARTIAL + FORMAT_PACKED_K * kidx[None, :] + 1, mask=kmask, other=0).to(tl.float32)
        if ACTIVATION == 2:
            ae = _act(ue, ue, 2, ALPHA, LIMIT)
            ao = _act(uo, uo, 2, ALPHA, LIMIT)
        elif HAS_GATE:
            ge = tl.load(hrow + FORMAT_PACKED_K * kidx[None, :], mask=kmask, other=0).to(tl.float32)
            go = tl.load(hrow + FORMAT_PACKED_K * kidx[None, :] + 1, mask=kmask, other=0).to(tl.float32)
            ae = _act(ge, ue, ACTIVATION, ALPHA, LIMIT)
            ao = _act(go, uo, ACTIVATION, ALPHA, LIMIT)
        else:
            ae = _act(ue, ue, ACTIVATION, ALPHA, LIMIT)
            ao = _act(uo, uo, ACTIVATION, ALPHA, LIMIT)
        d_s_raw = tl.load(down_sc_p + coff[None, :, None] * (SCALE_GROUPS_BLK * 32)
                          + sgrp[:, None, None] * 32 + col32[None, None, :],
                          mask=gsel, other=0)
        if FORMAT == "nvfp4":
            d_s = _e4m3(d_s_raw)
        else:
            d_s = _e8m0(d_s_raw)
        d_s = d_s.reshape(KB, NDIM).to(tl.bfloat16) * down_g.to(tl.bfloat16)
        d_byte = tl.load(down_col_p + coff[None, :, None] * KCOL_STRIDE
                         + kidx[:, None, None] * 32 + col32[None, None, :],
                         mask=sel, other=0).reshape(KB, NDIM)
        acc += tl.dot(ae.to(tl.bfloat16), _e2m1_bits(d_byte).to(tl.bfloat16) * d_s, out_dtype=tl.float32)
        acc += tl.dot(ao.to(tl.bfloat16), _e2m1_bits(d_byte >> 4).to(tl.bfloat16) * d_s, out_dtype=tl.float32)
    if FORMAT == "fp8":
        down_sc_f = tl.cast(down_sc_p, tl.pointer_type(tl.float32))
        for it in range((K_PER_BLOCK + KB - 1) // KB):
            kidx = it * KB + kb
            keep_k = kidx < K_PER_BLOCK
            sel = keep_k[:, None, None] & tl.full((1, COLS, 1), 1, tl.int1)
            sgrp = kidx // SCALE_GROUP
            gsel = (sgrp < SCALE_GROUPS_BLK)[:, None, None] & tl.full((1, COLS, 1), 1, tl.int1)
            kmask = live[:, None] & keep_k[None, :]
            u = tl.load(hrow + N_PARTIAL + kidx[None, :], mask=kmask, other=0).to(tl.float32)
            if ACTIVATION == 2:
                act8 = _act(u, u, 2, ALPHA, LIMIT)
            elif HAS_GATE:
                g8 = tl.load(hrow + kidx[None, :], mask=kmask, other=0).to(tl.float32)
                act8 = _act(g8, u, ACTIVATION, ALPHA, LIMIT)
            else:
                act8 = _act(u, u, ACTIVATION, ALPHA, LIMIT)
            d_sf = tl.load(down_sc_f + coff[None, :, None] * (SCALE_GROUPS_BLK * 32)
                           + sgrp[:, None, None] * 32 + col32[None, None, :],
                           mask=gsel, other=0).reshape(KB, NDIM) * down_g
            d_byte = tl.load(down_col_p + coff[None, :, None] * KCOL_STRIDE
                             + kidx[:, None, None] * 32 + col32[None, None, :],
                             mask=sel, other=0).reshape(KB, NDIM)
            d_val = _e4m3(d_byte).to(tl.float32) * d_sf
            d_hi = (d_val.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
            acc += tl.dot(act8, d_hi, out_dtype=tl.float32, input_precision="tf32")
            acc += tl.dot(act8, d_val - d_hi, out_dtype=tl.float32, input_precision="tf32")
    tl.store(gout_part_p + slot64[:, None, None] * NCOL + (col_blk + coff)[None, :, None] * 32
             + col32[None, None, :],
             (acc * rw[:, None]).reshape(M_TILE, COLS, 32),
             mask=live[:, None, None] & tl.full((1, COLS, 1), 1, tl.int1))
