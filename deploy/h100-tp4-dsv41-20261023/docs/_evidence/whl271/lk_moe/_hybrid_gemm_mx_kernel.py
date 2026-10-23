"""
copyright (c) 2025-2026 guqiong9696@gmail.com, All Rights Reserved.
MOE V2 - Expert-pool GPU-side GEMM for the PREFILL half (M >= 16).

Blackwell mxfp4 tensor-core path (triton_kernels-style): the e2m1 nibbles and
their e4m3 block scales feed tl.dot_scaled with NO dequant; x rows are
quantized to e4m3 with per-16k power-of-2 scales. x is pre-quantized by
moe_quant_x (gateup) / by the activation epilogue path (down re-quantizes
hbuf on the fly, 16-element groups).

Pool slot layout matches _hybrid_gemv_kernel (set[slot][col_blk][K/2][32] and
set_s[slot][col_blk][K/16][16][32]); a program owns M_TILE token-major slots
of ONE expert (slot_blk = token block when M_TILE divides topk) and one
32-column panel. The per-16-group scale interleave [g][16][32] is transposed
in-kernel to [32][g][16] for dot_scaled.

  hbuf: [n_slots, 2*I] (gate half 0, up half 1)   gout_part: [n_slots, H]
Addresses arrive as i64 (UVA allocations can sit above 2 GiB).
"""
import triton
import triton.language as tl
from ._pool_kernel_common import _act


@triton.jit
def _quant_mx(x, BG: tl.constexpr):
    """bf16/fp32 [R, BG*16] -> (fp8 bytes [R, BG*16], pow2 scales [R, BG] uint8)."""
    xg = tl.reshape(x, (x.shape[0], BG, 16))
    amax = tl.max(tl.abs(xg.to(tl.float32)), axis=2, keep_dims=True)
    deq = (amax.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    deq_f = deq.to(tl.float32, bitcast=True)
    q = tl.where(deq_f == 0, 0.0, 1.0 / deq_f)
    xq = (xg * q).to(tl.float8e4nv).to(tl.uint8, bitcast=True).reshape(x.shape[0], BG * 16)
    xs = (deq >> 23).reshape(x.shape[0], BG).to(tl.uint8)
    return xq, xs


@triton.jit
def moe_quant_x(x_ptr, xq_ptr, xs_ptr, n_rows, K: tl.constexpr, BT: tl.constexpr):
    row_blk = tl.program_id(0)
    rows = row_blk * BT + tl.arange(0, BT)
    rmask = rows < n_rows
    cols = tl.arange(0, K)
    x = tl.load(x_ptr + rows.to(tl.int64)[:, None] * K + cols[None, :],
                mask=rmask[:, None], other=0.0)
    xq, xs = _quant_mx(x, K // 16)
    tl.store(xq_ptr + rows.to(tl.int64)[:, None] * K + cols[None, :], xq, mask=rmask[:, None])
    gcols = tl.arange(0, K // 16)
    tl.store(xs_ptr + rows.to(tl.int64)[:, None] * (K // 16) + gcols[None, :], xs, mask=rmask[:, None])


@triton.jit
def _scales_t3(ws_u8, gcols, col32):
    # pool interleave [g][16][32] -> [32][g][16] (dim-labeled, reshape-free)
    b16 = tl.arange(0, 16)
    ws = tl.load(ws_u8 + (gcols[:, None, None] * 16 + b16[None, :, None]) * 32
                 + col32[None, None, :])
    return tl.trans(ws, (1, 2, 0))


@triton.jit
def moe_gateup_gemm_mx(
    ids_ptr, slot_tab_ptr,
    gate_w, gate_s, up_w, up_s, gateup_gds,
    xq_ptr, xs_ptr, hbuf_ptr,
    n_slots, topk, rows_per_token,
    GDS_COLS: tl.constexpr,
    NCOL: tl.constexpr, K_PER_BLOCK: tl.constexpr, SLOT_BYTES: tl.constexpr,
    SLOT_SCALE_BYTES: tl.constexpr, KCOL_STRIDE: tl.constexpr,
    HAS_GATE: tl.constexpr, ACTIVATION: tl.constexpr, ALPHA: tl.constexpr, LIMIT: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
):
    tok_blk = tl.program_id(0)
    col_blk = tl.program_id(1)
    slots = tok_blk * BT * topk + tl.arange(0, BT) * topk
    smask = slots < n_slots
    eid0 = tl.load(ids_ptr + slots, mask=smask, other=-1)
    if tl.max(tl.sum(tl.where(eid0 != eid0[0] if BT > 1 else 0, 1, 0), axis=0), axis=0) if False else False:
        return
    if tl.max(eid0, axis=0) <= 0:
        return
    ps = tl.load(slot_tab_ptr + slots, mask=smask, other=-1)
    if tl.min(tl.where(smask, ps, 2147483647), axis=0) < 0:
        return
    tok0 = (tok_blk * BT) * rows_per_token
    rows = tok0 + tl.arange(0, BT)

    gate_p = tl.cast(gate_w.to(tl.int64), tl.pointer_type(tl.uint8))
    gate_sp = tl.cast(gate_s.to(tl.int64), tl.pointer_type(tl.uint8))
    up_p = tl.cast(up_w.to(tl.int64), tl.pointer_type(tl.uint8))
    up_sp = tl.cast(up_s.to(tl.int64), tl.pointer_type(tl.uint8))
    gds_p = tl.cast(gateup_gds.to(tl.int64), tl.pointer_type(tl.float32))
    up_g = tl.load(gds_p + eid0[0] * GDS_COLS + (GDS_COLS - 1), mask=True) if False else 1.0

    col32 = tl.arange(0, 32)
    kb2 = tl.arange(0, BK // 2)
    kg = tl.arange(0, BK // 16)
    xq_p = tl.cast(xq_ptr, tl.pointer_type(tl.uint8)) + rows.to(tl.int64)[:, None] * (2 * KCOL_STRIDE // 2)
    xq_cols = tl.arange(0, BK)
    xs_p = tl.cast(xs_ptr, tl.pointer_type(tl.uint8)) + rows.to(tl.int64)[:, None] * (K_PER_BLOCK * 2 // 16)
    xg_cols = tl.arange(0, BK // 16)

    acc = tl.zeros((BT, 32), dtype=tl.float32)
    for it in range(K_PER_BLOCK // BK):
        col0 = it * BK
        xq = tl.load(xq_p + (2 * col0 + xq_cols)[None, :])
        xs = tl.load(xs_p + ((col0 + kg * 8) // 8 * 8 // 8 + kg)[None, :])
        # scale cols: global group index = col0//16 + kg*?  (BK groups: BK//16)
        up_byte = tl.load(up_p + ps[0].to(tl.int64) * SLOT_BYTES + col_blk * KCOL_STRIDE
                          + (col0 // 2 + kb2)[:, None] * 32 + col32[None, :])
        up_wT = tl.trans(up_byte)
        up_sT = _scales_t3(up_sp + ps[0].to(tl.int64) * SLOT_SCALE_BYTES
                           + col_blk * (K_PER_BLOCK // 16 * 512) + (col0 // 16 + kg) * 512, kg, col32)
        acc = tl.dot_scaled(xq, xs, "e4m3", up_wT, up_sT, "e2m1", acc=acc)

    hbuf_p = tl.cast(hbuf_ptr, tl.pointer_type(tl.float32))
    up = acc
    if HAS_GATE:
        acc = tl.zeros((BT, 32), dtype=tl.float32)
        for it in range(K_PER_BLOCK // BK):
            col0 = it * BK
            xq = tl.load(xq_p + (2 * col0 + xq_cols)[None, :])
            xs = tl.load(xs_p + (col0 // 16 + kg)[None, :])
            g_byte = tl.load(gate_p + ps[0].to(tl.int64) * SLOT_BYTES + col_blk * KCOL_STRIDE
                             + (col0 // 2 + kb2)[:, None] * 32 + col32[None, :])
            g_wT = tl.trans(g_byte)
            g_sT = _scales_t3(gate_sp + ps[0].to(tl.int64) * SLOT_SCALE_BYTES
                              + col_blk * (K_PER_BLOCK // 16 * 512) + (col0 // 16 + kg) * 512, kg, col32)
            acc = tl.dot_scaled(xq, xs, "e4m3", g_wT, g_sT, "e2m1", acc=acc)
        gate = acc
        out = _act(gate, up, ACTIVATION, ALPHA, LIMIT)
    else:
        out = _act(up, up, ACTIVATION, ALPHA, LIMIT)
    tl.store(hbuf_p + rows.to(tl.int64)[:, None] * (2 * NCOL) + NCOL + col_blk * 32 + col32[None, :], out)


@triton.jit
def moe_down_gemm_mx(
    ids_ptr, slot_tab_ptr, weights_ptr,
    down_w, down_s, down_gds,
    hbuf_ptr, gout_ptr,
    n_tokens, topk,
    NCOL: tl.constexpr, N_PARTIAL: tl.constexpr, K_PER_BLOCK: tl.constexpr,
    SLOT_BYTES: tl.constexpr, SLOT_SCALE_BYTES: tl.constexpr, KCOL_STRIDE: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
):
    tok_blk = tl.program_id(0)
    col_blk = tl.program_id(1)
    tok0 = tok_blk * BT
    rows = tok0 + tl.arange(0, BT)
    tmask = rows < n_tokens
    down_p = tl.cast(down_w.to(tl.int64), tl.pointer_type(tl.uint8))
    down_sp = tl.cast(down_s.to(tl.int64), tl.pointer_type(tl.uint8))
    hbuf_p = tl.cast(hbuf_ptr, tl.pointer_type(tl.float32))
    col32 = tl.arange(0, 32)
    kb2 = tl.arange(0, BK // 2)
    kg = tl.arange(0, BK // 16)
    acc = tl.zeros((BT, 32), dtype=tl.float32)
    for j in range(topk):
        slots = rows * topk + j
        eid = tl.load(ids_ptr + slots, mask=tmask, other=-1)
        ps = tl.load(slot_tab_ptr + slots, mask=tmask, other=-1)
        ok = tmask & (eid >= 0) & (ps >= 0)
        route_w = tl.load(weights_ptr + slots, mask=ok, other=0.0)
        hbuf_slot_p = hbuf_p + slots.to(tl.int64)[:, None] * (2 * N_PARTIAL)
        accj = tl.zeros((BT, 32), dtype=tl.float32)
        for it in range(K_PER_BLOCK // BK):
            col0 = it * BK
            e_elem = col0 + 2 * tl.arange(0, BK // 2)
            a_even = tl.load(hbuf_slot_p + N_PARTIAL + e_elem[None, :], mask=ok[:, None], other=0.0)
            a_odd = tl.load(hbuf_slot_p + N_PARTIAL + e_elem[None, :] + 1, mask=ok[:, None], other=0.0)
            a_pair = tl.reshape(tl.interleave(a_even, a_odd), (BT, BK))
            aq, as_ = _quant_mx(a_pair, BK // 16)
            gds_p = tl.cast(down_gds.to(tl.int64), tl.pointer_type(tl.float32))
            d_byte = tl.load(down_p + tl.where(ok, ps, 0).to(tl.int64)[:, None] * SLOT_BYTES
                             + (col0 // 2 + kb2)[:, None][None, :, :] * 0 + col_blk * KCOL_STRIDE
                             + (col0 // 2 + kb2)[:, None] * 32 + col32[None, :],
                             mask=ok[0] if BT == 1 else None)
            d_wT = tl.trans(d_byte)
            d_sT = _scales_t3(down_sp + tl.where(ok, ps, 0)[0].to(tl.int64) * SLOT_SCALE_BYTES
                              + col_blk * (K_PER_BLOCK // 16 * 512) + (col0 // 16 + kg) * 512, kg, col32)
            accj = tl.dot_scaled(aq, as_, "e4m3", d_wT, d_sT, "e2m1", acc=accj)
        acc += accj * route_w[:, None]
    tl.store(gout_ptr + rows.to(tl.int64)[:, None] * NCOL + col_blk * 32 + col32[None, :], acc,
             mask=tmask[:, None])
