"""GPU compute of LK pool misses on DSPARK draft rows, reading MXFP4 experts over UVA.

A target-verify batch holds ``rows_per_request`` rows per request: row 0 is the real
token, the others are draft candidates. Pool-resident routes are computed by the LK
pool GEMV; this module computes the draft-row routes the pool snapshot does not hold,
straight from the layer's pinned host copy of its experts, so those rows never wait
for the CPU. One read of ``slot_tab`` (the pool snapshot gather) decides both halves.

Host layout (sglang's MXFP4 CPU parameters of a non-resident layer, gate rows first):
  w13 [E, 2I, H/2] uint8 (two e2m1 per byte, low nibble = even k), s13 [E, 2I, H/32] e8m0
  w2  [E, H,  I/2] uint8,                                          s2  [E, H,  I/32] e8m0
"""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

# Activation shared with the LK pool GEMV and the LK CPU kernels, so draft rows
# computed here use the same SwiGLU clamp semantics as the other routes.
from lk_moe._pool_kernel_common import _act

# Row tile of the expert-grouped GEMMs; tl.dot needs M >= 16.
_BM = 16


class HostExperts(msgspec.Struct, frozen=True):
    """Device-visible (UVA) addresses of one layer's pinned MXFP4 experts."""

    w13_addr: int
    s13_addr: int
    w2_addr: int
    s2_addr: int
    num_experts: int
    hidden: int
    intermediate: int


class UvaScratch(msgspec.Struct, frozen=True):
    """Per-process buffers shared by all layers (they run in order on one stream)."""

    counts: torch.Tensor  # [E] int32
    cursor: torch.Tensor  # [E] int32
    offsets: torch.Tensor  # [E] int32
    order: torch.Tensor  # [max_slots] int32, slots grouped by expert
    tile_expert: torch.Tensor  # [max_slots] int32
    tile_start: torch.Tensor  # [max_slots] int32
    tile_count: torch.Tensor  # [max_slots] int32
    mid: torch.Tensor  # [max_slots, I] fp32, activated gate/up per slot

    @classmethod
    def allocate(cls, *, num_experts: int, max_slots: int, intermediate: int, device):
        def i32(n):
            return torch.zeros((n,), dtype=torch.int32, device=device)

        return cls(
            counts=i32(num_experts),
            cursor=i32(num_experts),
            offsets=i32(num_experts),
            order=i32(max_slots),
            tile_expert=i32(max_slots),
            tile_start=i32(max_slots),
            tile_count=i32(max_slots),
            mid=torch.zeros(
                (max_slots, intermediate), dtype=torch.float32, device=device
            ),
        )


def register_host_experts(w13, s13, w2, s2) -> HostExperts:
    """Pin the CPU expert tensors in place and return their UVA addresses.

    The caller must keep the tensors alive for the process lifetime.
    """
    for t in (w13, s13, w2, s2):
        assert t.device.type == "cpu" and t.is_contiguous(), t.shape
        err = torch.cuda.cudart().cudaHostRegister(
            t.data_ptr(), t.numel() * t.element_size(), 0
        )
        if int(err) != 0:
            raise RuntimeError(
                f"cudaHostRegister({t.numel() * t.element_size()} bytes) failed: {err}"
            )
    num_experts, two_i, half_h = w13.shape
    return HostExperts(
        # Under UVA a registered host address is also its device address.
        w13_addr=w13.data_ptr(),
        s13_addr=s13.data_ptr(),
        w2_addr=w2.data_ptr(),
        s2_addr=s2.data_ptr(),
        num_experts=num_experts,
        hidden=half_h * 2,
        intermediate=two_i // 2,
    )


@triton.jit
def _e2m1(code):
    """e2m1 nibble -> fp32: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign."""
    mag = code & 7
    val = tl.where(mag == 7, 6.0, tl.where(mag == 6, 4.0, tl.where(mag == 5, 3.0, 0.5 * mag.to(tl.float32))))
    return tl.where((code & 8) != 0, -val, val)


@triton.jit
def _e8m0(code):
    """ue8m0 scale byte -> fp32 2^(code-127); code 0 keeps 2^-127 instead of flushing."""
    c = code.to(tl.uint32)
    v = (c << 23).to(tl.float32, bitcast=True)
    return tl.where(c == 0, 5.8774718e-39, v)


@triton.jit
def _plan_kernel(
    ids_ptr,
    slot_tab_ptr,
    counts_ptr,
    cursor_ptr,
    offsets_ptr,
    order_ptr,
    tile_expert_ptr,
    tile_start_ptr,
    tile_count_ptr,
    n_slots,
    top_k,
    rows_per_request,
    num_experts,
    n_tiles_cap,
    BM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
    MAX_T: tl.constexpr,
):
    s = tl.arange(0, BLOCK_S)
    in_range = s < n_slots
    eid = tl.load(ids_ptr + s, mask=in_range, other=-1)
    pool_slot = tl.load(slot_tab_ptr + s, mask=in_range, other=0)
    draft_row = ((s // top_k) % rows_per_request) != 0
    need = in_range & (eid >= 0) & (pool_slot < 0) & draft_row

    e = tl.arange(0, BLOCK_E)
    in_e = e < num_experts
    t = tl.arange(0, BLOCK_T)
    tl.store(counts_ptr + e, 0, mask=in_e)
    tl.store(cursor_ptr + e, 0, mask=in_e)
    tl.store(tile_count_ptr + t, 0, mask=t < n_tiles_cap)
    tl.debug_barrier()
    tl.atomic_add(counts_ptr + eid, 1, mask=need)
    tl.debug_barrier()
    # Volatile: the atomics landed in L2, a cached load could return the zeroing store.
    counts = tl.load(counts_ptr + e, mask=in_e, other=0, volatile=True)
    offsets = tl.cumsum(counts, 0) - counts
    tl.store(offsets_ptr + e, offsets, mask=in_e)
    n_tiles = (counts + BM - 1) // BM
    first_tile = tl.cumsum(n_tiles, 0) - n_tiles
    for i in tl.static_range(MAX_T):
        valid = in_e & (i < n_tiles)
        idx = first_tile + i
        tl.store(tile_expert_ptr + idx, e, mask=valid)
        tl.store(tile_start_ptr + idx, offsets + i * BM, mask=valid)
        tl.store(tile_count_ptr + idx, tl.minimum(counts - i * BM, BM), mask=valid)
    tl.debug_barrier()
    base = tl.load(offsets_ptr + eid, mask=need, other=0, volatile=True)
    pos = tl.atomic_add(cursor_ptr + eid, 1, mask=need)
    tl.store(order_ptr + base + pos, s, mask=need)


@triton.jit
def _mxfp4_tile(w_ptr, s_ptr, rows, k0, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """[BN, BK/2] fp32 weights of k columns k0+2j (lo) and k0+2j+1 (hi), scales applied."""
    j = tl.arange(0, BK // 2)
    packed = tl.load(w_ptr + rows[:, None] * (K // 2) + (k0 // 2 + j)[None, :])
    g = tl.arange(0, BK // 32)
    scale = _e8m0(tl.load(s_ptr + rows[:, None] * (K // 32) + (k0 // 32 + g)[None, :]))
    scale = tl.reshape(
        tl.broadcast_to(scale[:, :, None], (BN, BK // 32, 16)), (BN, BK // 2)
    )
    lo = _e2m1(packed & 0xF) * scale
    hi = _e2m1((packed >> 4) & 0xF) * scale
    return lo, hi


@triton.jit
def _gate_up_kernel(
    x_ptr,
    order_ptr,
    tile_expert_ptr,
    tile_start_ptr,
    tile_count_ptr,
    w13_addr,
    s13_addr,
    mid_ptr,
    top_k,
    H: tl.constexpr,
    I: tl.constexpr,
    ACTIVATION: tl.constexpr,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    count = tl.load(tile_count_ptr + tile)
    if count == 0:
        return
    expert = tl.load(tile_expert_ptr + tile).to(tl.int64)
    start = tl.load(tile_start_ptr + tile)
    r = tl.arange(0, BM)
    live = r < count
    slot = tl.load(order_ptr + start + r, mask=live, other=0).to(tl.int64)
    x_row = x_ptr + (slot // top_k) * H
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    w = tl.cast(w13_addr.to(tl.int64), tl.pointer_type(tl.uint8))
    s = tl.cast(s13_addr.to(tl.int64), tl.pointer_type(tl.uint8))
    gate_rows = expert * (2 * I) + n
    up_rows = gate_rows + I
    j = tl.arange(0, BK // 2)
    acc_gate = tl.zeros((BM, BN), dtype=tl.float32)
    acc_up = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        k_even = k0 + 2 * j
        x_even = tl.load(x_row[:, None] + k_even[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        x_odd = tl.load(x_row[:, None] + k_even[None, :] + 1, mask=live[:, None], other=0.0).to(tl.float32)
        g_lo, g_hi = _mxfp4_tile(w, s, gate_rows, k0, H, BN, BK)
        u_lo, u_hi = _mxfp4_tile(w, s, up_rows, k0, H, BN, BK)
        acc_gate = tl.dot(x_even, tl.trans(g_lo), acc_gate, input_precision="ieee")
        acc_gate = tl.dot(x_odd, tl.trans(g_hi), acc_gate, input_precision="ieee")
        acc_up = tl.dot(x_even, tl.trans(u_lo), acc_up, input_precision="ieee")
        acc_up = tl.dot(x_odd, tl.trans(u_hi), acc_up, input_precision="ieee")
    act = _act(acc_gate, acc_up, ACTIVATION, ALPHA, LIMIT)
    tl.store(mid_ptr + slot[:, None] * I + n[None, :], act, mask=live[:, None])


@triton.jit
def _down_kernel(
    order_ptr,
    tile_expert_ptr,
    tile_start_ptr,
    tile_count_ptr,
    route_weights_ptr,
    w2_addr,
    s2_addr,
    mid_ptr,
    gout_part_ptr,
    H: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    count = tl.load(tile_count_ptr + tile)
    if count == 0:
        return
    expert = tl.load(tile_expert_ptr + tile).to(tl.int64)
    start = tl.load(tile_start_ptr + tile)
    r = tl.arange(0, BM)
    live = r < count
    slot = tl.load(order_ptr + start + r, mask=live, other=0).to(tl.int64)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    w = tl.cast(w2_addr.to(tl.int64), tl.pointer_type(tl.uint8))
    s = tl.cast(s2_addr.to(tl.int64), tl.pointer_type(tl.uint8))
    rows = expert * H + n
    j = tl.arange(0, BK // 2)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, I, BK):
        k_even = k0 + 2 * j
        a_even = tl.load(mid_ptr + slot[:, None] * I + k_even[None, :], mask=live[:, None], other=0.0)
        a_odd = tl.load(mid_ptr + slot[:, None] * I + k_even[None, :] + 1, mask=live[:, None], other=0.0)
        d_lo, d_hi = _mxfp4_tile(w, s, rows, k0, I, BN, BK)
        acc = tl.dot(a_even, tl.trans(d_lo), acc, input_precision="ieee")
        acc = tl.dot(a_odd, tl.trans(d_hi), acc, input_precision="ieee")
    route_w = tl.load(route_weights_ptr + slot, mask=live, other=0.0)
    tl.store(
        gout_part_ptr + slot[:, None] * H + n[None, :],
        acc * route_w[:, None],
        mask=live[:, None],
    )


def _tile_k(k: int) -> int:
    # 128 keeps the fp32 weight tiles of the gate/up kernel in registers.
    for bk in (128, 64):
        if k % bk == 0:
            return bk
    raise ValueError(f"LK verify UVA: reduction dim {k} is not a multiple of 64")


def launch_draft_miss_uva(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    slot_tab: torch.Tensor,
    gout_part: torch.Tensor,
    rows_per_request: int,
    host: HostExperts,
    scratch: UvaScratch,
    activation: int,
    alpha: float,
    limit: float,
) -> None:
    """Write route-weighted outputs of draft-row pool misses into ``gout_part[slot]``.

    Enqueued on the current stream. ``gout_part`` rows of other slots are untouched,
    so the caller zeroes them and the pool GEMV fills the resident ones.
    """
    rows, top_k = topk_ids.shape
    H, I, E = host.hidden, host.intermediate, host.num_experts
    assert hidden_states.shape == (rows, H) and hidden_states.is_contiguous()
    assert topk_ids.is_contiguous() and topk_weights.is_contiguous()
    assert topk_weights.dtype == torch.float32
    n_slots = rows * top_k
    draft_rows = rows - rows // rows_per_request
    n_tiles_cap = max(1, draft_rows * top_k)
    assert n_slots <= scratch.order.numel() and E <= scratch.counts.numel()
    _plan_kernel[(1,)](
        topk_ids,
        slot_tab,
        scratch.counts,
        scratch.cursor,
        scratch.offsets,
        scratch.order,
        scratch.tile_expert,
        scratch.tile_start,
        scratch.tile_count,
        n_slots,
        top_k,
        rows_per_request,
        E,
        n_tiles_cap,
        BM=_BM,
        BLOCK_S=triton.next_power_of_2(n_slots),
        BLOCK_E=triton.next_power_of_2(E),
        BLOCK_T=triton.next_power_of_2(n_tiles_cap),
        # An expert appears at most once per row.
        MAX_T=triton.cdiv(max(1, draft_rows), _BM),
    )
    gate_up_bn = 32
    _gate_up_kernel[(n_tiles_cap, triton.cdiv(I, gate_up_bn))](
        hidden_states,
        scratch.order,
        scratch.tile_expert,
        scratch.tile_start,
        scratch.tile_count,
        host.w13_addr,
        host.s13_addr,
        scratch.mid,
        top_k,
        H=H,
        I=I,
        ACTIVATION=activation,
        ALPHA=alpha,
        LIMIT=limit,
        BM=_BM,
        BN=gate_up_bn,
        BK=_tile_k(H),
        num_warps=4,
    )
    down_bn = 64
    _down_kernel[(n_tiles_cap, triton.cdiv(H, down_bn))](
        scratch.order,
        scratch.tile_expert,
        scratch.tile_start,
        scratch.tile_count,
        topk_weights,
        host.w2_addr,
        host.s2_addr,
        scratch.mid,
        gout_part,
        H=H,
        I=I,
        BM=_BM,
        BN=down_bn,
        BK=_tile_k(I),
        num_warps=4,
    )
