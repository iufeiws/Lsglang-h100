"""GPU compute of LK pool misses on DSPARK draft rows, reading MXFP4 experts over UVA.

A target-verify batch holds ``rows_per_request`` rows per request: row 0 is the real
token, the others are draft candidates. Pool-resident routes are computed by the LK
pool GEMV; this module computes the draft-row routes the pool snapshot does not hold,
straight from the layer's pinned host copy of its experts, so those rows never wait
for the CPU. One read of ``slot_tab`` (the pool snapshot gather) decides both halves.

Per-expert layout at each address (that of sglang's MXFP4 CPU parameters, gate rows first):
  w13 [2I, H/2] uint8 (two e2m1 per byte, low nibble = even k), s13 [2I, H/32] e8m0
  w2  [H,  I/2] uint8,                                          s2  [H,  I/32] e8m0
Experts may live anywhere: every expert has its own address, so per-NUMA placement works.
"""

from __future__ import annotations

import os
from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

# Row tile of the expert-grouped GEMMs; tl.dot needs M >= 16.
_BM = 16


class GemmConfig(msgspec.Struct, frozen=True):
    bn: int
    bk: int
    num_warps: int


# No register spills under ptxas sm_90a at BM 16 and 64 (Triton 3.8); not timed on H100 yet.
GATE_UP_CONFIG = GemmConfig(bn=32, bk=64, num_warps=8)
DOWN_CONFIG = GemmConfig(bn=64, bk=64, num_warps=8)


class HostExperts(msgspec.Struct, frozen=True):
    """Device-visible (UVA) address of every MXFP4 expert block of one layer."""

    # [E] int64 on the GPU: address of expert e's block of each tensor.
    w13_ptrs: torch.Tensor
    s13_ptrs: torch.Tensor
    w2_ptrs: torch.Tensor
    s2_ptrs: torch.Tensor
    num_experts: int
    hidden: int
    intermediate: int

    @classmethod
    def from_addresses(cls, *, w13, s13, w2, s2, hidden: int, intermediate: int, device):
        """Each argument is a sequence of E integer addresses, readable by the GPU."""
        tables = [torch.tensor(list(a), dtype=torch.int64, device=device) for a in (w13, s13, w2, s2)]
        num_experts = tables[0].numel()
        assert all(t.numel() == num_experts for t in tables)
        return cls(*tables, num_experts=num_experts, hidden=hidden, intermediate=intermediate)


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


def register_host_experts(w13, s13, w2, s2, *, device) -> HostExperts:
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

    def per_expert(t):
        # Under UVA a registered host address is also its device address.
        step = t[0].numel() * t.element_size()
        return [t.data_ptr() + e * step for e in range(num_experts)]

    return HostExperts.from_addresses(
        w13=per_expert(w13), s13=per_expert(s13), w2=per_expert(w2), s2=per_expert(s2),
        hidden=half_h * 2, intermediate=two_i // 2, device=device,
    )


@triton.jit
def _copy_bytes_kernel(src_ptrs, index, dst_ptr, n_bytes, BLOCK: tl.constexpr):
    src = tl.cast(tl.load(src_ptrs + index), tl.pointer_type(tl.uint8))
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst_ptr + off, tl.load(src + off, mask=off < n_bytes), mask=off < n_bytes)


def matches_host_experts(host: HostExperts, *, w13, s13, w2, s2, experts) -> bool:
    """Byte-compare the GPU's view of ``host`` with reference CPU tensors [E, ...]."""
    for table, ref in ((host.w13_ptrs, w13), (host.s13_ptrs, s13),
                       (host.w2_ptrs, w2), (host.s2_ptrs, s2)):
        for e in experts:
            want = ref[e].contiguous().view(torch.uint8).reshape(-1)
            got = torch.empty(want.numel(), dtype=torch.uint8, device=table.device)
            _copy_bytes_kernel[(triton.cdiv(want.numel(), 4096),)](
                table, e, got, want.numel(), BLOCK=4096)
            if not torch.equal(got.cpu(), want):
                return False
    return True


@triton.jit
def _e2m1_bf16(code):
    """e2m1 nibble -> bf16 (0, 0.5, 1, 1.5, 2, 3, 4, 6, signed by bit 3), built from bits."""
    c = code.to(tl.int32)
    exponent = (c >> 1) & 3
    mantissa = c & 1
    bits = tl.where(exponent == 0, mantissa * 0x3F00, ((exponent + 126) << 7) | (mantissa << 6))
    bits = bits | ((c & 8) << 12)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _e8m0_bf16(code):
    """ue8m0 scale byte -> bf16 2^(code-127); code 0 keeps 2^-127 (a bf16 subnormal)."""
    c = code.to(tl.int32)
    bits = tl.where(c == 0, 0x0040, c << 7)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _clamped_swiglu(gate, up, LIMIT: tl.constexpr):
    """silu(min(gate, L)) * clamp(up, -L, L): DSv4 semantics of expert_pack._clamped_swiglu."""
    if LIMIT > 0:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    return up * (gate / (1 + tl.exp(-gate)))


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
    hist_ptr,
    n_slots,
    top_k,
    rows_per_request,
    num_experts,
    n_tiles_cap,
    HAS_SLOT_TAB: tl.constexpr,
    HAS_HIST: tl.constexpr,
    BM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
    MAX_T: tl.constexpr,
    INCLUDE_FIRST_ROW: tl.constexpr,
):
    s = tl.arange(0, BLOCK_S)
    in_range = s < n_slots
    eid = tl.load(ids_ptr + s, mask=in_range, other=-1)
    if HAS_SLOT_TAB:
        pool_slot = tl.load(slot_tab_ptr + s, mask=in_range, other=0)
    else:
        pool_slot = tl.full((BLOCK_S,), -1, tl.int32)
    row_selected = (((s // top_k) % rows_per_request) != 0) | INCLUDE_FIRST_ROW
    need = in_range & (eid >= 0) & (pool_slot < 0) & row_selected

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
    if HAS_HIST:
        tl.atomic_add(hist_ptr + e, counts, mask=in_e & (counts > 0))
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
    """[BN, BK/2] bf16 weights of k columns k0+2j (lo) and k0+2j+1 (hi), scales applied.

    Exact: an e2m1 value times a power of two needs at most 2 significant bits.
    """
    j = tl.arange(0, BK // 2)
    packed = tl.load(w_ptr + rows[:, None] * (K // 2) + (k0 // 2 + j)[None, :])
    g = tl.arange(0, BK // 32)
    scale = _e8m0_bf16(tl.load(s_ptr + rows[:, None] * (K // 32) + (k0 // 32 + g)[None, :]))
    scale = tl.reshape(
        tl.broadcast_to(scale[:, :, None], (BN, BK // 32, 16)), (BN, BK // 2)
    ).to(tl.float32)
    lo = (_e2m1_bf16(packed & 0xF).to(tl.float32) * scale).to(tl.bfloat16)
    hi = (_e2m1_bf16((packed >> 4) & 0xF).to(tl.float32) * scale).to(tl.bfloat16)
    return lo, hi


@triton.jit
def _dot_bf16(a, b_t, acc, TC: tl.constexpr):
    """acc += a @ b_t for bf16 operands with FP32 accumulation; TC=False is the FP32 fallback."""
    if TC:
        acc = tl.dot(a, b_t, acc)
    else:
        acc = tl.dot(a.to(tl.float32), b_t.to(tl.float32), acc, input_precision="ieee")
    return acc


@triton.jit
def _dot_f32_split(a, b_t, acc, TC: tl.constexpr):
    """acc += a @ b_t for FP32 ``a``: split into bf16 high and low parts (~16 mantissa bits)."""
    if TC:
        a_hi = a.to(tl.bfloat16)
        a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(a_hi, b_t, acc)
        acc = tl.dot(a_lo, b_t, acc)
    else:
        acc = tl.dot(a, b_t.to(tl.float32), acc, input_precision="ieee")
    return acc


@triton.jit
def _gate_up_kernel(
    x_ptr,
    order_ptr,
    tile_expert_ptr,
    tile_start_ptr,
    tile_count_ptr,
    w13_ptrs,
    s13_ptrs,
    mid_ptr,
    top_k,
    H: tl.constexpr,
    I: tl.constexpr,
    LIMIT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TC: tl.constexpr,
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
    w = tl.cast(tl.load(w13_ptrs + expert), tl.pointer_type(tl.uint8))
    s = tl.cast(tl.load(s13_ptrs + expert), tl.pointer_type(tl.uint8))
    gate_rows = n
    up_rows = n + I
    j = tl.arange(0, BK // 2)
    acc_gate = tl.zeros((BM, BN), dtype=tl.float32)
    acc_up = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        k_even = k0 + 2 * j
        x_even = tl.load(x_row[:, None] + k_even[None, :], mask=live[:, None], other=0.0).to(tl.bfloat16)
        x_odd = tl.load(x_row[:, None] + k_even[None, :] + 1, mask=live[:, None], other=0.0).to(tl.bfloat16)
        g_lo, g_hi = _mxfp4_tile(w, s, gate_rows, k0, H, BN, BK)
        u_lo, u_hi = _mxfp4_tile(w, s, up_rows, k0, H, BN, BK)
        acc_gate = _dot_bf16(x_even, tl.trans(g_lo), acc_gate, TC)
        acc_gate = _dot_bf16(x_odd, tl.trans(g_hi), acc_gate, TC)
        acc_up = _dot_bf16(x_even, tl.trans(u_lo), acc_up, TC)
        acc_up = _dot_bf16(x_odd, tl.trans(u_hi), acc_up, TC)
    act = _clamped_swiglu(acc_gate, acc_up, LIMIT)
    tl.store(mid_ptr + slot[:, None] * I + n[None, :], act, mask=live[:, None])


@triton.jit
def _down_kernel(
    order_ptr,
    tile_expert_ptr,
    tile_start_ptr,
    tile_count_ptr,
    route_weights_ptr,
    w2_ptrs,
    s2_ptrs,
    mid_ptr,
    gout_part_ptr,
    H: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TC: tl.constexpr,
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
    w = tl.cast(tl.load(w2_ptrs + expert), tl.pointer_type(tl.uint8))
    s = tl.cast(tl.load(s2_ptrs + expert), tl.pointer_type(tl.uint8))
    rows = n
    j = tl.arange(0, BK // 2)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, I, BK):
        k_even = k0 + 2 * j
        a_even = tl.load(mid_ptr + slot[:, None] * I + k_even[None, :], mask=live[:, None], other=0.0)
        a_odd = tl.load(mid_ptr + slot[:, None] * I + k_even[None, :] + 1, mask=live[:, None], other=0.0)
        d_lo, d_hi = _mxfp4_tile(w, s, rows, k0, I, BN, BK)
        acc = _dot_f32_split(a_even, tl.trans(d_lo), acc, TC)
        acc = _dot_f32_split(a_odd, tl.trans(d_hi), acc, TC)
    route_w = tl.load(route_weights_ptr + slot, mask=live, other=0.0)
    tl.store(
        gout_part_ptr + slot[:, None] * H + n[None, :],
        acc * route_w[:, None],
        mask=live[:, None],
    )


def _use_tensor_cores() -> bool:
    # The Triton CPU interpreter mis-emulates bf16 tl.dot; it takes the FP32 path.
    return os.environ.get("TRITON_INTERPRET") != "1"


class _Tiles(msgspec.Struct, frozen=True):
    order: torch.Tensor
    tile_expert: torch.Tensor
    tile_start: torch.Tensor
    tile_count: torch.Tensor
    mid: torch.Tensor
    n_tiles: int
    bm: int


def _plan_graph(*, topk_ids, slot_tab, rows_per_request, include_first_row, hist,
                scratch, num_experts) -> _Tiles:
    """Single-program plan into fixed scratch buffers: CUDA-graph capturable."""
    rows, top_k = topk_ids.shape
    n_slots = rows * top_k
    computed_rows = rows if include_first_row else rows - rows // rows_per_request
    n_tiles_cap = max(1, computed_rows * top_k)
    assert n_slots <= scratch.order.numel() and num_experts <= scratch.counts.numel()
    _plan_kernel[(1,)](
        topk_ids,
        # Unused placeholders keep the signature fixed when a feature is off.
        slot_tab if slot_tab is not None else scratch.order,
        scratch.counts,
        scratch.cursor,
        scratch.offsets,
        scratch.order,
        scratch.tile_expert,
        scratch.tile_start,
        scratch.tile_count,
        hist if hist is not None else scratch.counts,
        n_slots,
        top_k,
        rows_per_request,
        num_experts,
        n_tiles_cap,
        HAS_SLOT_TAB=slot_tab is not None,
        HAS_HIST=hist is not None,
        BM=_BM,
        BLOCK_S=triton.next_power_of_2(n_slots),
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_T=triton.next_power_of_2(n_tiles_cap),
        # An expert appears at most once per row.
        MAX_T=triton.cdiv(max(1, computed_rows), _BM),
        INCLUDE_FIRST_ROW=include_first_row,
    )
    return _Tiles(scratch.order, scratch.tile_expert, scratch.tile_start,
                  scratch.tile_count, scratch.mid, n_tiles_cap, _BM)


def _plan_eager(*, topk_ids, slot_tab, hist, num_experts, intermediate) -> _Tiles:
    """Host-sized plan for any batch (e.g. prefill tails); not graph capturable."""
    bm = 64
    ids = topk_ids.reshape(-1)
    need = ids >= 0
    if slot_tab is not None:
        need &= slot_tab < 0
    slots = torch.nonzero(need).squeeze(1)
    experts = ids[slots].long()
    experts, perm = torch.sort(experts, stable=True)
    order = slots[perm].to(torch.int32)
    counts = torch.bincount(experts, minlength=num_experts)
    if hist is not None:
        hist.add_(counts.to(hist.dtype))
    n_tiles_per = (counts + bm - 1) // bm
    tile_expert = torch.repeat_interleave(
        torch.arange(num_experts, device=ids.device), n_tiles_per)
    first_tile = torch.cumsum(n_tiles_per, 0) - n_tiles_per
    within = torch.arange(tile_expert.numel(), device=ids.device) - first_tile[tile_expert]
    offsets = torch.cumsum(counts, 0) - counts
    tile_start = offsets[tile_expert] + within * bm
    tile_count = torch.clamp(counts[tile_expert] - within * bm, max=bm)
    mid = torch.empty((ids.numel(), intermediate), dtype=torch.float32, device=ids.device)
    return _Tiles(order, tile_expert.to(torch.int32), tile_start.to(torch.int32),
                  tile_count.to(torch.int32), mid, tile_expert.numel(), bm)


def launch_draft_miss_uva(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    slot_tab: Optional[torch.Tensor],
    gout_part: torch.Tensor,
    rows_per_request: int,
    host: HostExperts,
    scratch: UvaScratch,
    activation: int,
    alpha: float,
    limit: float,
    include_first_row: bool = False,
    hist: Optional[torch.Tensor] = None,
    eager_plan: bool = False,
) -> None:
    """Write route-weighted outputs of draft-row pool misses into ``gout_part[slot]``.

    ``slot_tab`` None computes every selected route (no LK pool); ``include_first_row``
    also computes row 0 of each request (every row of a plain decode batch with
    ``rows_per_request=1``); ``hist`` [E] int32 accumulates the computed routes per
    expert. ``eager_plan`` plans on the host for batches larger than ``scratch``
    (every row; not capturable). Enqueued on the current stream.
    ``gout_part`` rows of other slots are untouched, so the caller zeroes them and
    the pool GEMV fills the resident ones.
    """
    # Only SiLU (LK activation 0) is implemented; ``alpha`` belongs to the other kinds.
    assert activation == 0, f"LK verify UVA: activation {activation} is not SiLU"
    del alpha
    rows, top_k = topk_ids.shape
    H, I, E = host.hidden, host.intermediate, host.num_experts
    assert hidden_states.shape == (rows, H) and hidden_states.is_contiguous()
    assert topk_ids.is_contiguous() and topk_weights.is_contiguous()
    assert topk_weights.dtype == torch.float32
    if eager_plan:
        assert include_first_row
        tiles = _plan_eager(topk_ids=topk_ids, slot_tab=slot_tab, hist=hist,
                            num_experts=E, intermediate=I)
        if tiles.n_tiles == 0:
            return
    else:
        tiles = _plan_graph(
            topk_ids=topk_ids, slot_tab=slot_tab, rows_per_request=rows_per_request,
            include_first_row=include_first_row, hist=hist, scratch=scratch,
            num_experts=E)
    tc = _use_tensor_cores()
    gu, dn = GATE_UP_CONFIG, DOWN_CONFIG
    assert I % gu.bn == 0 and H % gu.bk == 0 and H % dn.bn == 0 and I % dn.bk == 0
    _gate_up_kernel[(tiles.n_tiles, I // gu.bn)](
        hidden_states,
        tiles.order,
        tiles.tile_expert,
        tiles.tile_start,
        tiles.tile_count,
        host.w13_ptrs,
        host.s13_ptrs,
        tiles.mid,
        top_k,
        H=H,
        I=I,
        LIMIT=limit,
        BM=tiles.bm,
        BN=gu.bn,
        BK=gu.bk,
        TC=tc,
        num_warps=gu.num_warps,
    )
    _down_kernel[(tiles.n_tiles, H // dn.bn)](
        tiles.order,
        tiles.tile_expert,
        tiles.tile_start,
        tiles.tile_count,
        topk_weights,
        host.w2_ptrs,
        host.s2_ptrs,
        tiles.mid,
        gout_part,
        H=H,
        I=I,
        BM=tiles.bm,
        BN=dn.bn,
        BK=dn.bk,
        TC=tc,
        num_warps=dn.num_warps,
    )


# Counters of ``admit_misses`` (int32 [STAT_COUNT], per layer): routes for the first two,
# distinct experts for the rest.
STAT_ROUTES, STAT_ROW_HITS, STAT_ADMITTED, STAT_OVERFLOW = range(4)
STAT_COUNT = 4
# Buckets of the distinct-missed-experts-per-forward histogram; the last one is ">=".
MISS_BUCKETS = 33


@triton.jit
def _admit_misses_kernel(
    ids_ptr,
    slot_of_ptr,
    row_expert_ptr,
    score_ptr,
    counts_ptr,
    copy_row_ptr,
    copy_expert_ptr,
    slot_ids_ptr,
    miss_tab_ptr,
    stats_ptr,
    miss_hist_ptr,
    n,
    num_experts,
    num_rows,
    decay,
    BLOCK_S: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_R: tl.constexpr,
    MAX_COPY: tl.constexpr,
    BUCKETS: tl.constexpr,
):
    s = tl.arange(0, BLOCK_S)
    live = s < n
    expert = tl.load(ids_ptr + s, mask=live, other=-1)
    local = live & (expert >= 0)
    hit = local & (tl.load(slot_of_ptr + expert, mask=local, other=-1) >= 0)
    e = tl.arange(0, BLOCK_E)
    in_e = e < num_experts
    tl.store(counts_ptr + e, 0, mask=in_e)
    c = tl.arange(0, MAX_COPY)
    tl.store(copy_row_ptr + c, -1)
    tl.store(copy_expert_ptr + c, -1)
    tl.debug_barrier()
    tl.atomic_add(counts_ptr + expert, 1, mask=local)
    tl.debug_barrier()
    counts = tl.load(counts_ptr + e, mask=in_e, other=0, volatile=True)
    score = tl.load(score_ptr + e, mask=in_e, other=0.0) * decay + counts.to(tl.float32)
    tl.store(score_ptr + e, score, mask=in_e)
    tl.debug_barrier()
    missed = in_e & (counts > 0) & (tl.load(slot_of_ptr + e, mask=in_e, other=0) < 0)
    # Victims: rows whose expert this forward does not route, empty rows first, then
    # the lowest score; the most valuable missed experts are admitted first.
    r = tl.arange(0, BLOCK_R)
    in_r = r < num_rows
    holder = tl.load(row_expert_ptr + r, mask=in_r, other=-1)
    held = in_r & (holder >= 0)
    busy = tl.load(counts_ptr + holder, mask=held, other=0, volatile=True) > 0
    holder_score = tl.load(score_ptr + holder, mask=held, other=0.0, volatile=True)
    victim_key = tl.where(in_r & ~busy, tl.where(held, holder_score, -1.0), float("inf"))
    want = tl.where(missed, score, -float("inf"))
    n_admitted = 0
    for i in tl.static_range(MAX_COPY):
        row = tl.argmin(victim_key, 0)
        new = tl.argmax(want, 0)
        ok = (tl.min(victim_key, 0) < float("inf")) & (tl.max(want, 0) > -float("inf"))
        old = tl.load(row_expert_ptr + row, mask=ok, other=-1)
        tl.store(slot_of_ptr + old, -1, mask=ok & (old >= 0))
        tl.store(slot_of_ptr + new, row, mask=ok)
        tl.store(row_expert_ptr + row, new, mask=ok)
        tl.store(copy_row_ptr + i, row, mask=ok)
        tl.store(copy_expert_ptr + i, new, mask=ok)
        n_admitted += ok.to(tl.int32)
        victim_key = tl.where(r == row, float("inf"), victim_key)
        want = tl.where(e == new, -float("inf"), want)
    tl.debug_barrier()
    slot = tl.load(slot_of_ptr + expert, mask=local, other=-1, volatile=True)
    overflow = local & (slot < 0)
    tl.store(slot_ids_ptr + s, tl.where(slot >= 0, slot, num_rows), mask=live)
    tl.store(miss_tab_ptr + s, tl.where(overflow, -1, 0), mask=live)
    n_missed = tl.sum(missed.to(tl.int32), 0)
    tl.atomic_add(miss_hist_ptr + tl.minimum(n_missed, BUCKETS - 1), 1)
    tl.atomic_add(stats_ptr + 0, tl.sum(local.to(tl.int32), 0))
    tl.atomic_add(stats_ptr + 1, tl.sum(hit.to(tl.int32), 0))
    tl.atomic_add(stats_ptr + 2, n_admitted)
    tl.atomic_add(stats_ptr + 3, n_missed - n_admitted)


def admit_misses(*, topk_ids, slot_of, row_expert, score, counts, copy_row, copy_expert,
                 slot_ids, miss_tab, stats, miss_hist, decay: float) -> None:
    """Admit this forward's missed experts into cache rows; graph capturable.

    ``slot_of`` [E] / ``row_expert`` [R] (expert <-> row, -1 none) and ``score`` [E]
    (decayed route counts, times ``decay`` per forward) persist per layer and are
    updated in place. Up to ``copy_row.numel()`` missed experts, highest score
    first, replace the experts of rows not routed in this forward (empty or lowest
    score first); ``copy_row`` / ``copy_expert`` list the copies to make (-1 unused).
    ``slot_ids`` gets each route's row, or ``R`` (out of range: skipped) for -1 and
    for misses left over, which ``miss_tab`` marks -1 for the UVA kernel.
    """
    n = topk_ids.numel()
    num_experts = slot_of.numel()
    num_rows = row_expert.numel()
    assert triton.next_power_of_2(copy_row.numel()) == copy_row.numel()
    _admit_misses_kernel[(1,)](
        topk_ids, slot_of, row_expert, score, counts, copy_row, copy_expert,
        slot_ids, miss_tab, stats, miss_hist, n, num_experts, num_rows, float(decay),
        BLOCK_S=triton.next_power_of_2(max(n, 2)),
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_R=triton.next_power_of_2(num_rows),
        MAX_COPY=copy_row.numel(),
        BUCKETS=MISS_BUCKETS,
    )


@triton.jit
def _copy_experts_kernel(copy_row_ptr, copy_expert_ptr, src_ptrs, dst_ptrs, words_ptr,
                         total_items, num_tensors, max_chunks, BLOCK: tl.constexpr,
                         NPROG: tl.constexpr):
    for item in range(tl.program_id(0), total_items, NPROG):
        j = item // (num_tensors * max_chunks)
        t = (item // max_chunks) % num_tensors
        start = (item % max_chunks) * BLOCK
        expert = tl.load(copy_expert_ptr + j)
        words = tl.load(words_ptr + t)
        if (expert >= 0) & (start < words):
            row = tl.load(copy_row_ptr + j).to(tl.int64)
            off = start + tl.arange(0, BLOCK)
            mask = off < words
            src = (tl.cast(tl.load(src_ptrs + t), tl.pointer_type(tl.int32))
                   + expert.to(tl.int64) * words)
            dst = tl.cast(tl.load(dst_ptrs + t), tl.pointer_type(tl.int32)) + row * words
            tl.store(dst + off, tl.load(src + off, mask=mask), mask=mask)


def copy_experts(*, copy_row, copy_expert, src_ptrs, dst_ptrs, words, max_words: int,
                 num_programs: int) -> None:
    """Copy expert ``copy_expert[j]``'s blocks (host, over UVA) into row ``copy_row[j]``.

    Tensor ``t`` of expert ``e`` is ``words[t]`` int32 words at ``src_ptrs[t]`` +
    e * words[t], row ``r`` at ``dst_ptrs[t]`` + r * words[t]. Persistent programs
    with wide coalesced reads keep PCIe saturated; entries of -1 are skipped.
    """
    block = 2048
    max_chunks = triton.cdiv(max_words, block)
    _copy_experts_kernel[(num_programs,)](
        copy_row, copy_expert, src_ptrs, dst_ptrs, words,
        copy_row.numel() * words.numel() * max_chunks, words.numel(), max_chunks,
        BLOCK=block, NPROG=num_programs)
