"""MXFP4 expert-pool verification on Hopper.

The slot table is the *same frozen split* used by lk_moe's CPU half. Group
routes within a 16-token tile, loading each resident expert's weights once
for its rows. No expert staging, host synchronization, or live residency
lookup belongs here. Misses remain the CPU half's responsibility.

The packed layout is [slot, output // 32, input // 2, output % 32], with
E8M0 scales [slot, output // 32, input // 32, output % 32]. This is W4A16:
both dot products use BF16 operands and FP32 accumulation. Split-K reduction
and SiLU run in FP32, then the down input is rounded/stored in BF16. CPU/GEMV
keeps its existing arithmetic, so bitwise agreement with it is not promised.
The resident FlashInfer Humming W4A8 setting does not change this pool kernel.
"""

import triton
import triton.language as tl


@triton.jit
def _group_rows(
    Slots,
    leader,
    N_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BK_ROUTES: tl.constexpr,
):
    tile_start = (leader // (BM * TOPK)) * BM
    rows = tile_start + tl.arange(0, BM)
    routes = rows[:, None] * TOPK + tl.arange(0, BK_ROUTES)[None, :]
    valid = (rows[:, None] < N_ROWS) & (tl.arange(0, BK_ROUTES)[None, :] < TOPK)
    slot = tl.load(Slots + leader)
    match = valid & (tl.load(Slots + routes, valid, -1) == slot) & (slot >= 0)
    selected = tl.min(tl.where(match, routes, 2147483647), axis=1)
    first = tl.min(selected, axis=0)
    live = selected != 2147483647
    return slot, rows, tl.where(live, selected, 0), live, first == leader


@triton.jit
def _load_mxfp4_reference(
    W,
    S,
    slot,
    kk,
    nn,
    K: tl.constexpr,
    N: tl.constexpr,
    W_BYTES: tl.constexpr,
    S_BYTES: tl.constexpr,
):
    wp = W.to(tl.pointer_type(tl.uint8))
    sp = S.to(tl.pointer_type(tl.uint8))
    mask = (kk[:, None] < K) & (nn[None, :] < N)
    off = (
        (nn[None, :] // 32) * (K // 2 * 32) + (kk[:, None] // 2) * 32 + nn[None, :] % 32
    )
    packed = tl.load(wp + slot.to(tl.int64) * W_BYTES + off, mask, 0)
    code = (packed >> ((kk[:, None] % 2) * 4)) & 15
    mag = code & 7
    # E2M1 magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6.
    value = tl.where(
        mag < 2,
        mag.to(tl.float32) * 0.5,
        (1.0 + (mag & 1).to(tl.float32) * 0.5)
        * tl.exp2(((mag >> 1) - 1).to(tl.float32)),
    )
    value = tl.where((code & 8) != 0, -value, value)
    soff = (
        (nn[None, :] // 32) * (K // 32 * 32)
        + (kk[:, None] // 32) * 32
        + nn[None, :] % 32
    )
    scale = tl.load(sp + slot.to(tl.int64) * S_BYTES + soff, mask, 0).to(tl.uint32)
    decoded_scale = (scale << 23).to(tl.float32, bitcast=True)
    decoded_scale = tl.where(scale == 0, 5.8774718e-39, decoded_scale)
    return value * decoded_scale


@triton.jit
def _unpack_e2m1_prmt(packed):
    # Eight FP16 magnitudes have zero low bytes. PRMT looks up the two
    # high bytes together, then inserts both sign bits. No FP arithmetic,
    # exponential, or rounding is involved in the E2M1 -> FP16 conversion.
    return tl.inline_asm_elementwise(
        """{
        .reg .b32 p, sel, hi, signs, tmp;
        mov.b32 p, $2;
        and.b32 sel, p, 7;
        shl.b32 tmp, p, 4;
        and.b32 tmp, tmp, 0x700;
        or.b32 sel, sel, tmp;
        prmt.b32 hi, 0x3e3c3800, 0x46444240, sel;
        shl.b32 hi, hi, 8;
        and.b32 signs, p, 8;
        shl.b32 signs, signs, 12;
        and.b32 tmp, p, 128;
        shl.b32 tmp, tmp, 24;
        or.b32 signs, signs, tmp;
        or.b32 hi, hi, signs;
        mov.b32 {$0, $1}, hi;
        }""",
        constraints="=h,=h,r",
        args=[packed.to(tl.uint32)],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _load_mxfp4(
    W, S, slot, kk, nn, K: tl.constexpr, N: tl.constexpr,
    W_BYTES: tl.constexpr, S_BYTES: tl.constexpr, USE_PRMT: tl.constexpr,
):
    if not USE_PRMT:
        return _load_mxfp4_reference(W, S, slot, kk, nn, K, N, W_BYTES, S_BYTES)
    # K tiles begin on an even element. Load each packed byte and its scale
    # once for both adjacent K elements, then restore the GEMM's [K, N] tile.
    even, _ = tl.split(tl.reshape(kk, (kk.shape[0] // 2, 2)))
    wp = W.to(tl.pointer_type(tl.uint8))
    sp = S.to(tl.pointer_type(tl.uint8))
    mask = (even[:, None] < K) & (nn[None, :] < N)
    off = (nn[None, :] // 32) * (K // 2 * 32) + (even[:, None] // 2) * 32 + nn[None, :] % 32
    packed = tl.load(wp + slot.to(tl.int64) * W_BYTES + off, mask, 0)
    low, high = _unpack_e2m1_prmt(packed)
    soff = (nn[None, :] // 32) * (K // 32 * 32) + (even[:, None] // 32) * 32 + nn[None, :] % 32
    exponent = tl.load(sp + slot.to(tl.int64) * S_BYTES + soff, mask, 0).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exponent == 0, 5.8774718e-39, scale)
    low = low.to(tl.float32) * scale
    high = high.to(tl.float32) * scale
    return tl.reshape(tl.permute(tl.join(low, high), (0, 2, 1)), (kk.shape[0], nn.shape[0]))


@triton.jit
def gate_up(
    X,
    Slots,
    GW,
    GS,
    UW,
    US,
    Partial,
    N_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    W_BYTES: tl.constexpr,
    S_BYTES: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 64,
    BK_ROUTES: tl.constexpr = 8,
    USE_PRMT: tl.constexpr = True,
):
    leader, col, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot, rows, routes, live, owns = _group_rows(
        Slots, leader, N_ROWS, TOPK, BM, BK_ROUTES
    )
    if slot < 0 or not owns:
        return
    nn = col * BN + tl.arange(0, BN)
    ag = tl.zeros((BM, BN), tl.float32)
    au = tl.zeros((BM, BN), tl.float32)
    tiles: tl.constexpr = triton.cdiv(H, BK * SPLIT_K)
    for it in range(tiles):
        kk = (split * tiles + it) * BK + tl.arange(0, BK)
        x = tl.load(
            X + rows[:, None] * H + kk[None, :], live[:, None] & (kk[None, :] < H), 0
        ).to(tl.bfloat16)
        g = _load_mxfp4(GW, GS, slot, kk, nn, H, I, W_BYTES, S_BYTES, USE_PRMT).to(tl.bfloat16)
        u = _load_mxfp4(UW, US, slot, kk, nn, H, I, W_BYTES, S_BYTES, USE_PRMT).to(tl.bfloat16)
        ag = tl.dot(x, g, ag)
        au = tl.dot(x, u, au)
    off = (split * N_ROWS * TOPK + routes[:, None]) * (2 * I) + nn[None, :]
    mask = live[:, None] & (nn[None, :] < I)
    tl.store(Partial + off, ag, mask)
    tl.store(Partial + off + I, au, mask)


@triton.jit
def activate(
    Partial,
    Slots,
    Activated,
    N_ROUTES: tl.constexpr,
    I: tl.constexpr,
    SPLIT_K: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr = 256,
):
    route = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = tl.load(Slots + route) >= 0
    mask = live & (col < I)
    g = tl.full((BLOCK,), 0, tl.float32)
    u = tl.full((BLOCK,), 0, tl.float32)
    for split in range(SPLIT_K):
        off = (split * N_ROUTES + route) * (2 * I) + col
        g += tl.load(Partial + off, mask, 0)
        u += tl.load(Partial + off + I, mask, 0)
    if LIMIT > 0:
        g = tl.minimum(tl.maximum(g, -LIMIT), LIMIT)
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
    a = u * (g / (1.0 + tl.exp(-g)))
    tl.store(Activated + route * I + col, a.to(tl.bfloat16), col < I)


@triton.jit
def down(
    Activated,
    Slots,
    Weights,
    DW,
    DS,
    Out,
    N_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    W_BYTES: tl.constexpr,
    S_BYTES: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 64,
    BK_ROUTES: tl.constexpr = 8,
    USE_PRMT: tl.constexpr = True,
):
    leader, col = tl.program_id(0), tl.program_id(1)
    slot, rows, routes, live, owns = _group_rows(
        Slots, leader, N_ROWS, TOPK, BM, BK_ROUTES
    )
    if slot < 0 or not owns:
        return
    nn = col * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for it in range(triton.cdiv(I, BK)):
        kk = it * BK + tl.arange(0, BK)
        a = tl.load(
            Activated + routes[:, None] * I + kk[None, :],
            live[:, None] & (kk[None, :] < I),
            0,
        ).to(tl.bfloat16)
        w = _load_mxfp4(DW, DS, slot, kk, nn, I, H, W_BYTES, S_BYTES, USE_PRMT).to(tl.bfloat16)
        acc = tl.dot(a, w, acc)
    weight = tl.load(Weights + routes, live, 0).to(tl.float32)
    tl.store(
        Out + routes[:, None] * H + nn[None, :],
        acc * weight[:, None],
        live[:, None] & (nn[None, :] < H),
    )


@triton.jit
def combine(
    Parts, Slots, Out, TOPK: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr = 256
):
    row, block = tl.program_id(0), tl.program_id(1)
    col = block * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for j in range(TOPK):
        route = row * TOPK + j
        live = tl.load(Slots + route) >= 0
        acc += tl.load(Parts + route * H + col, live & (col < H), 0)
    tl.store(Out + row * H + col, acc, col < H)


def launch(hidden, slot_table, weights, geo, partial, activated, parts, out, *, use_prmt=True):
    """Run on the caller's pool stream, using preallocated graph-safe buffers."""
    rows, h = hidden.shape
    i, topk, splits = geo["I"], geo["k"], geo["NGK"]
    routes = rows * topk
    common = dict(
        N_ROWS=rows, TOPK=topk, H=h, I=i, BK_ROUTES=triton.next_power_of_2(topk),
        USE_PRMT=use_prmt,
    )
    gate_up[(routes, triton.cdiv(i, 32), splits)](
        hidden,
        slot_table,
        geo["weight_addr"][0],
        geo["scale_addr"][0],
        geo["weight_addr"][1],
        geo["scale_addr"][1],
        partial,
        W_BYTES=geo["slot_bytes"][0],
        S_BYTES=geo["slot_scale_bytes"][0],
        SPLIT_K=splits,
        **common,
    )
    activate[(routes, triton.cdiv(i, 256))](
        partial,
        slot_table,
        activated,
        N_ROUTES=routes,
        I=i,
        SPLIT_K=splits,
        LIMIT=geo["swiglu_limit"],
    )
    down[(routes, triton.cdiv(h, 64))](
        activated,
        slot_table,
        weights,
        geo["weight_addr"][2],
        geo["scale_addr"][2],
        parts,
        W_BYTES=geo["slot_bytes"][2],
        S_BYTES=geo["slot_scale_bytes"][2],
        **common,
    )
    combine[(rows, triton.cdiv(h, 256))](parts, slot_table, out, TOPK=topk, H=h)
