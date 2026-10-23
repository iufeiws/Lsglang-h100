"""
copyright (c) 2025-2026 guqiong9696@gmail.com, All Rights Reserved.
MOE V2 - Expert-pool GPU-side shared decoders and activation, format-driven
(mirrors moe_v2_activate.h / csrc/cuda/moe_v2_gpu_pool_kernels.cuh).
"""
import torch
import triton
import triton.language as tl


# tl.dot k-block for both group-GEMMs: the only bit-exact value on the
# dual-domain branch and the fastest on Blackwell. A tuned constant.
KB = 16


def m_tile():
    """Group-GEMM row tile, the ONE home shared by the staging and resident
    drivers (it sizes row_of/tile_id and enters the triton key, so the two
    halves must never disagree). Tuned per architecture, not a user knob:
    SM86 measures 2.5% better at 32 (2026-10-16 serve A/B), Blackwell
    regresses at 32 (tl.dot M32 ~3.5x slower). New architectures get a
    row here."""
    return 32 if torch.cuda.get_device_capability()[0] == 8 else 16


@triton.jit
def _e2m1_bits(packed_byte):
    """e2m1 4-bit code (uint8) -> fp16 by bit-twiddle (2x faster than a where
    tree). magnitude mag=code&7 has fp16 exponent (mag>>1)+14 and mantissa bit
    mag&1 (except mag==1 -> 0.5); sign is code bit 3."""
    code = packed_byte & 0xF
    mag = code & 0x7
    mag_i = mag.to(tl.int32)
    h = (((mag_i >> 1) + 14) << 10) | ((mag_i & 1) << 9)
    h = tl.where(mag == 1, 0x3800, h)
    h = tl.where(mag == 0, 0, h).to(tl.uint16)
    v = h.to(tl.float16, bitcast=True)
    return tl.where(code < 8, v, -v)


@triton.jit
def _e4m3(raw):
    """fp8 e4m3fn block-scale byte -> fp32 (software decode; no fp8 bitcast on
    every arch). sign(1)|exp(4,bias7)|mant(3)."""
    bits = raw.to(tl.int32)
    sign = (bits >> 7) & 1
    expo = (bits >> 3) & 0xF
    mant = bits & 7
    mag = tl.where(expo == 0,
                   tl.exp2(-6.0) * (mant.to(tl.float32) / 8.0),
                   tl.exp2((expo - 7).to(tl.float32)) * (1.0 + mant.to(tl.float32) / 8.0))
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def _e8m0(raw):
    """ue8m0 (e8m0) block-scale byte -> fp32: value = 2^(code-127), decoded by
    placing the exponent straight into an fp32 word (mantissa=0). code==255 is
    NaN upstream; here it becomes inf, same as the C++ e8m0_to_f32 tables."""
    code = raw.to(tl.uint32)
    v = (code << 23).to(tl.float32, bitcast=True)  # 2^(code-127) for code>0
    return tl.where(code == 0, 5.8774718e-39, v)   # 2^-127 (bitcast would give 0.0)


@triton.jit
def _act(gate, up, ACTIVATION: tl.constexpr, ALPHA: tl.constexpr, LIMIT: tl.constexpr):
    # Mirror moe_v2_activate.h (scalar path), which is what the CPU side runs.
    if ACTIVATION == 0:
        if LIMIT > 0:
            gate = tl.minimum(tl.maximum(gate, -LIMIT), LIMIT)
            up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        return up * (gate / (1 + tl.exp(-gate)))
    if ACTIVATION == 1:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        return (up + 1.0) * (gate / (1 + tl.exp(-gate * ALPHA)))
    if LIMIT > 0:
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    relu = tl.maximum(up, 0.0)
    return relu * relu
