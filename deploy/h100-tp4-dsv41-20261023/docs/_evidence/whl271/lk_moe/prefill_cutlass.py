"""
copyright (c) 2025-2026 guqiong9696@gmail.com, All Rights Reserved.
MOE V2 - CUTLASS grouped-GEMM GPU prefill driver (restored option).

Same framework contract as StagingPrefill/PrefillResident:
    drv = CutlassPrefill.probe(moe)            # None -> stays CPU
    ev, out = drv.prefill_gpu(hs, topk_ids, topk_weights)

Unlike the Triton staging tier, this engine drives the restored C++ CUTLASS
path (MOE_V2::gpu_prefill -> sort + grouped GEMM) directly off the layer
weights. The framework selects it via cfg.use_gpu_prefill (engine switch
LK_GPU_PREFILL_ENGINE, default cutlass); when a layer is a pool member the
engine stages through borrowed pool VRAM, which the framework brackets with
frozen_chunk() around one prefill forward.
"""

import contextlib
import weakref

import torch

try:
    from sgl_kernel.logging_utils import logger  # optional, framework side
except Exception:  # pragma: no cover - framework-free import
    import logging
    logger = logging.getLogger("lk_moe.prefill_cutlass")

# Instances whose cfg may borrow pool VRAM; frozen_chunk() brackets one
# prefill forward so staging rows stay valid across every layer of the chunk.
_INSTANCES = weakref.WeakSet()
_CHUNK_DEPTH = 0


def register_instance(moe):
    """Register a cfg-backed lk_moe instance for forward-length borrow windows.

    Idempotent; the driver registers itself too. Frameworks that call the C++
    gpu_prefill directly (vLLM) register here right after construction.
    """
    _INSTANCES.add(moe)


@contextlib.contextmanager
def frozen_chunk(stream=None):
    # Re-entrant: a nested model forward (e.g. a spec-decode drafter) must not
    # re-open or re-pin an already frozen chunk.
    global _CHUNK_DEPTH
    live = [m for m in list(_INSTANCES) if m is not None]
    opened = _CHUNK_DEPTH == 0 and bool(live)
    s = stream
    restored = []
    if opened:
        s = stream if stream is not None else torch.cuda.current_stream().cuda_stream
        for m in live:
            m.gpu_prefill_freeze(s)
            # Existing switch: force the decode half to CPU for the whole
            # window (it must not read rows the prefill staging overwrites).
            if m.pool_gpu_live:
                restored.append(m)
                m.pool_gpu_report(False)
    _CHUNK_DEPTH += 1
    try:
        yield
    finally:
        _CHUNK_DEPTH -= 1
        if opened:
            for m in live:
                m.gpu_prefill_thaw(s)
            for m in restored:
                m.pool_gpu_report(True)


class CutlassPrefill:
    """One per FusedMoE layer; probe() returns a driver or None (CPU)."""

    _buf_cache = {}  # dev -> fp32 output scratch [cap, H]

    @staticmethod
    def probe(moe):
        cfg = moe.cfg
        if not hasattr(moe, "gpu_prefill"):
            logger.warning(
                "cutlass prefill layer %s: build lacks gpu_prefill binding "
                "-> prefill stays CPU", getattr(cfg, "layer_id", "?"))
            return None
        try:
            return CutlassPrefill(moe)
        except Exception as e:  # broken deployment: log loudly, keep CPU
            logger.warning(
                "cutlass prefill layer %s: probe failed (%s) -> prefill "
                "stays CPU", getattr(cfg, "layer_id", "?"), e)
            return None

    def __init__(self, moe):
        cfg = moe.cfg
        self.dev = torch.device("cuda", int(cfg.gpu_id))
        self.H = int(cfg.hidden_size)
        self.topk = int(cfg.top_k)
        self.layer = int(cfg.layer_id)
        self.max_batch = int(getattr(cfg, "max_batch_size", 0)) or 8192
        self.block = max(1, self.max_batch)
        self.moe_ref = weakref.ref(moe)
        register_instance(moe)
        # Engine gate must be on so the ctor allocated the StaticGPUMemory
        # arena; if the framework built the layer without it, flag now.
        if not bool(getattr(cfg, "use_gpu_prefill", False)):
            raise RuntimeError(
                "cutlass prefill: cfg.use_gpu_prefill is False, GPU arena "
                "was never initialized")

    def _out_buf(self, M, dtype):
        # C++ gpu_prefill writes ActType (== the layer's activation dtype).
        key = (self.dev, dtype)
        buf = CutlassPrefill._buf_cache.get(key)
        if buf is None or buf.size(0) < M:
            buf = torch.empty(max(M, self.block), self.H,
                              dtype=dtype, device=self.dev)
            CutlassPrefill._buf_cache[key] = buf
        return buf

    def prefill_gpu(self, hidden_states, topk_ids, topk_weights):
        moe = self.moe_ref()
        if moe is None:
            raise RuntimeError(f"cutlass prefill layer {self.layer}: moe gone")
        M = hidden_states.size(0)
        ids = topk_ids.reshape(-1)
        ids = ids if ids.dtype == torch.int32 else ids.to(torch.int32)
        w = topk_weights.reshape(-1)
        w = w if w.dtype == torch.float32 else w.to(torch.float32)
        x = hidden_states.contiguous()
        stream = torch.cuda.current_stream(self.dev)
        if M <= self.block:
            out = self._out_buf(M, hidden_states.dtype)[:M]
            moe.gpu_prefill(x.data_ptr(), out.data_ptr(), ids.data_ptr(),
                            w.data_ptr(), M, self.topk, stream.cuda_stream)
            ev = torch.cuda.Event()
            ev.record(stream)
            return ev, out
        # Over the arena cap: WALK token blocks on the fp32 output (the cutlass
        # gpu_prefill is fully synchronous per call, so a plain loop is safe).
        out = torch.empty(M, self.H, dtype=hidden_states.dtype, device=self.dev)
        for m0 in range(0, M, self.block):
            m1 = min(m0 + self.block, M)
            xb = x[m0:m1].contiguous()
            if topk_ids.dim() == 2:
                ib, wb = topk_ids[m0:m1], topk_weights[m0:m1]
            else:
                ib = ids[m0 * self.topk:m1 * self.topk]
                wb = w[m0 * self.topk:m1 * self.topk]
            moe.gpu_prefill(xb.data_ptr(), out[m0:m1].data_ptr(),
                            ib.data_ptr(), wb.data_ptr(), m1 - m0, self.topk,
                            stream.cuda_stream)
        ev = torch.cuda.Event()
        ev.record(stream)
        return ev, out
