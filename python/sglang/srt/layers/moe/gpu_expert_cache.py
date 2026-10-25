"""GPU-resident hot-expert cache for LK offloaded MoE layers, misses read over UVA.

Each offloaded layer keeps R expert rows in VRAM in FlashInfer's SM90 layout (W4A16,
or Humming W4A8), the one the GPU-resident layers use, and every expert twice in
pinned host memory: that layout (the source of row refills) and the checkpoint
layout (read over UVA by the overflow kernel). ``slot_of`` [E] / ``row_expert`` [R]
map experts to rows and back, ``score`` [E] holds decayed route counts.

Residency is decided on the GPU, in the forward that misses: one kernel admits up to
C missed experts (highest score first) into the rows of experts this forward does
not route (empty, else lowest score), copies them in from pinned host memory, and
CUTLASS then computes hits and admitted misses together. A miss's copy is its swap:
no background swapping, no second copy. Misses beyond C (or with no evictable row)
run in the UVA kernel on a side stream. Rows change only in stream order on the
forward's own stream, so no reader can see a row being refilled.

A background thread only reads the counters and logs them; its CUDA calls run under
``park()``, which graph capture and graph replay launch also hold (the same rule as
the lk_moe pump).
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Optional

import msgspec
import torch

logger = logging.getLogger(__name__)

_PARK_LOCK = threading.Lock()


@contextmanager
def park():
    """Exclude the logging thread's CUDA calls; held around capture and replay launch."""
    with _PARK_LOCK:
        yield


def _to_cutlass_layout(w13, s13, w2, s2, *, fp8: bool):
    """Checkpoint-layout uint8 CUDA experts [c, ...] (w13 gate rows first) -> FlashInfer SM90.

    W4A16: (w13, s13, w2, s2). W4A8 (Humming): the same four plus each expert's FP32
    residual scales of w13 and w2 as [c, 4] bytes, already times Humming's 2^6.
    """
    from flashinfer.fused_moe import (
        interleave_moe_scales_for_sm90_mixed_gemm,
        interleave_moe_weights_for_sm90_mixed_gemm,
    )

    half = w13.shape[1] // 2

    def up_first(t):
        # GPU-resident layers load w13 as [up; gate] (load_up_proj_weight_first).
        return torch.cat([t[:, half:], t[:, :half]], dim=1)

    w13, s13, w2, s2 = up_first(w13), up_first(s13), w2.contiguous(), s2.contiguous()
    if not fp8:
        return (
            interleave_moe_weights_for_sm90_mixed_gemm(w13, "fp4"),
            interleave_moe_scales_for_sm90_mixed_gemm(s13, group_size=32),
            interleave_moe_weights_for_sm90_mixed_gemm(w2, "fp4"),
            interleave_moe_scales_for_sm90_mixed_gemm(s2, group_size=32),
        )
    from flashinfer.fused_moe import preprocess_moe_weights_for_sm90_mixed_gemm_humming

    w13_il, s13_il, r13 = preprocess_moe_weights_for_sm90_mixed_gemm_humming(w13, s13)
    w2_il, s2_il, r2 = preprocess_moe_weights_for_sm90_mixed_gemm_humming(w2, s2)

    def residual_bytes(r):
        return (r.to(torch.float32) * 64.0).contiguous().view(torch.uint8).view(-1, 4)

    return w13_il, s13_il, w2_il, s2_il, residual_bytes(r13), residual_bytes(r2)


def _build_row_source(host_tensors, *, fp8: bool, device, chunk: int = 16):
    """Pinned host copy of every expert in the row layout; ~0.3 GiB VRAM transient."""
    num_experts = host_tensors[0].shape[0]
    first = _to_cutlass_layout(*(t[:1].to(device) for t in host_tensors), fp8=fp8)
    out = [torch.empty((num_experts, *f.shape[1:]), dtype=torch.uint8, pin_memory=True)
           for f in first]
    for start in range(0, num_experts, chunk):
        stop = min(num_experts, start + chunk)
        converted = _to_cutlass_layout(
            *(t[start:stop].to(device) for t in host_tensors), fp8=fp8)
        for dst, src in zip(out, converted):
            dst[start:stop].copy_(src)
    return out


class LayerCache:
    """VRAM rows, the expert/row tables, scores and counters of one layer."""

    def __init__(self, *, layer_id: int, host_tensors, num_rows: int, max_copy: int,
                 score_decay: float, fp8: bool, device):
        # Lazy: park() must stay importable without the lk_moe wheel.
        from sglang.kernels.ops.moe.lk_verify_uva import (
            MISS_BUCKETS,
            STAT_COUNT,
            register_host_experts,
        )

        w13, s13, w2, s2 = host_tensors
        self.layer_id = layer_id
        self.num_experts = w13.shape[0]
        self.num_rows = min(num_rows, self.num_experts)
        self.score_decay = score_decay
        self.fp8 = fp8
        # Checkpoint layout, read in place over UVA for misses that are not admitted.
        self.experts = register_host_experts(w13, s13, w2, s2, device=device)
        self._host = host_tensors
        self._row_source = _build_row_source(host_tensors, fp8=fp8, device=device)
        self.slabs = [
            torch.empty((self.num_rows, *t.shape[1:]), dtype=torch.uint8, device=device)
            for t in self._row_source
        ]
        words = [t[0].numel() // 4 for t in self._row_source]
        assert all(t[0].numel() % 4 == 0 for t in self._row_source)
        self.max_words = max(words)
        self.copy_words = torch.tensor(words, dtype=torch.int32, device=device)
        # Pinned (cudaHostAlloc) host memory has the same address on the device under UVA.
        self.copy_src = torch.tensor([t.data_ptr() for t in self._row_source],
                                     dtype=torch.int64, device=device)
        self.copy_dst = torch.tensor([slab.data_ptr() for slab in self.slabs],
                                     dtype=torch.int64, device=device)
        # The admit kernel's copy list has a power-of-two length (a tl.arange).
        copies = 1 << max(0, max_copy - 1).bit_length()
        self.copy_row = torch.full((copies,), -1, dtype=torch.int32, device=device)
        self.copy_expert = torch.full((copies,), -1, dtype=torch.int32, device=device)
        self.counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        self.score = torch.zeros(self.num_experts, dtype=torch.float32, device=device)
        self.stats = torch.zeros(STAT_COUNT, dtype=torch.int32, device=device)
        self.miss_hist = torch.zeros(MISS_BUCKETS, dtype=torch.int32, device=device)
        self._counters_host = torch.zeros(STAT_COUNT + MISS_BUCKETS, dtype=torch.int32).pin_memory()
        # Cold start: the first R experts; misses replace them by score.
        self.slot_of = torch.full((self.num_experts,), -1, dtype=torch.int32, device=device)
        self.slot_of[: self.num_rows] = torch.arange(self.num_rows, dtype=torch.int32,
                                                     device=device)
        self.row_expert = torch.arange(self.num_rows, dtype=torch.int32, device=device)
        for row in range(self.num_rows):
            self._copy_in(row, row)
        torch.cuda.synchronize(device)

    def host_bytes(self) -> int:
        return sum(t.numel() for t in (*self._host, *self._row_source))

    def row_bytes(self) -> int:
        return sum(t[0].numel() for t in self._row_source)

    def _copy_in(self, row: int, expert: int) -> None:
        """Enqueue the H2D copy of ``expert`` into ``row`` on the current stream."""
        for slab, source in zip(self.slabs, self._row_source):
            slab[row].copy_(source[expert], non_blocking=True)

    def read_counters_async(self) -> None:
        n = self.stats.numel()
        self._counters_host[:n].copy_(self.stats, non_blocking=True)
        self._counters_host[n:].copy_(self.miss_hist, non_blocking=True)

    def counters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(stats, miss_hist) as last read, int64, since process start (mod 2^32)."""
        counters = self._counters_host.to(torch.int64) & 0xFFFFFFFF
        return counters[: self.stats.numel()], counters[self.stats.numel():]


class CacheLogger:
    """One per process: logs every registered layer's counters periodically."""

    _instance: Optional["CacheLogger"] = None

    def __init__(self, *, device, log_interval_s):
        self._device = device
        self._log_interval_s = log_interval_s
        self._layers: list[LayerCache] = []
        self._prev: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._stream = torch.cuda.Stream(device=device)
        if log_interval_s > 0:
            self._thread = threading.Thread(target=self._run, name="gpu-expert-cache",
                                            daemon=True)
            self._thread.start()

    @classmethod
    def register(cls, layer: LayerCache, **config) -> None:
        if cls._instance is None:
            cls._instance = cls(**config)
        cls._instance._layers.append(layer)

    def _read(self) -> None:
        event = torch.cuda.Event()
        with _PARK_LOCK, torch.cuda.stream(self._stream):
            for layer in self._layers:
                layer.read_counters_async()
            event.record(self._stream)
        while True:
            with _PARK_LOCK:
                if event.query():
                    return
            time.sleep(0.0005)

    def _log(self, seconds: float) -> None:
        from sglang.kernels.ops.moe.lk_verify_uva import (
            STAT_ADMITTED,
            STAT_OVERFLOW,
            STAT_ROUTES,
            STAT_ROW_HITS,
        )

        self._read()
        total, miss_hist = 0, 0
        for layer in self._layers:
            stats, hist = layer.counters()
            prev = self._prev.get(layer.layer_id, (stats * 0, hist * 0))
            self._prev[layer.layer_id] = (stats, hist)
            total = total + ((stats - prev[0]) & 0xFFFFFFFF)
            miss_hist = miss_hist + ((hist - prev[1]) & 0xFFFFFFFF)
        routes = int(total[STAT_ROUTES])
        admitted = int(total[STAT_ADMITTED])
        mib_s = admitted * self._layers[0].row_bytes() / 2**20 / max(seconds, 1e-9)
        logger.info(
            "[gpu-expert-cache] %d layers, last %.0fs: graph routes=%d row-hit=%.1f%% | "
            "missed experts: admitted=%d (%.0f MiB/s) overflow(UVA)=%d",
            len(self._layers), seconds, routes,
            100 * int(total[STAT_ROW_HITS]) / max(routes, 1), admitted, mib_s,
            int(total[STAT_OVERFLOW]))
        forwards = int(miss_hist.sum())
        if forwards == 0:
            return
        cdf = torch.cumsum(miss_hist, 0).double() / forwards

        def quantile(q):
            return int(torch.searchsorted(cdf, torch.tensor(q, dtype=torch.float64)))

        max_copy = self._layers[0].copy_row.numel()
        beyond = 1.0 - float(cdf[min(max_copy, cdf.numel() - 1)])
        nonzero = [(i, int(c)) for i, c in enumerate(miss_hist.tolist()) if c]
        logger.info(
            "[gpu-expert-cache] distinct missed experts per layer-forward (graph): "
            "p50=%d p90=%d p99=%d p99.9=%d; > max copy(%d): %.2f%% of %d; histogram %s",
            quantile(0.5), quantile(0.9), quantile(0.99), quantile(0.999), max_copy,
            100 * beyond, forwards, nonzero)

    def _run(self) -> None:
        torch.cuda.set_device(self._device)
        last = time.time()
        while True:
            time.sleep(self._log_interval_s)
            try:
                self._log(time.time() - last)
            except Exception:
                logger.exception("gpu-expert-cache: logging failed; stopping it")
                return
            last = time.time()


class ForwardBuffers:
    """Process-wide buffers of the graph path; layers run one after another."""

    def __init__(self, *, num_experts, max_rows, top_k, hidden, intermediate, device):
        from sglang.kernels.ops.moe.lk_verify_uva import UvaScratch

        routes = max_rows * top_k
        self.max_rows = max_rows
        self.scratch = UvaScratch.allocate(
            num_experts=num_experts, max_slots=routes, intermediate=intermediate,
            device=device)
        self.gout_part = torch.zeros((routes, hidden), dtype=torch.float32, device=device)
        self.slot_ids = torch.zeros(routes, dtype=torch.int32, device=device)
        self.miss_tab = torch.zeros(routes, dtype=torch.int32, device=device)
        self.side_stream = torch.cuda.Stream(device=device)


class MoeParams(msgspec.Struct, frozen=True):
    swiglu_limit: Optional[torch.Tensor]  # [rows] fp32 for CUTLASS, or None
    limit: float  # the same clamp for the UVA kernel; 0 disables it
    tp_size: int
    tp_rank: int
    fc2_act_scale: Optional[torch.Tensor]  # W4A8 only: FP32 scalar FC2 activation scale


def _cutlass(cache: LayerCache, x, slot_ids, weights, params: MoeParams):
    """Routes whose id is a row; ``num_rows`` marks a skipped route."""
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    from sglang.srt.environ import envs
    from sglang.srt.utils.common import next_power_of_2

    w13, s13, w2, s2 = cache.slabs[:4]
    if cache.fp8:
        r13, r2 = (t.view(torch.float32).view(-1) for t in cache.slabs[4:])
        quant_scales = [s13.view(torch.int32), r13, params.fc2_act_scale,
                        s2.view(torch.int32), r2]
        extra = {"use_wfp4afp8_humming": True}
    else:
        quant_scales = [s13.view(torch.int32), s2.view(torch.int32)]
        extra = {}
    out = torch.empty_like(x)
    cutlass_fused_moe(
        input=x,
        token_selected_experts=slot_ids,
        token_final_scales=weights,
        fc1_expert_weights=w13,
        fc2_expert_weights=w2,
        output_dtype=x.dtype,
        quant_scales=quant_scales,
        swiglu_limit=params.swiglu_limit,
        tp_size=params.tp_size,
        tp_rank=params.tp_rank,
        # Rows are rank 0's experts of two ranks; the skip id belongs to rank 1.
        ep_size=2,
        ep_rank=0,
        use_w4_group_scaling=True,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=next_power_of_2(x.shape[0]),
        output=out,
        use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
        **extra,
    )
    return out


# Programs of the admitted-miss copy: nothing else runs meanwhile, so one per H100 SM.
# Arbitrary; PCIe, not SM count, bounds it.
_COPY_PROGRAMS = 132


def _graph_forward(cache, *, x, topk_ids, topk_weights, buffers: ForwardBuffers,
                   params: MoeParams):
    """Fixed launches: hits and admitted misses in CUTLASS, the rest in the UVA kernel."""
    from sglang.kernels.ops.moe.lk_verify_uva import (
        admit_misses,
        copy_experts,
        launch_draft_miss_uva,
    )

    rows, top_k = topk_ids.shape
    n = rows * top_k
    assert n <= buffers.slot_ids.numel(), (rows, buffers.max_rows)
    slot_ids, miss_tab = buffers.slot_ids[:n], buffers.miss_tab[:n]
    gout_part = buffers.gout_part[:n]
    admit_misses(topk_ids=topk_ids, slot_of=cache.slot_of, row_expert=cache.row_expert,
                 score=cache.score, counts=cache.counts, copy_row=cache.copy_row,
                 copy_expert=cache.copy_expert, slot_ids=slot_ids, miss_tab=miss_tab,
                 stats=cache.stats, miss_hist=cache.miss_hist, decay=cache.score_decay)
    main = torch.cuda.current_stream()
    side_stream = buffers.side_stream
    side_stream.wait_stream(main)
    with torch.cuda.stream(side_stream):
        gout_part.zero_()
        launch_draft_miss_uva(
            hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights,
            slot_tab=miss_tab, gout_part=gout_part, rows_per_request=1,
            host=cache.experts, scratch=buffers.scratch, activation=0, alpha=1.702,
            limit=params.limit, include_first_row=True)
    copy_experts(copy_row=cache.copy_row, copy_expert=cache.copy_expert,
                 src_ptrs=cache.copy_src, dst_ptrs=cache.copy_dst, words=cache.copy_words,
                 max_words=cache.max_words, num_programs=_COPY_PROGRAMS)
    hits = _cutlass(cache, x, slot_ids.view(rows, top_k), topk_weights, params)
    main.wait_stream(side_stream)
    return gout_part.view(rows, top_k, -1).sum(dim=1).add_(hits)


def _eager_forward(cache, *, x, topk_ids, topk_weights, params: MoeParams):
    """Any batch outside a graph: admit every miss, in groups when rows run short."""
    ids = topk_ids.long()
    local = ids >= 0
    counts = torch.bincount(ids[local], minlength=cache.num_experts)
    cache.score.mul_(cache.score_decay).add_(counts.to(torch.float32))
    routed = counts > 0
    missing = torch.nonzero(routed & (cache.slot_of < 0)).squeeze(1)
    missing = missing[torch.argsort(cache.score[missing], descending=True)]
    holder = cache.row_expert.long()
    victim_key = torch.where(holder >= 0, cache.score[holder.clamp(min=0)], -1.0)
    victim_key = torch.where(routed[holder.clamp(min=0)] & (holder >= 0), float("inf"),
                             victim_key)
    victims = torch.argsort(victim_key)[: int((victim_key < float("inf")).sum())]
    missing, victims = missing.tolist(), victims.tolist()
    skip = cache.num_rows
    out = None
    group_size = max(len(victims), 1)
    for start in range(0, max(len(missing), 1), group_size):
        group = missing[start:start + group_size]
        assert not group or victims, "no evictable row: every row's expert is routed"
        if group:
            rows_t = torch.tensor(victims[: len(group)], device=x.device)
            new_t = torch.tensor(group, device=x.device)
            old = cache.row_expert[rows_t].long()
            cache.slot_of[old[old >= 0]] = -1
            cache.slot_of[new_t] = rows_t.to(torch.int32)
            cache.row_expert[rows_t] = new_t.to(torch.int32)
            for row, expert in zip(victims, group):
                cache._copy_in(row, expert)
        slot = torch.where(local, cache.slot_of[ids.clamp(min=0)], -1).long()
        if start > 0:
            # Later groups compute only their own experts; the hits ran with the first.
            in_group = torch.zeros(cache.num_experts, dtype=torch.bool, device=x.device)
            in_group[torch.tensor(group, device=x.device)] = True
            slot = torch.where(in_group[ids.clamp(min=0)] & local, slot, -1)
        slot_ids = torch.where(slot >= 0, slot, skip).to(torch.int32)
        part = _cutlass(cache, x, slot_ids, topk_weights, params).float()
        out = part if out is None else out.add_(part)
    return out


def cache_forward(cache: LayerCache, *, hidden_states, topk_ids, topk_weights,
                  buffers: ForwardBuffers, params: MoeParams, graph: bool):
    """MoE output [rows, H] fp32 of one offloaded layer; ``graph`` uses fixed buffers."""
    x = hidden_states.contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    if graph:
        return _graph_forward(cache, x=x, topk_ids=topk_ids, topk_weights=topk_weights,
                              buffers=buffers, params=params)
    return _eager_forward(cache, x=x, topk_ids=topk_ids, topk_weights=topk_weights,
                          params=params)
