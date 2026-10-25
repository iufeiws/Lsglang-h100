"""GPU-resident hot-expert cache for LK offloaded MoE layers, misses read over UVA.

Each offloaded layer keeps S expert slots in VRAM in FlashInfer's SM90 W4A16 layout,
the one the GPU-resident layers use, and every expert twice in pinned host memory:
that layout (the source of slot refills) and the checkpoint layout (read over UVA by
the miss kernel). ``LayerCache.slot_of`` maps each expert to its slot or -1. Per
forward, one kernel reads it once per route: slotted routes run in FlashInfer's
CUTLASS MoE kernel, the others in the UVA kernel on a side stream, so no route is
computed twice or dropped. The CPU only stores weights.

Residency adapts in a background thread from route counts the forward accumulates.
A swap never changes bytes a kernel may be reading: the outgoing expert is first
unmapped (-1), then every launch made before that point must finish, and only then
is its slot overwritten; the incoming expert is mapped after its copy has landed.

The thread issues CUDA calls only under ``park()``, which graph capture and graph
replay launch also hold (the same rule as the lk_moe pump), and starts swapping only
after the first replay, i.e. once every capture and its warmups are done. Eager
forwards (prefill tails of late layers) hold ``park()`` from reading ``slot_of`` to
their last launch, since they sync with the host in between; their streams join the
fence, so a fence recorded afterwards covers every launch that used the old table.
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
# Streams that launch cache reads (graph replays and eager forwards); fenced before a slot is reused.
_replay_streams: list[torch.cuda.Stream] = []
_first_replay = threading.Event()


@contextmanager
def park():
    """Exclude the cache thread's CUDA calls; held around capture and replay launch."""
    with _PARK_LOCK:
        yield


def _add_stream(stream: torch.cuda.Stream) -> None:
    """Fence ``stream`` before slot reuse; the caller holds ``_PARK_LOCK``."""
    if not any(s == stream for s in _replay_streams):
        _replay_streams.append(stream)


def note_replay(stream: torch.cuda.Stream) -> None:
    """Record a graph-replay stream; call inside ``park()`` right before the launch."""
    _add_stream(stream)
    _first_replay.set()


def plan_swaps(
    scores: torch.Tensor, slot_owner: list[int], *, budget: int, min_gain: float
) -> list[tuple[int, int, int]]:
    """(slot, outgoing expert or -1, incoming expert) of up to ``budget`` swaps.

    The hottest non-resident experts replace the coldest residents (empty slots
    first) while the incoming score exceeds ``min_gain`` times the outgoing one.
    """
    resident = torch.zeros(scores.numel(), dtype=torch.bool)
    owned = [e for e in slot_owner if e >= 0]
    if owned:
        resident[torch.tensor(owned)] = True
    outside = torch.where(~resident, scores, torch.tensor(float("-inf")))
    incoming = torch.argsort(outside, descending=True).tolist()
    slot_score = [
        float("-inf") if e < 0 else float(scores[e]) for e in slot_owner
    ]
    victims = sorted(range(len(slot_owner)), key=lambda v: slot_score[v])
    swaps = []
    for in_e, v in zip(incoming[:budget], victims):
        gain = float(outside[in_e])
        if gain <= 0 or (slot_owner[v] >= 0 and gain <= min_gain * slot_score[v]):
            break
        swaps.append((v, slot_owner[v], in_e))
    return swaps


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


def _build_slot_source(host_tensors, *, fp8: bool, device, chunk: int = 16):
    """Pinned host copy of every expert in the slot layout; ~0.3 GiB VRAM transient."""
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
    """VRAM slots, staging rows, the expert-to-slot table and route counts of one layer."""

    def __init__(self, *, layer_id: int, host_tensors, num_slots: int, num_staging: int,
                 fp8: bool, device):
        # Lazy: park() must stay importable without the lk_moe wheel.
        from sglang.kernels.ops.moe.lk_verify_uva import (
            MISS_BUCKETS,
            STAT_COUNT,
            register_host_experts,
            stage_items_per_row,
        )

        w13, s13, w2, s2 = host_tensors
        self.layer_id = layer_id
        self.num_experts = w13.shape[0]
        self.num_slots = min(num_slots, self.num_experts)
        self.num_staging = num_staging
        self.fp8 = fp8
        # Checkpoint layout, read in place over UVA for misses beyond the staging rows.
        self.experts = register_host_experts(w13, s13, w2, s2, device=device)
        self._host = host_tensors
        self._slot_source = _build_slot_source(host_tensors, fp8=fp8, device=device)
        # Rows [num_slots, num_slots + num_staging) take missed and prefetched experts;
        # the residency thread never touches them.
        self.slabs = [
            torch.empty((self.num_slots + num_staging, *t.shape[1:]), dtype=torch.uint8,
                        device=device)
            for t in self._slot_source
        ]
        words = [t[0].numel() // 4 for t in self._slot_source]
        assert all(t[0].numel() % 4 == 0 for t in self._slot_source)
        self.max_words = max(words)
        self.stage_words = torch.tensor(words, dtype=torch.int32, device=device)
        # Pinned (cudaHostAlloc) host memory has the same address on the device under UVA.
        self.stage_src = torch.tensor([t.data_ptr() for t in self._slot_source],
                                      dtype=torch.int64, device=device)
        self.stage_dst = torch.tensor([slab[self.num_slots].data_ptr() for slab in self.slabs],
                                      dtype=torch.int64, device=device)
        # Expert held by each staging row (-1: none); survives across forwards.
        rows = max(num_staging, 1)
        self.stage_expert = torch.full((rows,), -1, dtype=torch.int32, device=device)
        # Rows a prefetch is filling, and their finished copy items (of items_per_row).
        self.row_pending = torch.zeros(rows, dtype=torch.int32, device=device)
        self.row_count = torch.zeros(rows, dtype=torch.int32, device=device)
        self.items_per_row = stage_items_per_row(
            num_tensors=len(self._slot_source), max_words=self.max_words)
        # Recorded on the prefetch stream once this layer's prefetch rows are assigned.
        self.prefetch_assigned = torch.cuda.Event()
        self.prefetch_issued = False
        self.slot_owner = [-1] * self.num_slots
        # Pinned mirror of slot_of; written only while no H2D from it is pending.
        self._slot_of_host = torch.full((self.num_experts,), -1, dtype=torch.int32).pin_memory()
        self.slot_of = self._slot_of_host.to(device)
        self.hist = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        self._hist_host = torch.zeros(self.num_experts, dtype=torch.int32).pin_memory()
        self._hist_prev = torch.zeros(self.num_experts, dtype=torch.int64)
        self.scores = torch.zeros(self.num_experts, dtype=torch.float32)
        self.stats = torch.zeros(STAT_COUNT, dtype=torch.int32, device=device)
        self.miss_hist = torch.zeros(MISS_BUCKETS, dtype=torch.int32, device=device)
        self._counters_host = torch.zeros(STAT_COUNT + MISS_BUCKETS, dtype=torch.int32).pin_memory()
        # Cold start: the first S experts; the thread replaces them by route counts.
        for v in range(self.num_slots):
            self._copy_in(v, v)
            self._slot_of_host[v] = v
            self.slot_owner[v] = v
        self.slot_of.copy_(self._slot_of_host)
        torch.cuda.synchronize(device)

    def host_bytes(self) -> int:
        return sum(t.numel() for t in (*self._host, *self._slot_source))

    def expert_bytes(self) -> int:
        return sum(t[0].numel() for t in self._slot_source)

    def _copy_in(self, slot: int, expert: int) -> None:
        """Enqueue the H2D copy of ``expert`` into slab row ``slot`` on the current stream."""
        for slab, source in zip(self.slabs, self._slot_source):
            slab[slot].copy_(source[expert], non_blocking=True)

    def _push_slot_of(self) -> None:
        self.slot_of.copy_(self._slot_of_host, non_blocking=True)

    def read_hist_async(self) -> None:
        self._hist_host.copy_(self.hist, non_blocking=True)

    def read_counters_async(self) -> None:
        n = self.stats.numel()
        self._counters_host[:n].copy_(self.stats, non_blocking=True)
        self._counters_host[n:].copy_(self.miss_hist, non_blocking=True)

    def counters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(stats, miss_hist) as last read, int64, since process start (mod 2^32)."""
        counters = self._counters_host.to(torch.int64) & 0xFFFFFFFF
        return counters[: self.stats.numel()], counters[self.stats.numel():]

    def update_scores(self, decay: float) -> int:
        now = self._hist_host.to(torch.int64) & 0xFFFFFFFF
        delta = (now - self._hist_prev) & 0xFFFFFFFF
        self._hist_prev = now
        self.scores.mul_(decay).add_(delta.to(torch.float32))
        return int(delta.sum())

    def resident_routes(self, delta_scores: torch.Tensor) -> float:
        owned = [e for e in self.slot_owner if e >= 0]
        return float(delta_scores[owned].sum()) if owned else 0.0


class CacheManager:
    """One per process: adapts the residency of every registered layer."""

    _instance: Optional["CacheManager"] = None

    def __init__(self, *, device, interval_s, budget, decay, min_gain, log_interval_s):
        self._device = device
        self._interval_s = interval_s
        self._budget = budget
        self._decay = decay
        self._min_gain = min_gain
        self._log_interval_s = log_interval_s
        self._layers: list[LayerCache] = []
        self._counters_prev: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._stream = torch.cuda.Stream(device=device)
        self._thread = threading.Thread(target=self._run, name="gpu-expert-cache", daemon=True)
        self._thread.start()

    @classmethod
    def register(cls, layer: LayerCache, **config) -> None:
        if cls._instance is None:
            cls._instance = cls(**config)
        cls._instance._layers.append(layer)

    def _wait(self, event: torch.cuda.Event) -> None:
        while True:
            with _PARK_LOCK:
                if event.query():
                    return
            time.sleep(0.0005)

    def _submit(self, fn) -> None:
        """Run ``fn`` on the cache stream under the park lock and wait for it."""
        event = torch.cuda.Event()
        with _PARK_LOCK, torch.cuda.stream(self._stream):
            fn()
            event.record(self._stream)
        self._wait(event)

    def _push_tables(self, layers) -> None:
        def push():
            for layer in layers:
                layer._push_slot_of()

        self._submit(push)

    def _fence_replays(self) -> None:
        """Wait until every graph launched so far has finished."""
        events = []
        with _PARK_LOCK:
            for stream in _replay_streams:
                event = torch.cuda.Event()
                event.record(stream)
                events.append(event)
        for event in events:
            self._wait(event)

    def _round(self, stats) -> None:
        layers = list(self._layers)

        def read():
            for layer in layers:
                layer.read_hist_async()

        self._submit(read)
        plans = []
        for layer in layers:
            prev = layer.scores.clone()
            stats["routes"] += layer.update_scores(self._decay)
            delta = layer.scores - prev * self._decay
            stats["resident"] += layer.resident_routes(delta)
            swaps = plan_swaps(layer.scores, layer.slot_owner,
                               budget=self._budget, min_gain=self._min_gain)
            if swaps:
                plans.append((layer, swaps))
        if not plans:
            return
        for layer, swaps in plans:
            for _, out_e, _ in swaps:
                if out_e >= 0:
                    layer._slot_of_host[out_e] = -1
        touched = [layer for layer, _ in plans]
        self._push_tables(touched)
        self._fence_replays()

        def copy():
            for layer, swaps in plans:
                for slot, _, in_e in swaps:
                    layer._copy_in(slot, in_e)

        self._submit(copy)
        for layer, swaps in plans:
            for slot, _, in_e in swaps:
                layer._slot_of_host[in_e] = slot
                layer.slot_owner[slot] = in_e
            stats["swaps"] += len(swaps)
        self._push_tables(touched)

    def _counter_deltas(self):
        """Per-layer (stats, miss_hist) since the previous call."""
        layers = list(self._layers)

        def read():
            for layer in layers:
                layer.read_counters_async()

        self._submit(read)
        deltas = []
        for layer in layers:
            stats, miss_hist = layer.counters()
            prev = self._counters_prev.get(layer.layer_id, (stats * 0, miss_hist * 0))
            self._counters_prev[layer.layer_id] = (stats, miss_hist)
            deltas.append((layer, (stats - prev[0]) & 0xFFFFFFFF,
                           (miss_hist - prev[1]) & 0xFFFFFFFF))
        return deltas

    def _log(self, stats, seconds: float) -> None:
        from sglang.kernels.ops.moe.lk_verify_uva import (
            STAT_OVERFLOW,
            STAT_PREFETCH_USED,
            STAT_PREFETCHED,
            STAT_ROUTES,
            STAT_SLOT_HITS,
            STAT_STAGE_COPIED,
            STAT_STAGE_KEPT,
        )

        deltas = self._counter_deltas()
        total = sum(d[1] for d in deltas)
        miss_hist = sum(d[2] for d in deltas)
        routes = int(total[STAT_ROUTES])
        swap_mb = stats["swaps"] * (self._layers[0].expert_bytes() if self._layers else 0) / 2**20
        logger.info(
            "[gpu-expert-cache] %d layers, last %.0fs: graph routes=%d slot-hit=%.1f%% | "
            "missed experts: staged-kept=%d copied=%d overflow(UVA)=%d prefetched=%d "
            "(used %d) | "
            "swaps=%d (%.0f MiB/s) | all routes hit=%.1f%%",
            len(self._layers), seconds, routes,
            100 * int(total[STAT_SLOT_HITS]) / max(routes, 1),
            int(total[STAT_STAGE_KEPT]), int(total[STAT_STAGE_COPIED]),
            int(total[STAT_OVERFLOW]), int(total[STAT_PREFETCHED]),
            int(total[STAT_PREFETCH_USED]),
            stats["swaps"], swap_mb / max(seconds, 1e-9),
            100 * stats["resident"] / max(stats["routes"], 1))
        forwards = int(miss_hist.sum())
        if forwards == 0:
            return
        cdf = torch.cumsum(miss_hist, 0).double() / forwards

        def quantile(q):
            return int(torch.searchsorted(cdf, torch.tensor(q, dtype=torch.float64)))

        staging = self._layers[0].num_staging
        beyond = 1.0 - float(cdf[min(staging, cdf.numel() - 1)])
        nonzero = [(i, int(c)) for i, c in enumerate(miss_hist.tolist()) if c]
        logger.info(
            "[gpu-expert-cache] distinct missed experts per layer-forward (graph): "
            "p50=%d p90=%d p99=%d p99.9=%d; > staging(%d): %.2f%% of %d; histogram %s",
            quantile(0.5), quantile(0.9), quantile(0.99), quantile(0.999), staging,
            100 * beyond, forwards, nonzero)

    def _run(self) -> None:
        torch.cuda.set_device(self._device)
        stats = dict(routes=0, resident=0.0, swaps=0)
        last_log = time.time()
        while True:
            time.sleep(self._interval_s)
            if not _first_replay.is_set():
                continue
            try:
                self._round(stats)
                if self._log_interval_s > 0 and time.time() - last_log >= self._log_interval_s:
                    self._log(stats, time.time() - last_log)
                    stats = dict(routes=0, resident=0.0, swaps=0)
                    last_log = time.time()
            except Exception:
                logger.exception("gpu-expert-cache: residency round failed; stopping swaps")
                return


class ForwardBuffers:
    """Process-wide buffers of the graph path; layers run one after another."""

    def __init__(self, *, num_experts, max_rows, top_k, hidden, intermediate, num_staging,
                 device):
        from sglang.kernels.ops.moe.lk_verify_uva import UvaScratch

        routes = max_rows * top_k
        rows = max(num_staging, 1)
        self.max_rows = max_rows
        self.scratch = UvaScratch.allocate(
            num_experts=num_experts, max_slots=routes, intermediate=intermediate,
            device=device)
        self.gout_part = torch.zeros((routes, hidden), dtype=torch.float32, device=device)
        self.slot_ids = torch.zeros(routes, dtype=torch.int32, device=device)
        self.miss_tab = torch.zeros(routes, dtype=torch.int32, device=device)
        self.side_stream = torch.cuda.Stream(device=device)
        self.prefetch_stream = torch.cuda.Stream(device=device)

        def scratch_set():
            return (torch.zeros(num_experts, dtype=torch.int32, device=device),
                    torch.zeros(rows, dtype=torch.int32, device=device),
                    torch.zeros(rows, dtype=torch.int32, device=device))

        # (mark, free_rows, copy_expert): the main and the prefetch stream each own one.
        self.main_set = scratch_set()
        self.prefetch_set = scratch_set()
        # A prefetch was issued in this forward and the prefetch stream is not joined yet.
        self.prefetch_outstanding = False


class MoeParams(msgspec.Struct, frozen=True):
    swiglu_limit: Optional[torch.Tensor]  # [slots + staging] fp32 for CUTLASS, or None
    limit: float  # the same clamp for the UVA kernel; 0 disables it
    tp_size: int
    tp_rank: int
    fc2_act_scale: Optional[torch.Tensor]  # W4A8 only: FP32 scalar FC2 activation scale


class PrefetchTarget:
    """What layer L needs to fetch layer L+1's predicted experts: its router and cache."""

    def __init__(self, *, cache: LayerCache, gate_weight, correction_bias, scoring_func: str,
                 top_k: int, ep_rank: int, programs: int, max_experts: int):
        self.cache = cache
        self.gate_weight = gate_weight
        self.correction_bias = correction_bias
        self.scoring_func = scoring_func
        self.top_k = top_k
        self.expert_offset = ep_rank * cache.num_experts
        self.programs = programs
        self.max_experts = max_experts


_prefetch_targets: dict[int, PrefetchTarget] = {}


def register_prefetch_target(layer_id: int, target: PrefetchTarget) -> None:
    _prefetch_targets[layer_id] = target


def _predict_local_experts(target: PrefetchTarget, hidden_states):
    """Top-k of layer L+1's router applied to layer L's MoE input, as local ids or -1."""
    weight = target.gate_weight
    logits = torch.nn.functional.linear(hidden_states.to(weight.dtype), weight).float()
    if target.scoring_func == "sqrtsoftplus":
        scores = torch.sqrt(torch.nn.functional.softplus(logits))
    elif target.scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    else:
        scores = torch.softmax(logits, dim=-1)
    if target.correction_bias is not None:
        scores = scores + target.correction_bias.float()
    local = torch.topk(scores, target.top_k, dim=-1).indices - target.expert_offset
    in_rank = (local >= 0) & (local < target.cache.num_experts)
    return torch.where(in_rank, local, -1).to(torch.int32).contiguous()


def prefetch_next_layer(layer_id: int, hidden_states, buffers: ForwardBuffers) -> None:
    """Graph path of layer ``layer_id``: start copying layer ``layer_id + 1``'s most-voted
    predicted misses into its staging rows on the prefetch stream.

    Layer ``layer_id + 1`` waits only for the row assignment and then for the rows it
    actually reads or refills; the stream is joined after the last cache layer.
    """
    from sglang.kernels.ops.moe.lk_verify_uva import prefetch_assign, stage_copy

    target = _prefetch_targets.get(layer_id + 1)
    if (target is None or hidden_states.shape[0] > buffers.max_rows
            or not hidden_states.dtype.is_floating_point or hidden_states.element_size() < 2):
        # A pre-quantized (FP8) MoE input lacks its scale here; skip the prediction.
        return
    cache = target.cache
    stream = buffers.prefetch_stream
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        predicted = _predict_local_experts(target, hidden_states)
        votes, free_rows, copy_expert = buffers.prefetch_set
        prefetch_assign(topk_ids=predicted, slot_of=cache.slot_of, votes=votes,
                        free_rows=free_rows, stage_expert=cache.stage_expert,
                        row_pending=cache.row_pending, row_count=cache.row_count,
                        copy_expert=copy_expert, stats=cache.stats,
                        num_staging=cache.num_staging, max_prefetch=target.max_experts)
        cache.prefetch_assigned.record(stream)
        stage_copy(copy_expert=copy_expert, src_ptrs=cache.stage_src, dst_ptrs=cache.stage_dst,
                   words=cache.stage_words, num_staging=cache.num_staging,
                   max_words=cache.max_words, num_programs=target.programs,
                   row_count=cache.row_count)
    cache.prefetch_issued = True
    buffers.prefetch_outstanding = True


def _cutlass(cache: LayerCache, x, slot_ids, weights, params: MoeParams):
    """Routes whose id is a slab row; ``num_slots + num_staging`` marks a skipped route."""
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
        # Slab rows are rank 0's experts of two ranks; the skip id belongs to rank 1.
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


# Copy programs of an on-demand staging copy: nothing else runs meanwhile, so one per
# H100 SM. Arbitrary; PCIe, not SM count, bounds it.
_ON_DEMAND_COPY_PROGRAMS = 132


def _graph_forward(cache, *, x, topk_ids, topk_weights, buffers: ForwardBuffers,
                   params: MoeParams):
    """Fixed launches: slot hits and staged misses in CUTLASS, further misses in the UVA
    kernel on a side stream."""
    from sglang.kernels.ops.moe.lk_verify_uva import (
        assign_slots,
        launch_draft_miss_uva,
        stage_copy,
    )

    rows, top_k = topk_ids.shape
    n = rows * top_k
    assert n <= buffers.slot_ids.numel(), (rows, buffers.max_rows)
    main = torch.cuda.current_stream()
    if cache.prefetch_issued:
        # The row assignment only; assign_slots waits for the rows it uses.
        main.wait_event(cache.prefetch_assigned)
        cache.prefetch_issued = False
    slot_ids, miss_tab = buffers.slot_ids[:n], buffers.miss_tab[:n]
    gout_part = buffers.gout_part[:n]
    mark, free_rows, copy_expert = buffers.main_set
    assign_slots(topk_ids=topk_ids, slot_of=cache.slot_of, mark=mark, free_rows=free_rows,
                 stage_expert=cache.stage_expert, row_pending=cache.row_pending,
                 row_count=cache.row_count, copy_expert=copy_expert,
                 slot_ids=slot_ids, miss_tab=miss_tab, hist=cache.hist, stats=cache.stats,
                 miss_hist=cache.miss_hist, num_slots=cache.num_slots,
                 num_staging=cache.num_staging, items_per_row=cache.items_per_row)
    side_stream = buffers.side_stream
    side_stream.wait_stream(main)
    with torch.cuda.stream(side_stream):
        # Reads only the host checkpoint copy, so slot swaps need not fence it.
        gout_part.zero_()
        launch_draft_miss_uva(
            hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights,
            slot_tab=miss_tab, gout_part=gout_part, rows_per_request=1,
            host=cache.experts, scratch=buffers.scratch, activation=0, alpha=1.702,
            limit=params.limit, include_first_row=True)
    if cache.num_staging > 0:
        stage_copy(copy_expert=copy_expert, src_ptrs=cache.stage_src,
                   dst_ptrs=cache.stage_dst, words=cache.stage_words,
                   num_staging=cache.num_staging, max_words=cache.max_words,
                   num_programs=_ON_DEMAND_COPY_PROGRAMS)
    hits = _cutlass(cache, x, slot_ids.view(rows, top_k), topk_weights, params)
    main.wait_stream(side_stream)
    if buffers.prefetch_outstanding and cache.layer_id + 1 not in _prefetch_targets:
        # Last prefetched layer: join the prefetch stream (graph capture needs it).
        main.wait_stream(buffers.prefetch_stream)
        buffers.prefetch_outstanding = False
    return gout_part.view(rows, top_k, -1).sum(dim=1).add_(hits)


def _eager_forward(cache, *, x, topk_ids, topk_weights, params: MoeParams):
    """Any batch outside a graph: misses copied in groups of ``num_staging``, all in CUTLASS.

    The caller holds ``park()``: host syncs separate reading ``slot_of`` from the launches.
    """
    _add_stream(torch.cuda.current_stream())
    ids = topk_ids.long()
    local = ids >= 0
    slot = torch.where(local, cache.slot_of[ids.clamp(min=0)], -1)
    cache.hist.add_(torch.bincount(ids[local], minlength=cache.num_experts).to(torch.int32))
    missing = torch.unique(ids[local & (slot < 0)]).tolist()
    group_size = max(cache.num_staging, 1)
    skip = cache.num_slots + cache.num_staging
    out = None
    for start in range(0, max(len(missing), 1), group_size):
        group = missing[start:start + group_size]
        assert not group or cache.num_staging > 0, "misses need LK_GPU_CACHE_STAGING > 0"
        stage_of = torch.full((cache.num_experts,), -1, dtype=torch.long, device=x.device)
        for j, expert in enumerate(group):
            cache._copy_in(cache.num_slots + j, expert)
            stage_of[expert] = j
        staged = stage_of[ids.clamp(min=0)]
        slot_ids = torch.where(local & (staged >= 0), cache.num_slots + staged, skip)
        if start == 0:
            slot_ids = torch.where(slot >= 0, slot, slot_ids)
        part = _cutlass(cache, x, slot_ids.to(torch.int32), topk_weights, params).float()
        out = part if out is None else out.add_(part)
    if missing:
        # The rows now hold the last group, not what graph forwards last left there.
        cache.stage_expert.fill_(-1)
        cache.row_pending.zero_()
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
    with park():
        return _eager_forward(cache, x=x, topk_ids=topk_ids, topk_weights=topk_weights,
                              params=params)
