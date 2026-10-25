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


def _to_cutlass_layout(w13, s13, w2, s2):
    """Checkpoint-layout uint8 CUDA experts [c, ...] (w13 gate rows first) -> SM90 W4A16."""
    from flashinfer.fused_moe import (
        interleave_moe_scales_for_sm90_mixed_gemm,
        interleave_moe_weights_for_sm90_mixed_gemm,
    )

    half = w13.shape[1] // 2

    def up_first(t):
        # GPU-resident layers load w13 as [up; gate] (load_up_proj_weight_first).
        return torch.cat([t[:, half:], t[:, :half]], dim=1)

    return (
        interleave_moe_weights_for_sm90_mixed_gemm(up_first(w13), "fp4"),
        interleave_moe_scales_for_sm90_mixed_gemm(up_first(s13), group_size=32),
        interleave_moe_weights_for_sm90_mixed_gemm(w2.contiguous(), "fp4"),
        interleave_moe_scales_for_sm90_mixed_gemm(s2.contiguous(), group_size=32),
    )


def _build_slot_source(host_tensors, *, device, chunk: int = 16):
    """Pinned host copy of every expert in the slot layout; ~0.3 GiB VRAM transient."""
    num_experts = host_tensors[0].shape[0]
    first = _to_cutlass_layout(*(t[:1].to(device) for t in host_tensors))
    out = [torch.empty((num_experts, *f.shape[1:]), dtype=torch.uint8, pin_memory=True)
           for f in first]
    for start in range(0, num_experts, chunk):
        stop = min(num_experts, start + chunk)
        converted = _to_cutlass_layout(*(t[start:stop].to(device) for t in host_tensors))
        for dst, src in zip(out, converted):
            dst[start:stop].copy_(src)
    return out


class LayerCache:
    """VRAM slots, the expert-to-slot table and route counts of one layer."""

    def __init__(self, *, layer_id: int, host_tensors, num_slots: int, num_staging: int,
                 device):
        # Lazy: park() must stay importable without the lk_moe wheel.
        from sglang.kernels.ops.moe.lk_verify_uva import register_host_experts

        w13, s13, w2, s2 = host_tensors
        self.layer_id = layer_id
        self.num_experts = w13.shape[0]
        self.num_slots = min(num_slots, self.num_experts)
        self.num_staging = num_staging
        # Checkpoint layout, read in place over UVA for routes without a slot.
        self.experts = register_host_experts(w13, s13, w2, s2, device=device)
        self._host = host_tensors
        self._slot_source = _build_slot_source(host_tensors, device=device)
        # Rows [num_slots, num_slots + num_staging) take one forward's missed experts;
        # the residency thread never touches them.
        self.slabs = [
            torch.empty((self.num_slots + num_staging, *t.shape[1:]), dtype=torch.uint8,
                        device=device)
            for t in self._slot_source
        ]
        words = [t[0].numel() // 8 for t in self._slot_source]
        assert all(t[0].numel() % 8 == 0 for t in self._slot_source)
        self.max_words = max(words)
        self.stage_words = torch.tensor(words, dtype=torch.int64, device=device)
        # Pinned (cudaHostAlloc) host memory has the same address on the device under UVA.
        self.stage_src = torch.tensor([t.data_ptr() for t in self._slot_source],
                                      dtype=torch.int64, device=device)
        self.stage_dst = torch.tensor([slab[self.num_slots].data_ptr() for slab in self.slabs],
                                      dtype=torch.int64, device=device)
        self.slot_owner = [-1] * self.num_slots
        # Pinned mirror of slot_of; written only while no H2D from it is pending.
        self._slot_of_host = torch.full((self.num_experts,), -1, dtype=torch.int32).pin_memory()
        self.slot_of = self._slot_of_host.to(device)
        self.hist = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        self._hist_host = torch.zeros(self.num_experts, dtype=torch.int32).pin_memory()
        self._hist_prev = torch.zeros(self.num_experts, dtype=torch.int64)
        self.scores = torch.zeros(self.num_experts, dtype=torch.float32)
        # Cold start: the first S experts; the thread replaces them by route counts.
        for v in range(self.num_slots):
            self._copy_in(v, v)
            self._slot_of_host[v] = v
            self.slot_owner[v] = v
        self.slot_of.copy_(self._slot_of_host)
        torch.cuda.synchronize(device)

    def host_bytes(self) -> int:
        return sum(t.numel() for t in (*self._host, *self._slot_source))

    def _copy_in(self, slot: int, expert: int) -> None:
        """Enqueue the H2D copy of ``expert`` into slab row ``slot`` on the current stream."""
        for slab, source in zip(self.slabs, self._slot_source):
            slab[slot].copy_(source[expert], non_blocking=True)

    def _push_slot_of(self) -> None:
        self.slot_of.copy_(self._slot_of_host, non_blocking=True)

    def read_hist_async(self) -> None:
        self._hist_host.copy_(self.hist, non_blocking=True)

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
            except Exception:
                logger.exception("gpu-expert-cache: residency round failed; stopping swaps")
                return
            if self._log_interval_s > 0 and time.time() - last_log >= self._log_interval_s:
                hit = stats["resident"] / stats["routes"] if stats["routes"] else 0.0
                logger.info(
                    "[gpu-expert-cache] %d layers, last %.0fs: routes=%d hit=%.1f%% swaps=%d",
                    len(self._layers), time.time() - last_log, stats["routes"],
                    100 * hit, stats["swaps"])
                stats = dict(routes=0, resident=0.0, swaps=0)
                last_log = time.time()


class ForwardBuffers:
    """Process-wide buffers of the graph path; layers run one after another."""

    def __init__(self, *, num_experts, max_rows, top_k, hidden, intermediate, num_staging,
                 device):
        from sglang.kernels.ops.moe.lk_verify_uva import UvaScratch

        routes = max_rows * top_k
        self.max_rows = max_rows
        self.scratch = UvaScratch.allocate(
            num_experts=num_experts, max_slots=routes, intermediate=intermediate,
            device=device)
        self.gout_part = torch.zeros((routes, hidden), dtype=torch.float32, device=device)
        self.slot_ids = torch.zeros(routes, dtype=torch.int32, device=device)
        self.miss_tab = torch.zeros(routes, dtype=torch.int32, device=device)
        self.mark = torch.zeros(num_experts, dtype=torch.int32, device=device)
        self.staging_expert = torch.zeros(max(num_staging, 1), dtype=torch.int32, device=device)
        self.side_stream = torch.cuda.Stream(device=device)


class MoeParams(msgspec.Struct, frozen=True):
    swiglu_limit: Optional[torch.Tensor]  # [slots + staging] fp32 for CUTLASS, or None
    limit: float  # the same clamp for the UVA kernel; 0 disables it
    tp_size: int
    tp_rank: int


def _cutlass(cache: LayerCache, x, slot_ids, weights, params: MoeParams):
    """Routes whose id is a slab row; ``num_slots + num_staging`` marks a skipped route."""
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    from sglang.srt.environ import envs
    from sglang.srt.utils.common import next_power_of_2

    w13, s13, w2, s2 = cache.slabs
    out = torch.empty_like(x)
    cutlass_fused_moe(
        input=x,
        token_selected_experts=slot_ids,
        token_final_scales=weights,
        fc1_expert_weights=w13,
        fc2_expert_weights=w2,
        output_dtype=x.dtype,
        quant_scales=[s13.view(torch.int32), s2.view(torch.int32)],
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
    )
    return out


def _graph_forward(cache, *, x, topk_ids, topk_weights, buffers: ForwardBuffers,
                   params: MoeParams):
    """Fixed launches: slot hits and up to ``num_staging`` missed experts in CUTLASS,
    any further misses in the UVA kernel on a side stream."""
    from sglang.kernels.ops.moe.lk_verify_uva import (
        assign_slots,
        launch_draft_miss_uva,
        stage_copy,
    )

    rows, top_k = topk_ids.shape
    n = rows * top_k
    assert n <= buffers.slot_ids.numel(), (rows, buffers.max_rows)
    slot_ids, miss_tab = buffers.slot_ids[:n], buffers.miss_tab[:n]
    gout_part = buffers.gout_part[:n]
    assign_slots(topk_ids=topk_ids, slot_of=cache.slot_of, mark=buffers.mark,
                 staging_expert=buffers.staging_expert, slot_ids=slot_ids,
                 miss_tab=miss_tab, hist=cache.hist, num_slots=cache.num_slots,
                 num_staging=cache.num_staging)
    main = torch.cuda.current_stream()
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
        stage_copy(staging_expert=buffers.staging_expert, src_ptrs=cache.stage_src,
                   dst_ptrs=cache.stage_dst, words=cache.stage_words,
                   num_staging=cache.num_staging, max_words=cache.max_words)
    hits = _cutlass(cache, x, slot_ids.view(rows, top_k), topk_weights, params)
    main.wait_stream(side_stream)
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
