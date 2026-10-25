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
forwards (prefill tails of late layers) read the same table; their streams join the
fence, and a launch racing an unmap either precedes the fence's event record or
starts after the table push has landed.
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


def note_replay(stream: torch.cuda.Stream) -> None:
    """Record a graph-replay stream; call inside ``park()`` right before the launch."""
    if not any(s == stream for s in _replay_streams):
        _replay_streams.append(stream)
    _first_replay.set()


def note_eager_stream(stream: torch.cuda.Stream) -> None:
    """Record a stream that runs cache reads outside a graph; no park() needed."""
    if not any(s == stream for s in _replay_streams):
        with _PARK_LOCK:
            if not any(s == stream for s in _replay_streams):
                _replay_streams.append(stream)


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

    def __init__(self, *, layer_id: int, host_tensors, num_slots: int, device):
        # Lazy: park() must stay importable without the lk_moe wheel.
        from sglang.kernels.ops.moe.lk_verify_uva import register_host_experts

        w13, s13, w2, s2 = host_tensors
        self.layer_id = layer_id
        self.num_experts = w13.shape[0]
        self.num_slots = min(num_slots, self.num_experts)
        # Checkpoint layout, read in place over UVA for routes without a slot.
        self.experts = register_host_experts(w13, s13, w2, s2, device=device)
        self._host = host_tensors
        self._slot_source = _build_slot_source(host_tensors, device=device)
        self.slabs = [
            torch.empty((self.num_slots, *t.shape[1:]), dtype=torch.uint8, device=device)
            for t in self._slot_source
        ]
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
    swiglu_limit: Optional[torch.Tensor]  # [S] fp32 for the CUTLASS kernel, or None
    limit: float  # the same clamp for the UVA kernel; 0 disables it
    tp_size: int
    tp_rank: int


def _cutlass_hits(cache: LayerCache, x, slot_ids, weights, params: MoeParams):
    """Routes with a slot; ``slot_ids == num_slots`` marks the others, which EP skips."""
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
        # Slots are rank 0's experts of two ranks; id num_slots belongs to rank 1.
        ep_size=2,
        ep_rank=0,
        use_w4_group_scaling=True,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=next_power_of_2(x.shape[0]),
        output=out,
        use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
    )
    return out


def _forward_rows(cache, *, x, topk_ids, topk_weights, slot_ids, miss_tab, gout_part,
                  scratch, side_stream, params: MoeParams, eager_plan: bool):
    from sglang.kernels.ops.moe.lk_verify_uva import launch_draft_miss_uva, route_to_slots

    rows, top_k = topk_ids.shape
    route_to_slots(topk_ids=topk_ids, slot_of=cache.slot_of, slot_ids=slot_ids,
                   miss_tab=miss_tab, hist=cache.hist, miss_id=cache.num_slots)
    main = torch.cuda.current_stream()
    side_stream.wait_stream(main)
    with torch.cuda.stream(side_stream):
        # Reads only the host checkpoint copy, so slot swaps need not fence it.
        gout_part.zero_()
        launch_draft_miss_uva(
            hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights,
            slot_tab=miss_tab, gout_part=gout_part, rows_per_request=1,
            host=cache.experts, scratch=scratch, activation=0, alpha=1.702,
            limit=params.limit, include_first_row=True, eager_plan=eager_plan)
    hits = _cutlass_hits(cache, x, slot_ids.view(rows, top_k), topk_weights, params)
    main.wait_stream(side_stream)
    return gout_part.view(rows, top_k, -1).sum(dim=1).add_(hits)


def cache_forward(cache: LayerCache, *, hidden_states, topk_ids, topk_weights,
                  buffers: ForwardBuffers, params: MoeParams, graph: bool):
    """MoE output [rows, H] fp32 of one offloaded layer; ``graph`` uses fixed buffers."""
    x = hidden_states.contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    rows, top_k = topk_ids.shape
    if graph:
        n = rows * top_k
        assert n <= buffers.slot_ids.numel(), (rows, buffers.max_rows)
        return _forward_rows(
            cache, x=x, topk_ids=topk_ids, topk_weights=topk_weights,
            slot_ids=buffers.slot_ids[:n], miss_tab=buffers.miss_tab[:n],
            gout_part=buffers.gout_part[:n], scratch=buffers.scratch,
            side_stream=buffers.side_stream, params=params, eager_plan=False)
    note_eager_stream(torch.cuda.current_stream())
    out = torch.empty((rows, x.shape[1]), dtype=torch.float32, device=x.device)
    # Bounds the transient fp32 scratch to ~0.3 GiB (gout_part + mid) per chunk.
    chunk = 2048
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        n = (stop - start) * top_k
        ids = topk_ids[start:stop]
        out[start:stop] = _forward_rows(
            cache, x=x[start:stop], topk_ids=ids, topk_weights=topk_weights[start:stop],
            slot_ids=torch.empty(n, dtype=torch.int32, device=x.device),
            miss_tab=torch.empty(n, dtype=torch.int32, device=x.device),
            gout_part=torch.empty((n, x.shape[1]), dtype=torch.float32, device=x.device),
            scratch=buffers.scratch, side_stream=buffers.side_stream, params=params,
            eager_plan=True)
    return out
