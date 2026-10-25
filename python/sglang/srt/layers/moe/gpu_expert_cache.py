"""GPU-resident hot-expert cache for LK offloaded MoE layers, misses read over UVA.

Each offloaded layer keeps S expert slots in VRAM and every expert in pinned host
memory. A per-expert address table (``LayerCache.table``) points each expert at its
VRAM slot or at its host block; the decode/verify kernels read that table once per
tile, so one read decides where every route's weights come from and no route is
computed twice or dropped. The CPU only stores weights; all MoE math runs on the GPU.

Residency adapts in a background thread from route counts the kernels accumulate.
A swap never changes bytes a kernel may be reading: the outgoing expert is first
pointed back at its host block, then every graph launched before that point must
finish, and only then is its slot overwritten; the incoming expert is published
after its copy has landed. Host and VRAM copies are byte-identical, so a tile may
read either side at any moment.

The thread issues CUDA calls only under ``park()``, which graph capture and graph
replay launch also hold (the same rule as the lk_moe pump), and starts swapping only
after the first replay, i.e. once every capture and its warmups are done. Eager
forwards (prefill tails of late layers) read the same table; their streams join the
fence, and a launch racing an unpublish either precedes the fence's event record or
starts after the table push has landed.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Optional

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


class LayerCache:
    """VRAM slots, the per-expert address table and route counts of one layer."""

    def __init__(self, *, layer_id: int, host_tensors, num_slots: int, device):
        # Lazy: park() must stay importable without the lk_moe wheel.
        from sglang.kernels.ops.moe.lk_verify_uva import HostExperts, register_host_experts

        w13, s13, w2, s2 = host_tensors
        self.layer_id = layer_id
        self.num_experts = w13.shape[0]
        num_slots = min(num_slots, self.num_experts)
        self._host = host_tensors
        # Addresses of each expert's host blocks, [4, E] (w13, s13, w2, s2).
        host = register_host_experts(w13, s13, w2, s2, device="cpu")
        self._host_addrs = torch.stack(
            [host.w13_ptrs, host.s13_ptrs, host.w2_ptrs, host.s2_ptrs])
        self._slabs = [
            torch.empty((num_slots, *t.shape[1:]), dtype=torch.uint8, device=device)
            for t in host_tensors
        ]
        self._slot_addrs = torch.tensor(
            [[slab[v].data_ptr() for v in range(num_slots)] for slab in self._slabs],
            dtype=torch.int64)
        self.slot_owner = [-1] * num_slots
        # Pinned mirror of the table; written only while no H2D from it is pending.
        self._table_host = self._host_addrs.clone().pin_memory()
        self.table = self._table_host.to(device)
        self.hist = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        self._hist_host = torch.zeros(self.num_experts, dtype=torch.int32).pin_memory()
        self._hist_prev = torch.zeros(self.num_experts, dtype=torch.int64)
        self.scores = torch.zeros(self.num_experts, dtype=torch.float32)
        self.experts = HostExperts(
            w13_ptrs=self.table[0], s13_ptrs=self.table[1],
            w2_ptrs=self.table[2], s2_ptrs=self.table[3],
            num_experts=self.num_experts, hidden=host.hidden,
            intermediate=host.intermediate)
        # Cold start: the first S experts; the thread replaces them by route counts.
        for v in range(num_slots):
            self._copy_in(v, v)
            self._table_host[:, v] = self._slot_addrs[:, v]
            self.slot_owner[v] = v
        self.table.copy_(self._table_host)
        torch.cuda.synchronize(device)

    def _copy_in(self, slot: int, expert: int) -> None:
        for slab, host in zip(self._slabs, self._host):
            slab[slot].copy_(host[expert], non_blocking=True)

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
                layer.table.copy_(layer._table_host, non_blocking=True)

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
                    layer._table_host[:, out_e] = layer._host_addrs[:, out_e]
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
                layer._table_host[:, in_e] = layer._slot_addrs[:, slot]
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
