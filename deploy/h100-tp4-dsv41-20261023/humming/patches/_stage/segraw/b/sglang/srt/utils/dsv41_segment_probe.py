"""DSV4.1 decode-step **segment probe** (gated, default OFF).

Purpose
-------
Split one dspark verify step of this Lsglang DSV4.1 deployment into attributable
segments, without perturbing the measured object.

Gate
----
``SGLANG_DSV41_PROBE_SEGMENTS=1``  -> probe on.
Unset / "0" / "false" / anything else -> probe off: every ``span()`` returns
``contextlib.nullcontext()``, no event/tensor/kernel is created, no sync, no
file, no print, no CUDA call.  Off is one ``is None`` test per call site.

Tunables (read only when the probe is on)
-----------------------------------------
``SGLANG_DSV41_PROBE_WINDOW``  steps per readout window (default 50)
``SGLANG_DSV41_PROBE_OUT``     jsonl output path
                               (default /mnt/.../Lsglang/logs/dsv41-segprobe.<pid>.jsonl)
``SGLANG_DSV41_PROBE_MAXSAMP`` per-(segment,mode) sample cap (default 200000)

Why two mechanisms
------------------
**In-graph spans cannot use CUDA events in this runtime.**  Measured on this box
(H100 / CUDA 13.2 / torch 2.14.1): an event recorded inside stream capture
accepts the record without error, but afterwards *both* ``cudaEventQuery`` and
``cudaEventElapsedTime`` fail with ``cudaErrorInvalidValue``, before and after
replay (see ``patches/smoke_event_semantics.py``).  So the in-graph mode uses a
1-program Triton kernel that stamps ``%globaltimer`` (ns) into a device ring
buffer:

    row = (device counter, +1 at the channel's stamp boundary) % W
    col = boundary id

Every boundary of a replayed graph is its own graph node, so the ring yields
**per-replay (per-step) rows**, read out with a single D2H per window.  This is
strictly more informative than one event pair per segment and equally
non-intrusive (one tiny kernel launch per boundary, no host sync inside the
step).

Eager spans (Python runs per step: graph-launch bracket sites) use
``torch.cuda.Event`` pairs from a round-robin slot pool -> per-step samples,
read once per window.

Reading happens only in :meth:`SegmentProbe.flush` (called every
``window`` steps): exactly **one** ``torch.cuda.synchronize()`` plus the D2H
copies.  No per-step ``.item()`` / ``synchronize()`` / print anywhere.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import statistics
import time
from contextlib import nullcontext

import torch

logger = logging.getLogger(__name__)

_ENV_FLAG = "SGLANG_DSV41_PROBE_SEGMENTS"
_ENV_WINDOW = "SGLANG_DSV41_PROBE_WINDOW"
_ENV_SLOTS = "SGLANG_DSV41_PROBE_SLOTS"
_ENV_OUT = "SGLANG_DSV41_PROBE_OUT"
_ENV_MAXSAMP = "SGLANG_DSV41_PROBE_MAXSAMP"
_ENV_RAW = "SGLANG_DSV41_PROBE_RAW"  # [seg-probe-raw] per-step rows in the window line

#: grep-able marker stamped by every patched site and every printed line.
MARK = "[seg-probe]"

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_MAX_TICK_COLS = 256  # boundary columns per channel


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in _TRUTHY


def _int_env(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name, "").strip()
        return int(raw) if raw else default
    except Exception:
        return default


def _raw_rows_enabled() -> bool:
    # [seg-probe-raw] SGLANG_DSV41_PROBE_RAW=1 adds the per-step (per-row) values of
    # this window to the window record, so a caller can compute per-step p50/p95 over
    # any sub-interval of the run without shrinking the readout window (the sync
    # cadence, and therefore the measured object, is unchanged).  Default OFF:
    # the window record is byte-identical to the unextended probe.
    try:
        return _truthy(os.environ.get(_ENV_RAW, ""))
    except Exception:
        return False


def _quantile(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


try:  # kernel definition is cheap and host-side only
    import triton
    import triton.language as tl

    @triton.jit
    def _probe_tick_kernel(buf_ptr, ctr_ptr, row_mark, col, nb, w):
        """Stamp %globaltimer (ns) into buf[row, col]; stamp advances the row."""
        if row_mark != 0:
            tl.atomic_add(ctr_ptr, 1, sem="release")
        step = tl.atomic_add(ctr_ptr, 0, sem="acquire")
        t = tl.extra.cuda.globaltimer()
        tl.store(buf_ptr + (step % w) * nb + col, t, cache_modifier=".cg")

    _TRITON_OK = True
except Exception as _exc:  # pragma: no cover - triton is present on this box
    _probe_tick_kernel = None
    _TRITON_OK = False
    _TRITON_ERR = repr(_exc)


class _Span:
    """One segment span: tick kernel (capturing) or event pair (eager)."""

    __slots__ = ("_probe", "_name", "_idx", "_stamp", "_pair", "_channel", "_col")

    def __init__(self, probe, name, idx, stamp):
        self._probe = probe
        self._name = name
        self._idx = -1 if idx is None else int(idx)
        self._stamp = bool(stamp)
        self._pair = None
        self._channel = None
        self._col = None

    def __enter__(self):
        probe = self._probe
        try:
            if probe._is_capturing():
                self._channel, self._col = probe._tick_start(
                    self._name, self._idx, self._stamp
                )
            else:
                self._pair = probe._eager_pair(self._name)
                self._pair[0].record()
        except Exception:
            self._pair = None
            self._channel = None
            probe._note_failure("enter", self._name)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._pair is not None:
            try:
                self._pair[1].record()
            except Exception:
                self._probe._note_failure("exit", self._name)
        elif self._channel is not None:
            try:
                self._probe._tick(self._channel, self._col, False)
            except Exception:
                self._probe._note_failure("exit", self._name)
        self._pair = None
        self._channel = None
        return False


class _Channel:
    """Device ring buffer + row counter for one in-graph tick group."""

    __slots__ = (
        "name", "w", "nb", "buf", "ctr", "bids", "next_col", "stamp_col",
        "capture_alloc",
    )

    def __init__(self, name: str, w: int, nb: int, device, capture_alloc=False):
        self.name = name
        self.w = w
        self.nb = nb
        self.buf = torch.zeros((w, nb), dtype=torch.int64, device=device)
        self.ctr = torch.zeros((1,), dtype=torch.int32, device=device)
        self.bids = {}  # (name, idx) -> (col_start, col_end)
        self.next_col = 1  # col 0 is reserved for the eager per-step row marker
        self.stamp_col = 0
        # True => buffer was allocated inside a capture, i.e. it lives in the
        # CUDA-graph private pool, whose contents are re-initialised on every
        # replay (measured).  The ring then degenerates to one row per window.
        self.capture_alloc = bool(capture_alloc)


class SegmentProbe:
    def __init__(self, window: int, out_path: str, max_samples: int):
        self.window = max(1, int(window))
        self.rows = max(4, 2 * self.window)  # ring rows (>= 2 replays per step)
        self.max_samples = max(1, int(max_samples))
        self.out_path = out_path
        self.device = None

        self._channels: dict[str, _Channel] = {}
        self._e: dict[str, list] = {}
        self._e_cursor: dict[str, int] = {}
        self._e_used: dict[str, set] = {}

        self._samples: dict[tuple[str, str], list] = {}
        self._step = 0
        self._win = 0
        self._failures: dict[str, int] = {}
        self._fail_logged = False
        self._tick_off = not _TRITON_OK
        self._fh = None
        self._closed = False
        self._last_row = {}

    # ------------------------------------------------------------------ setup
    def _is_capturing(self) -> bool:
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    def _make_channel(self, chan_name: str, device: int, capture_alloc: bool) -> _Channel:
        chan = _Channel(
            chan_name,
            self.rows,
            _MAX_TICK_COLS,
            torch.device("cuda", device),
            capture_alloc=capture_alloc,
        )
        self._channels[chan_name] = chan
        logger.warning(
            "%s channel=%s ring=%dx%d on cuda:%d capture_alloc=%s",
            MARK,
            chan_name,
            chan.w,
            chan.nb,
            device,
            chan.capture_alloc,
        )
        return chan

    def _channel(self, name: str) -> _Channel:
        chan_name = _CHANNEL_OF.get(name, "misc")
        chan = self._channels.get(chan_name)
        if chan is None:
            if self._tick_off:
                raise RuntimeError("tick path disabled")
            chan = self._make_channel(
                chan_name, torch.cuda.current_device(), self._is_capturing()
            )
        return chan

    def ensure_channels(self) -> None:
        """Allocate every tick ring eagerly (never inside a capture).

        Must be called from an eager site on the final device before the first
        graph capture; see ``_init_pool`` in the patch.  Idempotent.
        """
        if self._tick_off or self._closed:
            return
        try:
            device = torch.cuda.current_device()
        except Exception:
            return
        for chan_name in ("pool", "draft"):
            if chan_name not in self._channels:
                self._make_channel(chan_name, device, False)

    def _tick(self, chan: _Channel, col: int, row_mark: bool) -> None:
        _probe_tick_kernel[(1,)](
            chan.buf, chan.ctr, 1 if row_mark else 0, col, chan.nb, chan.w, num_warps=1
        )

    def _tick_start(self, name: str, idx: int, stamp: bool = False):
        chan = self._channel(name)
        key = (name, idx)
        cols = chan.bids.get(key)
        if cols is None:
            if chan.next_col + 2 > chan.nb:
                raise RuntimeError(f"{chan.name}: out of tick columns")
            cols = (chan.next_col, chan.next_col + 1)
            chan.next_col += 2
            chan.bids[key] = cols
        self._tick(chan, cols[0], False)
        return chan, cols[1]

    def _eager_pair(self, name: str):
        pool = self._e.get(name)
        if pool is None:
            slots = max(64, 24 * self.window)
            pool = [
                [
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                ]
                for _ in range(slots)
            ]
            self._e[name] = pool
            self._e_cursor[name] = 0
            self._e_used[name] = set()
        slot = self._e_cursor[name] % len(pool)
        self._e_cursor[name] += 1
        self._e_used[name].add(slot)
        return pool[slot]

    def _note_failure(self, where: str, name: str) -> None:
        key = f"{where}:{name}"
        self._failures[key] = self._failures.get(key, 0) + 1
        if not self._fail_logged:
            self._fail_logged = True
            logger.warning(
                "%s WARNING span %s failed for %s; segment degrades to no-data, "
                "forward path unaffected",
                MARK,
                where,
                name,
            )

    # ------------------------------------------------------------- public API
    def span(self, name: str, idx=None, stamp: bool = False):
        if self._closed:
            return nullcontext()
        return _Span(self, name, idx, stamp)

    def mark_rows(self) -> None:
        """Eagerly stamp one ring row per channel at the start of a step.

        The in-graph ticks only *read* the row counter, so every tick of the
        replays that follow this mark lands in the same row.  Called from
        ``on_step`` (Python, outside capture) -- never from inside a graph.
        """
        if self._tick_off:
            return
        for chan in list(self._channels.values()):
            try:
                self._tick(chan, chan.stamp_col, True)
            except Exception:
                self._note_failure("mark", chan.name)

    def on_step(self) -> None:
        """Called once per decode step from Python (outside any capture)."""
        if self._closed or self._is_capturing():
            return
        self._step += 1
        if self._step % self.window == 0:
            self.flush()
        self.mark_rows()
        if self._step == 1:
            self._emit(
                {
                    "hello": "segment probe armed",
                    "tick_path": not self._tick_off,
                    "channels": sorted(self._channels),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                }
            )

    # ---------------------------------------------------------------- compute
    def _record(self, name: str, mode: str, value: float) -> None:
        if value != value or value < 0.0:
            return
        key = (name, mode)
        bucket = self._samples.get(key)
        if bucket is None:
            bucket = []
            self._samples[key] = bucket
        if len(bucket) < self.max_samples:
            bucket.append(value)

    def flush(self) -> dict:
        if self._closed:
            return {}
        try:
            torch.cuda.synchronize()
        except Exception as exc:  # pragma: no cover
            logger.warning("%s sync failed: %r", MARK, exc)
            return {}
        self._win += 1

        tick_rows: dict[str, dict] = {}
        if self._channels:
            for cname, chan in self._channels.items():
                if chan.stamp_col is None:
                    continue
                try:
                    host = chan.buf.cpu().numpy()
                except Exception as exc:
                    logger.warning("%s D2H failed for %s: %r", MARK, cname, exc)
                    continue
                try:
                    chan.buf.zero_()
                    chan.ctr.zero_()
                except Exception:
                    pass
                agg: dict[str, list] = {}
                n_rows = 0
                for r in range(chan.w):
                    if chan.capture_alloc:
                        # graph-pool memory is re-initialised on every replay, so
                        # the eager row marker cannot survive: accept any row that
                        # holds data (degenerates to one sample per window).
                        if not host[r].any():
                            continue
                    elif host[r][chan.stamp_col] == 0:
                        continue
                    n_rows += 1
                    per_name: dict[str, dict] = {}
                    for (name, idx), (cs, ce) in chan.bids.items():
                        t0 = host[r][cs]
                        t1 = host[r][ce]
                        if t0 and t1 and t1 >= t0:
                            per_name.setdefault(name, {})[idx] = (t1 - t0) / 1e6
                    for name, vals in per_name.items():
                        for idx, v in vals.items():
                            self._record(f"{name}[{idx}]", "graph", v)
                        s = sum(vals.values())
                        self._record(name + ".sum", "graph", s)
                        agg.setdefault(name, []).append((s, vals))
                tick_rows[cname] = {
                    "rows": n_rows,
                    "capture_alloc": chan.capture_alloc,
                    "seg": agg,
                }
                if _raw_rows_enabled():  # [seg-probe-raw] per-step sums of this window
                    tick_rows[cname]["raw_sum"] = {
                        name: [round(float(s), 4) for s, _ in items]
                        for name, items in agg.items()
                    }

        eager_rows: dict[str, dict] = {}
        for name, pool in self._e.items():
            vals = []
            for slot in sorted(self._e_used.get(name, ())):
                start, end = pool[slot]
                try:
                    if start.query() and end.query():
                        vals.append(start.elapsed_time(end))
                except Exception:
                    continue
            self._e_used[name] = set()
            for v in vals:
                self._record(name, "eager", v)
            if vals:
                eager_rows[name] = {
                    "n": len(vals),
                    "mean": round(statistics.fmean(vals), 4),
                    "max": round(max(vals), 4),
                }
                if _raw_rows_enabled():  # [seg-probe-raw] per-step values of this window
                    eager_rows[name]["raw"] = [round(float(v), 4) for v in vals]

        row = {
            "win": self._win,
            "step": self._step,
            "pid": os.getpid(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "mono": round(time.monotonic(), 3),
            "graph": {
                cname: {
                    "rows": block["rows"],
                    "capture_alloc": block["capture_alloc"],
                    "seg": {
                        name: {
                            "n_rows": len(items),
                            "sum_mean": round(
                                statistics.fmean(s for s, _ in items), 4
                            ),
                            "per_idx_mean": {
                                str(idx): round(
                                    statistics.fmean(
                                        vals.get(idx, float("nan"))
                                        for _, vals in items
                                    ),
                                    4,
                                )
                                for idx in sorted(
                                    {i for _, vals in items for i in vals}
                                )
                            },
                        }
                        for name, items in block["seg"].items()
                    },
                }
                for cname, block in tick_rows.items()
            },
            "eager": eager_rows,
        }
        self._emit(row)
        self._last_row = row
        return row

    # ----------------------------------------------------------------- output
    def _emit(self, row: dict) -> None:
        try:
            logger.warning(
                "%s %s", MARK, json.dumps(row, sort_keys=True, separators=(",", ":"))
            )
        except Exception:
            pass
        try:
            if self._fh is None and self.out_path:
                os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
                self._fh = open(self.out_path, "a", buffering=1, encoding="utf-8")
            if self._fh is not None:
                self._fh.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning("%s cannot write %s: %r", MARK, self.out_path, exc)
            self.out_path = ""

    def summary_lines(self) -> list:
        out = []
        for (name, mode), vals in sorted(self._samples.items()):
            if not vals:
                continue
            s = sorted(vals)
            out.append(
                "%s SUMMARY segment=%s mode=%s n=%d mean_ms=%.4f p50_ms=%.4f "
                "p95_ms=%.4f max_ms=%.4f"
                % (
                    MARK,
                    name,
                    mode,
                    len(s),
                    statistics.fmean(s),
                    _quantile(s, 0.50),
                    _quantile(s, 0.95),
                    s[-1],
                )
            )
        if self._failures:
            out.append("%s SUMMARY failures=%s" % (MARK, json.dumps(self._failures)))
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        except Exception:
            pass
        for line in self.summary_lines():
            try:
                logger.warning(line)
                if self._fh is not None:
                    self._fh.write(json.dumps({"summary": line}) + "\n")
            except Exception:
                pass
        try:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
        except Exception:
            pass


#: which tick channel a segment belongs to (one row counter per channel)
_CHANNEL_OF = {
    "pool_layer": "pool",
    "draft5_total": "draft",
    "draft_step": "draft",
    "draft_bcast": "draft",
    "vocab_ag": "draft",
}


def _default_out_path() -> str:
    return os.environ.get(_ENV_OUT, "").strip() or (
        f"/mnt/SSD01/models/.llm-env/Lsglang/logs/dsv41-segprobe.{os.getpid()}.jsonl"
    )


def _install_exit_hooks(probe: SegmentProbe) -> None:
    atexit.register(probe.close)
    try:
        prev = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame):
            try:
                probe.close()
            except Exception:
                pass
            if callable(prev):
                prev(signum, frame)
            else:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


def _build():
    if not _truthy(os.environ.get(_ENV_FLAG, "")):
        return None
    try:
        if not torch.cuda.is_available():
            return None
    except Exception:
        return None
    window = _int_env(_ENV_WINDOW, 50)
    probe = SegmentProbe(window, _default_out_path(), _int_env(_ENV_MAXSAMP, 200000))
    logger.warning(
        "%s ENABLED pid=%d window=%d rows=%d tick_path=%s out=%s",
        MARK,
        os.getpid(),
        probe.window,
        probe.rows,
        not probe._tick_off,
        probe.out_path,
    )
    if _raw_rows_enabled():  # [seg-probe-raw]
        logger.warning(
            "[seg-probe-raw] per-step window rows ON pid=%d (SGLANG_DSV41_PROBE_RAW=1)",
            os.getpid(),
        )
    _install_exit_hooks(probe)
    return probe


#: Module-level singleton; None when the gate is off.
PROBE = _build()

#: False => patched sites take the literal original code path.
PROBE_ENABLED = PROBE is not None


def span(name: str, idx=None, stamp: bool = False):
    """Context manager for one segment. Off => nullcontext."""
    probe = PROBE
    if probe is None:
        return nullcontext()
    return probe.span(name, idx, stamp)


def call(name: str, fn, *args, idx=None, **kwargs):
    """Call ``fn`` inside one segment span (off => direct call, no wrapper)."""
    probe = PROBE
    if probe is None:
        return fn(*args, **kwargs)
    with probe.span(name, idx):
        return fn(*args, **kwargs)


def ensure_channels() -> None:
    """Allocate tick rings eagerly (call from an eager init site)."""
    probe = PROBE
    if probe is None:
        return
    probe.ensure_channels()


def on_step() -> None:
    """One call per decode step; no-op when the probe is off."""
    probe = PROBE
    if probe is None:
        return
    probe.on_step()
