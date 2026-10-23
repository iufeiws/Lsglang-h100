"""DSV4.1 6-row GPU-pool verify contract probe -- GATED, default OFF.

Gate: ``LK_POOL_6ROW_PROBE=1`` (``true``/``on``/``yes``/``y``). Unset / ``0``
=> :func:`probe` returns ``None``; the caller in ``layer.py`` short-circuits on
``_LK_POOL_6ROW_ON`` before every entry point, so nothing here runs (no CUDA
call, no tensor, no file, no print, no device synchronisation).

Tunables (all optional, read once at arm time)
----------------------------------------------
``LK_POOL_6ROW_EVERY``     verify steps per emitted window          (default 50)
``LK_POOL_6ROW_OUT``       jsonl path   (default logs/dsv41-pool6row.<pid>.jsonl)
``LK_POOL_6ROW_IDSAMPLE``  (layer,expert) slot probes per layer/step (24; 0=off)
``LK_POOL_6ROW_HEAD``      stop emitting after N windows             (0 = off)
``LK_POOL_SLOTS``          pool slots per layer (from the serving env)

What is measured (host side only; the probe never synchronises the device)
--------------------------------------------------------------------------
One :func:`layer_done` call per ``_cpu_decode`` invocation == one pool-layer
boundary of one decode step.  Per call:

``launch_ms``  host wall time of the whole ``_cpu_decode`` call (start of the
               instrumentation line to just before the return).  It is the
               HOST time attributable to that layer's pool launch + CPU-half
               dispatch, bounded above by any queue-full block in the Triton
               launches; it is NOT device time.
``rows``/``splits``  batch shape handed to the GPU half / CPU half

Counters accumulated per call, emitted once per window (one line per window,
never per step, never with a device sync):

``gpu_tok_ids``      count of top-k ids read back from the tensor that is passed
                     to the GPU kernels (``topk_ids``), one k-tuple per real row
                     of the batch per layer.  Derived:
  ``id_dup_rows``    rows whose k ids contain a repeat.  The slot table maps
                     EXPERT IDS, so a repeated id makes two programs read the
                     SAME pool slot and each add its own routed contribution =>
                     a duplicated expert contribution.
  ``id_nodup_rows``  rows with k distinct ids (complement of the above).
  ``id_missing``     entries equal to -1.
  ``id_gt_pool``     ids outside ``[0, pool_slots)`` (cannot be resident).
  ``rows_mismatch``  layer calls whose row count differs from the first row
                     count seen in the process (layout change between layers).
``snap_mismatch``    (layer, expert) probes where ``pool_ready_slot >= 0``
                     (the pump reports the bytes landed) while
                     ``pool_snapshot_slot < 0`` at the same instant, i.e. the
                     frozen split word the GPU half reads hands the expert to
                     NEITHER half for that step.
``snap_claim``       probes where ``pool_snapshot_slot >= 0`` (GPU half owns).
``snap_absent``      probes where the expert is not in the pool at all.
``slots_probed``     probe population behind the three counters above.

Output
------
One JSON object per window plus one ``SUMMARY`` at exit; every window also
carries a ``line`` field printed through the layer logger, so the serve log
alone is sufficient.  ``_emit``/``close`` are the only writers.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import signal
import statistics
import time

_ENV_FLAG = "LK_POOL_6ROW_PROBE"
_TRUTHY = ("1", "true", "on", "yes", "y")


def _truthy(raw) -> bool:
    return str(raw).strip().lower() in _TRUTHY


def enabled() -> bool:
    return _truthy(os.environ.get(_ENV_FLAG, ""))


def _int_env(name, default):
    try:
        v = int(str(os.environ.get(name, "")).strip())
        return v if v >= 0 else default
    except Exception:
        return default


def _q(values, q):
    if not values:
        return float("nan")
    xs = sorted(values)
    i = int(round(q * (len(xs) - 1)))
    return float(xs[max(0, min(len(xs) - 1, i))])


def _log(msg):
    try:
        logging.getLogger(
            "sglang.srt.layers.moe.fused_moe_triton.layer").warning(msg)
    except Exception:
        print(msg, flush=True)


class Probe:
    def __init__(self):
        self.pid = os.getpid()
        self.t0 = time.time()
        self.every = max(1, _int_env("LK_POOL_6ROW_EVERY", 50))
        self.head = _int_env("LK_POOL_6ROW_HEAD", 0)
        self.idsample = _int_env("LK_POOL_6ROW_IDSAMPLE", 24)
        self.pool_slots = _int_env("LK_POOL_SLOTS", 0)
        self.out = (os.environ.get("LK_POOL_6ROW_OUT", "").strip()
                    or ("/mnt/SSD01/models/.llm-env/Lsglang/logs/"
                        "dsv41-pool6row.%d.jsonl" % self.pid))
        self.calls = 0
        self.windows = 0
        self.expect_rows = None
        self.first_rows = None
        self._fh = None
        self._lock_steps = 0
        self._probe_rot = 0
        self._c = {}
        self._digest = []
        self._launch = []
        self._layer_calls = []
        try:
            self._fh = open(self.out, "a", buffering=1)
        except Exception as exc:
            _log("[pool6row] cannot open %s: %r" % (self.out, exc))

    # ------------------------------------------------------------------ state
    def add(self, key, n=1):
        self._c[key] = self._c.get(key, 0) + n

    def _emit(self, obj):
        line = obj.pop("line", "")
        if self._fh is not None:
            try:
                self._fh.write(json.dumps(obj, ensure_ascii=False,
                                          sort_keys=True) + "\n")
            except Exception as exc:
                _log("[pool6row] write failed: %r" % (exc,))
        if line:
            _log("[pool6row] " + line)

    def _window(self):
        d = hashlib.sha1("|".join(self._digest).encode()).hexdigest()[:16] if self._digest else ""
        row = {
            "kind": "window",
            "pid": self.pid,
            "gate": _ENV_FLAG,
            "t_s": round(time.time() - self.t0, 1),
            "steps": self.every,
            "layer_calls": self.calls,
            "rows_per_token": self.first_rows,
            "pool_slots": self.pool_slots,
            "gpu_tok_ids": self._c.get("gpu_tok_ids", 0),
            "id_dup_rows": self._c.get("id_dup_rows", 0),
            "id_nodup_rows": self._c.get("id_nodup_rows", 0),
            "id_missing": self._c.get("id_missing", 0),
            "id_gt_pool": self._c.get("id_gt_pool", 0),
            "rows_mismatch": self._c.get("rows_mismatch", 0),
            "snap_mismatch": self._c.get("snap_mismatch", 0),
            "snap_claim": self._c.get("snap_claim", 0),
            "snap_absent": self._c.get("snap_absent", 0),
            "slots_probed": self._c.get("slots_probed", 0),
            "id_digest": d,
        }
        lc = self._layer_calls
        row["launch_ms_mean"] = round(statistics.fmean(lc), 4) if lc else float("nan")
        row["launch_ms_p50"] = round(_q(lc, 0.50), 4)
        row["launch_ms_p95"] = round(_q(lc, 0.95), 4)
        row["launch_ms_mean_per_step"] = round(len(lc) * row["launch_ms_mean"] / self.every, 4) if lc else float("nan")
        row["launch_ms_sum_per_step"] = round(sum(lc) / self.every, 4) if lc else float("nan")
        row["line"] = (
            "win#{w} calls={cl} rows={r} launch_ms/step={lps:.3f} "
            "p50={lp50:.3f} | snap_mismatch={sm}/{sp} claim={sc} absent={sa} "
            "dup_rows={dr} nodup={nd} missing={mi} gt_pool={gp} "
            "rows_mismatch={rm} digest={dg}"
        ).format(
            w=self.windows, cl=row["layer_calls"], r=self.first_rows,
            lps=(row["launch_ms_sum_per_step"] if lc else 0.0),
            lp50=(row["launch_ms_p50"] if lc else 0.0),
            sm=row["snap_mismatch"], sp=row["slots_probed"], sc=row["snap_claim"],
            sa=row["snap_absent"], dr=row["id_dup_rows"], nd=row["id_nodup_rows"],
            mi=row["id_missing"], gp=row["id_gt_pool"],
            rm=row["rows_mismatch"], dg=d,
        )
        self._emit(row)
        self.windows += 1
        self._c = {}
        self._digest = []
        self._layer_calls = []

    # ------------------------------------------------------------------- API
    def layer_done(self, module, bs, rows_per_token, hidden_states, topk_ids, mult):
        """Called by ``_cpu_decode`` once per pool layer per decode step.

        ``hidden_states``/``topk_ids`` are the live tensors the two halves were
        handed; nothing here synchronises the device (``.tolist()`` on tiny
        slices issues a copy but the caller reaches it after the CPU half has
        already been launched, and every value read is host-side C++ state).
        """
        t_enter = time.perf_counter()
        self.calls += 1
        self._lock_steps += 1
        try:
            split = hidden_states.size(0)
        except Exception:
            split = int(bs)
        if self.first_rows is None:
            self.first_rows = split
        elif split != self.first_rows:
            self.add("rows_mismatch")

        # ---- A. the ids the GPU kernels are handed (sampled rows; one bulk D2H)
        if topk_ids is not None and split:
            try:
                t = topk_ids.reshape(-1)
                k = int(getattr(module, "top_k", 0) or 1)
                real = split // rows_per_token if rows_per_token else split
                real = max(0, min(real, t.numel() // max(1, k)))
                want = max(1, self.idsample)  # rows inspected per layer call
                # evenly spread rows -> one .tolist() == one small D2H copy
                rows = sorted({int(round(i * (real - 1) / max(1, want - 1)))
                               for i in range(want)}) if real > 1 else [0] if real == 1 else []
                if rows:
                    blk = t.reshape(-1, k)[rows]
                    self.add("rows_read", len(rows))
                    for ids in [[int(x) for x in r] for r in blk.tolist()]:
                        self.add("gpu_tok_ids", len(ids))
                        dup = len(set(ids)) < len(ids)
                        self.add("id_dup_rows", 1 if dup else 0)
                        self.add("id_nodup_rows", 0 if dup else 1)
                        self.add("id_missing", sum(1 for x in ids if x < 0))
                        if self.pool_slots:
                            self.add("id_gt_pool",
                                     sum(1 for x in ids if not (0 <= x < self.pool_slots)))
                        self._digest.append(",".join(str(x) for x in ids))
            except Exception:
                pass

        # ---- B. (layer, expert) probes over a rotating window of sampled ids
        if self.idsample and topk_ids is not None and split:
            try:
                import lk_moe
                t = topk_ids.reshape(-1)
                layer = self._layer_index(module)
                k = int(getattr(module, "top_k", 0) or 1)
                n = int(t.numel())
                if n and k:
                    rot = self._probe_rot
                    idx = [(rot + j) % n for j in range(min(n, self.idsample))]
                    self._probe_rot = (rot + self.idsample) % n
                    for eid in [int(x) for x in t[idx].tolist()]:
                        if eid < 0:
                            continue
                        snap = int(lk_moe.pool_snapshot_slot(layer, eid))
                        ready = int(lk_moe.pool_ready_slot(layer, eid))
                        self.add("slots_probed")
                        if snap >= 0:
                            self.add("snap_claim")
                        else:
                            self.add("snap_absent")
                        if snap < 0 and ready >= 0:
                            self.add("snap_mismatch")
            except Exception:
                pass

        self.layer_time((time.perf_counter() - t_enter) * 1000.0)
        # ---- C. emit cadence: exactly once per `every` layer calls
        if self._lock_steps % self.every == 0:
            if not (self.head and self.windows >= self.head):
                self._window()

    @staticmethod
    def _layer_index(module):
        try:
            geo = getattr(module, "_pool_layer_info", None)
            if isinstance(geo, dict) and "layer" in geo:
                return int(geo["layer"])
        except Exception:
            pass
        return int(getattr(module, "layer_id", -1))

    def layer_time(self, ms):
        try:
            self._layer_calls.append(float(ms))
        except Exception:
            pass

    def close(self):
        if self._fh is None:
            return
        try:
            if self._c or self._layer_calls:
                self._window()
            self._emit({"kind": "SUMMARY", "pid": self.pid,
                        "t_s": round(time.time() - self.t0, 1),
                        "windows": self.windows, "layer_calls": self.calls,
                        "out": self.out,
                        "line": "SUMMARY windows=%d layer_calls=%d out=%s"
                                % (self.windows, self.calls, self.out)})
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        finally:
            self._fh = None


_probe = None


def probe():
    """Process-wide probe instance, or None while the gate is off."""
    global _probe
    if not enabled():
        return None
    if _probe is None:
        _probe = Probe()
        atexit.register(_probe.close)

        def _sig(signum, frame):
            try:
                _probe.close()
            except Exception:
                pass
        for s in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(s, _sig)
            except Exception:
                pass
        _log("[pool6row] armed pid=%d every=%d idsample=%d pool_slots=%d out=%s"
             % (_probe.pid, _probe.every, _probe.idsample, _probe.pool_slots,
                _probe.out))
    return _probe
