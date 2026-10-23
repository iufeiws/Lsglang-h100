"""
copyright (c) 2025-2026 guqiong9696@gmail.com, All Rights Reserved.
GPU side of a split prefill over pinned pool slots (group-GEMM). Per layer:
pin the pool+cache snapshot, stage only (topk ids - resident experts) into
the staging ring, enqueue the gemm on a private stream, return (event,
gpu_out); the pin retires on the gemm stream (C++ deferred unpin). No
prefetch-ahead and no cross-layer window.
"""
import logging
import weakref

import torch
import triton

logger = logging.getLogger(__name__)

from ._hybrid_gemv_kernel import reduce_down
from ._hybrid_gemm_kernel import (
    route_scan,
    moe_seg_count, moe_seg_scan, moe_seg_place,
    moe_gateup_group_gemm, moe_down_group_gemm,
)

from ._pool_kernel_common import KB, m_tile


class StagingPrefill:
    """One per FusedMoE layer. probe() returns a driver or raises."""

    @staticmethod
    def probe(moe):
        # Build the GPU-prefill driver for a registered pool layer, or
        # raise. An unqualified GPU-prefill layer is a BROKEN deployment
        # (a demoted layer runs the CPU path ~10x slower -- wrong, not
        # graceful), so every mismatch dies here, at boot, with a cause.
        info = moe.pool_layer_info()
        if not info:
            raise RuntimeError(
                f"GPU prefill layer {moe.cfg.layer_id}: prefill pool infra "
                f"not registered (role="
                f"{'draft' if moe.cfg.role_draft else 'main'}, pool_slots="
                f"{moe.cfg.pool_slots}, prefill_gpu={moe.cfg.prefill_gpu}); "
                f"the prefill tier owns its staging ring and is independent "
                f"of the cache tier -- check that cfg.prefill_gpu reached "
                f"the C++ ctor and pool_init() ran")
        if not info["gpu_view_ok"]:
            raise RuntimeError(
                f"GPU prefill layer {moe.cfg.layer_id}: no device view")
        if not info["snapshot_addr"] or not info["freeze_addr"]:
            raise RuntimeError(
                f"GPU prefill layer {moe.cfg.layer_id}: snapshot/freeze "
                f"table unmapped (pin path broken)")
        # slot_count == 0 is a valid config: the pool caches nothing, so
        # every topk-selected expert of a layer rides the staging ring.
        if 0 < int(info["slot_count"]) < int(moe.cfg.top_k):
            raise RuntimeError(
                f"GPU prefill layer {moe.cfg.layer_id}: pool slots "
                f"{info['slot_count']} < top_k {moe.cfg.top_k}")
        # Draft instances (C++ keys them 1000+) are the ONE role whose
        # GPU-prefill membership is opportunistic: draft-extend ran on the
        # CPU path long before the pool existed, so a draft whose checkpoint
        # format has no staging GEMM simply KEEPS that baseline -- decided
        # once at boot, logged loudly, never re-decided at runtime. A MAIN
        # layer with the same gap is a broken deployment: raise.
        draft = int(info.get("layer", moe.cfg.layer_id)) >= 1000
        caps = info["kernel_caps"]
        if caps.get("format") not in ("nvfp4", "mxfp4"):
            if draft:
                logger.warning(
                    "GPU prefill draft layer %s: staging kernels are "
                    "nvfp4-only (format=%s) -> draft prefill stays CPU",
                    moe.cfg.layer_id, caps.get("format"))
                return None
            logger.warning(
                "GPU prefill layer %s: staging kernels unsupported for "
                "format=%s -> prefill stays CPU",
                moe.cfg.layer_id, caps.get("format"))
            return None
        gds = moe.pool_gds()
        if caps.get("format") == "nvfp4" and (not gds or not gds["gateup_global_scale_addr"]):
            if draft:
                logger.warning(
                    "GPU prefill draft layer %s: global-scale tables "
                    "absent -> draft prefill stays CPU",
                    moe.cfg.layer_id)
                return None
            raise RuntimeError(
                f"GPU prefill layer {moe.cfg.layer_id}: global-scale "
                f"tables absent (pool unstaged)")
        return StagingPrefill(moe, info, gds, caps)

    def __init__(self, moe, info=None, gds=None, caps=None):
        cfg = moe.cfg
        if info is None:
            info = moe.pool_layer_info()
        if gds is None:
            gds = moe.pool_gds()
        if caps is None:
            caps = info.get("kernel_caps") or {}
        self.dev = torch.device("cuda", cfg.gpu_id)
        self.H = int(cfg.hidden_size)
        self.I = int(cfg.intermediate_size)
        self.E = int(cfg.expert_num)
        self.topk = int(cfg.top_k)
        self.has_gate = bool(cfg.has_gate_proj)
        self.act = int(cfg.activation_type)
        self.alpha = float(cfg.swiglu_alpha)
        self.limit = float(cfg.swiglu_limit) if cfg.swiglu_limit else 0.0
        self.layer = int(info.get("layer", -1))
        self.dbg = bool(cfg.prefill_dbg)
        # ONE launch block = 8192 slots (M_TILE-quantized); longer
        # inputs WALK this fixed buffer, never CPU.
        self.snap_addr = int(info["snapshot_addr"])
        self.freeze_addr = int(info["freeze_addr"])
        if not self.freeze_addr:
            raise RuntimeError("pool freeze table unmapped")
        self.moe_ref = weakref.ref(moe)
        self._scan_exec = -1
        self.gw, self.gws = int(info["weight_addr"][0]), int(info["scale_addr"][0])
        self.uw, self.uws = (int(info["weight_addr"][1]), int(info["scale_addr"][1])
                             if self.has_gate else (int(info["weight_addr"][0]),
                                                    int(info["scale_addr"][0])))
        di = 2 if self.has_gate else 1
        self.dw, self.dws = int(info["weight_addr"][di]), int(info["scale_addr"][di])
        self.gds13 = int(gds["gateup_global_scale_addr"]) if gds and gds.get("gateup_global_scale_addr") else 0
        self.gds2 = int(gds["down_global_scale_addr"]) if gds and gds.get("down_global_scale_addr") else 0
        stw = info.get("staging_weight_addr") or [0, 0, 0]
        sts = info.get("staging_scale_addr") or [0, 0, 0]
        self.st_gw, self.st_gws = int(stw[0]), int(sts[0])
        self.st_uw, self.st_uws = (int(stw[1]), int(sts[1]) if self.has_gate
                                   else (int(stw[0]), int(sts[0])))
        self.st_dw, self.st_dws = int(stw[di]), int(sts[di])
        self.fmt = caps.get("format", "nvfp4")
        self.pk = 2 if self.fmt in ("nvfp4", "mxfp4") else 1
        self.ssz = 1 if self.fmt != "fp8" else 4
        self.sg = int(caps["scale_group"]) if caps.get("scale_group") else 16
        # slot geometry straight from the pool: [col_blk][K/2][32] bytes
        self.nb_g = int(info["slot_weight_bytes"][0]) // ((self.H // self.pk) * 32)
        self.nb_d = int(info["slot_weight_bytes"][2 if self.has_gate else 1]) // ((self.I // self.pk) * 32)
        # m_tile: shared platform home; kb: the only bit-exact value on the
        # dual-domain branch (M32/KB32/KB64 drift rel~1e-3, test_staging_
        # mixed M1024) and fastest on Blackwell; col merges: pure geometry.
        self.m_tile = m_tile()
        self.kb = KB
        self.cols_g = 2 if self.nb_g % 2 == 0 else 1
        self.cols_d = next(c for c in (4, 2, 1) if self.nb_d % c == 0)
        assert self.nb_g % self.cols_g == 0 and self.nb_d % self.cols_d == 0
        assert int(info["slot_weight_bytes"][0]) == self.nb_g * (self.H // self.pk) * 32
        assert int(info["slot_scale_bytes"][0]) == self.nb_g * (self.H // self.sg) * 32 * self.ssz
        di = 2 if self.has_gate else 1
        assert int(info["slot_weight_bytes"][di]) == self.nb_d * (self.I // self.pk) * 32
        assert int(info["slot_scale_bytes"][di]) == self.nb_d * (self.I // self.sg) * 32 * self.ssz
        self.slot_count = int(info["slot_count"])         # pool domain C
        self._W = min(2, max(1, int(cfg.prefill_window) or 1))
        self.slot_domain = self.slot_count + self.E * self._W  # C++ block_slots_ = experts

    _side = None
    _copy = None  # prefill-private H2D lane: staging overlaps with the
                  # gemm on _side; both are fenced vs the decode lane in C++

    # Grow-only buffer set shared by ALL layers (per-layer copies would cost
    # 670MB x n_layers). Safe via the shared side stream + caller wait_event
    # ordering.
    _shared = None

    def _ensure_route(self):
        if self._bufs is None:
            self._bufs = {}
        for i in range(self._W):
            key = "route" if i == 0 else f"route{i}"
            if key not in self._bufs:
                self._bufs[key] = torch.full((self.E,), -1, dtype=torch.int32,
                                             device=self.dev)

    def _ensure(self, n_slots):
        cls = StagingPrefill
        if cls._shared is not None and n_slots <= cls._shared["cap"]:
            self._bufs = cls._shared
            return
        cap = max(n_slots, 8192)
        dev = self.dev
        tiles = cap // self.m_tile + self.E + 1
        cls._shared = dict(
            slot_tab=torch.empty(cap, dtype=torch.int32, device=dev),
            counts=torch.empty(self.E, dtype=torch.int32, device=dev),
            counts_bak=torch.empty(self.E, dtype=torch.int32, device=dev),
            offsets=torch.empty(self.E, dtype=torch.int32, device=dev),
            cursor=torch.empty(self.E, dtype=torch.int32, device=dev),
            row_of=torch.empty(tiles * self.m_tile, dtype=torch.int32, device=dev),
            tile_id=torch.empty(tiles, dtype=torch.int32, device=dev),
            n_seg=torch.empty(1, dtype=torch.int32, device=dev),
            hbuf=torch.empty(cap, 2 * self.I, dtype=torch.float32, device=dev),
            gout_part=torch.empty(cap, self.H, dtype=torch.float32, device=dev),
            out=torch.empty(cap // self.topk + 1, self.H, dtype=torch.float32, device=dev),
            cap=cap,
        )
        self._bufs = cls._shared

    _ids_host = None  # pinned D2H staging for this layer's topk ids
    _pool_stream = None  # decode gemv lane (layer.py wires it): staging H2Ds
                         # are ISSUED ON it so cuda-graph replay readers of a
                         # victim slot are ordered by same-lane FIFO, no event
                         # bookkeeping can miss them.

    def prefill_gpu(self, hidden_states, topk_ids, topk_weights):
        """Pure-GPU prefill layer, self-contained per layer.

        Stage ONLY (this layer's topk experts - pool-cached experts). There
        is no prefetch-ahead and no cross-layer window: routing for the
        current layer is the only input, storage/ring geometry is unchanged,
        and the gemm never reads a row outside its ids. The pool is pinned
        for this layer only; prefill_begin republishes the snapshot so
        route_scan sees exactly the residency the staging decision used.
        Every failure raises; the under-gate decision lives in the framework
        gate, not in this layer."""
        moe = self.moe_ref()
        if StagingPrefill._side is None:
            StagingPrefill._side = torch.cuda.Stream(device=self.dev)
        if StagingPrefill._copy is None:
            StagingPrefill._copy = torch.cuda.Stream(device=self.dev)
        ids_flat = topk_ids.reshape(-1)
        if ids_flat.dtype != torch.int32:
            ids_flat = ids_flat.to(torch.int32)
        # C++ filters the copy set on the host, so the ids must be here (the
        # ring rows themselves are unchanged; duplicates are dropped C-side).
        n_ids = ids_flat.numel()
        if (StagingPrefill._ids_host is None
                or StagingPrefill._ids_host.numel() < n_ids):
            StagingPrefill._ids_host = torch.empty(
                max(n_ids, 8192), dtype=torch.int32, pin_memory=True)
        ids_host = StagingPrefill._ids_host[:n_ids]
        ids_host.copy_(ids_flat, non_blocking=False)
        gemm_stream = StagingPrefill._side.cuda_stream
        if not moe.prefill_begin(gemm_stream):
            try:
                info = moe.pool_layer_info()
            except Exception as e:
                info = f"info_err:{e}"
            raise RuntimeError(
                f"GPU prefill layer {self.layer}: prefill_begin failed; "
                f"pool_info={info}")
        try:
            if not moe.prefill_stage(
                    ids_host.data_ptr(), n_ids,
                    StagingPrefill._copy.cuda_stream, gemm_stream):
                raise RuntimeError(
                    f"GPU prefill layer {self.layer}: stage failed")
            got = self._launch(moe, hidden_states, topk_ids, topk_weights)
            if got is None:
                raise RuntimeError(
                    f"GPU prefill layer {self.layer}: gemm launch failed")
        finally:
            # Deferred unpin: the ticket rides the gemm stream, so the pin
            # covers this layer's gemm without blocking the host.
            moe.prefill_end(gemm_stream)
        ev, out = got
        return ev, out

    def _launch(self, moe, hidden_states, topk_ids, topk_weights):
        M = hidden_states.size(0)
        block_tokens = 8192 // self.topk
        if M <= block_tokens:
            return self._launch_block(moe, hidden_states, topk_ids,
                                      topk_weights)
        # Over the buffer cap: WALK token blocks on the fixed buffers (the
        # staging copy already covered the whole layer exactly once --
        # stage-once). The GPU tier NEVER falls back to CPU.
        out = torch.empty(M, self.H, dtype=torch.float32,
                          device=hidden_states.device)
        ev = None
        for m0 in range(0, M, block_tokens):
            m1 = min(m0 + block_tokens, M)
            if topk_ids.dim() == 2:          # (M, k) layout
                ids_b, w_b = topk_ids[m0:m1], topk_weights[m0:m1]
            else:                            # flat (M*k,) layout
                ids_b = topk_ids[m0 * self.topk:m1 * self.topk]
                w_b = topk_weights[m0 * self.topk:m1 * self.topk]
            got = self._launch_block(moe, hidden_states[m0:m1], ids_b, w_b,
                                     out=out[m0:m1])
            if got is None:
                return None    # only a launch failure (never capacity)
            ev = got[0]
        return ev, out

    def _launch_block(self, moe, hidden_states, topk_ids, topk_weights,
                      out=None):
        M = hidden_states.size(0)
        n_slots = M * self.topk
        self._ensure(n_slots)
        self._ensure_route()
        b = self._bufs
        dev = self.dev
        if StagingPrefill._side is None:
            StagingPrefill._side = torch.cuda.Stream(device=dev)
        self._side = StagingPrefill._side
        ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        weights = topk_weights if topk_weights.dtype == torch.float32 \
            else topk_weights.to(torch.float32)
        x = hidden_states.contiguous()
        if out is None:
            out = b["out"][:M]
        tiles = n_slots // self.m_tile + self.E + 1
        main = torch.cuda.current_stream(dev)
        self._side.wait_stream(main)
        _exec = moe.prefill_execution()
        par = _exec % self._W
        # PARITY/ROW algebra here MIRRORS PrefillStaging::staging_row /
        # staging_pos (C++): row = parity*E + eid, pos = C + row. Change one,
        # change both.
        rtab = b["route"] if par == 0 else b[f"route{par}"]
        if self._scan_exec != _exec:
            self._scan_exec = _exec
            # scan rides the COPY lane: stage() already queued this block's
            # overwrite wait on it (last consumer gemm of the same parity),
            # so these bytes move WHILE THE PREVIOUS LAYER'S GEMM IS STILL
            # RUNNING on the side lane.
            with torch.cuda.stream(StagingPrefill._copy):
                route_scan[(1,)](self.snap_addr, rtab, self.slot_count,
                                 par * self.E, E=self.E, BLOCK=1024)
                _evg = torch.cuda.Event()
                _evg.record(StagingPrefill._copy)
            self._side.wait_event(_evg)
        with torch.cuda.stream(self._side):
            b["counts"].zero_()
            moe_seg_count[(triton.cdiv(n_slots, 1024),)](
                ids, b["counts"], n_slots, E=self.E, BLOCK=1024)
            b["tile_id"].fill_(-1)
            b["row_of"].fill_(-1)
            b["cursor"].zero_()
            moe_seg_scan[(1,)](b["counts"], b["counts_bak"], b["offsets"],
                               b["tile_id"], b["n_seg"], E=self.E,
                               EP=triton.next_power_of_2(self.E), M_TILE=self.m_tile)
            moe_seg_place[(triton.cdiv(n_slots, 1024),)](
                ids, b["offsets"], b["cursor"], b["row_of"], n_slots, 0, BLOCK=1024)
            moe_gateup_group_gemm[(tiles, self.nb_g // self.cols_g)](
                rtab, b["row_of"], b["tile_id"],
                self.gw, self.gws, self.uw, self.uws, self.gds13,
                self.st_gw, self.st_gws, self.st_uw, self.st_uws,
                b["hbuf"], x, n_slots, self.topk, 1,
                GDS_COLS=2 if self.has_gate else 1, N_TILE_ROWS_MAX=b["row_of"].numel(),
                NCOL=self.I, K_PER_BLOCK=self.H // self.pk,
                SCALE_GROUPS_BLK=self.H // self.sg,
                SLOT_BYTES=self.nb_g * (self.H // self.pk) * 32,
                SLOT_SCALE_BYTES=self.nb_g * (self.H // self.sg) * 32 * self.ssz,
                KCOL_STRIDE=(self.H // self.pk) * 32, HIDDEN=self.H,
                SCALE_GROUP=self.sg, FORMAT=self.fmt, FORMAT_PACKED_K=self.pk, SCALE_SIZE=self.ssz,
                HAS_GATE=self.has_gate, ACTIVATION=self.act,
                ALPHA=self.alpha, LIMIT=self.limit, M_TILE=self.m_tile, SLOT_COUNT=self.slot_domain,
                POOL_C=self.slot_count, KB=self.kb, COLS=self.cols_g)
            moe_down_group_gemm[(tiles, self.nb_d // self.cols_d)](
                rtab, weights, b["row_of"], b["tile_id"],
                self.dw, self.dws, self.gds2,
                self.st_dw, self.st_dws,
                b["hbuf"], b["gout_part"], n_slots,
                GDS_COLS=2 if self.has_gate else 1, N_TILE_ROWS_MAX=b["row_of"].numel(),
                NCOL=self.H, N_PARTIAL=self.I, K_PER_BLOCK=self.I // self.pk,
                SCALE_GROUPS_BLK=self.I // self.sg,
                SLOT_BYTES=self.nb_d * (self.I // self.pk) * 32,
                SLOT_SCALE_BYTES=self.nb_d * (self.I // self.sg) * 32 * self.ssz,
                KCOL_STRIDE=(self.I // self.pk) * 32,
                SCALE_GROUP=self.sg, FORMAT=self.fmt, FORMAT_PACKED_K=self.pk, SCALE_SIZE=self.ssz,
                HAS_GATE=self.has_gate, ACTIVATION=self.act,
                ALPHA=self.alpha, LIMIT=self.limit, M_TILE=self.m_tile, SLOT_COUNT=self.slot_domain,
                POOL_C=self.slot_count, KB=self.kb, COLS=self.cols_d)
            reduce_down[(triton.cdiv(self.H, 1024), triton.cdiv(M, 64))](
                b["gout_part"], out, rtab, HIDDEN=self.H,
                SLOTS_PER_TOKEN=self.topk, BATCH=64, BLOCK=1024, RESIDENT_ONLY=0,
                TOKS=64, NTOK=M)
            # private snapshot of gout ON THE SIDE STREAM: the next layer's
            # gemm reuses gout there and must not race a caller's later copy.
            out = out.clone()
            ev = torch.cuda.Event()
            ev.record(self._side)
            # fence for worker copies: the pool records THIS stream's
            # progress so no inbound H2D can land before the gemm reads.
            moe.pool_record_main(self._side.cuda_stream)
        # keep referenced tensors alive until the side stream is done
        for t in (x, ids, weights, out):
            t.record_stream(self._side)
        return ev, out
