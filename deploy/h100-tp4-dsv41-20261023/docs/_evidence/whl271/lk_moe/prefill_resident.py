"""
copyright (c) 2025-2026 guqiong9696@gmail.com, All Rights Reserved.
GPU side of a split prefill: pool-resident expert slots via the group-GEMM
kernels, paired with cpu_prefill(pool_skip=True) over the misses. Residency is
read as-is from the frozen snapshot (nothing claimed or staged). launch() runs
on a private stream and returns (event, gpu_out); the caller waits on the event
then adds gpu_out to the CPU part. Buffers are shared across layers, grown on
demand.
"""
import torch
import triton

from ._hybrid_gemv_kernel import slot_table_fill, reduce_down
from ._hybrid_gemm_kernel import (
    moe_seg_count, moe_seg_scan, moe_seg_place,
    moe_gateup_group_gemm, moe_down_group_gemm,
)
from ._pool_kernel_common import KB, m_tile


class PrefillResident:
    """One per FusedMoE layer. None from probe() => caller stays CPU-only."""

    @staticmethod
    def probe(moe):
        # Returns a driver or raises: a GPU layer that quietly stays on
        # CPU is 10x-slower WRONG, not graceful.
        info = moe.pool_layer_info()
        if not info:
            raise RuntimeError(
                f"resident layer {moe.cfg.layer_id}: pool not registered")
        if not info["gpu_view_ok"]:
            raise RuntimeError(
                f"resident layer {moe.cfg.layer_id}: no device view")
        if not info["snapshot_addr"] or not info["freeze_addr"]:
            raise RuntimeError(
                f"resident layer {moe.cfg.layer_id}: snapshot/freeze "
                f"table unmapped")
        if int(info["slot_count"]) < int(moe.cfg.top_k):
            raise RuntimeError(
                f"resident layer {moe.cfg.layer_id}: pool slots "
                f"{info['slot_count']} < top_k {moe.cfg.top_k}")
        caps = info["kernel_caps"]
        if caps.get("format") not in ("nvfp4", "mxfp4"):
            logger.warning(
                "resident layer %s: resident kernels unsupported for "
                "format=%s -> prefill stays CPU",
                moe.cfg.layer_id, caps.get("format"))
            return None
        gds = moe.pool_gds()
        if caps.get("format") == "nvfp4" and (not gds or not gds["gateup_global_scale_addr"]):
            raise RuntimeError(
                f"resident layer {moe.cfg.layer_id}: global-scale tables "
                f"absent")
        return PrefillResident(moe, info, gds, caps)

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
        self.snap_addr = int(info["snapshot_addr"])
        self.gw, self.gws = int(info["weight_addr"][0]), int(info["scale_addr"][0])
        self.uw, self.uws = (int(info["weight_addr"][1]), int(info["scale_addr"][1])
                             if self.has_gate else (int(info["weight_addr"][0]),
                                                    int(info["scale_addr"][0])))
        di = 2 if self.has_gate else 1
        self.dw, self.dws = int(info["weight_addr"][di]), int(info["scale_addr"][di])
        self.gds13 = int(gds["gateup_global_scale_addr"]) if gds and gds.get("gateup_global_scale_addr") else 0
        self.gds2 = int(gds["down_global_scale_addr"]) if gds and gds.get("down_global_scale_addr") else 0
        self.fmt = caps.get("format", "nvfp4")
        self.pk = 2 if self.fmt in ("nvfp4", "mxfp4") else 1
        self.ssz = 1 if self.fmt != "fp8" else 4
        self.sg = int(caps["scale_group"]) if caps.get("scale_group") else 16
        # slot geometry straight from the pool: [col_blk][K/2][32] bytes
        self.nb_g = int(info["slot_weight_bytes"][0]) // ((self.H // self.pk) * 32)
        self.nb_d = int(info["slot_weight_bytes"][2 if self.has_gate else 1]) // ((self.I // self.pk) * 32)
        assert int(info["slot_weight_bytes"][0]) == self.nb_g * (self.H // self.pk) * 32
        assert int(info["slot_scale_bytes"][0]) == self.nb_g * (self.H // self.sg) * 32 * self.ssz
        di = 2 if self.has_gate else 1
        assert int(info["slot_weight_bytes"][di]) == self.nb_d * (self.I // self.pk) * 32
        assert int(info["slot_scale_bytes"][di]) == self.nb_d * (self.I // self.sg) * 32 * self.ssz
        self.slot_count = int(info["slot_count"])
        # m_tile / kb: shared homes in _pool_kernel_common; col merges are
        # pure geometry.
        self.m_tile = m_tile()
        self.kb = KB
        self.cols_g = 2 if self.nb_g % 2 == 0 else 1
        self.cols_d = next(c for c in (4, 2, 1) if self.nb_d % c == 0)
        assert self.nb_g % self.cols_g == 0 and self.nb_d % self.cols_d == 0

    _side = None

    # Grow-only buffer set shared by ALL layers (per-layer copies would cost
    # 670MB x n_layers). Safe via the shared side stream + caller wait_event
    # ordering.
    _shared = None

    def _ensure(self, n_slots):
        cls = PrefillResident
        if cls._shared is not None and n_slots <= cls._shared["cap"]:
            self._bufs = cls._shared
            return
        cap = max(n_slots, cls._shared["cap"] if cls._shared else 0, 8192)
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

    def launch(self, hidden_states, topk_ids, topk_weights):
        """Enqueue the resident-slot side on a side stream. Returns
        (done_event, out[M, H] fp32). A pool whose slot space cannot be
        addressed raises instead of silently staying CPU-only."""
        if self.slot_count > self.E:
            raise RuntimeError(
                f"resident layer {self.layer}: pool slots "
                f"{self.slot_count} exceed E {self.E} (snapshot word "
                f"domain outside the weight sets)")
        M = hidden_states.size(0)
        n_slots = M * self.topk
        self._ensure(n_slots)
        b = self._bufs
        dev = self.dev
        if PrefillResident._side is None:
            PrefillResident._side = torch.cuda.Stream(device=dev)
        self._side = PrefillResident._side
        ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        weights = topk_weights if topk_weights.dtype == torch.float32 \
            else topk_weights.to(torch.float32)
        x = hidden_states.contiguous()
        out = b["out"][:M]
        tiles = n_slots // self.m_tile + self.E + 1
        main = torch.cuda.current_stream(dev)
        self._side.wait_stream(main)
        with torch.cuda.stream(self._side):
            slot_table_fill[(triton.cdiv(n_slots, 1024),)](
                ids, self.snap_addr, b["slot_tab"], n_slots, BLOCK=1024)
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
                b["slot_tab"], b["row_of"], b["tile_id"],
                self.gw, self.gws, self.uw, self.uws, self.gds13,
                b["hbuf"], x, n_slots, self.topk, 1,
                GDS_COLS=2 if self.has_gate else 1, N_TILE_ROWS_MAX=b["row_of"].numel(),
                NCOL=self.I, K_PER_BLOCK=self.H // self.pk,
                SCALE_GROUPS_BLK=self.H // self.sg,
                SLOT_BYTES=self.nb_g * (self.H // self.pk) * 32,
                SLOT_SCALE_BYTES=self.nb_g * (self.H // self.sg) * 32 * self.ssz,
                KCOL_STRIDE=(self.H // self.pk) * 32, HIDDEN=self.H,
                SCALE_GROUP=self.sg, FORMAT=self.fmt, FORMAT_PACKED_K=self.pk, SCALE_SIZE=self.ssz,
                HAS_GATE=self.has_gate, ACTIVATION=self.act,
                ALPHA=self.alpha, LIMIT=self.limit, M_TILE=self.m_tile, KB=self.kb, COLS=self.cols_g)
            moe_down_group_gemm[(tiles, self.nb_d // self.cols_d)](
                b["slot_tab"], weights, b["row_of"], b["tile_id"],
                self.dw, self.dws, self.gds2,
                b["hbuf"], b["gout_part"], n_slots,
                GDS_COLS=2 if self.has_gate else 1, N_TILE_ROWS_MAX=b["row_of"].numel(),
                NCOL=self.H, N_PARTIAL=self.I, K_PER_BLOCK=self.I // self.pk,
                SCALE_GROUPS_BLK=self.I // self.sg,
                SLOT_BYTES=self.nb_d * (self.I // self.pk) * 32,
                SLOT_SCALE_BYTES=self.nb_d * (self.I // self.sg) * 32 * self.ssz,
                KCOL_STRIDE=(self.I // self.pk) * 32,
                SCALE_GROUP=self.sg, FORMAT=self.fmt, FORMAT_PACKED_K=self.pk, SCALE_SIZE=self.ssz,
                HAS_GATE=self.has_gate, ACTIVATION=self.act,
                ALPHA=self.alpha, LIMIT=self.limit, M_TILE=self.m_tile, KB=self.kb, COLS=self.cols_d)
            reduce_down[(triton.cdiv(self.H, 1024), triton.cdiv(M, 64))](
                b["gout_part"], out, b["slot_tab"], HIDDEN=self.H,
                SLOTS_PER_TOKEN=self.topk, BATCH=64, BLOCK=1024, RESIDENT_ONLY=1,
                TOKS=64, NTOK=M)
            ev = torch.cuda.Event()
            ev.record(self._side)
        # keep referenced tensors alive until the side stream is done
        for t in (x, ids, weights, out):
            t.record_stream(self._side)
        return ev, out
