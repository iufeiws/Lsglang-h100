"""LK target verification must use complete, independent MoE rows.

Runs the production Python adapter with a recording native boundary and a
small dense expert oracle. It does not emulate the proprietary native verify
implementation or claim to test that implementation's numerical accuracy.
No model, CUDA runtime, or server startup is needed.
"""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[2] / "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"


class Stream:
    cuda_stream = 0

    def wait_stream(self, other):
        pass


class CPUCudaTorch:
    """Only allocation/stream plumbing is replaced; tensor math is PyTorch."""

    cuda = SimpleNamespace(
        current_device=lambda: 0,
        current_stream=Stream,
        Stream=Stream,
        stream=lambda stream: nullcontext(),
        get_device_capability=lambda: (9, 0),
    )

    def __getattr__(self, name):
        value = getattr(torch, name)
        if name not in ("zeros", "empty", "full"):
            return value

        def allocate(*args, **kwargs):
            kwargs["device"] = "cpu"
            return value(*args, **kwargs)

        return allocate


def load_adapter():
    source = ast.parse(SOURCE.read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "FusedMoE")
    names = {"_init_pool_decode", "_pool_decode_launch", "_cpu_decode"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    pool = SimpleNamespace(_pool_verify_rows_mult=1)
    ns = dict(
        FusedMoE=pool,
        torch=CPUCudaTorch(),
        get_verify_resident=lambda: 0,
        get_verify_rows_per_request=lambda rows: 6,
        get_bool_env_var=lambda name, default: default == "true",
        _LK_POOL_SPLIT_PROBE=False,
        _LK_POOL_DET_SPLIT=False,
        triton=SimpleNamespace(
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
            cdiv=lambda n, d: (n + d - 1) // d,
        ),
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns, pool


def initialize(ns, pool, monkeypatch, *, resident, cap=2, width=6):
    staging = ModuleType("lk_moe.prefill_staging")
    staging.StagingPrefill = SimpleNamespace(_pool_stream=None)
    monkeypatch.setitem(sys.modules, "lk_moe.prefill_staging", staging)
    reports = []
    caps = dict(
        stride_n=32, packed_k_per_iter=2, scale_group=32, scale_size=1,
        format="mxfp4", has_gate=True, activation=0,
    )
    layer = SimpleNamespace(
        layer_id=20,
        max_running_requests=cap,
        speculative_num_draft_tokens=width,
        speculative_dspark_block_size=width - 1,
        lk_moe=SimpleNamespace(
            cfg=SimpleNamespace(hidden_size=64, intermediate_size=32, top_k=3),
            pool_layer_info=lambda: dict(
                kernel_caps=caps, snapshot_addr=0, weight_addr=[0] * 3,
                scale_addr=[0] * 3, slot_weight_bytes=[0] * 3,
                slot_scale_bytes=[0] * 3,
            ),
            pool_gds=lambda: dict(
                pool_decode_on=1, gateup_global_scale_addr=0, down_global_scale_addr=0,
            ),
            pool_gpu_report=reports.append,
            pool_record_aux=lambda stream: None,
        ),
    )
    ns["get_verify_resident"] = lambda: resident
    ns["_init_pool_decode"](layer)
    layer._pool_decode_launch = lambda *args, **kw: ns["_pool_decode_launch"](layer, *args, **kw)
    return layer, reports


@pytest.mark.parametrize("resident", [0, 1])
@pytest.mark.parametrize("requests", [1, 2, 16])
def test_target_verify_calls_native_with_every_row_complete(monkeypatch, resident, requests):
    ns, pool = load_adapter()
    layer, reports = initialize(ns, pool, monkeypatch, resident=resident, cap=requests)
    rows, h, k = requests * 6, 64, 3
    x = torch.arange(rows * h, dtype=torch.float32).reshape(rows, h).remainder(19).bfloat16()
    ids = torch.arange(rows * k).reshape(rows, k).remainder(4).int()
    ids[:, 0] = -1  # EP routes owned by another rank.
    ids[0] = -1    # A whole row with no locally owned experts.
    weights = torch.arange(1, rows * k + 1).reshape(rows, k).float() / (rows * k)
    pool.output_gpu = torch.full((rows, h), float("nan"))
    layer.top_k, layer.check_nan_in_output = k, False
    native_calls = []

    # Distinct row-dependent contributions make missing, duplicated, or shifted
    # rows visible. Even experts are resident, odd experts are CPU misses.
    def oracle(resident_half):
        out = torch.zeros_like(x, dtype=torch.float32)
        for row in range(rows):
            for route in range(k):
                expert = int(ids[row, route])
                if expert < 0 or (resident_half is not None and (expert % 2 == 0) != resident_half):
                    continue
                out[row] += torch.nn.functional.silu(x[row].float() + expert) * weights[row, route]
        return out

    class Slots:
        def __getitem__(self, grid):
            def fill(gathered, addr, slots, n_slots, **kwargs):
                flat = gathered.reshape(-1)
                slots[:n_slots] = torch.where((flat >= 0) & (flat % 2 == 0), flat, -1)
            return fill

    ns["slot_table_fill"] = Slots()

    class Gather:
        def __getitem__(self, grid):
            def gather(ids, weights, gathered_ids, gathered_weights, topk, stride, n_slots, **kwargs):
                gathered_ids.copy_(ids[::stride].reshape(-1))
                gathered_weights.copy_(weights[::stride].reshape(-1))
            return gather

    ns["gather_rows"] = Gather()

    def gemv(hidden, gathered_ids, gathered_weights, slots, n_slots, real_tokens, rows_per_token):
        assert real_tokens == rows and rows_per_token == 1
        assert n_slots == rows * k
        torch.testing.assert_close(gathered_ids, ids, rtol=0, atol=0)
        torch.testing.assert_close(gathered_weights, weights, rtol=0, atol=0)
        pool.pool_gout[:rows].copy_(oracle(True))

    layer._pool_gemv_launch = gemv

    def native(stream, batch, topk, xptr, idptr, wptr, optr, *, real_tokens, rows_per_token):
        native_calls.append((batch, real_tokens, rows_per_token))
        # The target model may not delegate any row to LK's speculative-row
        # policy, irrespective of greedy/sampling acceptance downstream.
        assert (batch, real_tokens, rows_per_token) == (rows, rows, 1)
        assert (xptr, idptr, wptr, optr) == (x.data_ptr(), ids.data_ptr(), weights.data_ptr(), pool.output_gpu.data_ptr())
        pool.output_gpu[:rows].copy_(oracle(False) if reports[-1] else oracle(None))

    layer.lk_moe.cpu_decode = native
    out = ns["_cpu_decode"](layer, x, weights, ids)
    assert native_calls == [(rows, rows, 1)]
    assert reports[-1] is True
    torch.testing.assert_close(out, (oracle(False) + oracle(True)).bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("resident", [0, 1])
def test_pool_has_room_for_all_verify_rows(monkeypatch, resident):
    ns, pool = load_adapter()
    initialize(ns, pool, monkeypatch, resident=resident, cap=16)
    rows, routes = 16 * 6, 16 * 6 * 3
    assert pool.pool_gout.shape == (rows, 64)
    assert pool.pool_slot_tab.shape == (routes,)
    assert pool.pool_hbuf.shape == (routes, 64)
    assert pool.pool_hpart.shape[1:] == (routes, 64)


def test_uncovered_batch_withdraws_gpu_claim(monkeypatch):
    ns, pool = load_adapter()
    layer, reports = initialize(ns, pool, monkeypatch, resident=1, cap=1, width=1)
    ns["get_verify_rows_per_request"] = lambda rows: 1
    layer.top_k, layer.check_nan_in_output = 3, False
    pool.output_gpu = torch.zeros(2, 64)
    x, ids, weights = torch.ones(2, 64), torch.zeros(2, 3, dtype=torch.int32), torch.ones(2, 3)

    def native(*args, real_tokens, rows_per_token):
        # All routes in this case are resident. A stale GPU claim makes the
        # CPU half skip them, although the adapter submitted no GPU work.
        pool.output_gpu.fill_(0 if reports[-1] else 7)

    layer.lk_moe.cpu_decode = native
    out = ns["_cpu_decode"](layer, x, weights, ids)
    torch.testing.assert_close(out, torch.full_like(x, 7))


@pytest.mark.parametrize("width,rows", [(6, 6), (6, 12), (1, 6), (1, 1)])
def test_without_gpu_pool_uses_full_native_rows(width, rows):
    ns, pool = load_adapter()
    ns["get_verify_rows_per_request"] = lambda n: width
    reports = [False]
    pool.output_gpu = torch.full((rows, 64), float("nan"))
    x = torch.ones(rows, 64, dtype=torch.bfloat16)

    def native(*args, real_tokens, rows_per_token):
        assert (real_tokens, rows_per_token) == (rows, 1)
        assert reports[-1] is False
        pool.output_gpu.copy_(x.float())

    layer = SimpleNamespace(
        _pool_layer_info=None, top_k=3, check_nan_in_output=False,
        lk_moe=SimpleNamespace(pool_gpu_report=reports.append, cpu_decode=native),
    )
    out = ns["_cpu_decode"](layer, x, torch.ones(rows, 3), torch.zeros(rows, 3, dtype=torch.int32))
    torch.testing.assert_close(out, x, rtol=0, atol=0)
