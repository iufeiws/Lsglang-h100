"""CPU checks of the production pool selector and all-row verify dispatch."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[2] / "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"


@pytest.mark.parametrize("opt_in", [None, "false", "true"])
@pytest.mark.parametrize("requests", [1, 2, 16])
def test_verify_default_uses_decode_gemv_for_every_row(opt_in, requests):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedMoE")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_init_pool_decode")
    select = next(
        n for n in init.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_pool_verify_gemm" for t in n.targets)
    )
    launch = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_pool_decode_launch")
    rows, k, h, i = requests * 6, 6, 64, 32
    calls = []

    class Stream:
        cuda_stream = 0

        def wait_stream(self, other):
            pass

    class Fill:
        def __getitem__(self, grid):
            def fill(ids, addr, slots, n_slots, **kwargs):
                calls.append(("slots", n_slots))
                assert ids.shape == (rows, k)
                # EP-style dropped routes and resident/miss routes coexist.
                slots[:n_slots].copy_(ids.reshape(-1))
            return fill

    pool = SimpleNamespace(
        _pool_verify_rows_mult=6,
        pool_hpart=None, pool_hbuf=None, pool_gout=torch.zeros(rows, h),
        pool_gout_part=None, pool_slot_tab=torch.empty(rows * k, dtype=torch.int32),
        pool_stream=Stream(), pool_verify_activated=None,
    )
    layer = SimpleNamespace(_pool_layer_info=dict(
        H=h, I=i, k=k, scale_group=32, packed_k_per_iter=2, snapshot_addr=0,
    ))

    def gemv(hidden, ids, weights, slots, n_slots, real_tokens, rows_per_token):
        calls.append(("gemv", real_tokens, rows_per_token))
        assert real_tokens == rows and rows_per_token == 1
        assert n_slots == rows * k
        assert torch.equal(slots[:n_slots], ids.reshape(-1))

    layer._pool_gemv_launch = gemv
    layer.lk_moe = SimpleNamespace(pool_record_aux=lambda stream: None)
    ns = dict(
        self=layer, H=h, I=i, PK=2, SSZ=1,
        caps=dict(format="mxfp4", scale_group=32, has_gate=True, activation=0),
        get_bool_env_var=lambda name, default: (default if opt_in is None else opt_in) == "true",
        torch=SimpleNamespace(cuda=SimpleNamespace(
            get_device_capability=lambda: (9, 0), current_stream=Stream,
            stream=lambda stream: nullcontext(),
        )),
        FusedMoE=pool, _LK_POOL_SPLIT_PROBE=False, _LK_POOL_DET_SPLIT=False,
        slot_table_fill=Fill(),
        triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        launch_lk_verify=lambda *args: calls.append(("gemm",)),
    )
    exec(compile(ast.Module(body=[select, launch], type_ignores=[]), str(SOURCE), "exec"), ns)
    ids = torch.arange(rows * k, dtype=torch.int32).reshape(rows, k) % 4
    ids[:, 0] = -1
    ns["_pool_decode_launch"](
        layer, torch.zeros(rows, h), ids, torch.ones(rows, k),
        real_tokens=requests, rows_per_token=6,
    )
    assert calls[0] == ("slots", rows * k)
    assert calls[1][0] == ("gemm" if opt_in == "true" else "gemv")
