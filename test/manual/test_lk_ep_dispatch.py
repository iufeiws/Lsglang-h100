"""CPU regression tests for LK/FlashInfer expert-ID contracts.

Execute the production dispatcher and run_moe_core methods via AST so this
test needs only torch/pytest, not CUDA, FlashInfer, or the native LK library.
"""

import ast
from collections import namedtuple
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
TopK = namedtuple("TopK", "topk_weights topk_ids router_logits")
Dispatch = namedtuple("Dispatch", "hidden_states hidden_states_scale topk_output")
Combine = namedtuple("Combine", "hidden_states")


class Backend:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        assert name.startswith("is_")
        return lambda: self.name == name[3:]


def load_dispatcher(ep_size, ep_rank, backend):
    path = ROOT / "python/sglang/srt/layers/moe/token_dispatcher/standard.py"
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in ("_map_topk_ids_to_local_experts", "StandardDispatcher")
    ]
    namespace = dict(
        torch=torch,
        BaseDispatcher=object,
        get_parallel=lambda: SimpleNamespace(moe_ep_size=ep_size, moe_ep_rank=ep_rank),
        get_moe_runner_backend=lambda: Backend(backend),
        get_moe_a2a_backend=lambda: SimpleNamespace(supports_aiter=lambda: False),
        _use_aiter=False,
        _MASK_DP_PAD_MOE=False,
        get_device=lambda: "cpu",
        should_use_flashinfer_cutlass_moe_fp4_allgather=lambda: False,
        TopKOutputChecker=SimpleNamespace(
            format_is_standard=lambda x: isinstance(x, TopK)
        ),
        StandardDispatchOutput=Dispatch,
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ] + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["StandardDispatcher"]


def dispatcher(ep_size, ep_rank, backend="flashinfer_mxfp4", resident=False, shared=0):
    config = SimpleNamespace(
        num_experts=384 + shared,
        num_local_experts=384 // ep_size + shared,
        num_fused_shared_experts=shared,
        layer=SimpleNamespace(is_gpu_resident_layer=resident),
    )
    return load_dispatcher(ep_size, ep_rank, backend)(config)


@pytest.mark.parametrize("ep_size", [1, 2, 4])
@pytest.mark.parametrize(
    "backend",
    ["flashinfer_mxfp4", "flashinfer_cutlass", "flashinfer_trtllm", "triton", "aiter"],
)
def test_lk_routes_are_local_on_every_ep_rank(ep_size, backend):
    ids = torch.tensor([[-1, 0, 95, 191, 192, 383]], dtype=torch.int32)
    weights = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.15, 0.25]])
    topk = TopK(weights, ids, None)
    for rank in range(ep_size):
        d = dispatcher(ep_size, rank, backend)
        out = d.dispatch(torch.zeros(1, 8), topk)
        start, count = rank * (384 // ep_size), 384 // ep_size
        expected = torch.tensor(
            [[eid - start if start <= eid < start + count else -1
              for eid in ids[0].tolist()]], dtype=torch.int32
        )
        torch.testing.assert_close(out.topk_output.topk_ids, expected)
        assert out.topk_output.topk_weights is weights  # no per-rank renormalization
        assert torch.equal(ids, torch.tensor([[-1, 0, 95, 191, 192, 383]], dtype=torch.int32))


@pytest.mark.parametrize("resident", [True, None])
@pytest.mark.parametrize("backend", ["flashinfer_mxfp4", "flashinfer_cutlass", "flashinfer_trtllm"])
def test_gpu_runner_keeps_global_expert_ids(resident, backend):
    config = SimpleNamespace(num_experts=384, num_local_experts=192, num_fused_shared_experts=0)
    if resident is not None:
        config.layer = SimpleNamespace(is_gpu_resident_layer=resident)
    d = load_dispatcher(2, 1, backend)(config)
    topk = TopK(torch.ones(1, 3), torch.tensor([[0, 192, 383]], dtype=torch.int32), None)
    assert d.dispatch(torch.zeros(1, 8), topk).topk_output is topk


def test_ep2_fused_shared_expert_has_local_tail_slot():
    topk = TopK(torch.ones(1, 4), torch.tensor([[-1, 0, 383, 384]], dtype=torch.int32), None)
    for rank, expected in [(0, [-1, 0, -1, 192]), (1, [-1, -1, 191, 192])]:
        out = dispatcher(2, rank, shared=1).dispatch(torch.zeros(1, 8), topk)
        assert out.topk_output.topk_ids.tolist() == [expected]


@pytest.mark.parametrize("rows", [1, 6, 96])
def test_ep2_contributions_sum_to_ep1_without_losing_or_duplicating_routes(rows):
    gen = torch.Generator().manual_seed(27)
    ids = torch.randint(0, 384, (rows, 6), generator=gen, dtype=torch.int32)
    ids[0].fill_(-1)  # dropped row must contribute exactly zero
    weights = torch.rand(rows, 6, generator=gen)
    weights /= weights.sum(-1, keepdim=True)
    experts = torch.randn(384, 8, generator=gen)
    topk = TopK(weights, ids, None)
    expected = (experts[ids.clamp_min(0)] * weights[..., None] * (ids >= 0)[..., None]).sum(1)
    actual = torch.zeros_like(expected)
    for rank in range(2):
        out = dispatcher(2, rank).dispatch(torch.zeros(rows, 8), topk)
        local = out.topk_output.topk_ids
        assert ((local == -1) | ((local >= 0) & (local < 192))).all()
        values = experts[rank * 192 : (rank + 1) * 192][local.clamp_min(0)]
        actual += (values * weights[..., None] * (local >= 0)[..., None]).sum(1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mode", ["prefill", "gpu_prefill", "capture"])
@pytest.mark.parametrize("rank", [0, 1])
def test_all_lk_entry_points_receive_local_ids(mode, rank):
    path = ROOT / "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedMoE")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run_moe_core")
    namespace = dict(
        torch=torch,
        get_is_capture_mode=lambda: mode == "capture",
        is_decode_or_verify_batch=lambda: False,
        Mxfp4FlashinferTrtllmMoEMethod=type(None),
        Mxfp4FlashinferCutlassMoEMethod=type(None),
        Mxfp4MarlinMoEMethod=type(None),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    seen = []

    def native_entry(name):
        def call(hidden, weights, ids):
            assert ((ids == -1) | ((ids >= 0) & (ids < 192))).all()
            seen.append((name, ids.clone()))
            return torch.zeros_like(hidden)
        return call

    layer = SimpleNamespace(
        is_gpu_resident_layer=False,
        should_prefill_gpu=lambda _: mode == "gpu_prefill",
        _cpu_decode=native_entry("capture"),
        _cpu_prefill=native_entry("prefill"),
        _gpu_prefill=native_entry("gpu_prefill"),
        quant_method=None,
        should_fuse_routed_scaling_factor_in_topk=True,
    )
    topk = TopK(torch.ones(6, 3), torch.tensor([[0, 192, 383]], dtype=torch.int32).repeat(6, 1), None)
    dispatched = dispatcher(2, rank).dispatch(torch.zeros(6, 8), topk)
    stub = ModuleType("sglang.srt.layers.moe.token_dispatcher")
    stub.StandardCombineInput = Combine
    with (
        patch.dict("sys.modules", {stub.__name__: stub}),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
    ):
        namespace["run_moe_core"](layer, dispatched)
    assert len(seen) == 1 and seen[0][0] == mode
    assert seen[0][1][0].tolist() == ([0, -1, -1] if rank == 0 else [-1, 0, 191])
