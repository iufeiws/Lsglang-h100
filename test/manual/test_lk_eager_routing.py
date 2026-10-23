"""CPU regression tests for the real LK branch selector and dummy-run scope.

Native compute is a spy: these tests establish routing/context correctness,
not the absence of a SIGSEGV inside the installed LK binary.
"""
import ast
import importlib.util
from contextlib import nullcontext
from enum import IntEnum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SRT = ROOT / "python/sglang/srt"
spec = importlib.util.spec_from_file_location(
    "lk_batch_scope_test", SRT / "model_executor/runner_utils/forward_batch.py"
)
scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scope)

# Execute the real enum without importing the CUDA/model dependency tree.
tree = ast.parse((SRT / "model_executor/forward_batch_info.py").read_text())
mode_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ForwardMode")
ns = dict(IntEnum=IntEnum, auto=auto)
exec(compile(ast.Module(body=[mode_node], type_ignores=[]), "ForwardMode", "exec"), ns)
Mode = ns["ForwardMode"]


def load_core(source=None):
    path = SRT / "layers/moe/fused_moe_triton/layer.py"
    tree = ast.parse(source if source is not None else path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedMoE")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run_moe_core")
    ns = dict(torch=torch, is_decode_or_verify_batch=scope.is_decode_or_verify_batch,
              get_is_capture_mode=lambda: False,
              Mxfp4FlashinferTrtllmMoEMethod=type(None),
              Mxfp4FlashinferCutlassMoEMethod=type(None), Mxfp4MarlinMoEMethod=type(None))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns


def run_core(core, *, resident=False, gpu_prefill=False, capture=False):
    seen = []

    def entry(name):
        def invoke(*args, **kwargs):
            seen.append((name, scope.get_verify_rows_per_request(96)))
            return torch.zeros(96, 8)
        return invoke

    layer = SimpleNamespace(is_gpu_resident_layer=resident,
                            should_prefill_gpu=lambda _: gpu_prefill,
                            quant_method=SimpleNamespace(apply=entry("resident")),
                            _cpu_decode=entry("decode"), _cpu_prefill=entry("cpu_prefill"),
                            _gpu_prefill=entry("gpu_prefill"),
                            should_fuse_routed_scaling_factor_in_topk=True)
    output = SimpleNamespace(hidden_states=torch.zeros(96, 8),
                             topk_output=SimpleNamespace(topk_weights=None, topk_ids=None))
    stub = ModuleType("sglang.srt.layers.moe.token_dispatcher")
    stub.StandardCombineInput = lambda **kw: kw
    with patch.dict("sys.modules", {stub.__name__: stub}), patch.object(
        torch.cuda, "is_current_stream_capturing", return_value=capture
    ):
        core["run_moe_core"](layer, output)
    return seen


@pytest.mark.parametrize("mode", [Mode.DECODE, Mode.TARGET_VERIFY, Mode.IDLE,
                                  Mode.EXTEND, Mode.MIXED, Mode.DRAFT_EXTEND_V2,
                                  Mode.SPLIT_PREFILL, Mode.DLLM_EXTEND])
@pytest.mark.parametrize("gpu_prefill", [False, True])
@pytest.mark.parametrize("resident", [False, True])
def test_eager_workload_selects_correct_abi(mode, gpu_prefill, resident):
    batch = SimpleNamespace(forward_mode=mode, spec_info=SimpleNamespace(draft_token_num=6))
    if resident:
        expected = "resident"
    elif mode in (Mode.DECODE, Mode.TARGET_VERIFY, Mode.IDLE):
        expected = "decode"
    else:
        expected = "gpu_prefill" if gpu_prefill else "cpu_prefill"
    with scope.set_forward_batch(batch):
        assert run_core(load_core(), resident=resident, gpu_prefill=gpu_prefill) == [
            (expected, 6 if mode == Mode.TARGET_VERIFY else 1)
        ]


@pytest.mark.parametrize("flag", ["cuda", "logical", "neither"])
def test_no_batch_preserves_capture_fallback(flag):
    core = load_core()
    core["get_is_capture_mode"] = lambda: flag == "logical"
    assert run_core(core, capture=flag == "cuda") == [
        ("cpu_prefill" if flag == "neither" else "decode", 1)
    ]


@pytest.mark.parametrize("mode", [Mode.DECODE, Mode.TARGET_VERIFY])
@pytest.mark.parametrize("raise_in_model", [False, True])
def test_actual_dummy_run_execution_scope_publishes_and_restores_batch(mode, raise_in_model):
    # Execute the production _dummy_run's final execution scope. The complete
    # CPU-backed builder is covered in test_dummy_forward_mode.py; this test
    # connects the scope directly to the production MoE selector.
    path = SRT / "model_executor/runner/base_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseRunner")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_dummy_run")
    execution_scope = fn.body[-1]
    assert isinstance(execution_scope, ast.With)
    outer = SimpleNamespace(forward_mode=Mode.EXTEND, spec_info=None)
    batch = SimpleNamespace(forward_mode=mode, spec_info=SimpleNamespace(draft_token_num=6))
    called = []

    def run_once():
        assert scope.get_forward_batch() is batch
        called.extend(run_core(load_core()))
        if raise_in_model:
            raise ValueError("model failure")

    ns = dict(forward_context=lambda _: nullcontext(), ForwardContext=SimpleNamespace,
              mr=SimpleNamespace(attn_backend=None), forward_batch=batch,
              set_forward_batch=scope.set_forward_batch, run_once=run_once,
              run_ctx=None, empty_context=nullcontext)
    with scope.set_forward_batch(outer):
        with pytest.raises(ValueError, match="model failure") if raise_in_model else nullcontext():
            exec(compile(ast.Module(body=[execution_scope], type_ignores=[]), str(path), "exec"), ns)
        assert scope.get_forward_batch() is outer
    assert called == [("decode", 6 if mode == Mode.TARGET_VERIFY else 1)]
    with pytest.raises(RuntimeError, match="No forward batch"):
        scope.get_forward_batch()
