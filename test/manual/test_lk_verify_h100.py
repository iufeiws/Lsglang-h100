"""Run on CUDA, or on CPU with TRITON_INTERPRET=1 python -m pytest <this file>.

Loads the kernel directly so the CPU oracle does not require the serving stack
or the native lk_moe extension. The same tests execute on a real GPU unchanged.
"""

import importlib.util
import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kernel = load_source("lk_verify", "python/sglang/kernels/ops/moe/lk_verify.py")
batch_scope = load_source(
    "batch_scope", "python/sglang/srt/model_executor/runner_utils/forward_batch.py"
)
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


@pytest.fixture(autouse=True)
def interpreter_bf16_dot(monkeypatch):
    if DEVICE == "cpu":
        # Triton 3.8 stores BF16 as uint16 but create_dot passes those bit
        # patterns directly to np.matmul. Decode only the dot operands in the
        # CPU harness; its FP32->BF16 cast also truncates instead of rounding
        # to nearest even. Use PyTorch's BF16 conversion for that operation.
        # Production kernels and BF16 storage remain unchanged.
        import triton.language as tl
        from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

        original = InterpreterBuilder.create_dot
        original_cast = InterpreterBuilder.cast_impl

        def cast(builder, src, dst):
            if src.dtype.scalar == tl.float32 and dst.scalar == tl.bfloat16:
                bits = torch.from_numpy(src.data.copy()).bfloat16().view(torch.uint16)
                return TensorHandle(bits.numpy(), tl.bfloat16)
            return original_cast(builder, src, dst)

        def dot(builder, a, b, d, input_precision, max_num_imprecise_acc):
            if a.dtype == tl.bfloat16:
                a = builder.cast_impl(a, tl.float32)
            if b.dtype == tl.bfloat16:
                b = builder.cast_impl(b, tl.float32)
            return original(builder, a, b, d, input_precision, max_num_imprecise_acc)

        monkeypatch.setattr(InterpreterBuilder, "create_dot", dot)
        monkeypatch.setattr(InterpreterBuilder, "cast_impl", cast)


def packed_weights(experts, n, k, seed):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (experts, n, k), dtype=torch.uint8, generator=gen)
    scales = torch.randint(
        121, 126, (experts, n, k // 32), dtype=torch.uint8, generator=gen
    )
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    dense = lut[codes.long()] * torch.pow(2.0, scales.float() - 127).repeat_interleave(
        32, -1
    )
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    packed = packed.view(experts, n // 32, 32, k // 2).transpose(2, 3).contiguous()
    scales = scales.view(experts, n // 32, 32, k // 32).transpose(2, 3).contiguous()
    return packed.to(DEVICE), scales.to(DEVICE), dense


@pytest.mark.parametrize(
    "rows,topk,h,i,splits,limit,misses",
    [
        (6, 2, 64, 32, 1, 0.0, False),
        (6, 3, 96, 96, 4, 1.5, True),
        (18, 2, 64, 32, 2, 0.0, True),
        (1, 2, 64, 32, 2, 0.0, True),
        (6, 2, 64, 32, 1, 0.0, "all"),
        # V4.1 TP4 intermediate dimension: not a power of two.
        (6, 2, 64, 576, 2, 0.0, True),
        (6, 6, 96, 576, 16, 0.0, True),
    ],
)
def test_grouped_verify_matches_w4a16_reference(rows, topk, h, i, splits, limit, misses):
    experts = max(4, topk + 1)
    torch.manual_seed(73)
    x_cpu = torch.randn(rows, h).bfloat16().float()
    x = x_cpu.to(device=DEVICE, dtype=torch.bfloat16)
    slots_cpu = (torch.arange(rows * topk).view(rows, topk) % experts).int()
    if misses:
        slots_cpu.view(-1)[::3] = -1
    if misses == "all":
        slots_cpu.fill_(-1)
    weights_cpu = torch.rand(rows, topk)
    weights_cpu /= weights_cpu.sum(-1, keepdim=True)
    w, s, dense = [], [], []
    for idx, (n, k) in enumerate(((i, h), (i, h), (h, i))):
        ww, ss, dd = packed_weights(experts, n, k, idx + 20)
        w.append(ww)
        s.append(ss)
        dense.append(dd)
    geo = dict(
        I=i,
        k=topk,
        NGK=splits,
        weight_addr=w,
        scale_addr=s,
        slot_bytes=[t[0].numel() for t in w],
        slot_scale_bytes=[t[0].numel() for t in s],
        swiglu_limit=limit,
    )
    routes = rows * topk
    partial = torch.full((splits, routes, 2 * i), float("nan"), device=DEVICE)
    # Exercise the compact BF16 view of the shared GEMV workspace.
    hbuf = torch.full((routes, 2 * i), float("nan"), device=DEVICE)
    activated = hbuf.view(torch.bfloat16)
    parts = torch.full((routes, h), float("nan"), device=DEVICE)
    out = torch.full((rows, h), float("nan"), device=DEVICE)
    kernel.launch(
        x,
        slots_cpu.to(DEVICE),
        weights_cpu.to(DEVICE),
        geo,
        partial,
        activated,
        parts,
        out,
    )
    expected = torch.zeros(rows, h)
    for row in range(rows):
        for j in range(topk):
            slot = int(slots_cpu[row, j])
            if slot < 0:
                continue
            gate = x_cpu[row] @ dense[0][slot].T
            up = x_cpu[row] @ dense[1][slot].T
            if limit:
                gate = gate.clamp(-limit, limit)
                up = up.clamp(-limit, limit)
            act = (up * torch.nn.functional.silu(gate)).bfloat16().float()
            expected[row] += (act @ dense[2][slot].T) * weights_cpu[row, j]
    torch.testing.assert_close(out.cpu(), expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize(
    "mode,rows,ragged,expected",
    [
        ("decode", 6, False, 1),
        ("decode", 12, False, 1),
        ("draft", 30, False, 1),
        ("verify", 6, False, 6),
        ("verify", 12, False, 6),
        ("verify", 12, True, 1),
        ("verify", 7, False, 1),
    ],
)
def test_only_target_verify_has_fixed_request_stride(mode, rows, ragged, expected):
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: mode == "verify"),
        spec_info=SimpleNamespace(
            draft_token_num=6, ragged_verify_layout=object() if ragged else None
        ),
    )
    assert batch_scope.get_verify_rows_per_request(rows) == 1
    with batch_scope.set_forward_batch(batch):
        assert batch_scope.get_verify_rows_per_request(rows) == expected
    assert batch_scope.get_verify_rows_per_request(rows) == 1


@pytest.mark.parametrize("uniform_fp8,expected_heads", [(False, 64), (True, 16)])
def test_v41_draft_attention_padding_and_sink(uniform_fp8, expected_heads):
    # Execute the production forward with small attention/projection stand-ins.
    # Extracting its AST avoids importing the entire GPU-only serving stack.
    path = ROOT / "python/sglang/srt/models/deepseek_v4_dspark.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DSparkAttention"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
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
    ast.fix_missing_locations(module)
    seen = {}

    def attend(**kwargs):
        seen["q"] = kwargs["q"].float().clone()
        seen["sink"] = kwargs["attn_sink"]
        return kwargs["q"].float().clone()

    def sink(heads=64):
        return torch.zeros(heads)

    def compute_q(x, positions, q_out):
        q_out.fill_(1.0)
        return q_out

    backend = SimpleNamespace(forward=attend)
    ctx = ModuleType("sglang.srt.model_executor.forward_context")
    ctx.get_attn_backend = lambda: backend
    namespace = dict(
        torch=torch,
        _is_cuda=False,
        _is_npu=False,
        _PAD_NUM_HEADS=64,
        _resolve_dspark_pool=lambda: SimpleNamespace(uniform_fp8=uniform_fp8),
        fused_rope_inplace=lambda *a, **k: None,
        _apply_wo_a_bf16_matmul=lambda o, w, **kw: torch.einsum("tgd,grd->tgr", o, w),
    )
    exec(compile(module, str(path), "exec"), namespace)
    layer = SimpleNamespace(
        is_dsv41=True,
        q_head_norm=False,
        alt_streams=None,
        n_local_heads=16,
        head_dim=8,
        rope_head_dim=2,
        n_local_groups=1,
        o_lora_rank=2,
        _use_fast_kernel=True,
        attn=object(),
        freqs_cis=torch.empty(0),
        attn_tp_size=4,
        _compute_q=compute_q,
        _local_attn_sink=sink,
        kv_proj_only=lambda x: x,
        _store_block_kv=lambda **kw: None,
        wo_a=SimpleNamespace(weight=torch.ones(1, 2, 16 * 8)),
        wo_b=lambda v, **kw: (v, None),
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_idle=lambda: False,
            is_decode=lambda: True,
            is_target_verify=lambda: False,
        )
    )
    original_empty = torch.Tensor.new_empty

    def poisoned_empty(t, *args, **kwargs):
        return original_empty(t, *args, **kwargs).fill_(float("nan"))

    with (
        patch.dict(sys.modules, {ctx.__name__: ctx}),
        patch.object(torch.Tensor, "new_empty", poisoned_empty),
    ):
        namespace["forward"](layer, torch.arange(5), torch.zeros(5, 8), batch)
    assert seen["q"].shape == (5, expected_heads, 8)
    assert seen["sink"].shape == (expected_heads,)
    assert torch.isfinite(seen["q"]).all()
    torch.testing.assert_close(seen["q"][:, :16], torch.ones(5, 16, 8))
    if expected_heads > 16:
        assert torch.count_nonzero(seen["q"][:, 16:]) == 0


def compile_sm90():
    import triton
    from triton.compiler.compiler import ASTSource
    from triton.backends.compiler import GPUTarget

    cases = [
        (
            kernel.gate_up,
            dict(
                X="*bf16",
                Slots="*i32",
                GW="i64",
                GS="i64",
                UW="i64",
                US="i64",
                Partial="*fp32",
            ),
            dict(
                N_ROWS=6,
                TOPK=6,
                H=5120,
                I=576,
                W_BYTES=576 * 5120 // 2,
                S_BYTES=576 * 5120 // 32,
                SPLIT_K=16,
                BM=16,
                BN=32,
                BK=64,
                BK_ROUTES=8,
            ),
        ),
        (
            kernel.activate,
            dict(Partial="*fp32", Slots="*i32", Activated="*bf16"),
            dict(N_ROUTES=36, I=576, SPLIT_K=16, LIMIT=0.0, BLOCK=256),
        ),
        (
            kernel.down,
            dict(
                Activated="*bf16",
                Slots="*i32",
                Weights="*fp32",
                DW="i64",
                DS="i64",
                Out="*fp32",
            ),
            dict(
                N_ROWS=6,
                TOPK=6,
                H=5120,
                I=576,
                W_BYTES=5120 * 576 // 2,
                S_BYTES=5120 * 576 // 32,
                BM=16,
                BN=64,
                BK=64,
                BK_ROUTES=8,
            ),
        ),
        (
            kernel.combine,
            dict(Parts="*fp32", Slots="*i32", Out="*fp32"),
            dict(TOPK=6, H=5120, BLOCK=256),
        ),
    ]
    for fn, signature, constants in cases:
        compiled = triton.compile(
            ASTSource(fn, signature, constants), target=GPUTarget("cuda", 90, 32)
        )
        ptx = compiled.asm["ptx"]
        if fn in (kernel.gate_up, kernel.down):
            assert "mma.sync" in ptx and ".f32.bf16.bf16.f32" in ptx
            assert ".tf32" not in ptx
        print(
            f"{fn.__name__}: SM90 compiled; shared={compiled.metadata.shared}; mma={ptx.count('mma.sync')}"
        )


if __name__ == "__main__" and sys.argv[1:] == ["--compile-sm90"]:
    compile_sm90()
