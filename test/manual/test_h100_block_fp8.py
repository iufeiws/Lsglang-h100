"""CPU interpreter / CUDA numerics, SM90 compile, and H100 graph benchmarks.

No serving-stack imports: execute the production functions via AST. On CUDA
the operands are real E4M3, and --bench measures the complete split/reduce path.
"""

import ast
import functools
import importlib.util
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "python/sglang/kernels/ops/gemm/fp8_kernel.py"
spec = importlib.util.spec_from_file_location(
    "h100_config", SOURCE.with_name("h100_fp8_config.py")
)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
NAMES = {
    "_w8a8_block_fp8_matmul", "_w8a8_block_fp8_matmul_hopper",
    "_reduce_block_fp8_split_k", "prepare_block_fp8_matmul_inputs",
    "select_w8a8_block_fp8_matmul_kernel", "get_w8a8_block_fp8_configs",
    "w8a8_block_fp8_matmul_triton",
}
ns = dict(
    torch=torch, triton=triton, tl=tl, functools=functools, os=os, json=json,
    __file__=str(SOURCE), _is_cuda=True, _is_gfx1250=False,
    get_platform=lambda: SimpleNamespace(is_sm90=True),
    get_device_name=lambda: "NVIDIA H100 80GB HBM3",
    logger=logging.getLogger(__name__), log_info_on_rank0=lambda *args: None,
    get_h100_block_fp8_config=profile.get_h100_block_fp8_config,
)
nodes = [n for n in ast.parse(SOURCE.read_text()).body
         if isinstance(n, ast.FunctionDef) and n.name in NAMES]
module = ast.Module(body=[ast.ImportFrom(
    module="__future__", names=[ast.alias(name="annotations")], level=0
)] + nodes, type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def inputs(m, n, k, group=32, device=DEVICE):
    gen = torch.Generator(device=device).manual_seed(119)
    a = torch.randn(m, k, generator=gen, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(n, k, generator=gen, device=device).to(torch.float8_e4m3fn)
    sa = torch.exp2(torch.randint(-5, 1, (m, triton.cdiv(k, group)), generator=gen, device=device).float())
    sb = torch.exp2(torch.randint(-5, 1, (triton.cdiv(n, group), triton.cdiv(k, group)), generator=gen, device=device).float())
    return a, b, sa, sb


def reference(a, b, sa, sb, group):
    k = a.shape[-1]
    aa = a.float() * sa.repeat_interleave(group, -1)[:, :k]
    bb = b.float() * sb.repeat_interleave(group, 0).repeat_interleave(group, 1)[:b.shape[0], :k]
    return aa @ bb.T


@pytest.mark.parametrize("m,n,k,group,splits,swap", [
    (1, 64, 32, 32, 8, True),       # empty K partitions must write zeros
    (6, 70, 288, 32, 4, True),     # tail N and uneven split
    (17, 96, 100, 32, 2, True),    # tail M/K
    (6, 70, 288, 128, 4, True),    # split begins inside a quantization block
    (17, 70, 128, 32, 1, True),
    (6, 70, 288, 32, 4, False),
])
def test_hopper_numerics(m, n, k, group, splits, swap):
    a, b, sa, sb = inputs(m, n, k, group)
    # Non-contiguous scales must be addressed with their supplied strides.
    sa, sb = sa.T.contiguous().T, sb.T.contiguous().T
    parts = torch.full((splits, m, n), float("nan"), device=DEVICE)
    out = torch.full((m, n), float("nan"), device=DEVICE)
    ns["_w8a8_block_fp8_matmul_hopper"][(triton.cdiv(m, 16) * triton.cdiv(n, 64), splits)](
        a, b, parts, sa, sb, m, n, k, group, group,
        a.stride(0), 1, 1, b.stride(0), n, 1,
        sa.stride(0), sa.stride(1), sb.stride(1), sb.stride(0),
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8, needs_masking=k % 32 != 0,
        SWAP_AB=swap, SPLIT_K=splits,
    )
    ns["_reduce_block_fp8_split_k"][(triton.cdiv(m * n, 256),)](
        parts, out, m * n, splits, 256
    )
    assert torch.isfinite(parts).all() and torch.isfinite(out).all()
    torch.testing.assert_close(out, reference(a, b, sa, sb, group), rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("m", [1, 6, 16, 17, 32, 33, 64, 65, 96])
def test_profile_shapes_and_bounds(m):
    for shape in profile._profiles()["shapes"]:
        n, k = map(int, shape.split(","))
        cfg = profile.get_h100_block_fp8_config(m, n, k, 32, 32, "NVIDIA H100 PCIe")
        assert cfg["BLOCK_SIZE_M"] <= 32 and cfg["BLOCK_SIZE_K"] == 32
        assert cfg["SWAP_AB"] and cfg["SPLIT_K"] <= 16
        assert cfg["SPLIT_K"] & (cfg["SPLIT_K"] - 1) == 0
    assert profile.get_h100_block_fp8_config(97, 576, 5120, 32, 32, "NVIDIA H100") is None
    assert profile.get_h100_block_fp8_config(6, 576, 5120, 128, 128, "NVIDIA H100") is None
    assert profile.get_h100_block_fp8_config(6, 576, 5120, 32, 32, "NVIDIA H200") is None
    assert profile.get_h100_block_fp8_config(6, 777, 5120, 32, 32, "NVIDIA H100") is None


@pytest.mark.parametrize("m,exact,compiling,expected_split", [
    (6, False, False, 16), (96, False, False, 2),
    (97, False, False, None), (6, True, False, None),
    (6, False, True, None),
])
def test_production_dispatch_and_exact_table_precedence(m, exact, compiling, expected_split):
    calls = []

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kw):
                calls.append(kw)
            return launch

    default = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
                   GROUP_SIZE_M=32, num_warps=4, num_stages=3)
    a, b, sa, sb = inputs(m, 576, 5120, device="cpu")
    with patch.dict(ns, {
        "get_w8a8_block_fp8_configs": lambda *args: {6: default} if exact else None,
        "_w8a8_block_fp8_matmul_hopper": Capture(),
        "_w8a8_block_fp8_matmul": Capture(),
        "_reduce_block_fp8_split_k": Capture(),
    }), patch.object(torch._dynamo, "is_compiling", return_value=compiling):
        ns["w8a8_block_fp8_matmul_triton"](a, b, sa, sb, [32, 32], torch.float32)
    assert calls[0].get("SPLIT_K") == expected_split


@pytest.mark.parametrize("n,k", [(576, 5120), (5120, 288)])
def test_full_profile_launch_matches_dequantized_reference(n, k):
    a, b, sa, sb = inputs(6, n, k)
    out = ns["w8a8_block_fp8_matmul_triton"](a, b, sa, sb, [32, 32], torch.float32)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, reference(a, b, sa, sb, 32), rtol=3e-5, atol=3e-5)


def compile_sm90():
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    fn = ns["_w8a8_block_fp8_matmul_hopper"]
    count = 0
    for shape in profile._profiles()["shapes"]:
        n, k = map(int, shape.split(","))
        for m in (6, 32, 64, 96):
            cfg = profile.get_h100_block_fp8_config(m, n, k, 32, 32, "NVIDIA H100")
            options = {key: cfg[key] for key in ("num_warps", "num_stages")}
            constants = {key: val for key, val in cfg.items() if key not in options}
            constants.update(M=m, N=n, K=k, group_n=32, group_k=32,
                             stride_am=k, stride_ak=1, stride_bk=1, stride_bn=k,
                             stride_cm=n, stride_cn=1, stride_As_m=triton.cdiv(k,32),
                             stride_As_k=1, stride_Bs_k=1, stride_Bs_n=triton.cdiv(k,32),
                             needs_masking=False)
            compiled = triton.compile(ASTSource(fn, dict(A="*fp8e4nv", B="*fp8e4nv", C="*fp32", As="*fp32", Bs="*fp32"), constants), target=GPUTarget("cuda",90,32), options=options)
            assert "e4m3" in compiled.asm["ptx"]
            count += 1
    print(f"SM90 FP8: {count} shape/bucket combinations compiled")


def benchmark(output):
    assert torch.cuda.get_device_capability()[0] == 9
    assert "H100" in torch.cuda.get_device_name()
    import triton.testing

    records = []
    for shape in profile._profiles()["shapes"]:
        n, k = map(int, shape.split(","))
        for m in (1, 6, 16, 96):
            a, b, sa, sb = inputs(m, n, k, device="cuda")
            call = lambda: ns["w8a8_block_fp8_matmul_triton"](a, b, sa, sb, [32,32], torch.bfloat16)
            baseline = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
                            GROUP_SIZE_M=32, num_warps=4, num_stages=3)
            with patch.dict(ns, {"get_w8a8_block_fp8_configs": lambda *args: {m: baseline}}):
                old = call()
                old_ms = triton.testing.do_bench_cudagraph(call)
            new = call()
            ref = reference(a, b, sa, sb, 32)
            # Relative RMS error tolerates BF16 rounding near cancellation;
            # also bound max error relative to the reference's global scale.
            for result in (old, new):
                assert torch.isfinite(result).all()
                err = (result.float() - ref).abs()
                assert err.square().mean().sqrt() <= 0.006 * ref.square().mean().sqrt() + 1e-5
                assert err.max() <= 0.01 * ref.abs().max() + 1e-5
            new_ms = triton.testing.do_bench_cudagraph(call)
            row = dict(M=m,N=n,K=k,baseline_us=old_ms*1000,profile_us=new_ms*1000)
            records.append(row)
            print(json.dumps(row), flush=True)
    Path(output).write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-sm90", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--output", default="h100_block_fp8_bench.json")
    args = parser.parse_args()
    if args.compile_sm90:
        compile_sm90()
    if args.bench:
        benchmark(args.output)
