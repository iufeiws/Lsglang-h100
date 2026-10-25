#!/usr/bin/env python3
"""单层 GPU 专家缓存 forward 的耗时 (H100, 不起服务)。

DSv4.1 Flash EP2 单卡形状: E=192 本地专家, H=5120, I=1152, top_k=6, 104 槽。
命中路由走 FlashInfer CUTLASS W4A16 (与常驻层同一个 kernel, 读显存槽), miss 走 UVA kernel
(侧流, 读主机)。每行 --local 条路由落本 rank (EP2 实测约 3), 其余为 -1。
L2: 同一输入反复计时; 每次读的权重 >= 6 行 x 3 专家 x 9 MiB = 160 MiB, 远大于 H100 的 50 MiB L2,
主机 (UVA) 读取不经过显存 L2, 所以热 L2 不影响结论。
graph: CUDA graph 内一层的耗时, ×20 ~= 一步里卸载层 MoE 的耗时; eager: prefill tail 每层耗时。
用法:
  python bench_gpu_cache_kernel.py --rows 6 12 24 48 96 --hit 1.0 0.98 0.9
"""
import argparse

import torch

from sglang.srt.layers.moe.gpu_expert_cache import (
    ForwardBuffers,
    LayerCache,
    MoeParams,
    cache_forward,
)

E, H, I, K, SLOTS = 192, 5120, 1152, 6, 104


def _experts():
    g = torch.Generator().manual_seed(0)
    return (
        torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=g),
        torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, generator=g),
        torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g),
        torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, generator=g),
    )


def _routes(rows, local, hit, gen):
    """``local`` distinct local experts per row; a share ``hit`` of them in VRAM slots."""
    ids = torch.full((rows, K), -1, dtype=torch.int32)
    for r in range(rows):
        n_hit = sum(int(torch.rand(1, generator=gen) < hit) for _ in range(local))
        hot = torch.randperm(SLOTS, generator=gen)[:n_hit]
        cold = SLOTS + torch.randperm(E - SLOTS, generator=gen)[: local - n_hit]
        ids[r, :local] = torch.cat([hot, cold]).int()
    return ids.cuda()


def _time(fn, iters, graph):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        fn()
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            fn()
        fn = g.replay
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[6, 12, 24, 48, 96])
    ap.add_argument("--hit", type=float, nargs="+", default=[1.0, 0.98, 0.9])
    ap.add_argument("--local", type=int, default=3, help="每行落本 rank 的路由数 (EP2 约 3)")
    ap.add_argument("--eager-rows", type=int, nargs="*", default=[128, 512, 2048])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    cache = LayerCache(layer_id=0, host_tensors=_experts(), num_slots=SLOTS, device="cuda")
    max_rows = max(args.rows + args.eager_rows)
    buffers = ForwardBuffers(num_experts=E, max_rows=max_rows, top_k=K, hidden=H,
                             intermediate=I, device="cuda")
    params = MoeParams(swiglu_limit=torch.full((SLOTS,), 10.0, device="cuda"), limit=10.0,
                       tp_size=1, tp_rank=0)
    gen = torch.Generator().manual_seed(1)

    def bench(rows, hit, graph):
        x = torch.randn(rows, H, device="cuda").bfloat16()
        w = torch.rand(rows, K, device="cuda")
        ids = _routes(rows, args.local, hit, gen)
        run = lambda: cache_forward(cache, hidden_states=x, topk_ids=ids, topk_weights=w,  # noqa: E731
                                    buffers=buffers, params=params, graph=graph)
        iters = args.iters if graph else max(3, args.iters // 5)
        return _time(run, iters, graph)

    print(f"{'mode':>6} {'rows':>5} {'hit':>5} {'ms/layer':>9} {'x20 ms':>7}")
    for rows in args.rows:
        for hit in args.hit:
            ms = bench(rows, hit, graph=True)
            print(f"{'graph':>6} {rows:>5} {hit:>5.2f} {ms:>9.3f} {20 * ms:>7.2f}")
    for rows in args.eager_rows:
        for hit in args.hit:
            ms = bench(rows, hit, graph=False)
            print(f"{'eager':>6} {rows:>5} {hit:>5.2f} {ms:>9.3f} {20 * ms:>7.2f}")


if __name__ == "__main__":
    main()
