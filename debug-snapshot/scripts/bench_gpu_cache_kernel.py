#!/usr/bin/env python3
"""单层 GPU 专家缓存 forward 的耗时 (H100, 不起服务)。

DSv4.1 Flash EP2 单卡形状: E=192 本地专家, H=5120, I=1152, top_k=6, --rows-vram 行显存。
命中路由走 FlashInfer CUTLASS (与常驻层同一个 kernel); miss 专家 (每次最多 --max-copy 个) 整块拷进
本次未用到的低分行再一起算, 超出的才走 UVA kernel (侧流)。每行 --local 条路由落本 rank (EP2 约 3)。
每次 replay 前把映射表恢复成初始状态, 所以重复计时测的是 miss 成本, 不是第一次之后的命中。
--miss-experts: 另测 rows=6 时每层恰好 m 个 miss 专家的耗时 (单个 miss 的边际成本)。
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
    gpu_numa_node,
)

E, H, I, K = 192, 5120, 1152, 6


def _experts():
    g = torch.Generator().manual_seed(0)
    return (
        torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=g),
        torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, generator=g),
        torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g),
        torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, generator=g),
    )


def _routes(rows, local, hit, gen, slots):
    """``local`` distinct local experts per row; a share ``hit`` of them in VRAM slots."""
    ids = torch.full((rows, K), -1, dtype=torch.int32)
    for r in range(rows):
        n_hit = sum(int(torch.rand(1, generator=gen) < hit) for _ in range(local))
        hot = torch.randperm(slots, generator=gen)[:n_hit]
        cold = slots + torch.randperm(E - slots, generator=gen)[: local - n_hit]
        ids[r, :local] = torch.cat([hot, cold]).int()
    return ids.cuda()


def _routes_with_misses(rows, local, misses, gen, slots):
    """Every route hits except one route in each of the first ``misses`` rows."""
    ids = _routes(rows, local, 1.0, gen, slots).cpu()
    cold = slots + torch.randperm(E - slots, generator=gen)[:misses]
    for m in range(misses):
        ids[m % rows, m // rows] = cold[m]
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
    ap.add_argument("--rows-vram", type=int, default=108)
    ap.add_argument("--max-copy", type=int, default=8)
    ap.add_argument("--host-numa", default="auto", help="auto = GPU 所在节点, -1 = 进程策略, 或节点号")
    ap.add_argument("--precision", choices=["bf16", "fp8"], default="bf16",
                    help="bf16 = W4A16, fp8 = Humming W4A8 (与 LK_GPU_CACHE_PRECISION 相同)")
    ap.add_argument("--miss-experts", type=int, nargs="*", default=[0, 1, 2, 4, 6, 8])
    args = ap.parse_args()

    slots = args.rows_vram
    fp8 = args.precision == "fp8"
    node = gpu_numa_node("cuda") if args.host_numa == "auto" else int(args.host_numa)
    cache = LayerCache(layer_id=0, host_tensors=_experts(), num_rows=slots,
                       max_copy=args.max_copy, score_decay=0.99, fp8=fp8,
                       host_numa_node=node, device="cuda")
    print("row source NUMA node", node, "first pages on", cache.row_source_nodes)
    slot_of0, row_expert0 = cache.slot_of.clone(), cache.row_expert.clone()
    max_rows = max(args.rows + args.eager_rows)
    buffers = ForwardBuffers(num_experts=E, max_rows=max_rows, top_k=K, hidden=H,
                             intermediate=I, device="cuda")
    params = MoeParams(swiglu_limit=torch.full((slots,), 10.0, device="cuda"), limit=10.0,
                       tp_size=1, tp_rank=0,
                       fc2_act_scale=torch.ones((), device="cuda") if fp8 else None)
    gen = torch.Generator().manual_seed(1)

    def bench(rows, hit, graph, ids=None):
        x = torch.randn(rows, H, device="cuda").bfloat16()
        w = torch.rand(rows, K, device="cuda")
        if ids is None:
            ids = _routes(rows, args.local, hit, gen, slots)

        def run():
            cache.slot_of.copy_(slot_of0)
            cache.row_expert.copy_(row_expert0)
            return cache_forward(cache, hidden_states=x, topk_ids=ids, topk_weights=w,
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
    print(f"{'misses':>6} {'rows':>5} {'ms/layer':>9}   (graph, rows=6; > --max-copy 的部分走 UVA)")
    for m in args.miss_experts:
        ms = bench(6, 1.0, graph=True, ids=_routes_with_misses(6, args.local, m, gen, slots))
        print(f"{m:>6} {6:>5} {ms:>9.3f}")


if __name__ == "__main__":
    main()
