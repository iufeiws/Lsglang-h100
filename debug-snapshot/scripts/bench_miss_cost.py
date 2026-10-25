#!/usr/bin/env python3
"""单层 forward 里 "每条 miss 的代价" —— 控制 miss 专家数, rows=6 (单流 verify 形状)。

服务端单流每层每步约 16-18 条本地路由, 命中率 95-99% ⇒ 每层约 0.3-1 个 miss。
用法: python bench_miss_cost.py [--rows 6] [--iters 50]
"""
import argparse
import statistics
import torch

from sglang.srt.layers.moe.gpu_expert_cache import (
    ForwardBuffers, LayerCache, MoeParams, cache_forward,
)

E, H, I, K, SLOTS = 192, 5120, 1152, 6, 104


def _experts(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=g),
            torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, generator=g),
            torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g),
            torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, generator=g))


def _routes(rows, local, n_miss, gen):
    """前 ``n_miss`` 条路由指向不同的非驻留专家 (>=SLOTS), 其余命中槽内专家 (<SLOTS)。"""
    ids = torch.full((rows, K), -1, dtype=torch.int32)
    miss = SLOTS + torch.randperm(E - SLOTS, generator=gen)[:n_miss]
    hot = torch.randperm(SLOTS, generator=gen)[: rows * local - n_miss]
    flat = []
    for i in range(n_miss):
        flat.append(int(miss[i]))
    for i in range(rows * local - n_miss):
        flat.append(int(hot[i]))
    flat = torch.tensor(flat, dtype=torch.int32)
    ids[:, :local] = flat.view(rows, local)
    return ids.cuda()


def _time_graph(run, iters):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(); run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        run()
    run = g.replay
    run()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        run()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--local", type=int, default=3, help="每行本 rank 路由数 (EP2 约 3)")
    ap.add_argument("--misses", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    cache = LayerCache(layer_id=0, host_tensors=_experts(), num_slots=SLOTS, device="cuda")
    buffers = ForwardBuffers(num_experts=E, max_rows=args.rows, top_k=K, hidden=H,
                             intermediate=I, device="cuda")
    params = MoeParams(swiglu_limit=torch.full((SLOTS,), 10.0, device="cuda"), limit=10.0,
                       tp_size=1, tp_rank=0)
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(args.rows, H, device="cuda").bfloat16()
    w = torch.rand(args.rows, K, device="cuda")
    print(f"rows={args.rows} local={args.local} 槽={SLOTS} 冷启动槽内容=专家 0..{SLOTS-1}")
    print(f"{'miss_experts':>12} {'ms/layer':>9} {'Δ vs 0 miss':>12} {'ms per miss':>12}")
    base = None
    for n in args.misses:
        ids = _routes(args.rows, args.local, n, gen)
        ms = statistics.median([_time_graph(
            lambda: cache_forward(cache, hidden_states=x, topk_ids=ids, topk_weights=w,
                                  buffers=buffers, params=params, graph=True),
            args.iters) for _ in range(3)])
        if base is None:
            base = ms
        per = (ms - base) / n if n else float("nan")
        print(f"{n:>12} {ms:>9.3f} {ms - base:>12.3f} {per:>12.3f}", flush=True)


if __name__ == "__main__":
    main()
