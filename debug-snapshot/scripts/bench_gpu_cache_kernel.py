#!/usr/bin/env python3
"""单层 GPU 专家缓存 kernel 的耗时与数值 (H100, 不起服务)。

DSv4.1 Flash EP2 单卡形状: E=192 本地专家, H=5120, I=1152, top_k=6; 104 个槽在显存, 其余经 UVA 读主机。
对每个 batch 行数 (verify = 请求数 x 6) 测 CUDA graph 内一层的耗时, 命中率 100% 与 --hit 两档,
并对比张量核路径与 FP32 路径的误差。20 层 x 本耗时 ~= 一步里卸载层 MoE 的耗时。
用法:
  python bench_gpu_cache_kernel.py --rows 6 12 24 48 --hit 0.9
  python bench_gpu_cache_kernel.py --gate-up 32:128:4 --down 64:64:8   # 换 tile 配置 (BN:BK:warps) 对比
"""
import argparse

import torch

import sglang.kernels.ops.moe.lk_verify_uva as uva

E, H, I, K, SLOTS = 192, 5120, 1152, 6, 104


def _experts(seed):
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=g)
    s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, generator=g)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g)
    s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, generator=g)
    return w13, s13, w2, s2


def _table(host_tensors, vram_tensors):
    """Experts < SLOTS point at VRAM, the rest at pinned host memory."""
    ptrs = []
    for h, v in zip(host_tensors, vram_tensors):
        step = h[0].numel()
        ptrs.append([v.data_ptr() + e * step if e < SLOTS else h.data_ptr() + e * step
                     for e in range(E)])
    return uva.HostExperts.from_addresses(
        w13=ptrs[0], s13=ptrs[1], w2=ptrs[2], s2=ptrs[3],
        hidden=H, intermediate=I, device="cuda")


def _routes(rows, hit, gen):
    """Distinct experts per row; a share ``hit`` of the routes lands on VRAM slots."""
    ids = torch.empty((rows, K), dtype=torch.int32)
    for r in range(rows):
        n_hit = int(round(hit * K))
        hot = torch.randperm(SLOTS, generator=gen)[:n_hit]
        cold = SLOTS + torch.randperm(E - SLOTS, generator=gen)[: K - n_hit]
        ids[r] = torch.cat([hot, cold])
    return ids.cuda()


def _run(host, scratch, x, ids, w, out):
    out.zero_()
    uva.launch_draft_miss_uva(
        hidden_states=x, topk_ids=ids, topk_weights=w, slot_tab=None, gout_part=out,
        rows_per_request=1, host=host, scratch=scratch, activation=0, alpha=1.702,
        limit=10.0, include_first_row=True)


def _time_eager(fn, iters):
    fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def _time_graph(fn, iters):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    graph.replay()
    start.record()
    for _ in range(iters):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[6, 12, 24, 48])
    ap.add_argument("--eager-rows", type=int, nargs="*", default=[256, 2048],
                    help="prefill tail 行数 (eager 路径, 主机规划)")
    ap.add_argument("--hit", type=float, default=0.9)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--gate-up", help="BN:BK:num_warps, 默认用 uva.GATE_UP_CONFIG")
    ap.add_argument("--down", help="BN:BK:num_warps, 默认用 uva.DOWN_CONFIG")
    args = ap.parse_args()
    if args.gate_up:
        uva.GATE_UP_CONFIG = uva.GemmConfig(*map(int, args.gate_up.split(":")))
    if args.down:
        uva.DOWN_CONFIG = uva.GemmConfig(*map(int, args.down.split(":")))
    print("gate_up", uva.GATE_UP_CONFIG, "down", uva.DOWN_CONFIG)

    host_tensors = [t.pin_memory() for t in _experts(0)]
    vram = [t[:SLOTS].cuda() for t in host_tensors]
    host = _table(host_tensors, vram)
    gen = torch.Generator().manual_seed(1)
    max_rows = max(args.rows)
    scratch = uva.UvaScratch.allocate(num_experts=E, max_slots=max_rows * K,
                                      intermediate=I, device="cuda")
    print(f"{'rows':>5} {'hit':>5} {'ms/layer':>9} {'x20 ms':>7} {'GB/s':>7}")
    for rows in args.rows:
        x = torch.randn(rows, H, device="cuda").bfloat16()
        w = torch.rand(rows, K, device="cuda")
        out = torch.empty(rows * K, H, device="cuda")
        for hit in (1.0, args.hit):
            ids = _routes(rows, hit, gen)
            ms = _time_graph(lambda: _run(host, scratch, x, ids, w, out), args.iters)
            n_exp = ids.unique().numel()
            gbs = n_exp * sum(t[0].numel() for t in host_tensors) / ms / 1e6
            print(f"{rows:>5} {hit:>5.2f} {ms:>9.3f} {20 * ms:>7.2f} {gbs:>7.0f}")
        _run(host, scratch, x, ids, w, out)
        tc = out.clone()
        uva._use_tensor_cores = lambda: False
        _run(host, scratch, x, ids, w, out)
        uva._use_tensor_cores = lambda: True
        torch.cuda.synchronize()
        rel = ((tc - out).norm() / out.norm()).item()
        print(f"      tensor-core vs fp32 relative error: {rel:.2e}")
    print(f"{'eager rows':>10} {'hit':>5} {'ms/layer':>9} {'x20 ms':>7}")
    for rows in args.eager_rows:
        x = torch.randn(rows, H, device="cuda").bfloat16()
        w = torch.rand(rows, K, device="cuda")
        ids = _routes(rows, args.hit, gen)

        def eager():
            out = torch.zeros(rows * K, H, device="cuda")
            uva.launch_draft_miss_uva(
                hidden_states=x, topk_ids=ids, topk_weights=w, slot_tab=None,
                gout_part=out, rows_per_request=1, host=host, scratch=scratch,
                activation=0, alpha=1.702, limit=10.0, include_first_row=True,
                eager_plan=True)

        ms = _time_eager(eager, max(3, args.iters // 10))
        print(f"{rows:>10} {args.hit:>5.2f} {ms:>9.3f} {20 * ms:>7.2f}")


if __name__ == "__main__":
    main()
