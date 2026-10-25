#!/usr/bin/env python3
"""EP 真实路由下的单层耗时 (旧 FP32 vs 新 bf16 张量核)。

与本目录 bench_kernel_ab.py 相同, 但路由按 EP2 真实分布: 全局 384 专家, 每 rank 本地 192;
每个 token 的 top-6 全局路由里只有约一半落在本 rank, 其余为 -1 (EP-remote, 内核跳过)。
服务端实测口径: 单流 decode/verify 每层每步约 17.4 条本地路由 (6 行 x ~2.9 条本地)。
用法:
  python bench_kernel_ep.py --old <旧 kernel> --new <当前 kernel> \
      --rows 6 12 24 48 --hit 1.0 0.9 0.0 --eager-rows 128 256 512
"""
import argparse
import importlib.machinery
import importlib.util
import statistics
import torch

E_LOCAL, E_GLOBAL, H, I, K, SLOTS = 192, 384, 5120, 1152, 6, 104


def load(path, name):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _experts(seed):
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randint(0, 256, (E_LOCAL, 2 * I, H // 2), dtype=torch.uint8, generator=g)
    s13 = torch.randint(118, 124, (E_LOCAL, 2 * I, H // 32), dtype=torch.uint8, generator=g)
    w2 = torch.randint(0, 256, (E_LOCAL, H, I // 2), dtype=torch.uint8, generator=g)
    s2 = torch.randint(118, 124, (E_LOCAL, H, I // 32), dtype=torch.uint8, generator=g)
    return w13, s13, w2, s2


def _table(uva, host_tensors, vram_tensors):
    ptrs = []
    for h, v in zip(host_tensors, vram_tensors):
        step = h[0].numel()
        ptrs.append([v.data_ptr() + e * step if e < SLOTS else h.data_ptr() + e * step
                     for e in range(E_LOCAL)])
    return uva.HostExperts.from_addresses(
        w13=ptrs[0], s13=ptrs[1], w2=ptrs[2], s2=ptrs[3],
        hidden=H, intermediate=I, device="cuda")


def _routes_ep(rows, hit, gen):
    """top-6 global routes per row; the ones owned by the other EP rank become -1."""
    ids = torch.full((rows, K), -1, dtype=torch.int32)
    for r in range(rows):
        g = torch.randperm(E_GLOBAL, generator=gen)[:K]
        loc = g[g < E_LOCAL]                       # this rank's share (~half)
        n = loc.numel()
        n_hit = int(round(hit * n))
        hot = torch.randperm(SLOTS, generator=gen)[:n_hit]
        cold = SLOTS + torch.randperm(E_LOCAL - SLOTS, generator=gen)[: n - n_hit]
        ids[r, :n] = torch.cat([hot, cold])
    return ids.cuda()


def _time_graph(fn, iters):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    graph.replay()
    start.record()
    for _ in range(iters):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def _time_eager(fn, iters):
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=None)
    ap.add_argument("--new", required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[6, 12, 24, 48])
    ap.add_argument("--hit", type=float, nargs="+", default=[1.0, 0.9, 0.0])
    ap.add_argument("--eager-rows", type=int, nargs="*", default=[128, 256, 512])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    mods = [("new_tc", load(args.new, "uva_new"))]
    if args.old:
        mods.insert(0, ("old_fp32", load(args.old, "uva_old")))
    host_tensors = [t.pin_memory() for t in _experts(0)]
    vram = [t[:SLOTS].cuda() for t in host_tensors]
    gen = torch.Generator().manual_seed(1)
    print(f"{'kernel':>9} {'rows':>5} {'hit':>5} {'local_routes':>12} {'tiles(est)':>10} "
          f"{'ms/layer':>9} {'x20 ms':>8}")
    for name, uva in mods:
        for rows in args.rows:
            host = _table(uva, host_tensors, vram)
            scratch = uva.UvaScratch.allocate(num_experts=E_LOCAL, max_slots=rows * K,
                                              intermediate=I, device="cuda")
            x = torch.randn(rows, H, device="cuda").bfloat16()
            w = torch.rand(rows, K, device="cuda")
            out = torch.empty(rows * K, H, device="cuda")

            def run(ids):
                out.zero_()
                uva.launch_draft_miss_uva(
                    hidden_states=x, topk_ids=ids, topk_weights=w, slot_tab=None,
                    gout_part=out, rows_per_request=1, host=host, scratch=scratch,
                    activation=0, alpha=1.702, limit=10.0, include_first_row=True)

            for hit in args.hit:
                ids = _routes_ep(rows, hit, gen)
                n_loc = int((ids >= 0).sum())
                ms = [_time_graph(lambda: run(ids), args.iters) for _ in range(3)]
                cand = statistics.median(ms)
                print(f"{name:>9} {rows:>5} {hit:>5.2f} {n_loc:>12} "
                      f"{n_loc / 16:.0f}{'':>4} {cand:>9.3f} {20 * cand:>8.2f}", flush=True)
    if args.eager_rows:
        name, uva = mods[-1]
        print(f"--- eager (prefill tail, 主机规划, 仅 {name}) ---")
        print(f"{'kernel':>9} {'rows':>5} {'hit':>5} {'local_routes':>12} {'ms/layer':>9} {'x20 ms':>8}")
        for rows in args.eager_rows:
            host = _table(uva, host_tensors, vram)
            scratch = uva.UvaScratch.allocate(num_experts=E_LOCAL, max_slots=rows * K,
                                              intermediate=I, device="cuda")
            x = torch.randn(rows, H, device="cuda").bfloat16()
            w = torch.rand(rows, K, device="cuda")
            ids = _routes_ep(rows, 0.9, gen)
            n_loc = int((ids >= 0).sum())

            def eager():
                out = torch.zeros(rows * K, H, device="cuda")
                uva.launch_draft_miss_uva(
                    hidden_states=x, topk_ids=ids, topk_weights=w, slot_tab=None,
                    gout_part=out, rows_per_request=1, host=host, scratch=scratch,
                    activation=0, alpha=1.702, limit=10.0, include_first_row=True,
                    eager_plan=True)

            ms = _time_eager(eager, max(3, args.iters // 10))
            print(f"{name:>9} {rows:>5} {0.9:>5.2f} {n_loc:>12} {ms:>9.3f} {20 * ms:>8.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
