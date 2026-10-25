#!/usr/bin/env python3
"""A/B 同一形状下 "旧 FP32 kernel" vs "新 bf16 张量核 kernel" 的单层耗时。

两个版本都用 CUDA graph 计时 (与 debug-snapshot/scripts/bench_gpu_cache_kernel.py 同法),
形状 = DSv4.1 Flash EP2 单卡 (E=192 本地专家, H=5120, I=1152, top_k=6, 104 槽在显存)。
典型 decode/verify 形状: bs 个请求 x (block5+1) 行 = rows; 单流 = 6 行。
用法:
  python bench_kernel_ab.py --old <旧 kernel .py> --new <当前 kernel .py> --rows 6 12 24 48 --hit 1.0 0.9 0.0
"""
import argparse
import importlib.machinery
import importlib.util
import statistics
import torch

E, H, I, K, SLOTS = 192, 5120, 1152, 6, 104


def load(path, name):
    # Non-.py names (e.g. *.site backups) need an explicit source loader.
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _experts(seed):
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=g)
    s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, generator=g)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g)
    s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, generator=g)
    return w13, s13, w2, s2


def _table(uva, host_tensors, vram_tensors):
    ptrs = []
    for h, v in zip(host_tensors, vram_tensors):
        step = h[0].numel()
        ptrs.append([v.data_ptr() + e * step if e < SLOTS else h.data_ptr() + e * step
                     for e in range(E)])
    return uva.HostExperts.from_addresses(
        w13=ptrs[0], s13=ptrs[1], w2=ptrs[2], s2=ptrs[3],
        hidden=H, intermediate=I, device="cuda")


def _routes(rows, hit, gen):
    ids = torch.empty((rows, K), dtype=torch.int32)
    for r in range(rows):
        n_hit = int(round(hit * K))
        hot = torch.randperm(SLOTS, generator=gen)[:n_hit]
        cold = SLOTS + torch.randperm(E - SLOTS, generator=gen)[: K - n_hit]
        ids[r] = torch.cat([hot, cold])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="旧 (FP32) kernel 文件路径")
    ap.add_argument("--new", required=True, help="新 (张量核) kernel 文件路径")
    ap.add_argument("--rows", type=int, nargs="+", default=[6, 12, 24, 48])
    ap.add_argument("--hit", type=float, nargs="+", default=[1.0, 0.9, 0.0])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    mods = [("old_fp32", load(args.old, "uva_old")),
            ("new_tc", load(args.new, "uva_new"))]
    host_tensors = [t.pin_memory() for t in _experts(0)]
    vram = [t[:SLOTS].cuda() for t in host_tensors]
    gen = torch.Generator().manual_seed(1)
    print(f"{'kernel':>9} {'rows':>5} {'hit':>5} {'distinct_e':>10} {'ms/layer':>9} "
          f"{'x20 ms':>8} {'med3':>8}")
    for name, uva in mods:
        for rows in args.rows:
            host = _table(uva, host_tensors, vram)
            scratch = uva.UvaScratch.allocate(num_experts=E, max_slots=rows * K,
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
                ids = _routes(rows, hit, gen)
                ms = [_time_graph(lambda: run(ids), args.iters) for _ in range(3)]
                cand = statistics.median(ms)
                print(f"{name:>9} {rows:>5} {hit:>5.2f} {ids.unique().numel():>10} "
                      f"{cand:>9.3f} {20 * cand:>8.2f} {min(ms):>8.3f}", flush=True)


if __name__ == "__main__":
    main()
