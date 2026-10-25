#!/usr/bin/env python3
"""主机 -> GPU 拷贝带宽: 单卡 / 两两同时 / 全部同时, 以及主机内存放在哪个 NUMA 节点的影响。

背景 (slots128 profile): 换入一个专家 (9.4 MB) 在服务里要 ~413 us (~23 GB/s), 单卡 bench 是
~200 us; 同 EP 组的两张卡 (TP0+TP1 / TP2+TP3) 总是同时拷同一批专家的两半, 两组之间互不影响。
如果是"同一个 PCIe switch 上的两张卡共享上行带宽", 把同 EP 组的两张卡换到不同 switch 下
(调整 CUDA_VISIBLE_DEVICES 的顺序) 就能让单组换入接近单卡速度。本脚本测出哪些卡互相抢带宽,
并给出建议的顺序。需在服务停止时运行 (占用所有可见 GPU 的拷贝引擎)。
用法:
  CUDA_VISIBLE_DEVICES=<与 serve 相同> python bench_h2d_topology.py
"""
import argparse
import itertools
import subprocess

import torch

from sglang.srt.layers.moe.gpu_expert_cache import _pinned_on_node, gpu_numa_node

MB = 1 << 20


def _bandwidth(gpus, sources, nbytes, iters):
    """GB/s per GPU while all ``gpus`` copy ``nbytes`` from ``sources[g]`` at once."""
    dst = {g: torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{g}") for g in gpus}
    streams = {g: torch.cuda.Stream(device=g) for g in gpus}
    events = {g: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for g in gpus}
    for g in gpus:
        torch.cuda.synchronize(g)
    for g in gpus:
        with torch.cuda.device(g), torch.cuda.stream(streams[g]):
            dst[g].copy_(sources[g][:nbytes], non_blocking=True)  # warm up
            events[g][0].record(streams[g])
            for _ in range(iters):
                dst[g].copy_(sources[g][:nbytes], non_blocking=True)
            events[g][1].record(streams[g])
    out = {}
    for g in gpus:
        torch.cuda.synchronize(g)
        ms = events[g][0].elapsed_time(events[g][1]) / iters
        out[g] = nbytes / ms / 1e6
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    n = torch.cuda.device_count()
    nbytes = args.mb * MB
    try:
        print(subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout)
    except OSError:
        pass
    nodes = {g: gpu_numa_node(g) for g in range(n)}
    print("GPU -> NUMA node:", nodes)
    all_nodes = sorted({v for v in nodes.values() if v >= 0}) or [-1]
    keep = []

    def source(g, node):
        with torch.cuda.device(g):
            t, alive, placed = _pinned_on_node(nbytes, node)
        keep.append(alive)
        t.fill_(1)
        return t, placed

    print(f"\n== 单卡, 源内存在各节点 (GB/s, {args.mb} MiB x {args.iters})")
    for g in range(n):
        row = []
        for node in all_nodes:
            src, placed = source(g, node)
            row.append(f"node {node} (page on {placed}): {_bandwidth([g], {g: src}, nbytes, args.iters)[g]:5.1f}")
        print(f"  GPU{g} (local node {nodes[g]}): " + "  ".join(row))

    local = {g: source(g, nodes[g])[0] for g in range(n)}
    solo = {g: _bandwidth([g], local, nbytes, args.iters)[g] for g in range(n)}
    print("\n== 两两同时 (源在各卡本地节点), 每卡 GB/s 及相对单卡的比例")
    contend = {}
    for a, b in itertools.combinations(range(n), 2):
        bw = _bandwidth([a, b], local, nbytes, args.iters)
        ratio = min(bw[a] / solo[a], bw[b] / solo[b])
        contend[(a, b)] = ratio
        print(f"  GPU{a}+GPU{b}: {bw[a]:5.1f} / {bw[b]:5.1f}   ({ratio:.2f} x 单卡)")
    bw = _bandwidth(list(range(n)), local, nbytes, args.iters)
    print("\n== 全部同时: " + "  ".join(f"GPU{g} {bw[g]:5.1f}" for g in range(n)))

    if n == 4:
        # TP ranks 0,1 form EP group 0 and 2,3 EP group 1; they copy in lockstep.
        best = max(([0, a, b, c] for a, b, c in itertools.permutations(range(1, 4))
                    if b < c), key=lambda o: contend[tuple(sorted(o[:2]))] +
                   contend[tuple(sorted(o[2:]))])
        print(f"\n建议: 让 TP0/TP1 = GPU{best[0]}/GPU{best[1]}, TP2/TP3 = GPU{best[2]}/GPU{best[3]}"
              f" (同组两卡同时拷贝时各自 {contend[tuple(sorted(best[:2]))]:.2f} / "
              f"{contend[tuple(sorted(best[2:]))]:.2f} x 单卡); "
              "即把 CUDA_VISIBLE_DEVICES 按这个下标顺序重排 (下标指当前列表里的位置)。")


if __name__ == "__main__":
    main()
