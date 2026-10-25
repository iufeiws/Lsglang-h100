#!/usr/bin/env python3
"""DSPARK decode 一步的时间都去了哪 (torch profiler trace, 每个 TP rank 一份)。

输出 (每个 verify 步, 再给中位数):
  - GPU: verify graph 的 GPU 跨度、两步之间 GPU 空闲 (CPU 是否跟不上 = ZOS 能省多少)
  - CPU: verify 的 cudaGraphLaunch 时长、从 verify 发射结束到下一次 draft 发射的间隔
  - 跨 rank: 各 rank verify 第一个 kernel 的开始时间差 (谁晚到, 其余 rank 就在 allreduce 里等)
  - allreduce: 每步总时长 (主要是等最慢 rank) 与纯通信下限 (次数 x p10)
  - 卸载层: 每层 MoE 段 (admit -> CUTLASS) 的耗时, 有/无换入拷贝分开统计
  - kernel 分族的 ms/步
用法:
  python analyze_step_timeline.py prof/*.trace.json.gz [--steps 3:30]
graph 内 kernel 靠 correlation id 归到 cudaGraphLaunch; >1000 个 kernel 的 graph 视为 verify, 其余视为 draft。
"""
import argparse
import collections
import gzip
import json
import re
import statistics


def _load(path):
    with gzip.open(path, "rt") as f:
        ev = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]
    launches = {e["args"]["correlation"]: e for e in ev
                if e.get("cat") == "cuda_runtime" and e["name"] == "cudaGraphLaunch"}
    launch_tid = collections.Counter(e["tid"] for e in launches.values()).most_common(1)[0][0]
    syncs = sorted((e for e in ev if e.get("cat") == "cuda_runtime" and e["tid"] == launch_tid
                    and e["name"] in ("cudaStreamSynchronize", "cudaEventSynchronize",
                                      "cudaDeviceSynchronize")), key=lambda e: e["ts"])
    kernels = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in ("kernel", "gpu_memcpy") and e.get("args", {}).get("correlation") in launches:
            kernels[e["args"]["correlation"]].append(e)
    graphs = []
    for c in sorted(launches, key=lambda c: launches[c]["ts"]):
        ks = sorted(kernels[c], key=lambda e: e["ts"])
        if ks:
            graphs.append(dict(launch=launches[c], kernels=ks, verify=len(ks) > 1000,
                               start=ks[0]["ts"], end=max(k["ts"] + k["dur"] for k in ks)))
    return graphs, syncs


def _sync_wait(syncs, a, b):
    """Host time in synchronize calls of the launching thread within [a, b]."""
    return sum(max(0, min(b, e["ts"] + e["dur"]) - max(a, e["ts"])) for e in syncs
               if e["ts"] < b and e["ts"] + e["dur"] > a)


# First match wins: attention and norm kernels carry "cutlass" in their template args.
_FAMILIES = [
    ("allreduce", r"allreduce"),
    ("cache: admit/copy", r"_admit_misses_kernel|_copy_experts_kernel|_stage_copy_kernel"
                          r"|_assign_slots_kernel"),
    ("cache: UVA overflow", r"^_gate_up_kernel$|^_down_kernel$|^_plan_kernel$"),
    ("attention", r"mla|flash_fwd|index_logits|topk_select|norm_rope|rope_kernel"),
    ("deep_gemm fp8", r"deep_gemm"),
    ("act quant", r"per_token_group_quant"),
    ("mHC", r"_hc_|hc_split|_mhc_"),
    ("cutlass MoE", r"GemmUniversal|cutlass_kernels|tensorrt_llm"),
    ("memcpy", r"^Memcpy"),
]


def _family(name):
    for fam, pat in _FAMILIES:
        if re.search(pat, name):
            return fam
    return "other"


def _median(values):
    return statistics.median(values) if values else float("nan")


def _layer_moe(ks):
    """(ms, copied) per offloaded layer: admit kernel start -> last CUTLASS end before
    the next allreduce; copied = the copy kernel ran longer than a no-op."""
    out = []
    for i, k in enumerate(ks):
        if k["name"] != "_admit_misses_kernel":
            continue
        end, copied = None, False
        for later in ks[i + 1:]:
            if "allreduce" in later["name"]:
                break
            if later["name"] == "_copy_experts_kernel" and later["dur"] > 20:
                copied = True
            if "cutlass" in later["name"] or "tensorrt_llm" in later["name"]:
                end = later["ts"] + later["dur"]
        if end is not None:
            out.append(((end - k["ts"]) / 1000, copied))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--steps", default="3:30", help="verify 步的切片 (跳过开头的预热)")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.steps.split(":"))
    loaded = {path: _load(path) for path in args.traces}
    ranks = {p: v[0] for p, v in loaded.items()}
    syncs = {p: v[1] for p, v in loaded.items()}
    names = {p: re.search(r"TP-\d+-EP-\d+", p).group(0) if re.search(r"TP-\d+-EP-\d+", p) else p
             for p in ranks}
    verify = {p: [g for g in gs if g["verify"]][lo:hi] for p, gs in ranks.items()}

    print("== 每个 rank (中位数, ms)")
    print(f"{'rank':<12} {'verify GPU':>10} {'GPU idle':>9} {'launch API':>10} {'cpu gap':>8} "
          f"{'sync wait':>9} {'allreduce':>9} {'ar floor':>8} {'step':>7}")
    for p, gs in ranks.items():
        vs = verify[p]
        span = [(g["end"] - g["start"]) / 1000 for g in vs]
        launch = [g["launch"]["dur"] / 1000 for g in vs]
        idle, gap, wait, step, ar, floor = [], [], [], [], [], []
        for g in vs:
            j = gs.index(g)
            if j + 1 < len(gs):
                nxt = gs[j + 1]
                idle.append(max(0.0, nxt["start"] - g["end"]) / 1000)
                a = g["launch"]["ts"] + g["launch"]["dur"]
                gap.append((nxt["launch"]["ts"] - a) / 1000)
                wait.append(_sync_wait(syncs[p], a, nxt["launch"]["ts"]) / 1000)
            later = [h for h in gs[j + 1:] if h["verify"]]
            if later:
                step.append((later[0]["start"] - g["start"]) / 1000)
            durs = sorted(k["dur"] for k in g["kernels"] if "allreduce" in k["name"])
            ar.append(sum(durs) / 1000)
            floor.append(len(durs) * durs[len(durs) // 10] / 1000 if durs else 0.0)
        print(f"{names[p]:<12} {_median(span):>10.2f} {_median(idle):>9.2f} {_median(launch):>10.2f} "
              f"{_median(gap):>8.2f} {_median(wait):>9.2f} {_median(ar):>9.2f} {_median(floor):>8.2f} "
              f"{_median(step):>7.2f}")
    print("  GPU idle = 本步 verify 结束到下一个 graph 开始; >0.5 ms 才说明 CPU 跟不上 (ZOS 的收益上限)")
    print("  cpu gap = verify 发射返回到下一次 draft 发射; 减去 sync wait (等 GPU 结果) 是 CPU 的真实工作,")
    print("  launch API + cpu gap - sync wait 接近 step 时 CPU 已是瓶颈。profiler 会放大 launch API")
    print("  allreduce - ar floor ~= 在 allreduce 里等最慢 rank 的时间")

    print("\n== 跨 rank: verify 第一个 kernel 的开始时间, 相对最早的 rank (ms, 中位数 / 最大)")
    n = min(len(v) for v in verify.values())
    skew = collections.defaultdict(list)
    for i in range(n):
        t0 = min(verify[p][i]["start"] for p in ranks)
        for p in ranks:
            skew[p].append((verify[p][i]["start"] - t0) / 1000)
    for p in ranks:
        print(f"  {names[p]:<12} {_median(skew[p]):6.2f} / {max(skew[p]) if skew[p] else float('nan'):6.2f}")

    print("\n== 卸载层 MoE 段 (admit -> CUTLASS 结束), ms")
    for p in ranks:
        layers = [m for g in verify[p] for m in _layer_moe(g["kernels"])]
        plain = [ms for ms, copied in layers if not copied]
        with_copy = [ms for ms, copied in layers if copied]
        print(f"  {names[p]:<12} 无换入 {_median(plain):6.3f} (n={len(plain)})   "
              f"有换入 {_median(with_copy):6.3f} (n={len(with_copy)})   "
              f"有换入的层占 {100 * len(with_copy) / max(len(layers), 1):5.1f}%")

    print("\n== kernel 分族, ms/步 (verify graph)")
    for p in ranks:
        fam = collections.Counter()
        for g in verify[p]:
            for k in g["kernels"]:
                fam[_family(k["name"])] += k["dur"]
        steps = max(len(verify[p]), 1)
        print(f"  {names[p]}: " + ", ".join(f"{f} {d / steps / 1000:.2f}" for f, d in fam.most_common()))


if __name__ == "__main__":
    main()
