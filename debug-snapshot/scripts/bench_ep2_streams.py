#!/usr/bin/env python3
"""EP2 压测客户端 —— 只发请求(SSE 流式), 不改服务端。

测项:
  * TTFT (首 token 延迟)
  * decode tok/s = (N-1) / (t_last - t_first)   ← 纯解码速率, 不含 prefill
  * ms/step = 1000 / decode_tok_s (与历史口径一致)
  * 多流聚合: 每流独立测, 另报"公共窗口"聚合(见 --plan)
  * 服务端 #cached-token / accept 相关行在外部日志解析(本脚本不写服务端日志)
用法:
  python bench_ep2_streams.py --k 16 --max-tokens 1024 --temp 0.7 --tag baseline
输出:
  CSV + JSON 摘要(绝对路径)
"""
from __future__ import annotations
import argparse, csv, json, os, statistics, subprocess, sys, threading, time
from datetime import datetime, timedelta, timezone

import requests

CST = timezone(timedelta(hours=8))

PROMPT_TEMPLATES = [
    "请用中文详细解释量子纠缠的物理机制, 并说明它在量子通信中的三种典型应用, 每种应用给出一个具体例子。",
    "写一篇关于深海热液喷口生态系统的综述, 包括能量来源、代表性生物类群、与地表生态的差异, 以及研究方法的演进。",
    "解释 Transformer 架构中自注意力机制的计算复杂度来源, 并对比稀疏注意力、线性注意力、状态空间模型三类改进路线的取舍。",
    "请设计一个面向百万级并发的在线推理服务的容量规划方案, 说明如何确定 GPU 数量、批大小、KV 缓存上限与超时策略。",
    "详细说明药物分子从靶点发现到临床试验的主要阶段, 每个阶段的关键技术手段与失败原因, 并举例说明。",
    "分析 2008 年金融危机中影子银行体系的作用链条, 包括回购市场、货币基金与证券化产品的相互放大机制。",
    "请解释混合专家(MoE)模型中专家并行与张量并行的区别, 并说明专家并行下负载不均衡的具体来源与缓解手段。",
    "写一份关于青藏高原多年冻土退化的观测报告, 包含观测指标、典型数据、生态与工程影响及不确定性来源。",
    "解释分布式训练中 Ring-AllReduce 与 Tree-AllReduce 的通信代价差异, 并分析在大规模 GPU 集群上的适用场景。",
    "请介绍蛋白质结构预测从同源建模到深度学习的范式转变, 说明关键模型的核心思想与主要局限。",
    "分析城市交通拥堵的经济学成因, 包括外部性、定价机制失灵与土地利用互动, 并比较三种治理政策的实证效果。",
    "请解释浮点数的表示原理(IEEE 754), 包括舍入误差、次正规数与特殊值, 并说明它对深度学习数值稳定性的影响。",
    "写一篇关于抗生素耐药性演化的科普文章, 说明耐药机制、传播途径、临床与农业用药的角色, 以及可能的干预措施。",
    "解释 CUDA 中流(stream)、事件(event)与图(graph)捕获的语义区别, 以及它们如何影响异步执行与性能测量。",
    "请设计一个跨数据中心的分布式存储一致性方案, 对比强一致、因果一致与最终一致的实现代价与业务适用性。",
    "详细说明光伏电池从硅料到组件的制造流程, 各环节的能耗与良率关键点, 以及当前效率纪录背后的技术路线。",
]


def now_iso():
    return datetime.now(CST).isoformat(timespec="seconds")


def one_stream(i, prompt, base, model, max_tokens, temp, results, lock, stop_evt, timeout, HDR=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    rec = dict(stream=i, t_start=time.time(), ttft=None, n_tok=0, n_content=0, n_reasoning=0,
               t_first=None, t_last=None, err=None, finish=None,
               prompt_tokens=None, completion_tokens=None)
    try:
        with requests.post(f"{base}/v1/chat/completions", json=body, stream=True, timeout=timeout, headers=HDR) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if stop_evt.is_set():
                    rec["err"] = "aborted"
                    break
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except Exception:
                    continue
                if ev.get("usage"):
                    rec["prompt_tokens"] = ev["usage"].get("prompt_tokens")
                    rec["completion_tokens"] = ev["usage"].get("completion_tokens")
                for ch in ev.get("choices", []):
                    if ch.get("finish_reason"):
                        rec["finish"] = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    # 分开计: 思考链 token 与正文 token (该模型 thinking 默认开启)
                    c_txt = delta.get("content") or ""
                    r_txt = delta.get("reasoning_content") or ""
                    if c_txt or r_txt:
                        t = time.time()
                        if rec["t_first"] is None:
                            rec["t_first"] = t
                            rec["ttft"] = t - rec["t_start"]
                        rec["t_last"] = t
                        rec["n_tok"] += 1
                        if c_txt:
                            rec["n_content"] += 1
                        if r_txt:
                            rec["n_reasoning"] += 1
    except Exception as e:
        rec["err"] = repr(e)[:200]
    rec["t_end"] = time.time()
    # sglang 会把一次 verify 接受的多个 token 合并进同一个 SSE chunk
    # ⇒ 事件计数 n_tok 会系统性低估 (倍数≈accept len)。权威计数用 usage.completion_tokens。
    n = rec.get("completion_tokens") or rec["n_tok"]
    rec["n_tok_used"] = n
    rec["events"] = rec["n_tok"]
    if rec["t_first"] is not None and rec["t_last"] is not None and n > 1:
        span = rec["t_last"] - rec["t_first"]
        rec["decode_tok_s"] = (n - 1) / span if span > 0 else None
        rec["ms_per_step"] = (span / (n - 1)) * 1000 if span > 0 else None
        rec["wall_s"] = rec["t_end"] - rec["t_start"]
        rec["total_tok_s"] = n / rec["wall_s"] if rec["wall_s"] > 0 else None
    else:
        rec["decode_tok_s"] = rec["ms_per_step"] = rec["wall_s"] = rec["total_tok_s"] = None
    with lock:
        results.append(rec)


class VramSampler(threading.Thread):
    def __init__(self, csv_path, gpus=(4, 5, 6, 7), interval=5.0):
        super().__init__(daemon=True)
        self.csv_path, self.gpus, self.interval = csv_path, gpus, interval
        self.stop_evt = threading.Event()
        self.rows = []

    def run(self):
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_unix", "elapsed_s"] + [f"gpu{g}_used_mib" for g in self.gpus]
                       + [f"gpu{g}_util" for g in self.gpus])
            f.flush()
            t0 = time.time()
            while not self.stop_evt.is_set():
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10).stdout.strip().split("\n")
                    d = {}
                    for line in out:
                        idx, mu, ut = [x.strip() for x in line.split(",")]
                        d[int(idx)] = (int(mu), int(ut))
                    row = [f"{time.time():.0f}", f"{time.time()-t0:.1f}"]
                    row += [d.get(g, (0, 0))[0] for g in self.gpus]
                    row += [d.get(g, (0, 0))[1] for g in self.gpus]
                    w.writerow(row); f.flush()
                    self.rows.append(row)
                except Exception:
                    pass
                self.stop_evt.wait(self.interval)


def pct(vals, p):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    k = max(0, min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1)))))
    return vals[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18010")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--k", type=int, default=16, help="并发流数")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--tag", default="ep2")
    ap.add_argument("--api-key", default="", help="Authorization: Bearer 值")
    ap.add_argument("--outdir", default="/mnt/SSD01/models/.llm-env/Lsglang/logs")
    args = ap.parse_args()

    ts = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
    stem = f"bench_{args.tag}_{ts}"
    csv_path = os.path.join(args.outdir, stem + ".csv")
    vram_path = os.path.join(args.outdir, stem + ".vram.csv")
    json_path = os.path.join(args.outdir, stem + ".json")

    HDR = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    k = args.k
    prompts = [PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)] for i in range(k)]
    print(f"[{now_iso()}] K={k} max_tokens={args.max_tokens} temp={args.temp} "
          f"prompt 互异={len(set(prompts))}/{k}", flush=True)

    sampler = VramSampler(vram_path)
    sampler.start()
    results, lock, stop_evt = [], threading.Lock(), threading.Event()
    threads = [threading.Thread(target=one_stream, args=(i, prompts[i], args.base, args.model,
                                                        args.max_tokens, args.temp, results,
                                                        lock, stop_evt, args.timeout, HDR))
               for i in range(k)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    sampler.stop_evt.set(); sampler.join(timeout=15)

    ok = [r for r in results if r.get("decode_tok_s")]
    bad = [r for r in results if not r.get("decode_tok_s")]
    with open(csv_path, "w", newline="") as f:
        cols = ["stream","ttft","n_tok_used","events","n_content","n_reasoning","decode_tok_s",
                "ms_per_step","wall_s","total_tok_s","prompt_tokens","completion_tokens","finish","err"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda x: x["stream"]):
            w.writerow(r)

    def agg(key):
        v = [r.get(key) for r in ok]
        return dict(p50=pct(v, 50), p10=pct(v, 10), p90=pct(v, 90),
                    mean=(statistics.fmean([x for x in v if x is not None]) if v else None),
                    n=len([x for x in v if x is not None]))

    # 公共重叠窗口聚合: 所有流都在产 token 的时间窗内累计 token
    tf = [r["t_first"] for r in ok]; tl = [r["t_last"] for r in ok]
    win = None
    if tf and tl and max(tf) < min(tl):
        t_lo, t_hi = max(tf), min(tl)
        tok_in = 0
        for r in ok:
            span = r["t_last"] - r["t_first"]
            n = (r.get("completion_tokens") or r["n_tok"]) - 1
            if span > 0:
                tok_in += n * ((t_hi - t_lo) / span)
        win = dict(window_s=t_hi - t_lo, tokens=tok_in, agg_tok_s=tok_in / (t_hi - t_lo))

    summary = dict(
        tag=args.tag, started=now_iso(), k=k, max_tokens=args.max_tokens, temperature=args.temp,
        prompt_uniq=len(set(prompts)), wall_s=wall,
        n_ok=len(ok), n_err=len(bad), errors=[r.get("err") for r in bad],
        ttft=agg("ttft"), decode_tok_s=agg("decode_tok_s"), ms_per_step=agg("ms_per_step"),
        total_tok_s=agg("total_tok_s"),
        sum_completion_tokens=sum((r.get("completion_tokens") or r["n_tok"]) for r in ok),
        note="decode tok/s 用 usage.completion_tokens 计; SSE 事件计数见 events 列 (会低估 accept-len 倍)",
        overlap_window=win,
        vram_csv=vram_path, per_stream_csv=csv_path,
        vram_max_mib={f"gpu{g}": (max((int(r[2+j]) for r in sampler.rows), default=None))
                      for j, g in enumerate((4,5,6,7))},
    )
    with open(json_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[{now_iso()}] 完成: {json_path}", flush=True)


if __name__ == "__main__":
    main()
