#!/usr/bin/env python3
"""长上下文压测客户端 (只发请求, 不改服务端): k 路并发, 每路 ~16k token 唯一 prompt。

测项与 bench_ep2_streams.py 同口径:
  TTFT / decode tok/s = (N-1)/(t_last-t_first) / ms/step = 1000/decode_tok_s / 公共窗口聚合。
prompt 长度用字符数逼近, 实际 token 数由响应 usage.prompt_tokens 回报。
用法: python bench_longctx_streams.py --k 16 --prompt-chars 18000 --max-tokens 256 --temp 0.7 --tag slots128_k16
"""
from __future__ import annotations
import argparse, csv, json, os, statistics, subprocess, threading, time
from datetime import datetime, timedelta, timezone

import requests

CST = timezone(timedelta(hours=8))
PARA = ("在混合专家模型的服务化部署中, 专家缓存与 KV 缓存共享同一块显存预算, 需要按命中率与换入带宽共同权衡; "
        "本节讨论路由计数、槽位准入、PCIe 带宽与批次形状之间的相互影响, 并给出可复现的测量口径与记录方式。")

HEADERS = ["i", "t_start", "ttft", "n_tok", "decode_tok_s", "ms_step", "wall", "prompt_tokens", "completion_tokens", "err"]


def now_iso():
    return datetime.now(CST).isoformat(timespec="seconds")


def make_prompt(i, chars):
    head = f"[流 {i} 的唯一前缀 {i * 7919}] "
    body = (PARA * (chars // len(PARA) + 1))[:chars]
    return head + body + f"\n\n以上是第 {i} 路材料。请用中文总结其要点, 并逐条列出你使用的判断依据。"


def one_stream(i, prompt, base, model, max_tokens, temp, results, lock, timeout, hdr):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp, "stream": True,
            "stream_options": {"include_usage": True}}
    rec = dict(i=i, t_start=time.time(), ttft=None, n_tok=0, t_first=None, t_last=None,
               err=None, prompt_tokens=None, completion_tokens=None)
    try:
        with requests.post(f"{base}/v1/chat/completions", json=body, stream=True,
                           timeout=timeout, headers=hdr) as r:
            if r.status_code != 200:
                rec["err"] = f"http {r.status_code}: {r.text[:200]}"
            else:
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    d = json.loads(payload)
                    if d.get("usage"):
                        rec["prompt_tokens"] = d["usage"].get("prompt_tokens")
                        rec["completion_tokens"] = d["usage"].get("completion_tokens")
                    for ch in d.get("choices") or []:
                        delta = ch.get("delta") or {}
                        piece = delta.get("content") or delta.get("reasoning_content") or ""
                        if piece:
                            t = time.time()
                            if rec["t_first"] is None:
                                rec["t_first"] = t
                                rec["ttft"] = t - rec["t_start"]
                            rec["t_last"] = t
                            rec["n_tok"] += 1
        rec["wall"] = time.time() - rec["t_start"]
    except Exception as e:  # noqa: BLE001
        rec["err"] = f"{type(e).__name__}: {e}"
        rec["wall"] = time.time() - rec["t_start"]
    n = rec["n_tok"]
    if n > 1 and rec["t_first"] and rec["t_last"] and rec["t_last"] > rec["t_first"]:
        rec["decode_tok_s"] = (n - 1) / (rec["t_last"] - rec["t_first"])
        rec["ms_step"] = 1000.0 / rec["decode_tok_s"]
    with lock:
        results.append(rec)


class VramSampler(threading.Thread):
    def __init__(self, path, gpus=(4, 5, 6, 7), interval=5.0):
        super().__init__(daemon=True)
        self.path, self.gpus, self.interval, self.rows, self.stop_evt = path, gpus, interval, [], threading.Event()

    def run(self):
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t", "iso"] + [f"gpu{g}" for g in self.gpus])
            while not self.stop_evt.is_set():
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True).stdout.split()
                if len(out) >= 8:
                    row = [time.time(), now_iso()] + [int(out[g]) for g in self.gpus]
                    self.rows.append(row); w.writerow(row); f.flush()
                self.stop_evt.wait(self.interval)


def pct(v, p):
    return None if not v else round(statistics.quantiles(v, n=100, method="inclusive")[p - 1], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18010")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--prompt-chars", type=int, default=18000, help="每路 prompt 的字符数 (≈16k token)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--tag", default="longctx")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--outdir", default="/data3/shared/wangyf/llm-serving/fix3-dspark-20261025/results_slots128")
    args = ap.parse_args()
    hdr = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    os.makedirs(args.outdir, exist_ok=True)
    stem = f"bench_{args.tag}_{datetime.now(CST).strftime('%Y%m%d_%H%M%S')}"
    vram_path = os.path.join(args.outdir, stem + ".vram.csv")
    sampler = VramSampler(vram_path); sampler.start()

    prompts = [make_prompt(i, args.prompt_chars) for i in range(args.k)]
    results, lock = [], threading.Lock()
    t0 = time.time()
    threads = [threading.Thread(target=one_stream, args=(i, p, args.base, args.model, args.max_tokens,
                                                         args.temp, results, lock, args.timeout, hdr))
               for i, p in enumerate(prompts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    sampler.stop_evt.set(); sampler.join(timeout=10)

    ok = [r for r in results if not r["err"]]
    agg = lambda key: dict(p50=pct(sorted(r[key] for r in ok if r.get(key)), 50),
                           p10=pct(sorted(r[key] for r in ok if r.get(key)), 10),
                           p90=pct(sorted(r[key] for r in ok if r.get(key)), 90),
                           n=len([r for r in ok if r.get(key)]))
    span = [max(r["t_first"] for r in ok), min(r["t_last"] for r in ok)] if ok else [0, 0]
    win = max(0.0, span[1] - span[0])
    csv_path = os.path.join(args.outdir, stem + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEADERS)
        for r in sorted(results, key=lambda x: x["i"]):
            w.writerow([r["i"], round(r["t_start"], 3), r.get("ttft"), r["n_tok"],
                        round(r.get("decode_tok_s") or 0, 3), round(r.get("ms_step") or 0, 3),
                        round(r.get("wall") or 0, 3), r.get("prompt_tokens"), r.get("completion_tokens"),
                        r.get("err") or ""])
    summary = dict(tag=args.tag, started=now_iso(), k=args.k, prompt_chars=args.prompt_chars,
                   max_tokens=args.max_tokens, temperature=args.temp, wall_s=wall,
                   prompt_tokens_per_stream=[r.get("prompt_tokens") for r in sorted(results, key=lambda x: x["i"])],
                   n_ok=len(ok), n_err=len(results) - len(ok),
                   errors=[r["err"] for r in results if r["err"]],
                   ttft=agg("ttft"), decode_tok_s=agg("decode_tok_s"), ms_step=agg("ms_step"),
                   completion_tokens_sum=sum(r.get("completion_tokens") or 0 for r in ok),
                   overlap_window=dict(window_s=round(win, 3),
                                       agg_tok_s=round(sum(r["decode_tok_s"] for r in ok if r.get("decode_tok_s")), 3),
                                       n_streams=len([r for r in ok if r.get("decode_tok_s")])),
                   vram_csv=vram_path, per_stream_csv=csv_path,
                   vram_max_mib={f"gpu{g}": max((int(r[2 + j]) for r in sampler.rows), default=None)
                                 for j, g in enumerate((4, 5, 6, 7))})
    json_path = os.path.join(args.outdir, stem + ".json")
    json.dump(summary, open(json_path, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[{now_iso()}] 完成: {json_path}", flush=True)


if __name__ == "__main__":
    main()
