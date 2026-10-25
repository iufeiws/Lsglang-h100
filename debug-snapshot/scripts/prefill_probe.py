#!/usr/bin/env python3
"""纯 prefill 探针: 服务空闲 + 唯一 prompt + max_tokens=1, 用墙钟算 tok/s。

长 prompt 由一个固定文本块重复到约 --long-tokens 个 token 构成 (重复块使不同轮次的
prompt 内容一致, 便于同口径比较)。短 prompt 用 --short-prompt 测 TTFT 类开销。
用法:
  python prefill_probe.py --api-key sk-... --outdir <dir> --tag uva_tc
"""
import argparse
import json
import os
import time

import requests
from transformers import AutoTokenizer

MODEL = "/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017"
BLOCK = ("在混合专家模型的推理系统中, 专家并行的通信开销与负载均衡是两个互相牵制的问题。"
         "当每层路由到不同专家的 token 数量差异较大时, 单卡上的专家计算量会出现明显的长尾, "
         "而调度器又必须在每个批次的截止时间内完成全部专家的前向计算。为了缓解这一问题, "
         "工程上通常采用容量因子、token 丢弃、专家复制与动态缓存等手段, 并在吞吐与精度之间取舍。") * 4
SHORT = "请用一句话说明什么是张量并行。"


def post(url, key, payload, stream=False):
    r = requests.post(url, headers={"Content-Type": "application/json",
                                    "Authorization": f"Bearer {key}"},
                      json=payload, timeout=1800, stream=stream)
    r.raise_for_status()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:18010")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--long-tokens", type=int, default=17000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="prefill")
    ap.add_argument("--salt", type=int, default=0,
                    help="非 0 时在 prompt 开头插入唯一串, 绕开前缀缓存 (冷 prefill)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    block_ids = tok(BLOCK)["input_ids"]
    reps = max(1, args.long_tokens // len(block_ids))
    salt = f"[测例 {args.salt}] " if args.salt else ""
    long_text = salt + BLOCK * reps
    n_prompt = len(tok(long_text)["input_ids"])

    url = f"{args.host}/v1/chat/completions"
    out = {"tag": args.tag, "block_tokens": len(block_ids), "reps": reps,
           "prompt_tokens_local_count": n_prompt, "runs": []}
    for name, text in (("long", long_text), ("short", SHORT)):
        payload = {"model": args.model, "messages": [{"role": "user", "content": text}],
                   "max_tokens": 1, "temperature": 0, "stream": False}
        t0 = time.time()
        r = post(url, args.api_key, payload)
        wall = time.time() - t0
        d = r.json()
        pt = d.get("usage", {}).get("prompt_tokens")
        rec = {"case": name, "salt": args.salt, "wall_s": wall, "prompt_tokens": pt,
               "tok_s": (pt / wall) if pt else None,
               "completion_tokens": d.get("usage", {}).get("completion_tokens")}
        out["runs"].append(rec)
        print(f"{name:>5}: prompt_tokens={pt} wall={wall:.3f}s "
              f"prefill={rec['tok_s']:.1f} tok/s", flush=True)
    path = os.path.join(args.outdir, f"prefill_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("json:", path)


if __name__ == "__main__":
    main()
