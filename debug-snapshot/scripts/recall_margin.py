#!/usr/bin/env python3
"""贪心复述 + logprob: 区分「近似平局的数值抖动」与「logits 被破坏」。

同一 prompt 贪心跑 N 次, 每个 token 取 top-5 logprob。对每个与多数输出不同的运行,
定位第一个分歧 token, 打印两边在该位置的 top-5:
  * 多数运行里, 错误 token 的概率若 < 1e-3 (margin 很大) 而分歧运行却选了它
    => 分歧运行在该步拿到的 logits 被破坏 (竞争/错专家/错 KV), 不是数值噪声
  * 两边 top-2 都接近 (例如 0.55 / 0.45) => 近似平局, 属于数值非确定性
无 DSPARK 与 DSPARK 两种服务都能跑; 结果逐条落盘 (JSONL), 便于事后对照。

用法:
  python recall_margin.py --n 50 --tag dspark10 --api-key sk-...
  python recall_margin.py --n 50 --tag nodspark --material /path/HANDOFF.md
"""
import argparse
import json
import math
import os
import time
from collections import Counter
from datetime import datetime

import requests


def run_once(base, headers, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 5,
    }
    r = requests.post(f"{base}/v1/chat/completions", headers=headers, json=body, timeout=600)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    content = (choice["message"].get("content") or "").strip()
    toks = []
    for item in (choice.get("logprobs") or {}).get("content") or []:
        toks.append({
            "token": item["token"],
            "logprob": item["logprob"],
            "top": [(t["token"], t["logprob"]) for t in item.get("top_logprobs") or []],
        })
    return content, toks


def fmt_top(top):
    return "  ".join(f"{tok!r}:{math.exp(lp):.4f}" for tok, lp in top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18010")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--api-key", default="sk-dsv41-h100-interop-2026")
    ap.add_argument("--material", default="/mnt/SSD01/models/.llm-env/Lsglang/docs/HANDOFF.md")
    ap.add_argument("--expect", default="DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--outdir", default="/mnt/SSD01/models/.llm-env/Lsglang/logs")
    a = ap.parse_args()

    txt = open(a.material, errors="replace").read()
    # Same prompt as _recall_det.py so the numbers stay comparable.
    prompt = ("请阅读以下材料，然后写出它的标题行原文。\n\n===材料开始===\n" + txt
              + "\n===材料结束===\n\n请完整写出标题行：")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {a.api_key}"}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(a.outdir, f"recall_margin_{a.tag}_{ts}.jsonl")
    runs = []
    with open(out_path, "w") as f:
        for i in range(a.n):
            t0 = time.time()
            content, toks = run_once(a.base, headers, a.model, prompt, a.max_tokens)
            runs.append((content, toks))
            f.write(json.dumps({"i": i, "content": content, "tokens": toks}, ensure_ascii=False) + "\n")
            f.flush()
            ok = a.expect in content
            print(f"  {i+1:3d}: {'OK ' if ok else 'BAD'} {time.time()-t0:5.2f}s {content[:60]!r}", flush=True)

    counts = Counter(c for c, _ in runs)
    major, n_major = counts.most_common(1)[0]
    print(f"\n不同输出种类: {len(counts)}  多数输出 x{n_major}  错误率(不含 expect): "
          f"{sum(1 for c, _ in runs if a.expect not in c)}/{a.n}")
    if not runs[0][1]:
        print("服务端未返回 logprobs, 无法做 margin 分析 (只有上面的计数有效)")
        print(f"落盘: {out_path}")
        return
    ref = next(t for c, t in runs if c == major)
    for i, (content, toks) in enumerate(runs):
        if content == major:
            continue
        pos = next((p for p in range(min(len(ref), len(toks)))
                    if ref[p]["token"] != toks[p]["token"]), None)
        if pos is None:
            continue
        r_top = ref[pos]["top"]
        wrong = toks[pos]["token"]
        p_wrong_in_ref = next((math.exp(lp) for tok, lp in r_top if tok == wrong), 0.0)
        verdict = ("近似平局(数值抖动)" if p_wrong_in_ref > 0.05
                   else "logits 被破坏(多数运行里该 token 概率极低)")
        print(f"\n运行 #{i+1} 在第 {pos} 个 token 分歧: 多数={ref[pos]['token']!r} 本次={wrong!r}"
              f"  多数运行中本次 token 概率={p_wrong_in_ref:.2e}  => {verdict}")
        print(f"  多数运行 top5: {fmt_top(r_top)}")
        print(f"  本次运行 top5: {fmt_top(toks[pos]['top'])}")
    print(f"\n落盘: {out_path}")


if __name__ == "__main__":
    main()
