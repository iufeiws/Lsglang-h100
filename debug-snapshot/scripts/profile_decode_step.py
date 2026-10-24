#!/usr/bin/env python3
"""抓一段纯 decode 的 torch profile, 定位 DSPARK 一步约 68 ms 花在哪。

先发一个长输出请求, 等它进入 decode 后再开 profile (避开 prefill), 抓 num_steps 步后
服务端自动停止并写出 trace。DSPARK 与无 DSPARK 各跑一次, 对照同一 prompt 的步内构成:
  草稿前向 / target verify / 20 个 LK 层 (GPU 池 kernel、cpu_decode 主机回调等待) / accept。
用法:
  python profile_decode_step.py --tag dspark10 --api-key sk-...
trace 目录: --outdir/<tag>_<时间>/ (每个 TP rank 一个 .trace.json.gz)
"""
import argparse
import os
import threading
import time
from datetime import datetime

import requests

PROMPT = "请用中文详细解释量子纠缠的物理机制, 并说明它在量子通信中的三种典型应用, 每种应用给出一个具体例子。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18010")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--api-key", default="sk-dsv41-h100-interop-2026")
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--warmup-s", type=float, default=4.0, help="请求发出后等多久再开 profile")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--outdir", default="/mnt/SSD01/models/.llm-env/Lsglang/logs/profiles")
    a = ap.parse_args()
    headers = {"Authorization": f"Bearer {a.api_key}"}
    out = os.path.join(a.outdir, f"{a.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)

    def generate():
        r = requests.post(f"{a.base}/v1/chat/completions", headers=headers, timeout=3600, json={
            "model": a.model, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 1500, "temperature": a.temp})
        print("request done:", r.status_code, r.json().get("usage"), flush=True)

    t = threading.Thread(target=generate)
    t.start()
    time.sleep(a.warmup_s)
    r = requests.post(f"{a.base}/start_profile", headers=headers, timeout=60, json={
        "output_dir": out, "num_steps": a.num_steps, "activities": ["CPU", "GPU"],
        "record_shapes": True, "profile_prefix": a.tag})
    print("start_profile:", r.status_code, r.text.strip(), flush=True)
    t.join()
    print("trace dir:", out)


if __name__ == "__main__":
    main()
