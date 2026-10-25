#!/usr/bin/env python3
"""图片(多模态)探针: 生成一张小 PNG, 以 OpenAI 兼容的 image_url(data URL) 发一次请求, 记录结果。

用法: python image_probe.py --api-key <key> --outdir <dir>
输出: <outdir>/probe_image.png (测试图) 与 <outdir>/image_probe_<ts>.json (请求与响应原文)
"""
from __future__ import annotations
import argparse, base64, json, time, os
from datetime import datetime, timedelta, timezone

import requests
from PIL import Image, ImageDraw

CST = timezone(timedelta(hours=8))


def make_png(path):
    im = Image.new("RGB", (128, 128), (245, 245, 240))
    d = ImageDraw.Draw(im)
    d.ellipse((18, 18, 110, 110), fill=(180, 90, 80), outline=(40, 40, 40), width=3)
    d.rectangle((52, 52, 76, 76), fill=(250, 250, 250))
    im.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18010")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--outdir", default="/data3/shared/wangyf/llm-serving/fix3-dspark-20261025/results_slots128")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    png = make_png(os.path.join(args.outdir, "probe_image.png"))
    b64 = base64.b64encode(open(png, "rb").read()).decode()
    hdr = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "这张图片里有什么? 请简短描述。"},
        ]}],
        "max_tokens": 128, "temperature": 0, "stream": False,
    }
    t0 = time.time()
    out = dict(started=datetime.now(CST).isoformat(timespec="seconds"), endpoint=f"{args.base}/v1/chat/completions",
               image=png, image_bytes=os.path.getsize(png), request=body)
    try:
        r = requests.post(f"{args.base}/v1/chat/completions", json=body, headers=hdr, timeout=600)
        out.update(http_status=r.status_code, elapsed_s=round(time.time() - t0, 3),
                   response_text=r.text[:4000])
        try:
            out["choices"] = r.json().get("choices")
            out["usage"] = r.json().get("usage")
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        out.update(http_status=None, elapsed_s=round(time.time() - t0, 3), error=f"{type(e).__name__}: {e}")
    body_safe = json.loads(json.dumps(out))
    body_safe["request"]["messages"][0]["content"][0]["image_url"]["url"] = f"<data:image/png;base64,{len(b64)} chars>"
    path = os.path.join(args.outdir, f"image_probe_{datetime.now(CST).strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(body_safe, open(path, "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: body_safe[k] for k in ("http_status", "elapsed_s", "usage") if k in body_safe}, ensure_ascii=False))
    print("response:", str(body_safe.get("response_text"))[:800].replace("\n", " "))
    print("saved:", path)


if __name__ == "__main__":
    main()
