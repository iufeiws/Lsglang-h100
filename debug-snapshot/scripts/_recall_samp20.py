import json,time,requests
from collections import Counter
B="http://127.0.0.1:18010"; H={"Content-Type":"application/json","Authorization":"Bearer sk-dsv41-h100-interop-2026"}
txt=open("/mnt/SSD01/models/.llm-env/Lsglang/docs/HANDOFF.md",errors="replace").read()
prompt=("请阅读以下材料，然后写出它的标题行原文。\n\n===材料开始===\n"+txt+"\n===材料结束===\n\n请完整写出标题行：")
EXP="DeepSeek-V4.1-Flash-1017"
N=20; outs=[]
for i in range(N):
    r=requests.post(f"{B}/v1/chat/completions",headers=H,timeout=600,
        json={"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":prompt}],
              "max_tokens":32,"temperature":0.7})
    t=(r.json()["choices"][0]["message"].get("content") or "").strip()
    outs.append(t)
    if EXP not in t: print(f"  {i+1:2d}: ❌ {t[:60]!r}", flush=True)
print(f"\n  temp=0.7, n={N}")
print(f"  不同输出种类: {len(set(outs))}")
for s,n in Counter(outs).most_common():
    print(f"    x{n:2d}  {'OK ' if EXP in s else 'ERR'}  {s[:64]!r}")
print(f"  错误率: {sum(1 for s in outs if EXP not in s)}/{N}")
