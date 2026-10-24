import json,time,requests
from collections import Counter
B="http://127.0.0.1:18010"; H={"Content-Type":"application/json","Authorization":"Bearer sk-dsv41-h100-interop-2026"}
txt=open("/mnt/SSD01/models/.llm-env/Lsglang/docs/HANDOFF.md",errors="replace").read()
prompt=("请阅读以下材料，然后写出它的标题行原文。\n\n===材料开始===\n"+txt+"\n===材料结束===\n\n请完整写出标题行：")
EXP="DeepSeek-V4.1-Flash-1017"
N=20
outs=[]
for i in range(N):
    t0=time.time()
    r=requests.post(f"{B}/v1/chat/completions",headers=H,timeout=600,
        json={"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":prompt}],
              "max_tokens":32,"temperature":0.0})
    d=r.json(); t=(d["choices"][0]["message"].get("content") or "").strip()
    ok = EXP in t
    outs.append(t)
    print(f"  {i+1:2d}: {'✅' if ok else '❌'} {time.time()-t0:5.2f}s {t[:58]!r}", flush=True)
c=Counter(outs)
print(f"\n  不同输出的种类: {len(c)}  ← 贪心下应为 1")
for s,n in c.most_common():
    print(f"    x{n}  {s[:70]!r}")
print(f"  错误率: {sum(1 for s in outs if EXP not in s)}/{N}")
