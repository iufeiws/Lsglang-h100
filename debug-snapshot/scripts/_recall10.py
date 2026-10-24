import json,time,requests
B="http://127.0.0.1:18010"; H={"Content-Type":"application/json","Authorization":"Bearer sk-dsv41-h100-interop-2026"}
txt=open("/mnt/SSD01/models/.llm-env/Lsglang/docs/HANDOFF.md",errors="replace").read()
prompt=("请阅读以下材料，然后写出它的标题行原文。\n\n===材料开始===\n"+txt+"\n===材料结束===\n\n请完整写出标题行：")
EXP="DeepSeek-V4.1-Flash-1017"
bad=0
for i in range(10):
    r=requests.post(f"{B}/v1/chat/completions",headers=H,timeout=600,
        json={"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":prompt}],
              "max_tokens":40,"temperature":0.0})
    d=r.json(); c=d["choices"][0]; out=(c["message"].get("content") or "")
    ok=EXP in out
    bad+= (not ok)
    print(f"  {i+1:2d}: {'✅' if ok else '❌'} {out[:70]!r}", flush=True)
print(f"\n  错误率: {bad}/10")
