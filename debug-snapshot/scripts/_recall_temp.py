import json,time,requests,sys
B="http://127.0.0.1:18010"; H={"Content-Type":"application/json","Authorization":"Bearer sk-dsv41-h100-interop-2026"}
txt=open("/mnt/SSD01/models/.llm-env/Lsglang/docs/HANDOFF.md",errors="replace").read()
prompt=("请阅读以下材料，然后写出它的标题行原文。\n\n===材料开始===\n"+txt+"\n===材料结束===\n\n请完整写出标题行：")
EXP="DeepSeek-V4.1-Flash-1017"
temp=float(sys.argv[1]) if len(sys.argv)>1 else 0.7
N=int(sys.argv[2]) if len(sys.argv)>2 else 6
print(f"=== temperature={temp} (实际配置见服务; 本行文案硬编码) ×{N} ===")
bad=0
for i in range(N):
    t0=time.time()
    r=requests.post(f"{B}/v1/chat/completions",headers=H,timeout=900,
        json={"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":prompt}],
              "max_tokens":48,"temperature":temp})
    d=r.json(); dt=time.time()-t0
    c=d["choices"][0]; out=(c["message"].get("content") or "")
    ok=EXP in out; bad += (not ok)
    print(f"  {i+1:2d}: {'✅' if ok else '❌'} {dt:5.2f}s toks={d['usage']['completion_tokens']:3d} {out[:66]!r}", flush=True)
print(f"  错误率: {bad}/{N}")
