import json, time, requests
BASE="http://127.0.0.1:18010"; H={"Content-Type":"application/json","Authorization":"Bearer sk-dsv41-h100-interop-2026"}
M="/mnt/SSD01/models/.llm-env/Lsglang/docs/HANDOFF.md"
material=open(M,errors="replace").read()
prompt=("请阅读以下材料，然后写出它的标题行原文。\n\n===材料开始===\n"+material+"\n===材料结束===\n\n请完整写出标题行：")
EXPECT="DeepSeek-V4.1-Flash-1017"
def chat(tag, req, expect=None):
    t0=time.time(); r=requests.post(f"{BASE}/v1/chat/completions",headers=H,json=req,timeout=1200); dt=time.time()-t0
    d=r.json()
    if "choices" not in d: print(f"[{tag}] {r.status_code} {json.dumps(d,ensure_ascii=False)[:150]}"); return
    c=d["choices"][0]; m=c["message"]; txt=m.get("content") or ""
    mark = "" if expect is None else ("  含'%s'=%s" % (expect, "✅" if expect in txt else "❌"))
    print(f"[{tag}] {dt:6.2f}s toks={d['usage']['completion_tokens']:4d} finish={c.get('finish_reason')}{mark}")
    print(f"        {txt[:150]!r}")
print("=== ① 精确复述 ×4 (同一 prompt 连发) ===")
for i in range(4):
    chat(f"复述{i+1}", {"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":prompt}],"max_tokens":48,"temperature":0.0}, EXPECT)
print("\n=== ② 工具调用 (auto) ===")
tools=[{"type":"function","function":{"name":"get_weather","description":"查询指定城市的当前天气","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
r=requests.post(f"{BASE}/v1/chat/completions",headers=H,timeout=600,json={"model":"DeepSeek-V4.1-Flash-1017",
  "messages":[{"role":"user","content":"北京现在天气怎么样？请调用工具查询。"}],"tools":tools,"tool_choice":"auto","max_tokens":256,"temperature":0})
d=r.json(); c=d["choices"][0]; m=c["message"]
print(f"[tool auto] finish={c.get('finish_reason')} tool_calls={json.dumps(m.get('tool_calls'),ensure_ascii=False)}")
print(f"        content={(m.get('content') or '')[:200]!r}")
print("\n=== ③ 普通文本 ===")
chat("杭州", {"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"用三句话介绍杭州。"}],"max_tokens":256,"temperature":0.7})
