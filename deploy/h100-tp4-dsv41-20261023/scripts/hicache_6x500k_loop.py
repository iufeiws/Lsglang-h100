import json,random,urllib.request,time
CHARS=346620
CH=[chr(c) for c in range(0x4E00,0x4E00+200)]
prompts={}
for i in range(1,7):
    rnd=random.Random(1000+i)
    prompts[i]="".join(rnd.choices(CH,k=CHARS))
def ask(i):
    p={"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":prompts[i] + "\n\n请用一个字回答：好"}],
       "max_tokens":8,"temperature":0.7}
    body=json.dumps(p).encode()
    r=urllib.request.Request("http://127.0.0.1:18010/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=3600) as resp: d=json.loads(resp.read())
    el=time.time()-t0; u=d.get("usage",{})
    det=u.get("prompt_tokens_details") or {}
    return u.get("prompt_tokens",0), det.get("cached_tokens"), el, d["choices"][0].get("finish_reason")
seq=[1,2,3,4,5,6,1,2,3,4,5,6]
rows=[]
print(f"[{time.strftime('%H:%M:%S')}] 6 条 prompt x {CHARS} 随机汉字 (≈500K token)；顺序 {seq}；KV 池=1,984,000 token", flush=True)
for k,i in enumerate(seq,1):
    pas = "第1遍" if k<=6 else "第2遍"
    pt,cached,el,fr = ask(i)
    rows.append((k,pas,i,pt,cached,el,fr))
    print(f"  {pas} #{i}: prompt_tokens={pt}  cached_tokens={cached}  耗时={el:7.3f}s  finish={fr}", flush=True)
print(f"\n[{time.strftime('%H:%M:%S')}] 汇总", flush=True)
for k,pas,i,pt,cached,el,fr in rows:
    r = (cached/pt) if (cached is not None and pt) else None
    print(f"  {pas} #{i}: cached={cached} ({'%.1f%%'%(100*r) if r is not None else 'n/a'})  耗时={el:.3f}s")
b=time.strftime("%Y%m%d_%H%M")
open(f"/mnt/SSD01/models/.llm-env/Lsglang/logs/hicache-6x500k-{b}.csv","w").write("k,pass,idx,prompt_tokens,cached_tokens,elapsed_s,finish\n"+"\n".join(",".join(map(str,r)) for r in rows))
print("CSV: /mnt/SSD01/models/.llm-env/Lsglang/logs/hicache-6x500k-%s.csv"%b)
