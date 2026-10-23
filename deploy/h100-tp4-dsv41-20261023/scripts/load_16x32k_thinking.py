import json,time,urllib.request,threading,subprocess,statistics,sys
NL="/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261023_192956.log"
OUT="/mnt/SSD01/models/.llm-env/Lsglang/logs/load-16x32k-thinking.csv"
para=("深海探测是指人类利用科学装备与技术手段深入海洋内部，观察、测量、取样并分析海洋环境及其资源的过程。"
      "海洋覆盖地球表面约七十一个百分点，平均水深超过一千米，绝大多数海域处于阳光无法抵达的状态。")
prompt=("请阅读以下材料，然后写一份详细的探测方案建议。\n\n"+ (para*160) + "\n\n建议：")
N=16; MAXTOK=32768
print(f"[{time.strftime('%H:%M:%S')}] prompt 字符={len(prompt)}  并发={N}  max_tokens={MAXTOK}  thinking=thinking/max", flush=True)
res={}; errs={}
def send(i):
    body=json.dumps({"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":prompt}],
        "max_tokens":MAXTOK,"temperature":0.7,"reasoning_effort":"max",
        "chat_template_kwargs":{"thinking_mode":"thinking","drop_thinking":False}}).encode()
    r=urllib.request.Request("http://127.0.0.1:18010/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.time()
    try:
        with urllib.request.urlopen(r,timeout=7200) as resp: d=json.loads(resp.read())
        m=d["choices"][0]["message"]; u=d.get("usage",{})
        res[i]=(u.get("prompt_tokens",0),u.get("completion_tokens",0),len(m.get("reasoning_content") or ""),time.time()-t0,d["choices"][0].get("finish_reason"))
    except Exception as e: errs[i]=repr(e)[:200]
vram=[]; stop=threading.Event()
def sample():
    f=open(OUT,"w"); f.write("t,gpu4,gpu5,gpu6,gpu7,util4,util5,util6,util7\n")
    t0=time.time()
    while not stop.is_set():
        try:
            s=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10).stdout.strip().split("\n")[4:8]
            row=[l.split(", ") for l in s]
            vram.append([int(x[1]) for x in row]); f.write(f"{time.time()-t0:.0f},"+",".join(x[1] for x in row)+","+",".join(x[2] for x in row)+"\n"); f.flush()
        except Exception: pass
        stop.wait(5)
    f.close()
th=threading.Thread(target=sample,daemon=True); th.start()
t0=time.time(); ths=[threading.Thread(target=send,args=(i,)) for i in range(N)]
for x in ths: x.start()
while any(x.is_alive() for x in ths):
    time.sleep(30); done=len(res)+len(errs)
    el=time.time()-t0
    print(f"[{time.strftime('%H:%M:%S')}] 完成 {done}/{N}  已跑 {el:.0f}s  当前显存={vram[-1] if vram else '?'}", flush=True)
for x in ths: x.join()
el=time.time()-t0; time.sleep(5); stop.set(); th.join(timeout=8)
print(f"\n=== 结束: 总耗时 {el:.0f}s  成功 {len(res)}  失败 {len(errs)} ===")
if errs: print("错误:", list(errs.values())[:2])
if res:
    pts=[v[0] for v in res.values()]; cts=[v[1] for v in res.values()]; rl=[v[2] for v in res.values()]; fr=[v[4] for v in res.values()]
    print(f"prompt_tokens/请求={statistics.median(pts):.0f} (合计 {sum(pts)})  completion/请求={statistics.median(cts):.0f} (合计 {sum(cts)})")
    print(f"reasoning 字符数/请求 中位={statistics.median(rl):.0f}  finish_reason 分布={ {x:fr.count(x) for x in set(fr)} }")
    print(f"聚合 decode ≈ {sum(cts)/el:.1f} tok/s   聚合含prefill ≈ {(sum(pts)+sum(cts))/el:.1f} tok/s")
if vram:
    for i,name in enumerate(("GPU4","GPU5","GPU6","GPU7")):
        v=[r[i] for r in vram]
        print(f"{name}: 峰值={max(v)} 中位={int(statistics.median(v))} 最小={min(v)} MiB")
print("显存曲线 CSV:", OUT)
