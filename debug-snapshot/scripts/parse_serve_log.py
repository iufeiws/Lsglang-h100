#!/usr/bin/env python3
"""从 sglang serve 日志抽取可比对的运行期指标 (只读, 不改服务端)。

抽取:
  * accept len / accept rate  (Decode batch 行, TP0)
  * gen throughput (tok/s)    (Decode batch 行尾 gen throughput)
  * #running-req / full token usage
  * [pool] win hit / psl / link / gather / copyiss
  * 启动期里程碑: [pool] slab, max_total_num_tokens, available, prewarm-skip, attn-prewarm done, capture elapsed
"""
from __future__ import annotations
import argparse, json, re, statistics, sys
from datetime import datetime

# 注意: 部分行前缀被截断 (缺少 "]"), 因此 TP 标签不要求前面有 "["
RE_TP = re.compile(r"(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP(?P<tp>\d+)")
RE_DECODE = re.compile(
    r"Decode batch, #running-req: (?P<run>\d+), "
    r"#full token: (?P<full>\d+), full token usage: (?P<fu>[\d.]+), #swa token: (?P<swa>\d+), "
    r"swa token usage: (?P<su>[\d.]+), accept len: (?P<alen>[\d.]+), accept rate: (?P<arate>[\d.]+)"
    r"(?:.*?gen throughput \(token/s\): (?P<thr>[\d.]+))?")
RE_POOL = re.compile(r"dstep=(?P<dstep>\d+) win hit=(?P<hit>[\d.]+)% nr=(?P<nr>[\d.]+)% psl=(?P<psl>[\d.]+) "
                     r"link=(?P<link>[\d.]+)GB/s gather=(?P<gather>[\d.]+)us copyiss=(?P<copyiss>[\d.]+)us")


def pct(v, p):
    v = sorted(v)
    if not v: return None
    k = max(0, min(len(v) - 1, int(round((p / 100) * (len(v) - 1)))))
    return v[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("--json-out", default=None)
    ap.add_argument("--tp", type=int, default=0, help="只统计该 TP rank 的 Decode 行")
    a = ap.parse_args()
    dec, pool, milestones = [], [], []
    with open(a.log, errors="replace") as f:
        for line in f:
            if "Decode batch" in line:
                m = RE_DECODE.search(line)
                mt = RE_TP.search(line)
                if m and mt and int(mt.group("tp")) == a.tp:
                    dec.append(dict(run=int(m.group("run")), alen=float(m.group("alen")),
                                    arate=float(m.group("arate")),
                                    thr=(float(m.group("thr")) if m.group("thr") else None)))
            if "[pool]" in line and "win hit=" in line:
                m = RE_POOL.search(line)
                if m:
                    pool.append({k: float(v) for k, v in m.groupdict().items()})
            for key in ("[pool] slab:", "max_total_num_tokens=", "available_bytes",
                        "[prewarm-skip]", "attn-prewarm", "Capture decode", "Capture prefill",
                        "engram host table", "Initialized lk_moe with", "Max total num tokens",
                        "hicache_attached", "max_total_num_tokens"):
                if key in line and len(milestones) < 400:
                    milestones.append(line.strip()[:200]); break
    out = dict(log=a.log, tp=a.tp, n_decode_lines=len(dec),
               accept_len=dict(p10=pct([d["alen"] for d in dec], 10), p50=pct([d["alen"] for d in dec], 50),
                               p90=pct([d["alen"] for d in dec], 90),
                               mean=(statistics.fmean([d["alen"] for d in dec]) if dec else None)),
               accept_rate=dict(p50=pct([d["arate"] for d in dec], 50)),
               gen_throughput_tok_s=dict(p50=pct([d["thr"] for d in dec if d["thr"]], 50),
                                         mean=(statistics.fmean([d["thr"] for d in dec if d["thr"]])
                                               if any(d["thr"] for d in dec) else None),
                                         n=len([d for d in dec if d["thr"]])),
               running_req=dict(p50=pct([d["run"] for d in dec], 50)),
               pool=dict(n=len(pool),
                         win_hit_p50=(pct([p["hit"] for p in pool], 50) if pool else None),
                         copyiss_us_p50=(pct([p["copyiss"] for p in pool], 50) if pool else None),
                         link_gbps_p50=(pct([p["link"] for p in pool], 50) if pool else None),
                         gather_us_p50=(pct([p["gather"] for p in pool], 50) if pool else None)),
               milestones=milestones)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json_out:
        with open(a.json_out, "w") as f: json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
