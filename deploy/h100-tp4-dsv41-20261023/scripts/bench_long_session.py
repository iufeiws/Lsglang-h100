#!/usr/bin/env python3
"""长会话续写基准 — 单流 / 多流（只发请求，不改服务端、不重启服务）

模式
  kprobe : 并发 K=1,2,4,8,16，每档发 K 条固定负载(输入 1024 / 输出 64)，测每流与聚合 decode tok/s
  single : 一条会话持续续写（第1轮要小说，之后把上一轮回复追加进 messages 继续写），
           直到累计上下文 >= --target-ctx 或达到 --budget-s 墙钟预算
  multi  : K 条会话并行续写，固定墙钟预算

记录每轮：轮次 / prompt_tokens / completion_tokens / 累计上下文 / TTFT / 总耗时 /
         输出 tok/s = completion_tokens/(总耗时-TTFT) / 显存 / 池命中率
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta

import requests

CST = timezone(timedelta(hours=8))
BASE = "http://127.0.0.1:18010"
CHAT = f"{BASE}/v1/chat/completions"
GEN = f"{BASE}/generate"
MODEL = "DeepSeek-V4.1-Flash-1017"
SERVE_LOG = "/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261023_025933.log"
GPU_UUIDS = [
    "GPU-7561b28f-b049-bbdb-31f8-16efeb06663d",
    "GPU-fb59db94-0e4e-4c4f-8ae5-4d0fdd1e0f0b",
    "GPU-3c5d7267-445c-4679-bce9-503bedd0c920",
    "GPU-3c0f2856-a6e3-b9d5-fd10-a08f55579fff",
]
R1_PROMPT = "写一篇大约 1000 字的小说"
CONT_PROMPT = "继续写下去，保持同样的风格"
_POOL_RE = re.compile(r"win hit=([\d.]+)%")
_POOL_CUM = re.compile(r"cum hit=([\d.]+)%")


class FW:
    """csv writer that flushes every row (long runs must survive a kill)."""

    def __init__(self, f):
        self.f = f
        self.w = csv.writer(f)

    def writerow(self, row):
        self.w.writerow(row)
        self.f.flush()


def now() -> str:
    return datetime.now(CST).strftime("%Y-%m-%dT%H:%M:%S%z")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return round(s[0], 3)
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos); hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (pos - lo), 3)


def gpu_mem() -> list:
    out = []
    for u in GPU_UUIDS:
        try:
            p = subprocess.run(["nvidia-smi", f"--id={u}", "--query-gpu=memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=30)
            a = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")]
            out.append((int(a[0]), int(a[1])))
        except Exception:
            out.append((None, None))
    return out


def pool_hit() -> tuple:
    """(win_hit_pct, cum_hit_pct) from the newest [pool] line in the serve log."""
    try:
        with open(SERVE_LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            data = f.read().decode("utf-8", "replace")
        lines = [l for l in data.splitlines() if l.startswith("[pool]")]
        if not lines:
            return (None, None)
        last = lines[-1]
        w = _POOL_RE.search(last); c = _POOL_CUM.search(last)
        return (float(w.group(1)) if w else None, float(c.group(1)) if c else None)
    except Exception:
        return (None, None)


def foreign_load() -> tuple:
    """(n_client_sockets, sorted foreign_pids) of established conns TO :18010 not from me.

    These are other agents' load generators. Round is only 'clean' when this is 0.
    """
    try:
        p = subprocess.run(["ss", "-tnp"], capture_output=True, text=True, timeout=20)
    except Exception:
        return (-1, [])
    me = os.getpid()
    n = 0
    pids = set()
    for ln in p.stdout.splitlines():
        if "ESTAB" not in ln:
            continue
        parts = ln.split()
        if len(parts) < 5 or not parts[4].endswith(":18010"):
            continue          # parts[4] = peer address -> client-side socket
        owners = re.findall(r"pid=(\d+)", ln)
        if owners and owners[0] == str(me):
            continue
        n += 1
        pids.update(owners)
    return (n, sorted(pids))


def wait_clean(quiet_s: float, max_wait_s: float, tag: str) -> bool:
    """Return True when it is safe to issue a request.

    Fast path: no foreign load right now -> go immediately.  `quiet_s` is only
    enforced when *resuming after* interference was seen; otherwise the gate
    would idle for quiet_s before every single round and deflate every
    wall-clock aggregate.
    """
    n, pids = foreign_load()
    if n == 0:
        return True
    log(f"  [{tag}] 检测到外部压测并发 {n} 条 (pids={pids})，暂停发请求…")
    t0 = time.time()
    clean_since = None
    while time.time() - t0 < max_wait_s:
        n, pids = foreign_load()
        if n == 0:
            if clean_since is None:
                clean_since = time.time()
            if time.time() - clean_since >= quiet_s:
                log(f"  [{tag}] 外部负载已消失 {quiet_s:.0f}s，恢复")
                return True
        else:
            clean_since = None
        time.sleep(15)
    log(f"  [{tag}] 等待干净窗口超时 {max_wait_s:.0f}s，放弃")
    return False


# ------------------------------------------------------------------ chat round
def chat_round(messages: list, max_tokens: int, temperature: float,
               read_timeout: float) -> dict:
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.time()
    ttft = None
    parts: list[str] = []
    usage = None
    finish = None
    try:
        with requests.post(CHAT, json=payload, stream=True,
                           timeout=(30, read_timeout)) as r:
            if r.status_code != 200:
                return {"status": f"http_{r.status_code}", "err": r.text[:300],
                        "total_s": time.time() - t0, "ttft_s": None,
                        "prompt_tokens": None, "completion_tokens": None,
                        "text": "", "finish": None, "reasoning": False}
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                body = raw[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    d = json.loads(body)
                except Exception:
                    continue
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices") or []:
                    dl = ch.get("delta") or {}
                    piece = dl.get("content") or dl.get("reasoning_content")
                    if piece:
                        if ttft is None:
                            ttft = time.time() - t0
                        parts.append(piece)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        total = time.time() - t0
        pt = (usage or {}).get("prompt_tokens")
        ct = (usage or {}).get("completion_tokens")
        if ct is None:
            ct = len(parts)
        return {"status": "ok", "err": "", "total_s": total, "ttft_s": ttft,
                "prompt_tokens": pt, "completion_tokens": ct,
                "text": "".join(parts), "finish": finish, "reasoning": False}
    except requests.exceptions.Timeout as e:
        return {"status": "timeout", "err": f"Timeout: {e}", "total_s": time.time() - t0,
                "ttft_s": ttft, "prompt_tokens": None, "completion_tokens": None,
                "text": "", "finish": None, "reasoning": False}
    except requests.exceptions.ConnectionError as e:
        return {"status": "conn_err", "err": f"ConnectionError: {str(e)[:200]}",
                "total_s": time.time() - t0, "ttft_s": ttft, "prompt_tokens": None,
                "completion_tokens": None, "text": "", "finish": None, "reasoning": False}
    except Exception as e:  # noqa: BLE001
        return {"status": "other_err", "err": f"{type(e).__name__}: {e}",
                "total_s": time.time() - t0, "ttft_s": ttft, "prompt_tokens": None,
                "completion_tokens": None, "text": "", "finish": None, "reasoning": False}


# ------------------------------------------------------------------ k probe
def gen_round(input_len: int, out_len: int, seed: int, read_timeout: float) -> dict:
    import random
    rng = random.Random(seed)
    ids = [rng.randrange(1000, 60000) for _ in range(input_len)]
    payload = {"input_ids": ids, "stream": True,
               "sampling_params": {"max_new_tokens": out_len, "temperature": 0.0,
                                   "ignore_eos": True}}
    t0 = time.time()
    ttft = None
    meta = None
    try:
        with requests.post(GEN, json=payload, stream=True,
                           timeout=(30, read_timeout)) as r:
            if r.status_code != 200:
                return {"status": f"http_{r.status_code}", "err": r.text[:300],
                        "total_s": time.time() - t0, "ttft_s": None, "ct": None}
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                b = raw[5:].strip()
                if b == "[DONE]":
                    break
                if ttft is None:
                    ttft = time.time() - t0
                try:
                    meta = json.loads(b).get("meta_info") or meta
                except Exception:
                    pass
        return {"status": "ok", "err": "", "total_s": time.time() - t0, "ttft_s": ttft,
                "ct": (meta or {}).get("completion_tokens"),
                "pt": (meta or {}).get("prompt_tokens")}
    except Exception as e:  # noqa: BLE001
        kind = ("timeout" if isinstance(e, requests.exceptions.Timeout)
                else "conn_err" if isinstance(e, requests.exceptions.ConnectionError)
                else "other_err")
        return {"status": kind, "err": f"{type(e).__name__}: {str(e)[:200]}",
                "total_s": time.time() - t0, "ttft_s": ttft, "ct": None}


def mode_kprobe(a, w) -> None:
    out = []
    for k in [int(x) for x in a.k_list.split(",")]:
        res: list = [None] * k
        bar = threading.Barrier(k)

        def run(i):
            bar.wait()
            res[i] = gen_round(a.input_len, 64, 5000 + k * 100 + i, a.read_timeout)

        t0 = time.time()
        ths = [threading.Thread(target=run, args=(i,)) for i in range(k)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.time() - t0
        ok = [x for x in res if x and x["status"] == "ok" and x.get("ct")]
        fail = [x for x in res if not (x and x["status"] == "ok" and x.get("ct"))]
        agg_out = sum(x["ct"] for x in ok)
        dec_rates = [x["ct"] / (x["total_s"] - x["ttft_s"]) for x in ok
                     if x["ttft_s"] and x["total_s"] > x["ttft_s"]]
        row = {"K": k, "ok": len(ok), "fail": len(fail), "wall_s": round(wall, 2),
               "agg_out_tok_s": round(agg_out / wall, 3) if wall else None,
               "agg_in_tok_s": round(sum(x.get("pt") or 0 for x in ok) / wall, 2) if wall else None,
               "per_stream_out_tok_s_mean": round(statistics.mean(dec_rates), 3) if dec_rates else None,
               "per_stream_out_tok_s_list": [round(x, 3) for x in dec_rates],
               "ttft_p50_s": pct([x["ttft_s"] for x in ok], 50),
               "ttft_p95_s": pct([x["ttft_s"] for x in ok], 95),
               "errs": [x["err"][:120] for x in fail]}
        out.append(row)
        log(f"K={k:2d} ok={len(ok)}/{k} wall={wall:.1f}s agg_out={row['agg_out_tok_s']} tok/s "
            f"per_stream={row['per_stream_out_tok_s_mean']} tok/s (n={len(ok)}) "
            f"ttft_p50={row['ttft_p50_s']}")
    json.dump(out, open(os.path.join(a.outdir, f"kprobe_{a.ts}.json"), "w"),
              ensure_ascii=False, indent=2)
    w.writerow(["=== KPROBE (input_ids 1024, output 64, ignore_eos, temp=0) ==="])
    for r in out:
        w.writerow([json.dumps(r, ensure_ascii=False)])
    w.writerow([])


# ------------------------------------------------------------------ sessions
class Session:
    def __init__(self, sid, topic, args):
        self.sid = sid
        self.args = args
        self.messages = [{"role": "user",
                          "content": f"{R1_PROMPT}，题材：{topic}"}]
        self.rnd = 0
        self.ctx = 0
        self.gate_failed = False
        self.lock = threading.Lock()
        self.rows = []
        self.errs = []

    def one_round(self, w, tag) -> bool:
        if self.args.gate and not wait_clean(self.args.gate_quiet_s,
                                            self.args.gate_max_wait_s, tag):
            self.gate_failed = True
            return False
        self.rnd += 1
        f0 = foreign_load()
        res = chat_round(self.messages, self.args.max_tokens, self.args.temperature,
                         self.args.read_timeout)
        f1 = foreign_load()
        gm = gpu_mem()
        winh, cumh = pool_hit()
        pt = res.get("prompt_tokens"); ct = res.get("completion_tokens")
        if res["status"] == "ok" and pt:
            self.ctx = pt + (ct or 0)
        row = {
            "tag": tag, "mode": self.args.mode, "K": self.args.concurrency,
            "stream": self.sid, "round": self.rnd, "ts": now(),
            "prompt_tokens": pt, "completion_tokens": ct,
            "ctx_after": self.ctx if res["status"] == "ok" else "",
            "ttft_s": round(res["ttft_s"], 3) if res["ttft_s"] else "",
            "prefill_tok_s": round(pt / res["ttft_s"], 3)
                            if (pt and res["ttft_s"]) else "",
            "total_s": round(res["total_s"], 3),
            "out_tok_s": round(ct / (res["total_s"] - res["ttft_s"]), 3)
                         if (ct and res["ttft_s"] and res["total_s"] > res["ttft_s"]) else "",
            "decode_tok_s": round((ct - 1) / (res["total_s"] - res["ttft_s"]), 3)
                           if (ct and ct > 1 and res["ttft_s"]
                               and res["total_s"] > res["ttft_s"]) else "",
            "decode_s": round(res["total_s"] - res["ttft_s"], 3)
                        if (res["ttft_s"] and res["total_s"] > res["ttft_s"]) else "",
            "in_tok_s": round(pt / res["total_s"], 2) if pt and res["total_s"] else "",
            "finish": res.get("finish") or "", "status": res["status"],
            "gpu4": gm[0][0], "gpu5": gm[1][0], "gpu6": gm[2][0], "gpu7": gm[3][0],
            "gpu_util": "/".join(str(x[1]) for x in gm),
            "win_hit_pct": winh if winh is not None else "",
            "cum_hit_pct": cumh if cumh is not None else "",
            "text_len": len(res.get("text") or ""), "err": res["err"][:200],
            "foreign_sock": max(f0[0], f1[0]),
            "foreign_pids": ";".join(sorted(set(map(str, f0[1] + f1[1])))),
        }
        with self.lock:
            self.rows.append(row)
            w.writerow(list(row.values()))
            if res["status"] != "ok":
                self.errs.append(f"r{self.rnd}:{res['status']}:{res['err'][:120]}")
        log(f"  [{tag} s{self.sid} r{self.rnd:3d}] pt={pt} ct={ct} ctx={self.ctx} "
            f"ttft={row['ttft_s']} tot={row['total_s']} "
            f"pref={row['prefill_tok_s']} out_tok/s={row['out_tok_s']} "
            f"dec={row['decode_tok_s']} fin={row['finish']} {row['err'][:80]}")
        # 追加本轮回复，形成"续写"会话
        reply = res.get("text") or ""
        if reply:
            self.messages.append({"role": "assistant", "content": reply})
            self.messages.append({"role": "user", "content": CONT_PROMPT})
        return res["status"] == "ok"


def run_sessions(args, w, topics, budget_s, tag, stop_ctx=None):
    sess = [Session(i, topics[i % len(topics)], args) for i in range(args.concurrency)]
    t0 = time.time()
    stop = threading.Event()
    deadline = t0 + budget_s

    def loop(s: Session):
        while not stop.is_set() and time.time() < deadline:
            if stop_ctx and s.ctx >= stop_ctx:
                break
            ok = s.one_round(w, tag)
            if not ok:
                if s.gate_failed:
                    break
                time.sleep(2)
                if len(s.errs) >= 3:
                    break

    ths = [threading.Thread(target=loop, args=(s,)) for s in sess]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    stop.set()
    wall = time.time() - t0

    rows = [r for s in sess for r in s.rows]
    okrows = [r for r in rows if r["status"] == "ok" and r["completion_tokens"]]
    fails = [r for r in rows if r["status"] != "ok"]
    agg_out = sum(r["completion_tokens"] or 0 for r in okrows)
    agg_in = sum(r["prompt_tokens"] or 0 for r in okrows)
    ctx_final = {s.sid: s.ctx for s in sess}
    summary = {
        "tag": tag, "mode": args.mode, "K": args.concurrency, "wall_s": round(wall, 1),
        "rounds_total": len(rows), "rounds_ok": len(okrows), "failures": len(fails),
        "failure_kinds": sorted({f"{r['status']}" for r in fails}),
        "failure_examples": [f"{r['stream']}/r{r['round']}:{r['status']}:{r['err'][:150]}"
                             for r in fails][:8],
        "agg_out_tok_s": round(agg_out / wall, 3) if wall else None,
        "agg_in_tok_s": round(agg_in / wall, 3) if wall else None,
        "per_stream_out_tok_s_mean": round(statistics.mean(
            [r["out_tok_s"] for r in okrows if r["out_tok_s"] != ""]), 3)
            if [r for r in okrows if r["out_tok_s"] != ""] else None,
        "per_stream_out_tok_s_p50": pct([r["out_tok_s"] for r in okrows if r["out_tok_s"] != ""], 50),
        "ttft_s": {"p50": pct([r["ttft_s"] for r in okrows if r["ttft_s"] != ""], 50),
                   "p95": pct([r["ttft_s"] for r in okrows if r["ttft_s"] != ""], 95),
                   "p99": pct([r["ttft_s"] for r in okrows if r["ttft_s"] != ""], 99),
                   "min": min([r["ttft_s"] for r in okrows if r["ttft_s"] != ""], default=None),
                   "max": max([r["ttft_s"] for r in okrows if r["ttft_s"] != ""], default=None)},
        "ctx_final_per_stream": ctx_final,
        "ctx_final_max": max(ctx_final.values()) if ctx_final else 0,
        "ctx_final_min": min(ctx_final.values()) if ctx_final else 0,
        "total_completion_tokens": agg_out,
        "gate_failed": any(s.gate_failed for s in sess),
        "foreign_sock_max": max([r["foreign_sock"] for r in rows], default=0),
        "clean_round_share_pct": round(100.0 * sum(1 for r in rows if r["foreign_sock"] == 0)
                                      / len(rows), 2) if rows else None,
    }
    return summary


TOPICS = ["一位老图书馆员的最后一班岗", "北方港口的冬季商队", "深海观测站的日志",
          "山间铁路的信号员", "考古队与沙暴", "钟表匠与失窃的机芯",
          "云端牧场的雨季", "旧城区的面包房", "极地科考站的发电机组",
          "南方水乡的摆渡人", "天文台的夜间观测", "机械蜂群与花园",
          "荒原上的灯塔守", "茶马古道上的马帮", "地下管网检修队",
          "海边小镇的灯塔"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["kprobe", "single", "multi"])
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--k-list", default="1,2,4,8,16")
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--target-ctx", type=int, default=100000)
    ap.add_argument("--budget-s", type=float, default=1500)
    ap.add_argument("--read-timeout", type=float, default=1800)
    ap.add_argument("--gen-mem-every", type=int, default=1)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gate", action="store_true",
                    help="每轮前等待无外部压测并发（保证数字干净）")
    ap.add_argument("--gate-quiet-s", type=float, default=30.0)
    ap.add_argument("--gate-max-wait-s", type=float, default=900.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    csv_path = os.path.join(args.outdir, f"longsession-{args.ts}.csv")
    txt_path = os.path.join(args.outdir, f"bench-longsession-{args.ts}.txt")
    hdr = ["tag", "mode", "K", "stream", "round", "ts", "prompt_tokens", "completion_tokens",
           "ctx_after", "ttft_s", "prefill_tok_s", "total_s", "out_tok_s", "decode_tok_s",
           "decode_s", "in_tok_s", "finish", "status",
           "gpu4", "gpu5", "gpu6", "gpu7", "gpu_util", "win_hit_pct", "cum_hit_pct",
           "text_len", "err", "foreign_sock", "foreign_pids"]
    cf = open(csv_path, "a")
    w = FW(cf)
    if os.path.getsize(csv_path) == 0:
        w.writerow(hdr)
    tf = open(txt_path, "a")
    tf.write(f"\n===== long-session bench start {now()} mode={args.mode} "
             f"K={args.concurrency} max_tokens={args.max_tokens} "
             f"model={MODEL} =====\n")
    tf.flush()

    log(f"model={MODEL} mode={args.mode} K={args.concurrency} "
        f"target_ctx={args.target_ctx} budget={args.budget_s}s")
    log(f"gpu_mem_start={gpu_mem()} pool_hit={pool_hit()}")

    if args.mode == "kprobe":
        mode_kprobe(args, w)
    else:
        topics = TOPICS
        if args.mode == "single":
            s = run_sessions(args, w, topics, args.budget_s,
                             "single", stop_ctx=args.target_ctx)
        else:
            s = run_sessions(args, w, topics, args.budget_s,
                             f"K{args.concurrency}")
        log(f"SUMMARY {args.mode} K={args.concurrency}: {json.dumps(s, ensure_ascii=False)}")
        tf.write(f"--- summary {json.dumps(s, ensure_ascii=False)}\n")
        json.dump(s, open(os.path.join(args.outdir,
                                       f"summary-{args.mode}-K{args.concurrency}-{args.ts}.json"), "w"),
                  ensure_ascii=False, indent=2)
    log(f"gpu_mem_end={gpu_mem()}")
    cf.close(); tf.close()
    log(f"CSV: {csv_path}")
    log(f"TXT: {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
