#!/usr/bin/env python3
"""run-256-slots 吞吐测量 — 单流长会话 + 多流 (只发请求, 不改服务端)

与 bench_long_session.py 的差别:
  * SERVE_LOG 指向本轮日志 (20261023_044623)
  * GPU/池采样移到后台线程 (每 5 s 一次, 单次 nvidia-smi 覆盖 4 卡),
    每轮只做 HTTP + 外部连接数检查 => 墙钟分母不被客户端开销吃掉
  * 每轮额外记录 ms_per_step = 1000/decode_tok_s 与 epoch 起止
  * 多流聚合用"公共重叠窗口"= [max_i(首轮起), min_i(末轮止)],
    只统计完全落在窗口内的轮次 => 避免收尾空档把聚合值压低

计划语法 --plan "mode:K:target_ctx:budget_s,mode:K:target_ctx:budget_s,..."
  mode = single | multi ; target_ctx=0 表示不按上下文停
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
from datetime import datetime, timedelta, timezone

import requests

CST = timezone(timedelta(hours=8))
BASE = "http://127.0.0.1:18010"
CHAT = f"{BASE}/v1/chat/completions"
MODEL = "DeepSeek-V4.1-Flash-1017"
SERVE_LOG = "/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261023_044623.log"
GPU_IDX = [4, 5, 6, 7]
R1_PROMPT = "写一篇大约 1000 字的小说"
CONT_PROMPT = "继续写下去，保持同样的风格"

TOPICS = ["一位老图书馆员的最后一班岗", "北方港口的冬季商队", "深海观测站的日志",
          "山间铁路的信号员", "考古队与沙暴", "钟表匠与失窃的机芯",
          "云端牧场的雨季", "旧城区的面包房", "极地科考站的发电机组",
          "南方水乡的摆渡人", "天文台的夜间观测", "机械蜂群与花园",
          "荒原上的灯塔守", "茶马古道上的马帮", "地下管网检修队",
          "海边小镇的灯塔"]

HDR = ["stage", "mode", "K", "stream", "round", "ts_start", "ts_end",
       "epoch_start", "epoch_end", "prompt_tokens", "completion_tokens", "ctx_after",
       "ttft_s", "e2e_s", "decode_s", "prefill_tok_s", "decode_tok_s", "ms_per_step",
       "in_tok_s", "finish", "status", "text_len", "err",
       "foreign_sock", "foreign_pids"]


def now() -> str:
    return datetime.now(CST).strftime("%Y-%m-%dT%H:%M:%S%z")


def log(msg: str) -> None:
    print(f"[{datetime.now(CST).strftime('%H:%M:%S')}] {msg}", flush=True)


class FW:
    def __init__(self, f):
        self.f = f
        self.w = csv.writer(f)
        self.lock = threading.Lock()

    def writerow(self, row):
        with self.lock:
            self.w.writerow(row)
            self.f.flush()


def foreign_load():
    """(n, pids): 对端为 :18010 且不属于本进程的 ESTAB 连接 (其他 agent 的压测)."""
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
            continue
        owners = re.findall(r"pid=(\d+)", ln)
        if owners and owners[0] == str(me):
            continue
        n += 1
        pids.update(owners)
    return (n, sorted(pids))


# ---------------------------------------------------------------- sampler thread
class Sampler(threading.Thread):
    """后台每 5 s 采一次 GPU 显存/util 与最新 [pool] 行; 不占用请求路径."""

    def __init__(self, gpu_csv: str, pool_csv: str, period: float = 5.0,
                 serve_log: str = SERVE_LOG):
        super().__init__(daemon=True)
        self.period = period
        self.serve_log = serve_log
        self.stop = threading.Event()
        self.gf = open(gpu_csv, "a")
        self.gw = FW(self.gf)
        self.pf = open(pool_csv, "a")
        self.pw = FW(self.pf)
        gp_hdr = ["ts"] + [f"gpu{i}_mem_mib" for i in GPU_IDX] + [f"gpu{i}_util" for i in GPU_IDX]
        po_hdr = ["ts", "raw"]
        if os.path.getsize(gpu_csv) == 0:
            self.gw.writerow(gp_hdr)
        if os.path.getsize(pool_csv) == 0:
            self.pw.writerow(po_hdr)

    def gpu_line(self):
        try:
            p = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=25)
            mem, uti = {}, {}
            for ln in p.stdout.strip().splitlines():
                a = [x.strip() for x in ln.split(",")]
                if len(a) < 3:
                    continue
                try:
                    i = int(a[0])
                except ValueError:
                    continue
                if i in GPU_IDX:
                    mem[i] = a[1]
                    uti[i] = a[2]
            return [mem.get(i, "") for i in GPU_IDX] + [uti.get(i, "") for i in GPU_IDX]
        except Exception:
            return [""] * 8

    def newest_pool(self):
        try:
            with open(self.serve_log, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 300_000))
                data = f.read().decode("utf-8", "replace")
            for ln in reversed(data.splitlines()):
                if ln.startswith("[pool] dstep="):
                    return ln
        except Exception:
            pass
        return ""

    def run(self):
        while not self.stop.is_set():
            t = now()
            self.gw.writerow([t] + self.gpu_line())
            pl = self.newest_pool()
            if pl:
                self.pw.writerow([t, pl])
            self.stop.wait(self.period)


# ---------------------------------------------------------------- one request
def chat_round(messages, max_tokens, temperature, read_timeout):
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.time()
    ttft = None
    parts = []
    usage = None
    finish = None
    try:
        with requests.post(CHAT, json=payload, stream=True,
                           timeout=(30, read_timeout)) as r:
            if r.status_code != 200:
                return {"status": f"http_{r.status_code}", "err": r.text[:300],
                        "total_s": time.time() - t0, "ttft_s": None,
                        "prompt_tokens": None, "completion_tokens": None,
                        "text": "", "finish": None}
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
                "text": "".join(parts), "finish": finish}
    except requests.exceptions.Timeout as e:
        return {"status": "timeout", "err": f"Timeout: {e}", "total_s": time.time() - t0,
                "ttft_s": ttft, "prompt_tokens": None, "completion_tokens": None,
                "text": "", "finish": None}
    except requests.exceptions.ConnectionError as e:
        return {"status": "conn_err", "err": f"ConnectionError: {str(e)[:200]}",
                "total_s": time.time() - t0, "ttft_s": ttft, "prompt_tokens": None,
                "completion_tokens": None, "text": "", "finish": None}
    except Exception as e:  # noqa: BLE001
        return {"status": "other_err", "err": f"{type(e).__name__}: {e}",
                "total_s": time.time() - t0, "ttft_s": ttft, "prompt_tokens": None,
                "completion_tokens": None, "text": "", "finish": None}


class Session:
    def __init__(self, sid, topic, args):
        self.sid = sid
        self.args = args
        self.messages = [{"role": "user", "content": f"{R1_PROMPT}，题材：{topic}"}]
        self.rnd = 0
        self.ctx = 0
        self.rows = []
        self.errs = []
        self.first_start = None
        self.last_end = None
        self.lock = threading.Lock()

    def one_round(self, w, stage):
        self.rnd += 1
        f0 = foreign_load()
        t_start = time.time()
        res = chat_round(self.messages, self.args.max_tokens, self.args.temperature,
                         self.args.read_timeout)
        t_end = time.time()
        f1 = foreign_load()
        pt = res.get("prompt_tokens")
        ct = res.get("completion_tokens")
        ts = res["ttft_s"]
        tot = res["total_s"]
        dec_s = (tot - ts) if (ts and tot > ts) else None
        decode_tok_s = ((ct - 1) / dec_s) if (ct and ct > 1 and dec_s and dec_s > 0) else None
        if res["status"] == "ok" and pt:
            self.ctx = pt + (ct or 0)
        row = {
            "stage": stage, "mode": self.args.mode, "K": self.args.K,
            "stream": self.sid, "round": self.rnd,
            "ts_start": datetime.fromtimestamp(t_start, CST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "ts_end": datetime.fromtimestamp(t_end, CST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "epoch_start": round(t_start, 3), "epoch_end": round(t_end, 3),
            "prompt_tokens": pt, "completion_tokens": ct,
            "ctx_after": self.ctx if res["status"] == "ok" else "",
            "ttft_s": round(ts, 3) if ts else "",
            "e2e_s": round(tot, 3),
            "decode_s": round(dec_s, 3) if dec_s else "",
            "prefill_tok_s": round(pt / ts, 3) if (pt and ts) else "",
            "decode_tok_s": round(decode_tok_s, 3) if decode_tok_s else "",
            "ms_per_step": round(1000.0 / decode_tok_s, 3) if decode_tok_s else "",
            "in_tok_s": round(pt / tot, 3) if (pt and tot) else "",
            "finish": res.get("finish") or "", "status": res["status"],
            "text_len": len(res.get("text") or ""), "err": res["err"][:200],
            "foreign_sock": max(f0[0], f1[0]),
            "foreign_pids": ";".join(sorted(set(map(str, f0[1] + f1[1])))),
        }
        with self.lock:
            self.rows.append(row)
            if self.first_start is None:
                self.first_start = t_start
            self.last_end = t_end
            w.writerow([row[k] for k in HDR])
            if res["status"] != "ok":
                self.errs.append(f"r{self.rnd}:{res['status']}:{res['err'][:120]}")
        log(f"  [{stage} s{self.sid} r{self.rnd:3d}] pt={pt} ct={ct} ctx={self.ctx} "
            f"ttft={row['ttft_s']} e2e={row['e2e_s']} pref={row['prefill_tok_s']} "
            f"dec={row['decode_tok_s']} ms/step={row['ms_per_step']} "
            f"fin={row['finish']} {row['err'][:70]}")
        reply = res.get("text") or ""
        if reply:
            self.messages.append({"role": "assistant", "content": reply})
            self.messages.append({"role": "user", "content": CONT_PROMPT})
        return res["status"] == "ok"


def pct(vals, q):
    s = sorted(v for v in vals if v is not None and v != "")
    if not s:
        return None
    if len(s) == 1:
        return round(float(s[0]), 3)
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return round(float(s[lo]) + (float(s[hi]) - float(s[lo])) * (pos - lo), 3)


def mean(vals):
    s = [float(v) for v in vals if v is not None and v != ""]
    return round(statistics.mean(s), 3) if s else None


def run_stage(args, w, stage, mode, K, target_ctx, budget_s):
    sess = [Session(i, TOPICS[i % len(TOPICS)], args) for i in range(K)]
    args.mode, args.K = mode, K
    for s in sess:
        s.args = args
    t0 = time.time()
    deadline = t0 + budget_s
    bar = threading.Barrier(K)
    stop = threading.Event()

    def loop(s: Session):
        bar.wait()
        while not stop.is_set() and time.time() < deadline:
            if target_ctx and s.ctx >= target_ctx:
                break
            ok = s.one_round(w, stage)
            if not ok and len(s.errs) >= 3:
                break

    ths = [threading.Thread(target=loop, args=(s,)) for s in sess]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    stop.set()

    rows = [r for s in sess for r in s.rows]
    okrows = [r for r in rows if r["status"] == "ok" and r["completion_tokens"]]
    fails = [r for r in rows if r["status"] != "ok"]

    # 公共重叠窗口: 所有流都已完成首轮起、且至少一条流未收尾
    starts = [s.first_start for s in sess if s.first_start]
    ends = [s.last_end for s in sess if s.last_end]
    w_start = max(starts) if starts else t0
    w_end = min(ends) if ends else time.time()
    win = max(0.0, w_end - w_start)
    inwin = [r for r in okrows
             if r["epoch_start"] >= w_start and r["epoch_end"] <= w_end]
    agg_out = sum(r["completion_tokens"] or 0 for r in inwin)
    agg_in = sum(r["prompt_tokens"] or 0 for r in inwin)
    span = (max(ends) - min(starts)) if (starts and ends) else 0.0
    span_out = sum(r["completion_tokens"] or 0 for r in okrows)

    # 每流活跃口径
    per_stream_active = []
    for s in sess:
        sr = [r for r in s.rows if r["status"] == "ok" and r["completion_tokens"]]
        tot = sum(r["e2e_s"] or 0 for r in sr)
        if tot > 0:
            per_stream_active.append(sum(r["completion_tokens"] or 0 for r in sr) / tot)

    dec_rates = [r["decode_tok_s"] for r in okrows if r["decode_tok_s"] != ""]
    ms_steps = [r["ms_per_step"] for r in okrows if r["ms_per_step"] != ""]
    ctxs = [r["ctx_after"] for r in okrows if r["ctx_after"] != ""]

    summary = {
        "stage": stage, "mode": mode, "K": K,
        "target_ctx": target_ctx, "budget_s": budget_s,
        "wall_common_window_s": round(win, 3),
        "wall_full_span_s": round(span, 3),
        "rounds_total": len(rows), "rounds_ok": len(okrows), "failures": len(fails),
        "failure_kinds": sorted({r["status"] for r in fails}),
        "failure_examples": [f"s{r['stream']}/r{r['round']}:{r['status']}:{r['err'][:150]}"
                             for r in fails][:8],
        "agg_out_tok_s_commonwindow": round(agg_out / win, 3) if win > 0 else None,
        "agg_in_tok_s_commonwindow": round(agg_in / win, 3) if win > 0 else None,
        "agg_out_tok_s_fullspan": round(span_out / span, 3) if span > 0 else None,
        "rounds_in_common_window": len(inwin),
        "tokens_out_in_common_window": agg_out,
        "tokens_in_in_common_window": agg_in,
        "per_stream_out_tok_s_mean": round(statistics.mean(dec_rates), 3) if dec_rates else None,
        "per_stream_out_tok_s_p50": pct(dec_rates, 50),
        "per_stream_ms_per_step_mean": round(statistics.mean(ms_steps), 3) if ms_steps else None,
        "per_stream_ms_per_step_p50": pct(ms_steps, 50),
        "per_stream_active_out_tok_s_mean": round(statistics.mean(per_stream_active), 3)
                                            if per_stream_active else None,
        "ttft_s": {"p50": pct([r["ttft_s"] for r in okrows if r["ttft_s"] != ""], 50),
                   "p95": pct([r["ttft_s"] for r in okrows if r["ttft_s"] != ""], 95)},
        "ctx_max": max(ctxs) if ctxs else 0,
        "ctx_min": min(ctxs) if ctxs else 0,
        "ctx_mean": mean(ctxs),
        "foreign_sock_max": max([r["foreign_sock"] for r in rows], default=0),
        "clean_round_share_pct": round(100.0 * sum(1 for r in rows if r["foreign_sock"] == 0)
                                       / len(rows), 2) if rows else None,
        "started": datetime.fromtimestamp(w_start, CST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ended": datetime.fromtimestamp(w_end, CST).strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--read-timeout", type=float, default=2400)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--prefix", default="bench-256")
    ap.add_argument("--serve-log", default=SERVE_LOG)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    args.mode = "multi"
    args.K = 1

    csv_path = os.path.join(args.outdir, f"{args.prefix}-{args.ts}.csv")
    gpu_csv = os.path.join(args.outdir, f"{args.prefix}-{args.ts}.gpu.csv")
    pool_csv = os.path.join(args.outdir, f"{args.prefix}-{args.ts}.pool.csv")
    sum_path = os.path.join(args.outdir, f"{args.prefix}-{args.ts}.summary.json")

    cf = open(csv_path, "a")
    w = FW(cf)
    if os.path.getsize(csv_path) == 0:
        w.writerow(HDR)

    samp = Sampler(gpu_csv, pool_csv, 5.0, args.serve_log)
    samp.start()

    log(f"model={MODEL} plan={args.plan} serve_log={args.serve_log}")
    log(f"foreign_conns_at_start={foreign_load()}")

    plan = []
    for spec in args.plan.split(","):
        mode, k, tc, bs = spec.split(":")
        plan.append((mode, int(k), int(tc), float(bs)))

    summaries = []
    for mode, K, tc, bs in plan:
        stage = f"{mode}-K{K}" + (f"-ctx{tc}" if tc else "")
        log(f"=== STAGE {stage} budget={bs}s target_ctx={tc or '-'} "
            f"foreign={foreign_load()[0]} ===")
        s = run_stage(args, w, stage, mode, K, tc, bs)
        summaries.append(s)
        log(f"SUMMARY {stage}: {json.dumps(s, ensure_ascii=False)}")
        json.dump(summaries, open(sum_path, "w"), ensure_ascii=False, indent=2)
        log(f"  -> sleep 20s 让服务端 dstep 落一行")
        time.sleep(20)

    samp.stop.set()
    samp.join(timeout=10)
    cf.close()
    log(f"CSV={csv_path}")
    log(f"SUMMARY={sum_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
