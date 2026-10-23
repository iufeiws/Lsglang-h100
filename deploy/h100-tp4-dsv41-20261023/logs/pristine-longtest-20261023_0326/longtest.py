#!/usr/bin/env python3
"""pristine long-test harness.

Sends fixed-input requests to the live sglang server at a fixed concurrency for a
fixed wall-clock duration, and appends one JSON line per request to a JSONL file.

Fixed input : input_ids_1024.json (exactly 1024 tokens, frozen)
Output      : max_new_tokens=128, ignore_eos=true (fixed 128 generated tokens)
Sampling    : temperature=0.0
Concurrency : 16 (async workers, one in-flight request each)

Only issues requests. Never touches the server process or its configuration.
"""
import argparse
import asyncio
import json
import os
import signal
import sys
import time

import aiohttp

STOP = False


def _handle_signal(signum, frame):
    global STOP
    STOP = True


async def one_request(session, base_url, model, input_ids, max_new_tokens,
                      timeout_s, seq, worker, lock, fh):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }
    rec = {
        "seq": seq,
        "worker": worker,
        "t_start": time.time(),
        "t_end": None,
        "latency_ms": None,
        "status": None,
        "outcome": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "finish_reason": None,
    }
    try:
        async with session.post(base_url + "/generate", json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            rec["status"] = resp.status
            body = await resp.text()
            if resp.status == 200:
                try:
                    j = json.loads(body)
                    mi = j.get("meta_info", {}) or {}
                    rec["prompt_tokens"] = mi.get("prompt_tokens")
                    rec["completion_tokens"] = mi.get("completion_tokens")
                    rec["finish_reason"] = (mi.get("finish_reason") or {}).get("type") \
                        if isinstance(mi.get("finish_reason"), dict) else mi.get("finish_reason")
                    rec["outcome"] = "ok"
                except Exception as e:
                    rec["outcome"] = "bad_json"
                    rec["error"] = repr(e)[:200]
            elif 400 <= resp.status < 500:
                rec["outcome"] = "http_4xx"
                rec["error"] = body[:300]
            elif 500 <= resp.status < 600:
                rec["outcome"] = "http_5xx"
                rec["error"] = body[:300]
            else:
                rec["outcome"] = "http_other"
                rec["error"] = body[:300]
    except asyncio.TimeoutError:
        rec["outcome"] = "timeout"
    except aiohttp.ServerDisconnectedError as e:
        rec["outcome"] = "disconnect"
        rec["error"] = repr(e)[:200]
    except aiohttp.ClientConnectionError as e:
        rec["outcome"] = "disconnect"
        rec["error"] = repr(e)[:200]
    except aiohttp.ClientError as e:
        rec["outcome"] = "client_error"
        rec["error"] = repr(e)[:200]
    except Exception as e:
        rec["outcome"] = "exception"
        rec["error"] = repr(e)[:200]

    rec["t_end"] = time.time()
    rec["latency_ms"] = round((rec["t_end"] - rec["t_start"]) * 1000.0, 3)
    async with lock:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
    return rec


async def worker_loop(wid, session, args, input_ids, seq_counter, fh, lock,
                      deadline, state):
    while not STOP and time.time() < deadline:
        async with lock:
            seq = seq_counter[0]
            seq_counter[0] += 1
        rec = await one_request(session, args.base_url, args.model, input_ids,
                                args.max_new_tokens, args.timeout_s, seq, wid, lock, fh)
        async with lock:
            state["done"] += 1
            if rec["outcome"] == "ok":
                state["ok"] += 1
                state["last_ok"] = time.time()
            else:
                state["fail"] += 1
                state["fail_by"][rec["outcome"]] = state["fail_by"].get(rec["outcome"], 0) + 1
        # back off only on connection-level failures so we do not spin on a dead server
        if rec["outcome"] in ("disconnect", "client_error"):
            await asyncio.sleep(1.0)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18010")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--input-ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--duration-s", type=float, default=3900)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--timeout-s", type=float, default=900)
    ap.add_argument("--calibrate", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=0)
    args = ap.parse_args()

    with open(args.input_ids) as f:
        input_ids = json.load(f)
    assert isinstance(input_ids, list) and len(input_ids) == 1024, len(input_ids)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    conn = aiohttp.TCPConnector(limit=args.concurrency + 4)
    async with aiohttp.ClientSession(connector=conn) as session:
        if args.calibrate:
            print(f"[calibrate] {args.calibrate} sequential request(s), "
                  f"in={len(input_ids)} out={args.max_new_tokens}")
            for i in range(args.calibrate):
                rec = await one_request(session, args.base_url, args.model, input_ids,
                                        args.max_new_tokens, args.timeout_s, i, -1,
                                        asyncio.Lock(), open(os.devnull, "w"))
                print(f"  #{i} outcome={rec['outcome']} status={rec['status']} "
                      f"latency={rec['latency_ms']/1000.0:.2f}s "
                      f"prompt_tokens={rec['prompt_tokens']} "
                      f"completion_tokens={rec['completion_tokens']} "
                      f"finish={rec['finish_reason']} err={rec.get('error','')[:120]}")
            return

        if args.warmup:
            print(f"[warmup] {args.warmup} request(s) at concurrency {args.concurrency}")
            t = [asyncio.create_task(one_request(session, args.base_url, args.model, input_ids,
                                                 args.max_new_tokens, args.timeout_s, i, -1,
                                                 asyncio.Lock(), open(os.devnull, "w")))
                 for i in range(args.warmup)]
            for r in await asyncio.gather(*t):
                print(f"  warmup outcome={r['outcome']} status={r['status']} "
                      f"latency={r['latency_ms']/1000.0:.2f}s")

        start = time.time()
        deadline = start + args.duration_s
        state = {"done": 0, "ok": 0, "fail": 0, "fail_by": {}, "last_ok": start,
                 "start": start, "deadline": deadline}
        lock = asyncio.Lock()
        seq_counter = [0]
        with open(args.out, "a") as fh:
            hdr = {
                "kind": "run_header", "t_start": start,
                "duration_s": args.duration_s, "concurrency": args.concurrency,
                "input_tokens": len(input_ids), "max_new_tokens": args.max_new_tokens,
                "base_url": args.base_url, "pid_of_harness": os.getpid(),
            }
            fh.write(json.dumps(hdr) + "\n")
            fh.flush()
            tasks = [asyncio.create_task(worker_loop(w, session, args, input_ids, seq_counter,
                                                     fh, lock, deadline, state))
                     for w in range(args.concurrency)]
            print(f"[run] concurrency={args.concurrency} duration={args.duration_s}s "
                  f"deadline={time.strftime('%H:%M:%S', time.localtime(deadline))} "
                  f"out={args.out}", flush=True)
            while not STOP and time.time() < deadline:
                await asyncio.sleep(30)
                el = time.time() - start
                print(f"[run] t+{el:7.0f}s done={state['done']} ok={state['ok']} "
                      f"fail={state['fail']} fail_by={state['fail_by']}", flush=True)
                # stop early if the server has been unreachable for 3 consecutive minutes
                if time.time() - state["last_ok"] > 180 and state["ok"] > 0:
                    print("[run] no success for >180s -> stopping early", flush=True)
                    state["stopped_early"] = "no_success_180s"
                    break
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            hdr2 = {
                "kind": "run_footer", "t_end": time.time(),
                "elapsed_s": round(time.time() - start, 3),
                "done": state["done"], "ok": state["ok"], "fail": state["fail"],
                "fail_by": state["fail_by"],
                "stopped_early": state.get("stopped_early"),
            }
            fh.write(json.dumps(hdr2) + "\n")
            fh.flush()
            print("[run] footer " + json.dumps(hdr2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
