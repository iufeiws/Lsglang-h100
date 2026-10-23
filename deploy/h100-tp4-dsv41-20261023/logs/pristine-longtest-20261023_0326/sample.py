#!/usr/bin/env python3
"""60s-resolution sampler for the pristine long test.

Each tick records: /health status code, harness cumulative ok/fail (by class),
server process liveness, GPU4/5/6/7 memory, cumulative [pool] hit= lines and
cumulative fatal-marker counts in the server log, plus window + cumulative
latency percentiles from the harness JSONL.

Appends one CSV row per tick. Read-only against the server.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
import urllib.error

FATAL_PATTERNS = ["SEGV", "OOM", "Traceback", "CUDA error", "illegal", "core dumped"]

CSV_HEADER = [
    "epoch", "iso", "health", "pid_alive", "harness_alive",
    "done", "ok", "fail", "fail_timeout", "fail_4xx", "fail_5xx",
    "fail_disconnect", "fail_other",
    "gpu4_mib", "gpu5_mib", "gpu6_mib", "gpu7_mib",
    "pool_hits", "fatal_hits", "fatal_new",
    "win_n", "win_p50_ms", "win_p95_ms", "cum_n", "cum_p50_ms", "cum_p95_ms",
    "log_lines",
]

# lines already counted, so "fatal_new" shows new occurrences since previous tick
_SEEN = {"pool_hits": 0, "fatal_hits": 0, "log_lines": 0}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def health(port, timeout=15):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return 1
    except Exception:
        return 0


def gpu_mem(idx):
    out = sh(f"nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sed -n '{idx+1}p'")
    try:
        return int(out.split(",")[1].strip())
    except Exception:
        return -1


def pct(vals, q):
    if not vals:
        return ""
    v = sorted(vals)
    k = (len(v) - 1) * q
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return round(v[f] + (v[c] - v[f]) * (k - f), 1)


def read_jsonl(path):
    ok_lat, ok_t, fail_by = [], [], {}
    done = 0
    footer = None
    if not os.path.exists(path):
        return dict(done=0, ok=0, fail=0, fail_by={}, ok_lat=[], ok_t=[], footer=None)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "run_footer":
                footer = r
                continue
            if r.get("kind"):
                continue
            if r.get("t_end") is None:
                continue
            done += 1
            if r.get("outcome") == "ok":
                ok_lat.append(r["latency_ms"])
                ok_t.append(r["t_end"])
            else:
                fail_by[r.get("outcome")] = fail_by.get(r.get("outcome"), 0) + 1
    return dict(done=done, ok=len(ok_lat), fail=done - len(ok_lat),
                fail_by=fail_by, ok_lat=ok_lat, ok_t=ok_t, footer=footer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--harness-pid", type=int, default=0)
    ap.add_argument("--port", type=int, default=18010)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    new = not os.path.exists(args.csv)
    with open(args.csv, "a") as f:
        if new:
            f.write(",".join(CSV_HEADER) + "\n")
            f.flush()
        for i in range(args.samples):
            t0 = time.time()
            ep = time.time()
            hc = health(args.port)
            alive = pid_alive(args.pid)
            halive = pid_alive(args.harness_pid) if args.harness_pid else -1
            g = [gpu_mem(i) for i in (4, 5, 6, 7)]
            pool_hits = int(sh(f"grep -c '\\[pool\\] hit=' {args.log}") or 0)
            fatal_hits = int(sh(f"grep -ciE '{'|'.join(FATAL_PATTERNS)}' {args.log}") or 0)
            log_lines = int(sh(f"wc -l < {args.log}") or 0)
            st = read_jsonl(args.jsonl)
            fb = st["fail_by"]
            fb_other = sum(v for k, v in fb.items()
                           if k not in ("timeout", "http_4xx", "http_5xx", "disconnect"))
            win = [l for tt, l in zip(st["ok_t"], st["ok_lat"]) if tt >= ep - args.interval]
            row = [
                f"{ep:.0f}", time.strftime("%H:%M:%S", time.localtime(ep)), hc, alive, halive,
                st["done"], st["ok"], st["fail"],
                fb.get("timeout", 0), fb.get("http_4xx", 0), fb.get("http_5xx", 0),
                fb.get("disconnect", 0), fb_other,
                g[0], g[1], g[2], g[3],
                pool_hits, fatal_hits, fatal_hits - _SEEN["fatal_hits"],
                len(win), pct(win, 0.5), pct(win, 0.95),
                len(st["ok_lat"]), pct(st["ok_lat"], 0.5), pct(st["ok_lat"], 0.95),
                log_lines,
            ]
            _SEEN["fatal_hits"] = fatal_hits
            _SEEN["pool_hits"] = pool_hits
            _SEEN["log_lines"] = log_lines
            f.write(",".join(str(x) for x in row) + "\n")
            f.flush()
            print(f"{args.label} {row[1]} health={hc} pid={alive} harn={halive} "
                  f"done={st['done']} ok={st['ok']} fail={st['fail']} fb={fb} "
                  f"gpu={g} poolhit={pool_hits} fatal={fatal_hits}(+{row[18]}) "
                  f"winN={len(win)} win_p50={row[21]} win_p95={row[22]} "
                  f"cum_p50={row[24]} cum_p95={row[25]}", flush=True)
            if st["footer"]:
                print(f"{args.label} HARNESS FINISHED: {json.dumps(st['footer'])}", flush=True)
            sl = args.interval - (time.time() - t0)
            if sl > 0 and i < args.samples - 1:
                time.sleep(sl)


if __name__ == "__main__":
    main()
