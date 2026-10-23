#!/usr/bin/env python3
"""60s-resolution sampler for the pristine long test (v2, corrected patterns).

Corrections vs v1:
  * fatal markers are now precise. The v1 pattern "OOM" matched the benign pool
    counter `oom=0` present in every `[pool] dstep=` line, so v1's fatal_hits
    was effectively tracking the pool-line count. v1's `[pool] hit=` pattern also
    missed the real pool lines, which are shaped `[pool] dstep=... win hit=...`.
  * pool stats are now parsed from the latest `[pool] dstep=` line.

Read-only against the server.
"""
import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
import urllib.error

# precise fatal markers (no bare "OOM": pool lines legitimately carry `oom=0`)
FATAL_RE = (r"Traceback \(most recent call last\)|CUDA error|core dumped|"
            r"illegal memory access|illegal instruction|illegal address|"
            r"out of memory|CUDA out of memory|RuntimeError|AssertionError|"
            r"Fatal Python error|SIGSEGV|SIGABRT|SIGBUS|watchdog timeout|"
            r"nccl.*error|NCCL error|CudaStreamCaptureStatusInvalidated")

POOL_HIT_RE = re.compile(r"\[pool\] dstep=(\d+).*?win hit=([\d.]+)%.*?cum hit=([\d.]+)%")
POOL_CNT_RE = re.compile(r"resident=(\d+).*?nr=(\d+) vf=(\d+) oom=(\d+) stalls=(\d+)")

CSV_HEADER = [
    "epoch", "iso", "health", "pid_alive", "harness_alive",
    "done", "ok", "fail", "fail_timeout", "fail_4xx", "fail_5xx",
    "fail_disconnect", "fail_other",
    "gpu4_mib", "gpu5_mib", "gpu6_mib", "gpu7_mib",
    "pool_lines", "dstep", "win_hit", "cum_hit", "pool_oom", "pool_stalls",
    "pool_vf", "pool_resident", "pool_nr",
    "fatal_hits", "fatal_new", "log_lines",
    "win_n", "win_p50_ms", "win_p95_ms", "cum_n", "cum_p50_ms", "cum_p95_ms",
]


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
    ok_lat, ok_t, fail_by, done, footer = [], [], {}, 0, None
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
            if r.get("kind") or r.get("t_end") is None:
                continue
            done += 1
            if r.get("outcome") == "ok":
                ok_lat.append(r["latency_ms"])
                ok_t.append(r["t_end"])
            else:
                fail_by[r.get("outcome")] = fail_by.get(r.get("outcome"), 0) + 1
    if footer:
        done = max(done, footer.get("done", 0))
    return dict(done=done, ok=len(ok_lat), fail=done - len(ok_lat),
                fail_by=fail_by, ok_lat=ok_lat, ok_t=ok_t, footer=footer)


_LAST = {"fatal": 0}


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
            g = [gpu_mem(x) for x in (4, 5, 6, 7)]

            pool_lines = int(sh(f"grep -c '\\[pool\\] dstep=' {args.log}") or 0)
            last_pool = sh(f"grep '\\[pool\\] dstep=' {args.log} | tail -1")
            m = POOL_HIT_RE.search(last_pool)
            dstep, win_hit, cum_hit = (m.group(1), m.group(2), m.group(3)) if m else ("", "", "")
            c = POOL_CNT_RE.search(last_pool)
            resident, nr, vf, poom, stalls = c.groups() if c else ("", "", "", "", "")

            fatal_hits = int(sh(f"grep -ciE \"{FATAL_RE}\" {args.log}") or 0)
            log_lines = int(sh(f"wc -l < {args.log}") or 0)

            st = read_jsonl(args.jsonl)
            fb = st["fail_by"]
            fb_other = sum(v for k, v in fb.items()
                           if k not in ("timeout", "http_4xx", "http_5xx", "disconnect"))
            win = [l for tt, l in zip(st["ok_t"], st["ok_lat"]) if tt >= ep - args.interval]
            fatal_new = fatal_hits - _LAST["fatal"]
            row = [
                f"{ep:.0f}", time.strftime("%H:%M:%S", time.localtime(ep)), hc, alive, halive,
                st["done"], st["ok"], st["fail"],
                fb.get("timeout", 0), fb.get("http_4xx", 0), fb.get("http_5xx", 0),
                fb.get("disconnect", 0), fb_other,
                g[0], g[1], g[2], g[3],
                pool_lines, dstep, win_hit, cum_hit, poom, stalls, vf, resident, nr,
                fatal_hits, fatal_new, log_lines,
                len(win), pct(win, 0.5), pct(win, 0.95),
                len(st["ok_lat"]), pct(st["ok_lat"], 0.5), pct(st["ok_lat"], 0.95),
            ]
            _LAST["fatal"] = fatal_hits
            f.write(",".join(str(x) for x in row) + "\n")
            f.flush()
            print(f"{row[1]} health={hc} pid={alive} harn={halive} done={st['done']} "
                  f"ok={st['ok']} fail={st['fail']} fb={fb} gpu={g} "
                  f"pool(dstep={dstep} win={win_hit}% cum={cum_hit}% oom={poom} "
                  f"stalls={stalls} vf={vf} res={resident}) fatal={fatal_hits}(+{fatal_new}) "
                  f"winN={len(win)} win_p50={row[30]} win_p95={row[31]} "
                  f"cum_p50={row[33]} cum_p95={row[34]}", flush=True)
            if st["footer"]:
                print(f"HARNESS FINISHED: {json.dumps(st['footer'])}", flush=True)
            sl = args.interval - (time.time() - t0)
            if sl > 0 and i < args.samples - 1:
                time.sleep(sl)


if __name__ == "__main__":
    main()
