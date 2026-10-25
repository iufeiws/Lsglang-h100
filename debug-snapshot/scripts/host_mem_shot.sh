#!/usr/bin/env bash
# 服务就绪后的主机内存快照: 4 个 TP rank 的 RSS/Pss/私有/共享 (smaps_rollup)、/dev/zero 映射、整机 free。
# 用法: host_mem_shot.sh <输出文件>
set -u
OUT="${1:-/dev/stdout}"
PID_FILE=/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100.pid
PID=$(awk -F= '/^PID=/{print $2}' "$PID_FILE")
{
echo "采点: $(date '+%F %T')   服务 launcher PID=${PID}"
echo
python3 - "$PID" <<'PY'
import os, re, sys
pid = sys.argv[1]
hdr = re.compile(r"^[0-9a-f]+-[0-9a-f]+ ")
kids = [l.split()[0] for l in open("/dev/stdin")] if False else \
       [l.split()[0] for l in os.popen(f"ps -o pid= --ppid {pid}").read().split("\n") if l.strip()]
for k in kids:
    cmd = open(f"/proc/{k}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
    if not cmd.startswith("sglang::scheduler"):
        continue
    roll = {}
    for line in open(f"/proc/{k}/smaps_rollup"):
        f = line.split()
        if len(f) >= 2 and f[0].endswith(":"):
            roll[f[0].rstrip(":")] = int(f[1])
    zero = 0
    cur = None
    with open(f"/proc/{k}/smaps") as fh:
        for line in fh:
            if hdr.match(line):
                cur = "/dev/zero" if "/dev/zero" in line else None
            elif line.startswith("Size:") and cur == "/dev/zero":
                zero += int(line.split()[1])
    mib = lambda kb: kb / 1024 / 1024
    print(f"{cmd:<22} pid={k}  RSS={mib(roll.get('Rss',0)):8.2f} GiB  Pss={mib(roll.get('Pss',0)):8.2f} GiB  "
          f"Private={mib(roll.get('Private_Clean',0)+roll.get('Private_Dirty',0)):8.2f} GiB  "
          f"Shared_Clean={mib(roll.get('Shared_Clean',0)):8.2f} GiB  Shared_Dirty={mib(roll.get('Shared_Dirty',0)):8.2f} GiB  "
          f"Anonymous={mib(roll.get('Anonymous',0)):8.2f} GiB  /dev/zero 映射={mib(zero):8.2f} GiB")
PY
echo
echo "整机 free -g:"
free -g
echo
echo "整机 MemAvailable/free -m:"
free -m | head -2
} | tee "$OUT"
