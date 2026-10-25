#!/usr/bin/env bash
# 096b001e + util 0.90 + SLOTS=128 + GPU 顺序 4,6,5,7 的 e2e: profile -> k=1 -> k=4 -> 16k 压测(2/8/16)
set -u
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
PY=/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python
RES=$W/results_096b001e
KEY=sk-dsv41-h100-interop-2026
mkdir -p $RES/prof; chmod 755 $RES/prof
cd $W
echo "### $(date +%F_%T) 1/4 profile (num_steps=40, activities=GPU) 期间单流 k=1"
curl -s -m 60 -X POST http://127.0.0.1:18010/start_profile -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" -d "{\"output_dir\":\"$RES/prof\",\"num_steps\":40,\"activities\":[\"GPU\"]}"; echo
timeout 1800 $PY scripts/bench_ep2_streams.py --k 1 --max-tokens 1000 --temp 0.7 --tag prof_window --api-key "$KEY" --outdir $RES 2>&1 | tail -4
for i in $(seq 1 40); do n=$(ls $RES/prof/*.trace.json.gz 2>/dev/null | wc -l); [[ "$n" -ge 4 ]] && break; sleep 5; done
echo "  traces: $(ls $RES/prof/*.trace.json.gz 2>/dev/null | wc -l)"
sleep 20
echo "### $(date +%F_%T) 2/4 干净 k=1 / k=4"
timeout 1800 $PY scripts/bench_ep2_streams.py --k 1 --max-tokens 1000 --temp 0.7 --tag k1_clean --api-key "$KEY" --outdir $RES 2>&1 | tail -4
sleep 20
timeout 1800 $PY scripts/bench_ep2_streams.py --k 4 --max-tokens 500 --temp 0.7 --tag k4_clean --api-key "$KEY" --outdir $RES 2>&1 | tail -4
sleep 20
echo "### $(date +%F_%T) 3/4 16k 长上下文压测 2/8/16"
for K in 2 8 16; do
  timeout 3600 $PY scripts/bench_longctx_streams.py --k $K --prompt-chars 25000 --max-tokens 256 --temp 0.7 \
    --vram-interval 1 --tag 16k_k${K} --api-key "$KEY" --outdir $RES 2>&1 | tail -20
  sleep 20
done
echo "### $(date +%F_%T) 4/4 抓缓存日志"
L=$(ls -t /mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_*.log | head -1)
grep -hE "TP0 EP0\] \[gpu-expert-cache\]" "$L" > $RES/cache_log_TP0.txt
grep -hE "\[gpu-expert-cache\] (20 layers|distinct missed)" "$L" > $RES/cache_log_new2_allranks.txt
grep -hE "20 layers, last 30s" "$L" | tail -12 > $RES/cache_log_allranks_last12.txt
echo "### $(date +%F_%T) 完成"
