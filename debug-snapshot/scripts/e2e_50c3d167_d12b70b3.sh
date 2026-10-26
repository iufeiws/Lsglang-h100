#!/usr/bin/env bash
# 本轮 (50c3d167 开销精简 + d12b70b3 verify 图按行分块) 的 e2e:
# 快照 -> profile(40 步) + 单流 k=1 -> 干净 k=1/k=4 -> recall 50 + 工具调用 10 -> 16k 压测 k=2/8/16/32(跑两遍) -> 抓缓存日志
set -u
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
PY=/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python
RP=/mnt/SSD01/models/.llm-env/Lsglang-h100/debug-snapshot/scripts
LOGDIR=/mnt/SSD01/models/.llm-env/Lsglang/logs
RES=$W/results_50c3d167_d12b70b3
TAG=d12b70b3
KEY=sk-dsv41-h100-interop-2026
mkdir -p $RES/prof; chmod 755 $RES/prof
cd $W
LOG=$(ls -t $LOGDIR/serve_dsv41_tp4_h100_*.log | head -1)
SNAP=$(awk -F= '/^SNAP_FILE=/{print $2}' $LOGDIR/serve_dsv41_tp4_h100.pid)

echo "### $(date +%F_%T) 0/6 serve 日志=$LOG"
echo "### 生效快照 $SNAP 里的关键键:"
grep -hE "mem-fraction|max-running|LK_GPU_CACHE|CUDA_GRAPH|GPU_UUIDS|GPU_EXPECTED" "$SNAP" 2>/dev/null | head -12
cp -f "$SNAP" $RES/serve_snapshot_effective_${TAG}.conf 2>/dev/null || true

echo "### $(date +%F_%T) 1/6 profile (num_steps=40, activities=GPU) 期间单流 k=1"
curl -s -m 60 -X POST http://127.0.0.1:18010/start_profile -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" -d "{\"output_dir\":\"$RES/prof\",\"num_steps\":40,\"activities\":[\"GPU\"]}"; echo
timeout 1800 $PY scripts/bench_ep2_streams.py --k 1 --max-tokens 1000 --temp 0.7 --tag prof_window --api-key "$KEY" --outdir $RES 2>&1 | tail -4
for i in $(seq 1 60); do n=$(ls $RES/prof/*.trace.json.gz 2>/dev/null | wc -l); [[ "$n" -ge 4 ]] && break; sleep 5; done
echo "  traces: $(ls $RES/prof/*.trace.json.gz 2>/dev/null | wc -l)"
sleep 20

echo "### $(date +%F_%T) 2/6 干净 k=1 / k=4"
timeout 1800 $PY scripts/bench_ep2_streams.py --k 1 --max-tokens 1000 --temp 0.7 --tag k1_clean --api-key "$KEY" --outdir $RES 2>&1 | tail -4
sleep 20
timeout 1800 $PY scripts/bench_ep2_streams.py --k 4 --max-tokens 500 --temp 0.7 --tag k4_clean --api-key "$KEY" --outdir $RES 2>&1 | tail -4
sleep 20

echo "### $(date +%F_%T) 3/6 复述 recall_margin n=50 (贪心+logprob)"
timeout 3600 $PY $RP/recall_margin.py --n 50 --tag ${TAG} --outdir $RES 2>&1 | tail -8
sleep 20

echo "### $(date +%F_%T) 4/6 工具调用 10 次 (temp0 ×5 + temp0.7 ×5)"
timeout 3600 $PY $W/scripts/toolcalls.py $RES/toolcalls_${TAG}.jsonl 2>&1 | tail -8
sleep 20

echo "### $(date +%F_%T) 5/6 16k 长上下文压测 k=2/8/16/32 (第一遍 + 复测一遍)"
for R in "" "rep_"; do
  for K in 2 8 16 32; do
    timeout 3600 $PY scripts/bench_longctx_streams.py --k $K --prompt-chars 25000 --max-tokens 256 --temp 0.7 \
      --vram-interval 1 --tag 16k_${R}k${K} --api-key "$KEY" --outdir $RES 2>&1 | tail -22
    sleep 20
  done
done

echo "### $(date +%F_%T) 6/6 抓缓存日志 / 缓存行 / 显存"
grep -hE "TP0 EP0\] \[gpu-expert-cache\]" "$LOG" > $RES/cache_log_TP0_${TAG}.txt
grep -hE "\[gpu-expert-cache\] (20 layers|distinct missed)" "$LOG" > $RES/cache_log_new2_${TAG}.txt
grep -hE "20 layers, last 30s" "$LOG" | tail -16 > $RES/cache_log_allranks_last16_${TAG}.txt
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv > $RES/vram_after_e2e_${TAG}.csv 2>&1
echo "### $(date +%F_%T) 完成"
