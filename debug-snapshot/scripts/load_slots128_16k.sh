#!/usr/bin/env bash
# SLOTS=128 压测: 并发 2 / 8 / 16, 每路 ~16k token 唯一 prompt, 输出 256 token, temp 0.7; 末段图片探针。
set -u
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
PY=/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python
RES=$W/results_slots128
KEY=sk-dsv41-h100-interop-2026
cd $W
for K in 2 8 16; do
  echo "### $(date +%F_%T) 并发 k=$K, ~16k token/路, mt=256"
  timeout 3600 $PY scripts/bench_longctx_streams.py --k $K --prompt-chars 25000 --max-tokens 256 \
    --temp 0.7 --tag slots128_16k_k${K} --api-key "$KEY" --outdir $RES 2>&1 | tail -30
  sleep 20
done
echo "### $(date +%F_%T) 图片探针"
timeout 900 $PY scripts/image_probe.py --api-key "$KEY" --outdir $RES 2>&1 | tail -5
echo "### $(date +%F_%T) 抓两行缓存日志 (TP0 与全 rank)"
L=$(ls -t /mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_*.log | head -1)
grep -hE "TP0 EP0\] \[gpu-expert-cache\]" "$L" > $RES/cache_log_TP0_slots128.txt
grep -hE "\[gpu-expert-cache\] (20 layers|distinct missed)" "$L" > $RES/cache_log_new2_slots128.txt
grep -hE "20 layers, last 30s" "$L" | tail -12 > $RES/cache_log_allranks_last12.txt
echo "### $(date +%F_%T) 完成"
