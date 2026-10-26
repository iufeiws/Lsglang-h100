#!/usr/bin/env bash
# 等本轮服务就绪 (bs64 / util 0.90 / mrr 32 / SLOTS 112), 或早退于进程消失; 就绪后抽取关键行。
set -u
LSG=/mnt/SSD01/models/.llm-env/Lsglang
RES=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025/results_50c3d167_d12b70b3
PID=$(awk -F= '/^PID=/{print $2}' $LSG/logs/serve_dsv41_tp4_h100.pid 2>/dev/null)
LOG=$(awk -F= '/^LOG_FILE_RECORDED=/{print $2}' $LSG/logs/serve_dsv41_tp4_h100.pid 2>/dev/null)
LOG="${LOG:-$(ls -t $LSG/logs/serve_dsv41_tp4_h100_*.log | head -1)}"
echo "### 等待就绪 log=$LOG pid=${PID:-?} $(date +%F_%T)"
ready=0
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:18010/health 2>/dev/null || echo 000)
  if [[ "$code" == "200" ]]; then ready=1; echo "### READY after $((i*30))s $(date +%F_%T)"; break; fi
  if ! kill -0 "${PID:-0}" 2>/dev/null; then echo "### 进程 ${PID:-?} 已消失 (第 $i 次探测, $(date +%F_%T))"; break; fi
  sleep 30
done
if [[ "$ready" != "1" ]]; then
  echo "### 未就绪; serve 日志尾部:"
  tail -40 "$LOG"
  exit 1
fi
echo "--- 真实命令行 ---"; grep -m1 -o "sglang serve .*" "$LOG"
echo "--- server_args 关键值 ---"; grep -m1 -o "'mem_fraction_static': [0-9.]*" "$LOG"; grep -m1 -o "'max_running_requests': [0-9]*" "$LOG"
echo "--- max_total_num_tokens ---"; grep -m2 "max_total_num_tokens" "$LOG"
echo "--- 图捕获 (全部 rank) ---"; grep -h "Capture target .* CUDA graph end" "$LOG"
echo "--- 缓存初始化行 ---"; grep -m2 "LK_GPU_CACHE_SLOTS: layer 2[01]" "$LOG"
echo "--- 自检 ---"; tail -12 $RES/serve_start_bs64.out
echo "### 完成 $(date +%F_%T)"
