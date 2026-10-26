#!/usr/bin/env bash
# 起服务 (本轮配置: util 0.90 / mrr 32 / cuda-graph-max-bs 64 / LK_GPU_CACHE_SLOTS 112 / GPU 顺序 4,6,5,7),
# 等 /health, 然后抽取 bs64 捕获显存行与 max_total_num_tokens。
# 前置: conf 已改好 (本脚本只做快照 + 启动 + 抽取)。
set -u
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
LSG=/mnt/SSD01/models/.llm-env/Lsglang
TAG=d12b70b3
RES=$W/results_50c3d167_d12b70b3
mkdir -p $RES; chmod 755 $RES

cp -f $LSG/configs/dsv41_tp4_h100.conf      $RES/conf_dsv41_tp4_h100.${TAG:-d12b70b3}.conf
cp -f $LSG/scripts/serve_dsv41_tp4_h100.sh  $RES/serve_dsv41_tp4_h100.${TAG:-d12b70b3}.sh
echo "### conf 关键行:"
grep -nE "mem-fraction-static|max-running-requests|CUDA_GRAPH_MAX_BS|LK_GPU_CACHE_SLOTS=|LK_GPU_CACHE_PRECISION|LK_GPU_CACHE_MAX_COPY" \
  $LSG/configs/dsv41_tp4_h100.conf | head -12

echo "### $(date +%F_%T) 启动服务"
bash $LSG/scripts/serve_dsv41_tp4_h100.sh > $RES/serve_start_bs64.out 2>&1
echo "serve 脚本退出码=$? (2 = 自检 grep|head SIGPIPE, 以 /health 为准)"
curl -s -o /dev/null -w "health=%{http_code}\n" --max-time 5 http://127.0.0.1:18010/health
LOG=$(ls -t $LSG/logs/serve_dsv41_tp4_h100_*.log | head -1)
echo "### 日志=$LOG"
echo "--- 真实命令行 ---"; grep -m1 -o "sglang serve .*" "$LOG"
echo "--- server_args 关键值 ---"; grep -m1 -o "'mem_fraction_static': [0-9.]*" "$LOG"; grep -m1 -o "'max_running_requests': [0-9]*" "$LOG"
echo "--- max_total_num_tokens / KV ---"; grep -m2 "max_total_num_tokens" "$LOG"
echo "--- bs64 图捕获显存 ---"; grep -h "Capture target .* CUDA graph end" "$LOG"
echo "--- 缓存行 (SLOTS/NUMA) ---"; grep -m2 "LK_GPU_CACHE_SLOTS: layer 2[01]" "$LOG"
echo "### $(date +%F_%T) 完成"
