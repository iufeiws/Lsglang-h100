#!/usr/bin/env bash
# 起服务(等就绪+自检) -> 跑端到端 -> 抓两行缓存日志。用法: run_cycle_113a9017.sh <标签>
set -u
TAG="${1:?标签: A_off|B_on|C_stage8}"
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
LG=/mnt/SSD01/models/.llm-env/Lsglang
RES=$W/results_113a9017
echo "########## $(date +%F_%T) 起服务 tag=$TAG"
( cd $LG && ./scripts/serve_dsv41_tp4_h100.sh ) > $RES/serve_start_${TAG}.out 2>&1
rc=$?
echo "########## serve 脚本退出码=$rc (自检段的 grep|head 可能触发 SIGPIPE; 以 /health 为准)"
echo "########## serve_start 尾部:"; tail -6 $RES/serve_start_${TAG}.out
h=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:18010/health)
echo "########## /health=$h"
[[ "$h" == "200" ]] || { echo "服务未就绪, 中止"; exit 1; }
echo "########## $(date +%F_%T) 端到端 tag=$TAG"
$W/scripts/run_113a9017_suite.sh "$TAG"
echo "########## $(date +%F_%T) cycle 完成 tag=$TAG"
