#!/usr/bin/env bash
# 本轮 (788fe82c) 端到端套件。用法: run_788fe82c_suite.sh <标签: w4a16|w4a8>
# 与上几轮同一批脚本/同一组参数, 便于同条件对照; 末段额外抓本轮新增的两行 [gpu-expert-cache] 日志。
set -u
TAG="${1:-w4a16}"
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
PY=/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python
RP=/mnt/SSD01/models/.llm-env/Lsglang-h100/debug-snapshot/scripts
RES=$W/results_788fe82c
KEY=sk-dsv41-h100-interop-2026
LOGDIR=/mnt/SSD01/models/.llm-env/Lsglang/logs
LOG=$LOGDIR/$(ls -t $LOGDIR/serve_dsv41_tp4_h100_*.log | head -1 | xargs basename)
SNAP=$(awk -F= '/^SNAP_FILE=/{print $2}' $LOGDIR/serve_dsv41_tp4_h100.pid)
mkdir -p $RES; chmod 755 $RES
cd $W

echo "### $(date +%F_%T) 轮次=$TAG  serve 日志=$LOG"
echo "### 生效快照 $SNAP 里的缓存键:"
grep -hE "LK_GPU_CACHE" "$SNAP" 2>/dev/null | head -6 || true

echo "### $(date +%F_%T) 1/5 纯 prefill (单 prompt ~17k token, max_tokens=1; 服务空闲)"
timeout 1800 $PY scripts/prefill_probe.py --api-key "$KEY" --outdir $RES --tag ${TAG} 2>&1 | tail -5

echo "### $(date +%F_%T) 2/5 单流 bench k=1 mt=1000 temp=0.7"
timeout 1800 $PY scripts/bench_ep2_streams.py --k 1 --max-tokens 1000 --temp 0.7 \
  --tag ${TAG}_single --api-key "$KEY" --outdir $RES 2>&1 | tail -8

echo "### $(date +%F_%T) 3/5 复述 recall_margin n=50 (贪心+logprob)"
timeout 3600 $PY $RP/recall_margin.py --n 50 --tag ${TAG} --outdir $RES 2>&1 | tail -6

echo "### $(date +%F_%T) 4/5 工具调用 1 次"
curl -s -m 600 http://127.0.0.1:18010/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"北京现在天气怎么样？请调用工具查询。"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"查天气","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":256,"temperature":0}' > $RES/toolcall_1_${TAG}.json
$PY - "$RES/toolcall_1_${TAG}.json" "${TAG}" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); c = d["choices"][0]; m = c["message"]
print(sys.argv[2], "finish:", c.get("finish_reason"), "| tool_calls:",
      json.dumps(m.get("tool_calls"), ensure_ascii=False)[:140], "| usage:", d.get("usage"))
PYEOF

echo "### $(date +%F_%T) 5/5 多流 bench k=4 mt=500 temp=0.7"
timeout 1800 $PY scripts/bench_ep2_streams.py --k 4 --max-tokens 500 --temp 0.7 \
  --tag ${TAG}_k4 --api-key "$KEY" --outdir $RES 2>&1 | tail -8

echo "### $(date +%F_%T) 6/6 抓本轮新增的 [gpu-expert-cache] 两行 (TP0 EP0)"
grep -hE "TP0 EP0\] \[gpu-expert-cache\]" "$LOG" > $RES/cache_log_${TAG}.txt
grep -hE "\[gpu-expert-cache\] (graph routes|distinct missed)" "$LOG" | tail -20 > $RES/cache_log_new2_${TAG}.txt
echo "--- 新格式两行 (全 rank, 最近 12 条) ---"
grep -hE "\[gpu-expert-cache\] (graph routes|distinct missed)" "$LOG" | tail -12
echo "### $(date +%F_%T) 完成 tag=$TAG"
