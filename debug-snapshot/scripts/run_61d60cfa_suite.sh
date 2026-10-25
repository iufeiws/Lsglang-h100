#!/usr/bin/env bash
# 本轮 (61d60cfa, miss 专家 staging 进 VRAM 走 CUTLASS) 端到端验证套件。
# 与上一轮 (d467311a / b9b8a46b / 1cf916ebc0) 用同一批脚本/同一组参数, 便于同条件对照。
set -u
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
PY=/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python
RP=/mnt/SSD01/models/.llm-env/Lsglang-h100/debug-snapshot/scripts
RES=$W/results_61d60cfa
KEY=sk-dsv41-h100-interop-2026
mkdir -p $RES; chmod 755 $RES
cd $W

echo "### $(date +%F_%T) 1/5 纯 prefill (单 prompt ~17k token, max_tokens=1; 服务空闲)"
timeout 1800 $PY scripts/prefill_probe.py --api-key "$KEY" --outdir $RES --tag 61d60cfa 2>&1 | tail -6

echo "### $(date +%F_%T) 2/5 单流 bench k=1 mt=1000 temp=0.7"
timeout 1800 $PY scripts/bench_ep2_streams.py --k 1 --max-tokens 1000 --temp 0.7 \
  --tag 61d60cfa_single --api-key "$KEY" --outdir $RES 2>&1 | tail -22

echo "### $(date +%F_%T) 3/5 复述 recall_margin n=50 (贪心+logprob)"
timeout 3600 $PY $RP/recall_margin.py --n 50 --tag 61d60cfa --outdir $RES 2>&1 | tail -28

echo "### $(date +%F_%T) 4/5 工具调用 1 次"
curl -s -m 600 http://127.0.0.1:18010/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"北京现在天气怎么样？请调用工具查询。"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"查天气","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":256,"temperature":0}' > $RES/toolcall_1.json
$PY - "$RES/toolcall_1.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
c = d["choices"][0]; m = c["message"]
print("finish:", c.get("finish_reason"), "| tool_calls:",
      json.dumps(m.get("tool_calls"), ensure_ascii=False)[:160], "| usage:", d.get("usage"))
PYEOF

echo "### $(date +%F_%T) 5/5 多流 bench k=4 mt=500 temp=0.7"
timeout 1800 $PY scripts/bench_ep2_streams.py --k 4 --max-tokens 500 --temp 0.7 \
  --tag 61d60cfa_k4 --api-key "$KEY" --outdir $RES 2>&1 | tail -22

echo "### $(date +%F_%T) 完成"
