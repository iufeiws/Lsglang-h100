#!/usr/bin/env bash
# SLOTS=128 服务上的 profile: /start_profile(num_steps=40, activities=GPU) 期间跑单流 k=1 decode。
set -u
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
PY=/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python
RES=$W/results_slots128
KEY=sk-dsv41-h100-interop-2026
mkdir -p $RES/prof; chmod 755 $RES/prof
cd $W
echo "### $(date +%F_%T) start_profile (num_steps=40, activities=GPU)"
curl -s -m 60 -X POST http://127.0.0.1:18010/start_profile \
  -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" \
  -d "{\"output_dir\":\"$RES/prof\",\"num_steps\":40,\"activities\":[\"GPU\"]}"; echo
echo "### $(date +%F_%T) 驱动流量: 单流 bench k=1 mt=1000 temp=0.7"
timeout 1800 $PY scripts/bench_ep2_streams.py --k 1 --max-tokens 1000 --temp 0.7 \
  --tag slots128_prof --api-key "$KEY" --outdir $RES 2>&1 | tail -6
echo "### $(date +%F_%T) 等 4 份 trace 落盘"
for i in $(seq 1 40); do n=$(ls $RES/prof/*.trace.json.gz 2>/dev/null | wc -l); [[ "$n" -ge 4 ]] && break; sleep 5; done
ls -la $RES/prof/
