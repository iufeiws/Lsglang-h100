#!/usr/bin/env bash
# 本轮 (113a9017) 准备: 快照上一轮日志 -> 停服务 -> 切 commit -> 部署 -> 全树核对 -> 单测
set -u
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
REPO=/mnt/SSD01/models/.llm-env/Lsglang-h100
LG=/mnt/SSD01/models/.llm-env/Lsglang
VENV=$LG/env
PY=$VENV/bin/python
RES=$W/results_113a9017
mkdir -p $RES; chmod 755 $RES

echo "### $(date +%F_%T) 0/5 快照上一轮 (788fe82c W4A8) 的 serve 日志尾部"
cp -a $LG/logs/serve_dsv41_tp4_h100_20261025_232327.log $RES/serve_dsv41_tp4_h100_20261025_232327.final.log
echo "  行数 $(wc -l < $RES/serve_dsv41_tp4_h100_20261025_232327.final.log)"

echo "### $(date +%F_%T) 1/5 停服务"
$LG/scripts/stop_dsv41_tp4_h100.sh > $RES/stop_before_113a9017.out 2>&1
for i in $(seq 1 30); do sleep 5; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tail -4 | tr '\n' ' '); [[ "$u" == "0 0 0 0 " ]] && { echo "  GPU4-7 已释放 (t=$((i*5))s)"; break; }; done

echo "### $(date +%F_%T) 2/5 切到 113a9017"
cd $REPO && git checkout -B diagnose/113a9017-logs-20261026 113a9017e4 2>&1 | tail -1 && git log --oneline -1

echo "### $(date +%F_%T) 3/5 部署 4 个 .py"
$W/scripts/deploy_113a9017.sh 2>&1 | tee $RES/deploy_113a9017.out | tail -8

echo "### $(date +%F_%T) 4/5 全树核对"
$PY $W/scripts/tree_hash_compare.py > $RES/tree_diff_113a9017.txt 2>&1; head -1 $RES/tree_diff_113a9017.txt

echo "### $(date +%F_%T) 5/5 单测"
export PATH="/usr/local/cuda/bin:$VENV/bin:$PATH" CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=GPU-7561b28f-b049-bbdb-31f8-16efeb06663d CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $REPO
echo "--- 整文件 forward 测试 ---"
timeout 2400 $PY -m pytest -q test/manual/layers/test_gpu_expert_cache_forward.py 2>&1 | tee $RES/pytest_forward_113a9017.out | tail -6
echo "--- lk_verify_uva ---"
timeout 2400 $PY -m pytest -q test/manual/kernels/test_lk_verify_uva.py 2>&1 | tee $RES/pytest_lk_verify_uva_113a9017.out | tail -4
echo "### $(date +%F_%T) 完成"
