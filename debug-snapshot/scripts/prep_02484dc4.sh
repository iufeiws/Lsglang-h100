#!/usr/bin/env bash
# 本轮 (02484dc4) 准备: 切 commit -> 部署 6 个 .py -> 全树核对 -> 单测 + 单层基准 (bf16/fp8)
set -u
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
REPO=/mnt/SSD01/models/.llm-env/Lsglang-h100
VENV=/mnt/SSD01/models/.llm-env/Lsglang/env
PY=$VENV/bin/python
RES=$W/results_02484dc4
mkdir -p $RES; chmod 755 $RES

echo "### $(date +%F_%T) 1/5 切到 02484dc4"
cd $REPO && git checkout -B diagnose/02484dc4-logs-20261026 02484dc460 2>&1 | tail -1 && git log --oneline -1

echo "### $(date +%F_%T) 2/5 部署 6 个 .py"
$W/scripts/deploy_02484dc4.sh 2>&1 | tee $RES/deploy_02484dc4.out | tail -10

echo "### $(date +%F_%T) 3/5 全树核对"
$PY $W/scripts/tree_hash_compare.py > $RES/tree_diff_02484dc4.txt 2>&1; head -1 $RES/tree_diff_02484dc4.txt

export PATH="/usr/local/cuda/bin:$VENV/bin:$PATH" CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=GPU-7561b28f-b049-bbdb-31f8-16efeb06663d CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $REPO
echo "### $(date +%F_%T) 4/5 单测"
timeout 2400 $PY -m pytest -q test/manual/layers/test_gpu_expert_cache_forward.py 2>&1 | tee $RES/pytest_forward_02484dc4.out | tail -6
timeout 2400 $PY -m pytest -q test/manual/kernels/test_lk_verify_uva.py 2>&1 | tee $RES/pytest_lk_verify_uva_02484dc4.out | tail -4

echo "### $(date +%F_%T) 5/5 单层基准 (bf16 与 fp8, 加 --rows-vram 108 --max-copy 8)"
timeout 3600 $PY debug-snapshot/scripts/bench_gpu_cache_kernel.py --rows 6 12 24 48 96 --hit 1.0 0.98 0.9 --rows-vram 108 --max-copy 8 2>&1 \
  | tee $RES/bench_kernel_bf16_02484dc4.out | tail -46
timeout 3600 $PY debug-snapshot/scripts/bench_gpu_cache_kernel.py --rows 6 12 24 48 96 --hit 1.0 0.98 0.9 --rows-vram 108 --max-copy 8 --precision fp8 2>&1 \
  | tee $RES/bench_kernel_fp8_02484dc4.out | tail -46
echo "### $(date +%F_%T) 完成"
