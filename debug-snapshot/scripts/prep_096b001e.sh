#!/usr/bin/env bash
# 本轮 (096b001e) 准备: 切 commit -> 部署 3 个运行期 .py -> 全树核对 -> H2D 拓扑测量 -> 单测 -> 单层基准
set -u
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
REPO=/mnt/SSD01/models/.llm-env/Lsglang-h100
VENV=/mnt/SSD01/models/.llm-env/Lsglang/env
PY=$VENV/bin/python
RES=$W/results_096b001e
mkdir -p $RES; chmod 755 $RES
UUIDS="GPU-7561b28f-b049-bbdb-31f8-16efeb06663d,GPU-fb59db94-0e4e-4c4f-8ae5-4d0fdd1e0f0b,GPU-3c5d7267-445c-4679-bce9-503bedd0c920,GPU-3c0f2856-a6e3-b9d5-fd10-a08f55579fff"
export CUDA_VISIBLE_DEVICES="$UUIDS" CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="/usr/local/cuda/bin:$VENV/bin:$PATH" CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"

echo "### $(date +%F_%T) 1/6 切到 096b001e"
cd $REPO && git checkout -B diagnose/096b001e-logs-20261026 096b001e60 2>&1 | tail -1 && git log --oneline -2

echo "### $(date +%F_%T) 2/6 部署 3 个运行期 .py"
$W/scripts/deploy_096b001e.sh 2>&1 | tee $RES/deploy_096b001e.out | tail -8

echo "### $(date +%F_%T) 3/6 全树核对"
$PY $W/scripts/tree_hash_compare.py > $RES/tree_diff_096b001e.txt 2>&1; head -1 $RES/tree_diff_096b001e.txt

echo "### $(date +%F_%T) 4/6 H2D 拓扑测量 (服务已停, CUDA_VISIBLE_DEVICES 与 serve 相同)"
cd $REPO && timeout 3600 $PY debug-snapshot/scripts/bench_h2d_topology.py 2>&1 | tee $RES/bench_h2d_topology.out | tail -60

echo "### $(date +%F_%T) 5/6 单测"
timeout 2400 $PY -m pytest -q test/manual/layers/test_gpu_expert_cache_forward.py 2>&1 | tee $RES/pytest_forward_096b001e.out | tail -6
timeout 2400 $PY -m pytest -q test/manual/kernels/test_lk_verify_uva.py 2>&1 | tee $RES/pytest_lk_verify_uva_096b001e.out | tail -4

echo "### $(date +%F_%T) 6/6 单层基准 (rows-vram 128 / max-copy 8, bf16 与 fp8)"
timeout 3600 $PY debug-snapshot/scripts/bench_gpu_cache_kernel.py --rows 6 12 24 48 96 --hit 1.0 0.98 0.9 --rows-vram 128 --max-copy 8 2>&1 | tee $RES/bench_kernel_bf16_096b001e.out | tail -46
timeout 3600 $PY debug-snapshot/scripts/bench_gpu_cache_kernel.py --rows 6 12 24 48 96 --hit 1.0 0.98 0.9 --rows-vram 128 --max-copy 8 --precision fp8 2>&1 | tee $RES/bench_kernel_fp8_096b001e.out | tail -46
echo "### $(date +%F_%T) 完成"
