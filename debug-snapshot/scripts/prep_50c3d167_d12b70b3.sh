#!/usr/bin/env bash
# 本轮 (50c3d167 + d12b70b3) 准备: 停服务 -> 切 commit -> 部署 7 个运行期 .py -> 全树核对 -> 单测 -> 单层基准
set -u
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
REPO=/mnt/SSD01/models/.llm-env/Lsglang-h100
VENV=/mnt/SSD01/models/.llm-env/Lsglang/env
PY=$VENV/bin/python
RES=$W/results_50c3d167_d12b70b3
mkdir -p $RES; chmod 755 $RES
# GPU 顺序 = 物理 4,6,5,7 (与 serve 的 GPU_UUIDS 一致)
UUIDS="GPU-7561b28f-b049-bbdb-31f8-16efeb06663d,GPU-3c5d7267-445c-4679-bce9-503bedd0c920,GPU-fb59db94-0e4e-4c4f-8ae5-4d0fdd1e0f0b,GPU-3c0f2856-a6e3-b9d5-fd10-a08f55579fff"
export CUDA_VISIBLE_DEVICES="$UUIDS" CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="/usr/local/cuda/bin:$VENV/bin:$PATH" CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"

echo "### $(date +%F_%T) 0/7 停 096b001e 服务 (腾出 4 张卡给单测/基准)"
bash /mnt/SSD01/models/.llm-env/Lsglang/scripts/stop_dsv41_tp4_h100.sh > $RES/stop_before_deploy.out 2>&1
tail -3 $RES/stop_before_deploy.out
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n '5,8p'

echo "### $(date +%F_%T) 1/7 切到 d12b70b3 (含祖先 50c3d167)"
cd $REPO && git checkout -B diagnose/50c3d167_d12b70b3-logs-20261026 d12b70b3 2>&1 | tail -1 && git log --oneline -3

echo "### $(date +%F_%T) 2/7 部署 7 个运行期 .py"
$W/scripts/deploy_50c3d167_d12b70b3.sh 2>&1 | tee $RES/deploy_d12b70b3.out | tail -10

echo "### $(date +%F_%T) 3/7 全树核对"
$PY $W/scripts/tree_hash_compare.py > $RES/tree_diff_d12b70b3.txt 2>&1; head -3 $RES/tree_diff_d12b70b3.txt

echo "### $(date +%F_%T) 4/7 单测 (50c3d167 改的两个测试文件 + d12b70b3 的 fp4_indexer)"
timeout 2400 $PY -m pytest -q test/manual/layers/test_gpu_expert_cache_forward.py 2>&1 | tee $RES/pytest_forward_d12b70b3.out | tail -6
timeout 2400 $PY -m pytest -q test/manual/kernels/test_lk_verify_uva.py 2>&1 | tee $RES/pytest_lk_verify_uva_d12b70b3.out | tail -6
timeout 3600 $PY -m pytest -q test/registered/kernels/ops/attention/test_fp4_indexer.py \
  -k "paged_logits_replay or hopper_indexer_backends_replay" 2>&1 | tee $RES/pytest_fp4_indexer_d12b70b3.out | tail -8

echo "### $(date +%F_%T) 5/7 单层基准 bf16 @ rows-vram 112 (本轮部署档) 与 128 (与上轮同条件)"
timeout 3600 $PY debug-snapshot/scripts/bench_gpu_cache_kernel.py --rows 6 12 24 48 96 --hit 1.0 0.98 0.9 --rows-vram 112 --max-copy 8 2>&1 | tee $RES/bench_kernel_bf16_rv112.out | tail -30

echo "### $(date +%F_%T) 6/7 单层基准 fp8 @ rows-vram 112 / bf16+fp8 @ rows-vram 128"
timeout 3600 $PY debug-snapshot/scripts/bench_gpu_cache_kernel.py --rows 6 12 24 48 96 --hit 1.0 0.98 0.9 --rows-vram 112 --max-copy 8 --precision fp8 2>&1 | tee $RES/bench_kernel_fp8_rv112.out | tail -30
timeout 3600 $PY debug-snapshot/scripts/bench_gpu_cache_kernel.py --rows 6 12 24 48 96 --hit 1.0 0.98 0.9 --rows-vram 128 --max-copy 8 2>&1 | tee $RES/bench_kernel_bf16_rv128.out | tail -30
timeout 3600 $PY debug-snapshot/scripts/bench_gpu_cache_kernel.py --rows 6 12 24 48 96 --hit 1.0 0.98 0.9 --rows-vram 128 --max-copy 8 --precision fp8 2>&1 | tee $RES/bench_kernel_fp8_rv128.out | tail -30

echo "### $(date +%F_%T) 7/7 完成"
