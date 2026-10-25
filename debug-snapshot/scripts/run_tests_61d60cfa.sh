#!/usr/bin/env bash
# 本轮 (61d60cfa) 步骤 1+2: 单测 + 单层 forward 基准。与上轮同命令同参数。
set -u
W=/data3/shared/wangyf/llm-serving/fix3-dspark-20261025
REPO=/mnt/SSD01/models/.llm-env/Lsglang-h100
VENV=/mnt/SSD01/models/.llm-env/Lsglang/env
PY=$VENV/bin/python
RES=$W/results_61d60cfa
GPU=GPU-7561b28f-b049-bbdb-31f8-16efeb06663d   # 物理 GPU4
mkdir -p $RES; chmod 755 $RES
export PATH="/usr/local/cuda/bin:$VENV/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=$GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $REPO

echo "### 环境"
echo "python: $($PY -V 2>&1)  git: $(git log --oneline -1)"
$PY -c "import sglang, flashinfer, torch; print('sglang from:', sglang.__file__); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('device:', torch.cuda.get_device_name(0))"

echo "### $(date +%F_%T) 1/3 单测: test_gpu_expert_cache_forward.py (整文件)"
timeout 2400 $PY -m pytest -q test/manual/layers/test_gpu_expert_cache_forward.py 2>&1 | tee $RES/pytest_forward_61d60cfa.out | tail -25
echo "exit=${PIPESTATUS[0]}"

echo "### $(date +%F_%T) 2/3 单测: test_lk_verify_uva.py"
timeout 2400 $PY -m pytest -q test/manual/kernels/test_lk_verify_uva.py 2>&1 | tee $RES/pytest_lk_verify_uva_61d60cfa.out | tail -10
echo "exit=${PIPESTATUS[0]}"

echo "### $(date +%F_%T) 3/3 单层 forward 基准"
timeout 3600 $PY debug-snapshot/scripts/bench_gpu_cache_kernel.py --rows 6 12 24 48 96 --hit 1.0 0.98 0.9 \
  2>&1 | tee $RES/bench_kernel_61d60cfa.out | tail -45
echo "exit=${PIPESTATUS[0]}"
echo "### $(date +%F_%T) 完成"
