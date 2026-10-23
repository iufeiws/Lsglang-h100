# Lsglang 1.6.1 环境安装报告

| 项 | 值 |
| --- | --- |
| 任务 | task-8 (T9 环境: 在 `.llm-env/Lsglang/env` 建 conda 环境并安装 Lsglang 1.6.1) |
| 执行者 | teammate `env-builder` |
| 状态 | **完成 (可用, 已通过启动门验证)** |
| 环境前缀 | `/mnt/SSD01/models/.llm-env/Lsglang/env` (11 GB) |
| 安装窗口 | 2026-10-22 13:22:40 ~ 13:38:53 (Asia/Shanghai), 有效安装用时约 16 min |
| 仓库 | `/mnt/SSD01/models/.llm-env/Lsglang/lsglang`, HEAD `60f6658524`, `git describe` = `lsglang-v1.6.1-2-g60f6658524` |
| 仓库改动 | **无** (`git status --porcelain` 为空; 全部依赖来自 PyPI 清华镜像 wheel, 未编译仓库源码) |

## 1. 结论速览 (实测命令与输出)

| 验收项 | 命令 | 结果 |
| --- | --- | --- |
| python 存在 | `<env>/bin/python -V` | `Python 3.12.11` ✅ |
| import sglang | `<env>/bin/python -c "import sglang; print(sglang.__version__)"` | `0.5.15.dev105+g46b087b58` ✅ (发行包 `lsglang==1.6.1`) |
| sglang serve CLI | `<env>/bin/sglang serve --help` | 输出 2713 行, rc=0 ✅ |
| 关键 flag 命中 | `sglang serve --help \| grep -E "flashinfer-mxfp4-moe-precision\|moe-runner-backend\|swa-full-tokens-ratio"` | 3 项全部命中 (rc=0) ✅ |
| import lk_moe | `<env>/bin/python -c "import lk_moe; print(lk_moe.__version__)"` | `2.7.1` ✅ |
| torch / CUDA | `<env>/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"` | `2.14.1+cu132` / `13.2` ✅ |
| nvcc | `/usr/local/cuda/bin/nvcc -V` | `release 13.2, V13.2.51` ✅ |
| GPU 专家池能力 | 见 §5 | **存在** (能力口径通过; 字面 grep 口径不成立) ✅ |

运行前置环境变量 (实测必需):

```bash
export PATH=/usr/local/cuda/bin:$PATH          # nvcc 不在默认 PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/mnt/SSD01/models/.llm-env/Lsglang/env/lib:$LD_LIBRARY_PATH
```

## 2. 完整命令与耗时

### 2.1 conda 环境 (Python 3.12.11)

```bash
/data3/users/wangyf/miniconda3/bin/conda create -p /mnt/SSD01/models/.llm-env/Lsglang/env \
    python=3.12.11 -y --no-default-packages
# → python 3.12.11 (defaults 渠道); 日志 logs/01_conda_create.log; 结束 13:23:29
```

### 2.2 README 要求的 C++ 运行库与 libnuma

```bash
conda install -p <env> -c conda-forge -y libstdcxx-ng numactl
# 日志 logs/02_conda_libstdcxx_numactl.log; 结束 13:24:18
```

| 包 | 版本 | 渠道 | 说明 |
| --- | --- | --- | --- |
| libstdcxx-ng / libstdcxx | 15.2.0 | defaults | solver 选择 defaults 的 15.2.0 (满足 README "新 libstdc++" 意图); `<env>/lib/libstdc++.so.6.0.34` |
| numactl | 2.0.18 | conda-forge | **替代 `apt-get install libnuma-dev`** (无 root) |
| libnuma | 2.0.18 | conda-forge | 提供 `<env>/include/numa.h`, `<env>/lib/libnuma.so`, `libnuma.so.1`, `<env>/bin/numactl` ✅ |

### 2.3 Python 依赖 (wheel, 清华镜像)

```bash
pip install -q --upgrade pip setuptools wheel                       # 13:24:53
pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    "torch==2.14.1+cu132" "torchvision==0.29.0+cu132" "torchaudio==2.11.0+cu132"   # 见 §4 失败 1
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu132 lsglang==1.6.1        # 13:25:01 → 13:33:16, rc=0
```

`lsglang` wheel 校验: `lsglang-1.6.1-py3-none-any.whl`, 40,142,979 字节,
sha256 = `da9fee8f2303de5f0662c0c26a8065c2fd3ccc98af8ea2adf1e50f496b06eeaa` (与任务给定值一致 ✅)。
wheel 元数据 138 条 `Requires-Dist`, **0 条 direct-URL (无 GitHub 依赖)** —— 全部可从镜像解析。

### 2.4 时间线

| 时间 | 事件 | 耗时 |
| --- | --- | --- |
| 13:22:40 | task-8 claim, 启动 `conda create` | 44 s |
| 13:23:29 | conda 环境就绪 (Python 3.12.11) | — |
| 13:23:3x | 启动 `conda install libstdcxx-ng numactl` | ~45 s |
| 13:24:18 | libstdcxx-ng + numactl/libnuma 完成 | — |
| 13:24:53 | 升级 pip/setuptools/wheel | 4 s |
| 13:24:57 | torch 三件套 `+cu132` 尝试 → **失败** (torchaudio 无 cu132 构建) | 4 s |
| 13:25:01 | `pip install lsglang==1.6.1` 开始 (解析+下载+安装) | **8 min 15 s** |
| 13:33:16 | lsglang 安装完成 rc=0 (安装 collect 阶段约 300+ 包) | — |
| 13:33:2x | 发现混栈故障: `import sglang` 抛 torch/torchaudio CUDA 版本不一致 | — |
| 13:34:31 | 尝试整体归一到 cu130 (torch/torchvision/torchaudio) → rc=0 | 2 min 07 s |
| 13:36:38 | cu130 归一完成 (torch 2.14.1+cu130, tv 0.29.1+cu130, ta 2.11.0+cu130) | — |
| 13:37:05 | 收到 lead 改向: 回滚到 cu132 + 卸载 torchaudio | — |
| 13:38:11 | torch 2.14.1+cu132 / torchvision 0.29.1+cu132 恢复 (rc=0, 走 pip 缓存) | 66 s |
| 13:38:16 | `pip uninstall -y torchaudio` (rc=0) | — |
| 13:38:53 | 验证三件套全绿 (import sglang / serve --help / 3 个 flag) | — |

## 3. 版本清单 (实测, 全量见 `docs/pip-freeze.txt`, 208 行)

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.11 (conda, defaults) |
| **lsglang (发行包)** | **1.6.1** (sha256 `da9fee8f…eeaa`) |
| sglang (模块 `__version__`) | 0.5.15.dev105+g46b087b58 (发行包内串与上游版本号不同, 属正常) |
| torch | 2.14.1+cu132 (`torch.version.cuda` = 13.2) |
| torchvision | 0.29.1+cu132 |
| torchcodec | 0.17.0+cu132 |
| torchaudio | **未安装** (见 §4 失败 2 与 §5 偏离) |
| triton | 3.8.0 (另有 tokenspeed-triton 3.9.10.post20261008) |
| cuda-toolkit (pip) | 13.2.2 |
| **lk_moe** | **2.7.1** |
| flashinfer-python | 0.7.2 |
| flash-attn-4 | 4.0.0b36 (`>=4.0.0b18` 由 pip 选最新 wheel) |
| sglang-kernel | 0.4.10 (cp310-abi3 manylinux2014_x86_64 wheel) |
| transformers / tokenizers | 5.19.0 / 0.23.2 |
| numba | 0.65.1 |
| nvidia-cutlass-dsl (+libs-base/core/cu12/cu13) | 4.8.0 |
| humming-kernels / quack-kernels | 0.1.12 / 0.6.5 |
| tokenspeed-mla | 0.1.8 |
| torch_memory_saver | 0.0.10 (`>=0.0.9.post1`) |
| sgl-deep-ep / sgl-deep-gemm | 0.1.2.post1 / 0.2.2 |
| nvshmem4py-cu13 / nccl4py | 0.4.1 / 0.6.0 |
| smg-grpc-servicer / smg-grpc-proto | 0.14.0 / 0.4.27 |
| nvcc (系统) | CUDA 13.2.51 (`/usr/local/cuda -> /usr/local/cuda-13.2`) |
| NVIDIA 驱动 | 595.45.04 |
| GPU | 8 × H100 80GB HBM3 (SM90, cc 9.0), 目标卡 4-7 |

## 4. 失败与重试记录

### 失败 1: `torchaudio==2.11.0+cu132` 不存在 (13:24:57, rc=1)

```
ERROR: Could not find a version that satisfies the requirement torchaudio==2.11.0+cu132
       (from versions: 2.2.0 ... 2.11.0)
ERROR: No matching distribution found for torchaudio==2.11.0+cu132
```

- 实测 `https://download.pytorch.org/whl/cu132/torchaudio/` 无 `2.11.0+cu132`; 对照仓库
  `docker/Dockerfile:212-227`: 发布镜像 `ARG CUDA_VERSION=13.0.3` → `CUINDEX=130`, 即**发布配方用 cu130**。
- 影响: 该次三件套安装整体失败; 随后 `pip install lsglang==1.6.1` 自行解析 (带上 cu132 extra index),
  装成 **torch 2.14.1+cu132 / torchvision 0.29.1+cu132 / torchaudio 2.11.0 (PyPI 普通版 = CUDA 13.0 构建)**。

### 失败 2: 混栈 CUDA 版本不一致导致 `import sglang` 直接抛错 (13:33:2x)

```
torchaudio/_extension/utils.py:124 in _check_cuda_version
RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions.
PyTorch has CUDA version 13.2 whereas TorchAudio has CUDA version 13.0.
```

- 根因: `pip install lsglang` 在 cu132 extra index 下把 torch 选为 `+cu132`, 而 torchaudio 只有
  `2.11.0`(cu130 构建) 可装, torchaudio 自带 CUDA 版本检查 → 硬失败。

### 处理 (最终采用 lead 裁决的最小改动方案)

1. 曾尝试整体归一 cu130 (13:34:31→13:36:38) 以消除混栈, **随后按 lead 指令回滚**:
   理由 — 环境内还有 `torchcodec 0.17.0+cu132`、`torch_c_dlpack_ext`、`torch_memory_saver`、
   `sglang-kernel 0.4.10`、`flash-attn-4`、`nvidia-cutlass-dsl` 等一批 cu132 构建, 降 torch 会把
   同类不匹配问题搬到下一处; 且 README 明确要求 CUDA 13.2.1 ↔ torch 2.14.1 的 CUDA build。
2. **回滚**: `pip install "torch==2.14.1+cu132" "torchvision==0.29.1+cu132"` (rc=0), 随后
   `pip uninstall -y torchaudio` (rc=0)。
3. 复验: `import sglang` ✅, `import torchaudio` → `ModuleNotFoundError` (预期),
   `sglang serve --help` + 3 个 flag ✅。

`pip check` 现状 (唯一一条, 已知且有意):

```
lsglang 1.6.1 requires torchaudio, which is not installed.
```

## 5. GPU 专家池能力核查 (LK_POOL_SLOTS) — 两种口径 + 证据

任务原文要求 "在装好的 lk_moe .so 里 `strings | grep LK_POOL_SLOTS`"。实测给出两个口径, **lead 已裁决按"能力口径"通过**。

### 口径 A: 字面 grep —— **不成立**

对 lk_moe 2.7.1 的全部 5 个 .so 做二进制扫描 (`grep -ao` / 正则扫描 printable 串):

| .so | 大小 | `LK_POOL_SLOTS` | 扫描到的 `LK_POOL*` 串 |
| --- | --- | --- | --- |
| `_lk_moe_C.cpython-312-x86_64-linux-gnu.so` | 23,153,736 B | 未命中 | `LK_POOL_STAGE` |
| `_lk_moe_C_avx2.so` | 23,153,736 B | 未命中 | `LK_POOL_STAGE` |
| `_lk_moe_C_avx512_base.so` | 23,506,056 B | 未命中 | `LK_POOL_STAGE` |
| `_lk_moe_C_avx512_vnni.so` | 23,538,824 B | 未命中 | `LK_POOL_STAGE` |
| `_lk_moe_C_avx512_amx.so` | 23,538,824 B | 未命中 | `LK_POOL_STAGE` |

### 口径 B: 能力/语义 —— **成立** (lead 裁决口径)

1. **池的开关在 Lsglang 侧 Python 读取** (环境变量本不应出现在闭源 .so 内):

```
# <env>/lib/python3.12/site-packages/sglang/srt/utils/lk_moe_env.py:277-282
def get_pool_slots() -> int:
    return get_int_env_var("LK_POOL_SLOTS", 0)

def is_lk_pool_enabled() -> bool:
    return is_lk_moe_feature_enabled() and get_pool_slots() > 0
```

同一文件 :125/:129 (仓库源码 :129 同) 注释原文: *"The ONLY pool switch is LK_POOL_SLOTS"*;
该值进入 `lk_moe_env.py:42-43` 的配置对象 (`pool_slots=..., pool_enabled=...`), 即开关→能力链路完整。
装出的包与仓库源码一致 (两处 file:line 已逐条比对)。

2. **池的能力在 .so 侧**: 上述 .so 内含整套池内部符号 (二进制扫描):
`pool_cache, pool_slot, pool_ready_slot, pool_snapshot_slot, pool_stats, pool_set_det_split,
pool_wait_fence, pool_wait_all_fences, pool_verify_sample, pool_layer_info, pool_member_layers,
pool_capture_pause, pool_capture_resume, pool_freeze_view, pool_gds, pool_pin, pool_inflight,
pool_gpu_live, pool_gpu_report, pool_decode, pool_decode_on, pool_record_main, pool_record_aux,
pool_skip, pool_log`。
其中 `pool_member_layers` 与本项目 "20 个非 resident 层" 的计费口径对应。

3. **对外导出**: `import lk_moe` 直接列出 17 个池/NVMe 函数:

```
pool_capture_pause, pool_capture_resume, pool_layer_info, pool_ready_slot, pool_set_det_split,
pool_slot, pool_snapshot_slot, pool_stats, pool_verify_sample, pool_wait_all_fences, pool_wait_fence,
nvme_filler_attach, nvme_filler_attach_gather, nvme_filler_detach, nvme_filler_detach_all,
nvme_filler_stats, nvme_gate
```

对照: 仓库 launch 脚本 `launch/sglang_serve_dsv41_tp4.sh` / README quick start 使用 `LK_POOL_SLOTS=96` 即走此链路。

**陈述**: "字面 grep LK_POOL_SLOTS" 不成立; "GPU 专家池能力存在" 成立。二者证据如上。

## 6. 与 README/仓库配方的偏离清单

| 项 | README / 仓库配方 | 实际采用 | 原因 |
| --- | --- | --- | --- |
| libnuma-dev | `sudo apt-get install libnuma-dev` | conda-forge `numactl 2.0.18` (+`libnuma 2.0.18`) | 无 root; 提供完全等效的 `numa.h` + `libnuma.so` |
| libstdcxx-ng | `conda install -c conda-forge libstdcxx-ng` | solver 落 `defaults` 的 libstdcxx-ng 15.2.0 | 结果满足 "新 libstdc++ 进 `<env>/lib`" 的意图; 版本 15.2.0 |
| torch CUDA 变体 | 仓库 Dockerfile 用 cu130 (CUDA 13.0.3); README 提 CUDA 13.2.1 | `torch 2.14.1+cu132` | 与本机 CUDA 13.2.51/驱动 595.45.04 及 README 对齐; lead 裁决保持 cu132 |
| torchaudio | `pip install lsglang` 会拉 `torchaudio==2.11.0` | **卸载** | 无 cu132 构建; 装上即触发 torch/torchaudio CUDA 版本不一致 (失败 2) |
| 编译 CUDA 扩展 | — | 未发生 | lsglang 1.6.1 全依赖均有预编译 wheel (含 sglang-kernel abi3 / lk_moe cp312) |

## 7. 后续安装注意 (防止环境被改坏)

1. 任何后续 `pip install` 请加 `--no-deps`, 或使用以下钉住组合, **避免 torch 被降级 / torchaudio 被拉回**:

```bash
# 允许的锁定组合 (当前已安装, 勿改动)
torch==2.14.1+cu132
torchvision==0.29.1+cu132
torchcodec==0.17.0+cu132
# torchaudio 明确不装
pip install --no-deps --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu132 <pkg>
```

2. `sglang serve --help` 需要 `PATH=/usr/local/cuda/bin:$PATH` (nvcc 13.2.51) 与
   `LD_LIBRARY_PATH=<env>/lib` (libstdc++ 15.2.0 / libnuma) —— 见 §1。

## 8. 本机网络事实 (安装期实测, 供后续复用)

| 渠道 | 状态 |
| --- | --- |
| `pypi.tuna.tsinghua.edu.cn` | ✅ (含 `lsglang==1.6.1`、`lk-moe==2.7.1`、`sglang-kernel==0.4.10`、`flash-attn-4` 全部 wheel) |
| `download.pytorch.org/whl/cu130` | ✅ |
| `download.pytorch.org/whl/cu132` | ✅ (无 `torchaudio 2.11.0+cu132`) |
| `github.com` 及 CDN (https) | ✗ 连接重置 (HTTP 000) |
| `github.com` ssh | ✅ (user `iufeiws`) |

## 9. 现场与产物

| 路径 | 内容 |
| --- | --- |
| `/mnt/SSD01/models/.llm-env/Lsglang/env` | conda 环境 (11 GB), 可直接使用 |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/env-report.md` | 本报告 |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/pip-freeze.txt` | 208 行全量版本快照 |
| `/mnt/SSD01/models/.llm-env/Lsglang/logs/01_conda_create.log` | conda create |
| `/mnt/SSD01/models/.llm-env/Lsglang/logs/02_conda_libstdcxx_numactl.log` | libstdcxx-ng + numactl |
| `/mnt/SSD01/models/.llm-env/Lsglang/logs/03_pip_install.log` | torch/lsglang 安装 (含失败 1/失败 2 原文) |
| `/mnt/SSD01/models/.llm-env/Lsglang/logs/04_normalize_cu130.log` | cu130 归一尝试 (已回滚) |
| `/mnt/SSD01/models/.llm-env/Lsglang/logs/05_restore_cu132_uninstall_torchaudio.log` | 回滚 cu132 + 卸载 torchaudio + 三件套验证 |
| `/mnt/SSD01/models/.llm-env/Lsglang/logs/artifacts/` | 各阶段可复现脚本 + 下载的 wheel (`lsglang-1.6.1-py3-none-any.whl`, 已校验 sha256) |
