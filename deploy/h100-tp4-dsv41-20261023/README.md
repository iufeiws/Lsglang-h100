# H100 部署快照：DeepSeek-V4.1-Flash-1017 · TP4（4×H100 SM90）

**状态**：可以跑起来（/health 200、KV 4M、dspark 已启用、CUDA 图抓取正常）。
本目录是**当前运行版本**的完整快照（脚本 + 配置 + 文档 + 未应用的补丁）。

## 硬件 / 软件
- 4× H100 80GB（SM90），**只用物理卡 4-7**，以 UUID 固定：
  `GPU-7561b28f…`(4) `GPU-fb59db94…`(5) `GPU-3c5d7267…`(6) `GPU-3c0f2856…`(7)
- 环境：`/mnt/SSD01/models/.llm-env/Lsglang/env` — lsglang 1.6.1 / sglang 0.5.15.dev105+g46b087b58 /
  torch 2.14.1+cu132 / flashinfer-python 0.7.2 / sgl-kernel 0.4.10 / lk_moe 2.7.1 / Python 3.12.11
  （非上游之处：**torchaudio 已卸载**（仅 cu130 轮子，与 cu132 冲突）；libnuma 用 conda-forge numactl）
- 权重：`/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017`（revision `1bae0e1088993b9b82de08f59de5be5a34e42170`，84 文件）
- **site-packages 状态（2026-10-23 更新）**：**已应用 1 个修复补丁** —— `reject_sampling q 容差`
  （文件 `sglang/kernels/ops/speculative/reject_sampling.py`；装后 sha256 `541e6ee8…`，备份 `.orig` = `62caac17…`；
  生效标记：`grep "ENV-PATCH] reject_sampling q-tolerance"`；apply/rollback 见 `humming/patches/`）。
  **其余补丁均未应用**；`humming/stash/` 为曾应用版本留档。
  ⚠ 该修复**在下次启动后才生效**（运行中的进程仍是启动时 import 的旧内核）。

## 当前生效配置（`configs/dsv41_tp4_h100.conf`）
- `--tensor-parallel-size 4`、无 `--ep-size`、无 `--attn-dp-size`
- `--mem-fraction-static 0.88`、`--max-running-requests 16`、`--context-length 1048576`、`--max-total-tokens 4194304`
- `--chunked-prefill-size 8192`、`--moe-runner-backend flashinfer_mxfp4`、**不传** `--flashinfer-mxfp4-moe-precision`（=W4A16）
- `--enable-decoder-swa-bounded-replay`、`--disable-flashinfer-autotune`、端口 18010
- **`CUDA_LAUNCH_BLOCKING=""`（2026-10-23 起置空；置空则脚本不导出。原为 `"1"` 仅用于故障归因，实测使解码慢 ~4.2 ms/step，见 `docs/ab-cuda-launch-blocking.md`）**
- 专家：`LK_GPU_RESIDENT_MOE_LAYERS="0-19"`（20 层全驻留）+ `LK_POOL_SLOTS=128`（后 20 层热池）
- `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`（消除权重转换的池化碎片，实测 −7.8 GB/卡）
- **dspark（spec decode）**：`--speculative-algorithm DSPARK --speculative-dspark-block-size 5 --speculative-attention-mode decode`
- `LK_VERIFY_RESIDENT=1`（让 6 行 verify 全部走 GPU 热池）

## 实测（4 卡）
| 项 | 值 |
|---|---|
| `max_total_num_tokens` | **4,194,304**（profile 5,585,408 被上限截断） |
| `[pool] slab` | 11.21 GiB（20 层 × 573.75 MiB） |
| decode（**无 dspark**） | **79.0 tok/s**、12.66 ms/step |
| decode（**有 dspark**） | ~32–34 tok/s、~30 ms/step；accept len p50 ≈1.85（block=5） |
| 每卡显存 | 无 dspark 71.1/68.2/68.2/67.7 GiB → 有 dspark 77.6/75.0/75.0/74.5 GiB |
| 池 | win hit ≈95%、nr ≈0%、copyiss ≈1.0–1.3 ms、que = 21、oom/stalls = 0 |
| 启动 | 加载+图抓取 ≈ 10–14 分钟（dspark 多两张 verify 图：target 34.5 s + draft 29.6 s） |

## HiCache（**准备态，未启用**）
见 `docs/hicache-prep.md`。要点：
- 必须显式 `--hicache-ratio`（DSV4 拒绝 `--hicache-size`）
- 被镜像 device 池 = 8.284 GB/rank ⇒ host = ratio × 8.284 GB/rank
- **ratio=6 ⇒ 49.7 GB/rank（合计 198.8 GB）；≈128 GB ⇒ ratio≈3.86**
- 变体配置：`configs/dsv41_tp4_h100.hicache.conf`（内含 `SERVE_HICACHE_ARGS="--enable-hierarchical-cache --hicache-ratio 3.86"`，**未验证**）

## 怎么跑
```bash
cd scripts && bash serve_dsv41_tp4_h100.sh        # 预检 → 启动 → 写日志/PID
bash stop_dsv41_tp4_h100.sh                       # 停
bash status.sh / bash health.sh                   # 看状态
```

## 已应用的修复（2026-10-23）

### reject_sampling q 容差（DSPARK 接受判定）
- **问题**：`q_is_prob = (q > 0.0) & (q <= 1.0)` 把"舍入导致 q 略大于 1"当成非法概率 ⇒ 接受长度被清零
  （实测：`q=1.0000001` 即从 **5/5 → 0/5**）。四处：普通验证、block 验证、residual 两遍扫描。
- **修法**：`q <= 1.0 + Q_PROB_TOL`（`Q_PROB_TOL_DEFAULT = 1e-2`），并在计算前把 q **clamp 到 [0,1]**
  （`p/qc`、`coin*qc` 全部用 clamp 后的值）；residual 由"越界清零"改为"容差带内夹到 1、负值/NaN/±inf 仍取 0"。
  `q<=0` 仍拒绝，`q > 1+EPS` 仍拒绝。
- **EPS 依据**：fp32 行和最坏前向误差界 `(V-1)·u`，V=129,280 ⇒ `7.71e-3`，取 `1e-2`（1.30× 余量）。
- **等价性**：`q∈[0,1]` 且无 NaN ⇒ 逐位等价（16,384 条随机链 accept_num/predicts 全同；接受律 B=262,144 两组同值）。
  唯一变化域 = `q ∈ (1, 1+EPS]`。
- **验证**：单卡（GPU3）单链 p=0.999/coin=0.5/gamma=5；`1.0000001 / 1.000006 / 1.00002 / 1.01` 修复后均 5/5；
  `1.0101/1.02/1.1/2.0/+inf/0/-1e-8/-0.5/NaN` 修复前后逐位一致。
- **产物**：`humming/patches/sglang-reject-sampling-q-tolerance.patch` + `apply_/rollback_*.sh`、
  `docs/reject-sampling-q-fix.md`、`scripts/test_reject_sampling_q_tolerance.py`、
  `scripts/repro_reject_q_gt1.py`、`docs/figs/reject-sampling-q-tolerance-20261023.png`、`docs/_evidence/q_tolerance/`
- **未覆盖**：真实 129k vocab 端到端验证未做（本机 softmax 810 次采样 `max(q)=1.000000000`，即触发条件未在本机观测到）。

## 目录
- `scripts/` 启动/停止/状态/健康检查 + 压测脚本（`bench256_streams.py` 等）
- `configs/` 当前 conf + 最近 3 个 effective 快照 + hicache 变体
- `humming/patches/` 我们的补丁与 apply/rollback（**未应用**）；`humming/stash/` 曾应用的版本
- `docs/` 全部分析报告（含 hicache-prep.md、显存清单、SEGV/时序时间线、dspark/吞吐报告与图）
（**未包含**：`logs/` 运行日志 81 MB、证据大件（lk_moe wheel/so、gsm8k parquet），均留在原机 `Lsglang/logs` 与 `Lsglang/docs/_big_not_committed/`）
