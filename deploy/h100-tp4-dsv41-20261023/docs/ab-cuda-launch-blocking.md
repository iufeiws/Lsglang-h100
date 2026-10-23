# A/B：CUDA_LAUNCH_BLOCKING 对 DSPARK 解码速度的影响（单变量）

- 固定条件：模型 `DeepSeek-V4.1-Flash-1017`（revision `1bae0e1088993b9b82de08f59de5be5a34e42170`）；TP4（物理卡 4-7，UUID 固定）；
  `resident=0-19` + `LK_POOL_SLOTS=128` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `LK_VERIFY_RESIDENT=1`；
  dspark 三件套（`--speculative-algorithm DSPARK --speculative-dspark-block-size 5 --speculative-attention-mode decode`）；
  `--max-total-tokens 4194304`（profile 5,585,408）；`max_running_requests=16`；端口 18010。
- 唯一变量：`CUDA_LAUNCH_BLOCKING` 由 `"1"` 置为 `""`（置空后启动脚本不导出该变量；进程 environ 实测该键计数 = 0）。
- 工作负载与口径：单流长会话续写（`max_tokens=1500`、`temperature=0.7`），`dec tok/s = (ct-1)/(e2e-TTFT)`，`ms/step = 1000/dec`；每轮一行；两轮均为 120–600 s 短程、无外部并发（`foreign=0`）。

## 结果

| 指标 | CUDA_LAUNCH_BLOCKING="1"（13:56 起 600 s） | CUDA_LAUNCH_BLOCKING=""（14:30 起 120 s） |
| --- | --- | --- |
| dec p50 / mean (tok/s) | 35.45 / 35.57 | **41.72 / 41.20** |
| ms/step p50 / mean (ms) | 28.20 / 28.13 | **23.97 / 24.30** |
| TTFT p50 (s) | 0.826 | **0.642** |
| accept len p50 (样本数) | 1.90 (94) | 1.80 (84) |
| 上下文档 | 1.3K → 7.2K | 1.2K → 6.1K |

逐轮 dec (tok/s)：`43.271, 41.799, 41.724, 39.090, 40.127`；逐轮 ms/step：`23.11, 23.92, 23.97, 25.58, 24.92`。

## 派生数字

- 该变量成本 ≈ **4.2 ms/step**（28.20 → 23.97）；`accept len` 不变 ⇒ 属纯开销。
- 无 dspark 基线（同配置、早先轮次）：**12.66 ms/step**、79.0 tok/s。
- 按 `accept len 1.80` 折算：**每个 verify 步 ≈ 43.1 ms**（= 1.80 × 23.97），约为普通 decode 一步的 **3.4×**。

## 配置现状

- `configs/dsv41_tp4_h100.conf:237` 现为 `CUDA_LAUNCH_BLOCKING=""`（原 `"1"`，仅用于故障归因）。
