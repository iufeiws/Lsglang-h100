# b9b8a46b (GPU 专家缓存 · 张量核版) 验证结果 —— 2026-10-25

本文件只记录测得的数字与测量条件。

## 0. 被测对象与部署

| 项 | 值 |
|---|---|
| 仓库 | `/mnt/SSD01/models/.llm-env/Lsglang-h100`, 分支 `debug/dspark-variant-matrix-20261025`, commit `b9b8a46bf2` |
| 参照轮 | `1cf916ebc0`（同配置的上一轮服务，日志 `serve_dsv41_tp4_h100_20261025_164259.log`） |
| 部署方式 | 把仓库 `python/sglang/` 下 3 份 .py 拷进 `/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/`，删对应 `__pycache__`；旧版留 `*.orig-1cf916ebc0.site` |
| md5（部署树 = 仓库） | `lk_verify_uva.py` `0e21f44943eca1e5e5b979e09cd988b4`；`fused_moe_triton/layer.py` `cb99583cd39bc252f822cd65022b48da`；`gpu_expert_cache.py` `90ebb23deb9718ca18d53a28292531cc` |
| 全树核对 | 仓库 4511 份 .py 与部署树逐份 md5 比较：differing = 0（94 份 `kernels/aot/benchmark/*` 部署树本就不存在） |
| conf 改动 | 仅 `SELFCHECK_EXPECT_STR`: `[pool] slab:` → `LK_GPU_CACHE_SLOTS: layer` |
| 生效配置（本轮） | `LK_POOL_SLOTS=0`、`LK_GPU_CACHE_SLOTS=104`、`LK_GPU_RESIDENT_MOE_LAYERS=0-19`、`LK_VERIFY_UVA`/`LK_VERIFY_RESIDENT` 未导出；TP4 EP2、DSPARK block5 + rejection sampling、mrr16、KV 硬钉 4194304；换槽参数全为代码默认（`INTERVAL_MS=200 SWAPS=4 DECAY=0.8 MIN_GAIN=1.5 LOG_S=30`） |
| 快照 | `/mnt/SSD01/models/.llm-env/Lsglang/configs/dsv41_tp4_h100.effective.20261025_184910.conf` |
| 服务 | PID 8745，`http://127.0.0.1:18010`，启动到 `/health` 就绪 **1205 s**；日志 `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261025_184910.log` |
| 就绪自检 | EXPECT=`LK_GPU_CACHE_SLOTS: layer` 命中 80 行；禁止串 0 行；退出码 0 |

## 1. 单层 kernel 基准（H100 / 物理 GPU4 / 不起服务 / CUDA graph 计时 50 iter × 3）

共用形状：E=192 本地专家、H=5120、I=1152、top_k=6、104 槽在显存；`debug-snapshot/scripts/bench_gpu_cache_kernel.py`。

### 1.1 默认 tile 配置（每条路由都算本 rank 上界）

| rows | hit | ms/层 | ×20 |
|---|---|---|---|
| 6 | 1.00 | 1.773 | 35.5 |
| 6 | 0.90 | 4.105 | 82.1 |
| 12 | 1.00 / 0.90 | 2.526 / 7.489 | 50.5 / 149.8 |
| 24 | 1.00 / 0.90 | 4.090 / 12.897 | 81.8 / 257.9 |
| 48 | 1.00 / 0.90 | 4.942 / 25.073 | 98.8 / 501.5 |

张量核 vs FP32 路径相对误差 4.2e-6 ~ 4.4e-6。
eager（prefill tail，主机规划）：256 行 61.8 ms/层；2048 行 77.0 ms/层。

### 1.2 EP2 真实路由（全局 384、本 rank 192，每行约 2.9 条本地；脚本 `scripts/bench_kernel_ep.py`）

旧 = `*.orig-1cf916ebc0.site`（FP32 内核），新 = `b9b8a46b`（bf16 张量核），同形状同权重同计时口径。

| 形状 | 旧 FP32 ms/层（×20） | 新张量核 ms/层（×20） | 倍数 |
|---|---|---|---|
| rows=6 hit=1.00（18 条本地路由） | 7.341（146.8） | 0.901（18.0） | 8.1× |
| rows=6 hit=0.90 | 9.948（199.0） | 1.427（28.5） | 7.0× |
| rows=6 hit=0.00 | 13.523（270.5） | 13.555（271.1） | 1.0× |
| rows=12 hit=1.00 | 12.944 | 1.735 | 7.5× |
| rows=12 hit=0.90 | 13.259 | 1.567 | 8.5× |
| rows=24 hit=0.90 | 21.690 | 2.970 | 7.3× |
| rows=48 hit=0.90 | 31.456 | 4.221 | 7.5× |

eager（新内核有该路径）：rows=128 hit=0.90 → 14.804 ms/层；rows=256 → 22.348；rows=512 → 30.674。

### 1.3 tile 配置对比（rows 6/12, hit 0.9）

| gate_up:down (BN:BK:warps) | r6 hit1.0 | r6 hit0.9 | r12 hit0.9 |
|---|---|---|---|
| 默认 32:64:8 / 64:64:8 | 1.772 | 4.120 | 7.489 |
| 32:128:4 / 64:128:8 | 1.407 | 3.457 | 6.351 |
| 32:128:8 / 64:128:8 | 2.201 | 3.143 | 5.598 |
| 64:64:8 / 128:64:8 | 1.267 | 4.128 | 7.586 |

## 2. 数值 / 策略测试（H100，未设 `TRITON_INTERPRET`）

| 测试 | 结果 |
|---|---|
| `test/manual/kernels/test_lk_verify_uva.py` | 11 passed（17.2 s），含 limit=0.5 的单边 gate 截断、专家打乱存放、eager 热专家跨 tile |
| `test/manual/layers/test_gpu_expert_cache_policy.py` | 3 passed（15.6 s） |

## 3. 端到端（同脚本、同参数、同部署配置；只换代码版本）

筛选条件：TP4 EP2、DSPARK block5 + rejection sampling、`--max-running-requests 16`、`LK_POOL_SLOTS=0` + `LK_GPU_CACHE_SLOTS=104`、物理 GPU4-7。

### 3.1 单流 bench（`bench_ep2_streams.py --k 1 --max-tokens 1000 --temp 0.7`，客户端口径）

| 指标 | 1cf916ebc0 | b9b8a46b |
|---|---|---|
| decode | 19.585 tok/s | **65.053 tok/s** |
| ms/token（客户端） | 51.059 | 15.372 |
| TTFT | 0.891 s | 0.624 s |
| 1000 token 墙钟 | 51.90 s | 15.98 s |
| completion_tokens | 1000 | 1000 |
| prompt_tokens | 34 | 34 |
| accept len（同窗日志中位） | 2.17 | 2.21 |
| 显存峰值 (gpu4/5/6/7 MiB) | 74330/74426/74426/73946 | 78014/78048/78048/77728 |

服务端 `Decode batch` 每 40 步一行口径（`scripts/step_ms.py`）：bench 段 125.0 ms/步 → 25.0 ms/步；复述段 250.0 → 100.0 ms/步。

### 3.2 多流 bench（本轮新增，无上一轮同条件对照）

`--k 4 --max-tokens 500 --temp 0.7`：4 流合计 2000 token / 20.95 s；decode 31.0 tok/s（p50，min 25.8 / max 34.3）；ms/token p50 37.4；TTFT p50 1.825 s；重叠窗口聚合 117.9 tok/s。

### 3.3 复述（`recall_margin.py --n 50`，贪心 + logprob）

| 指标 | 1cf916ebc0 | b9b8a46b |
|---|---|---|
| 结果 | 50/50 同一输出，错误 0/50 | 50/50 同一输出，错误 0/50 |
| 单条耗时 | — | 0.71–0.80 s |
| accept len（同窗日志中位） | 4.27 | 4.28 |

### 3.4 工具调用（`get_weather`，temperature 0）

| 指标 | 1cf916ebc0 | b9b8a46b |
|---|---|---|
| finish_reason | tool_calls | tool_calls |
| 调用 | 1/1 | 1/1（`{"city": "北京"}`） |

### 3.5 prefill

| 测法 | 条件 | 结果 |
|---|---|---|
| 冷 prefill（首次、冷内核） | 单 prompt 16688 token（8192+8192+304，`#cached-token: 0`），max_tokens=1 | 3.199 s → 5217 tok/s |
| 冷 prefill（保持热态，3 次） | 单 prompt 16695 token，前缀加唯一串绕开 HiCache，max_tokens=1 | 1.865 / 1.609 / 1.661 s → 8951 / 10379 / 10048 tok/s |
| 短 prompt | 13 token，max_tokens=1 | 0.422–0.533 s |
| 上一轮同类首次测法（交接文档记载） | 单 prompt 17013 token | 4.053 s → 约 4200 tok/s |

### 3.6 prefill tail（late layer 20-39）是否走 GPU 缓存的日志计数

| 场景 | 计数 |
|---|---|
| 空闲服务 + 单 prompt 304 新 token（16384 命中 HiCache） | `[gpu-expert-cache]` 一个 30 s 窗口内 routes=8932（≈ 20 层 × 128 行 tail × 3.5 条本地路由） |
| 1cf916ebc0 同类窗口（17013 token 纯 prefill 期间） | routes=377（池层 miss 由 CPU `_cpu_prefill` 承担，不进该计数） |

## 4. 运行期日志计数（TP0 EP0，本轮 vs 参照轮）

| 计数项 | 1cf916ebc0 | b9b8a46b |
|---|---|---|
| `LK_GPU_CACHE_SLOTS: layer N: 104 VRAM slots (0.91 GiB), 192 experts pinned on host (1.68 GiB)` | 80（20 层 × 4 rank） | 80 |
| `layer stays on lk_moe` 告警 | 0 | 0 |
| `[pool] slab:` | 1 | 0 |
| `Backend_NUMA`（lk_moe NUMA 后端 init） | 100 | 0 |
| `Using thread binding` | 4 | 0 |
| `Initialized lk_moe ... [CPU]`（20-39 层） | 80 | 80 |
| `Initialized lk_moe ... [GPU]`（0-19 层） | 92 | 89 |
| `max_total_num_tokens` | 3732480 | 3776512 |
| `available_gpu_mem` | 7.26 GB | 7.77 GB |
| 调度进程 RSS（每 rank） | 未采集 | 271.0 / 270.9 / 270.8 / 271.0 GiB |

命中率与路由（TP0 EP0，按 routes 加权）：

| 段 | 1cf916ebc0 | b9b8a46b |
|---|---|---|
| k=1 bench：每步路由 / 每层 / hit | 309 / 15.5 / 97.8% | 372 / 18.6 / 96.0% |
| 复述段：每步路由 / 每层 / hit | 432 / 21.6 / 99.3% | 1040 / 52.0 / 99.1% |
| k=4 段：每步路由 / 每层 / hit | — | 1686 / 84.3 / 97.5% |
| 整个套件窗口加权 hit | 98.4% | 97.8% |

换槽速度（`swaps` 每 30 s，TP0 EP0）：1cf916ebc0 bench 段 972/1017/610；b9b8a46b bench 段 2619/857、复述段 3394、k=4 段 857。

## 5. 产物路径（绝对路径）

| 内容 | 路径 |
|---|---|
| 本文件 | `/data3/shared/wangyf/llm-serving/fix3-dspark-20261025/RESULTS_uva_tensorcore_20261025.md` |
| 单层 bench 原始输出 | `/data3/shared/wangyf/llm-serving/fix3-dspark-20261025/results_uva_tensorcore/bench_kernel_default.out` |
| 旧/新内核同形状（全本地路由） | `.../results_uva_tensorcore/bench_kernel_ab.out` |
| 旧/新内核（EP2 真实路由） | `.../results_uva_tensorcore/bench_kernel_ep.out` |
| tile 配置对比 | `.../results_uva_tensorcore/bench_tile_configs.out` |
| 端到端套件输出 | `.../results_uva_tensorcore/suite.out` |
| 单流 bench JSON/CSV | `.../results_uva_tensorcore/bench_uva_tc_single_20261025_191017.{json,csv,vram.csv}` |
| 多流 bench JSON | `.../results_uva_tensorcore/bench_uva_tc_k4_20261025_191115.json` |
| 复述结果 | `.../results_uva_tensorcore/recall_margin_uva_tc_20261025_191033.jsonl` |
| 工具调用 | `.../results_uva_tensorcore/toolcall_1.json` |
| prefill 探针 | `.../results_uva_tensorcore/prefill_uva_tc_20261025_191017.json`、`prefill_uva_tc_idle_20261025_191355.json`、`prefill_uva_tc_cold2_20261025_191601.json`、`prefill_uva_tc_cold_s{1,2,3}_20261025_*.json` |
| 数值/策略测试输出 | `.../results_uva_tensorcore/pytest_lk_verify_uva_h100.out`、`pytest_swap_policy.out` |
| 部署日志与全树核对 | `.../results_uva_tensorcore/deploy_b9b8a46b.out`、`tree_diff_before.txt`、`tree_diff_after.txt` |
| 参照轮产物 | `/data3/shared/wangyf/llm-serving/fix3-dspark-20261025/results_gpucache/` |
| 服务日志 | `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261025_184910.log`（参照轮 `..._20261025_164259.log`） |

## 6. 测量条件说明（供引用时标注）

- 3.1 / 3.3 / 3.4 的对照两侧使用同一脚本、同一参数、同一部署配置与同一组物理卡，唯一变量是代码版本（`1cf916ebc0` → `b9b8a46b`）。
- 3.5 的 4200 tok/s 来自交接文档记载的上一轮单次测量（prompt 内容不同）；本轮 5217 tok/s 为首次冷内核测量，10.0k tok/s 为冷内核 + 唯一 prompt 的 3 次中位，两者不是同一 prompt，故只按各自条件引用。
- 3.2（k=4）本轮首次采集，无同条件对照。
- 单层 bench（第 1 节）与服务端口径存在差异：按 bench 数值得 1cf916ebc0 卸载层 MoE 约 147–199 ms/步，而其服务端实测整步为 110.8（客户端 51.06 ms/token × accept 2.17）/ 119.2 ms，bench 偏高约 1.3–1.8 倍；两版内核的 bench 倍数（7.0–8.5×）大于端到端倍数（3.2×）。
- `Initialized lk_moe ...` 行由 `process_weights_after_loading()` 在 `_process_mxfp4()` 返回后无条件打印，与是否真的构造 lk_moe 无关，不能作为该判据。
