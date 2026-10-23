# Lsglang 启动参数核对表

- 栈: **Lsglang 1.6.1** (`guqiong96/Lsglang`, `git describe` = `lsglang-v1.6.1-2-g60f6658524`, HEAD `60f6658524`)
- 依赖: `lk_moe==2.7.1`（`python/pyproject.toml:82`）
- 模型: `/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017`（476 GB / 475.26 GiB, 48 shard）
- 硬件: 物理卡 4-7（H100 80GB HBM3, SM90, NV18 全互联, NUMA node1）
- 模板基线: `lsglang/launch/sglang_serve_dsv41_tp4.sh`（SM86 3090+5060Ti 调参）
- 落盘: `scripts/serve_dsv41_tp4_h100.sh` + `stop_dsv41_tp4_h100.sh` + `status.sh` + `health.sh`；配置 `configs/dsv41_tp4_h100.conf`

> **核对方式说明**：核对时 `/mnt/SSD01/models/.llm-env/Lsglang/env/bin/sglang` 尚未生成
> （env-builder 正在 `logs/03_pip_install.log` 安装中），因此 `sglang serve --help` 无法执行。
> 改为**源码级核对**：本仓库 1.6.1 的 CLI 由 `add_cli_args_from_dataclass(parser, ServerArgs)`
> 自动生成（`python/sglang/srt/server_args.py:352`），字段定义在
> `python/sglang/srt/arg_groups/fields/*.py`。**该文件的字段名即 CLI flag 名**（下划线转连字符），
> 与 `--help` 输出同源。环境装好后建议再跑一次 `sglang serve --help` 做二次确认（见文末清单）。

---

## 1. CLI 参数逐条核对

| 目标 flag | 存在 | 定义位置 | 取值/校验 | 结论 |
|---|---|---|---|---|
| `--model <path>` | ✅ | `arg_groups/fields/model.py:45` `model_path`，`aliases=["--model"]` | `str` | 保留（模板同写法）。规范名是 `--model-path` |
| `--tensor-parallel-size 4` | ✅ | `arg_groups/fields/parallel.py:47-53` `tp_size`，`aliases=["--tensor-parallel-size"]` | `int`, 默认 1 | 保留（模板同写法）。规范名是 `--tp-size` |
| `--host 0.0.0.0` | ✅ | `arg_groups/fields/serving.py:65` | `str`, 默认 `127.0.0.1` | 保留 |
| `--port 18010` | ✅ | `arg_groups/fields/serving.py:66` | `int`, 默认 `30000` | **改**：模板 8070 → 18010 |
| `--trust-remote-code` | ✅ | `arg_groups/fields/model.py:100` | `bool`, 默认 `False` | 保留 |
| `--served-model-name` | ✅ | `arg_groups/fields/serving.py:162` | `str` | 保留（模板值不变） |
| `--context-length 1048576` | ✅ | `arg_groups/fields/model.py:104` | `Optional[int]`，默认 `None` = 用 config.json 的 `max_position_embeddings` | **新增**（模板无）。本模型 config 该值本就是 1048576，显式写等价但更明确 |
| `--max-running-requests 16` | ✅ | `arg_groups/fields/schedule.py:31` | `Optional[int]` | **改**：模板 2 → 16 |
| `--mem-fraction-static 0.88` | ✅ | `arg_groups/fields/schedule.py:27` | `Optional[float]` | **改**：模板 0.80 → 0.88 |
| `--max-total-tokens` | ✅ | `arg_groups/fields/schedule.py:39` | `Optional[int]`，注意带 `type_parser=human_readable_int`（可写 `1M`） | **本套件默认不传**（模板 262144）→ 见第 4 节核算 |
| `--chunked-prefill-size 32000` | ✅ | `arg_groups/fields/schedule.py:52` | `Optional[int]`，`-1` = 关闭 | **改**：模板 1024 → 32000 |
| `--swa-full-tokens-ratio 0.1` | ✅ | `arg_groups/fields/schedule.py:146` | `Optional[float]`，未设时 fallback `0.8` | 保留（模板值） |
| `--speculative-attention-mode decode` | ✅ | `arg_groups/fields/spec.py:129` | `choices=["prefill","decode"]`，默认 `prefill` | **未使用**（本套件不开推测解码；模板虽写了但模板也没开 `--speculative-algorithm`，属空转项） |
| `--enable-decoder-swa-bounded-replay` | ✅ | `arg_groups/fields/exec_.py:140` | `Optional[bool]` + `BooleanOptionalAction`，默认 `None`="supported configuration 下默认开启" | 保留（模板值）。对应关闭写法 `--no-enable-decoder-swa-bounded-replay` |
| `--disable-shared-experts-fusion` | ✅ | `arg_groups/fields/exec_.py:940` | `bool`, 默认 `False` | 保留 |
| `--disable-custom-all-reduce` | ✅ | `arg_groups/fields/exec_.py:681` | `bool`, 默认 `False` | **本机删除**（理由见第 3 节） |
| `--reasoning-parser auto` | ✅ | `server_args.py:369`（手工注册，因 choices 动态） | `choices=["auto"] + <已注册 parser>` | 保留 |
| `--tool-call-parser auto` | ✅ | `server_args.py:381`（同上） | `choices=["auto"] + <已注册 parser>` | 保留 |
| `--speculative-algorithm` | ✅ | `arg_groups/fields/spec.py:31` | — | 未使用（备查） |

**未发现"参数不存在"的情况** —— 模板里的 flag 全部真实存在且拼写正确（含 `--tensor-parallel-size` / `--model`
这两个"别名而非规范名"的写法）。

---

## 2. 环境变量逐条核对

依据：`lsglang/README.md` L148-244（官方 env 表）+ `python/sglang/srt/utils/lk_moe_env.py` + `python/sglang/srt/environ.py`。
**注意总开关**：`LK_MOE_HYBRID_ENABLED=0`（默认）时 lk_moe 根本不会被 import，**下面所有 `LK_*` 全部失效**（README:150-160）。

### 2.1 Python 侧可核对的（源码里能 grep 到读取点）

| 变量 | 存在 | 证据 | 模板值 | 本套件取值 | 说明 |
|---|---|---|---|---|---|
| `LK_MOE_HYBRID_ENABLED` | ✅ | README:167；`common.py:1557` 等 | `1` | `1` | 唯一总开关 |
| `LK_THREADS` | ✅ | README:173；lk_moe 引导期直读 | `48` | **`24`** | **改**，见第 3 节 |
| `LK_THREAD_BINDING` | ✅ | README:175 | `NUMA_NODE` | **`CPU_CORE`** | **改**，见第 3 节 |
| `LK_NUMA_INTERLEAVE` | ✅ | README:176；`common.py:1558` | `1` | `1` | 保留（默认已 1，显式写出防上游改默认） |
| `LK_POWER_SAVING` | ✅ | README:178 | `1` | `1` | 保留 |
| `OMP_NUM_THREADS` | ✅ | README:180 | `1` | `1` | 保留 |
| `LK_POOL_SLOTS` | ✅ | README:205；`lk_moe_env.py:282` | `16` | **`288`** | **改**（用户要求），显存核算见第 4 节 |
| `LK_POOL_DECODE` | ✅ | README:206 | — | 留空 | 留空 = 跟随 pool |
| `LK_POOL_WINDOW` | ✅ | README:207；`lk_moe_env.py:132` | — | `32` | 官方默认值 32（clamp 下限 8） |
| `LK_POOL_INFLIGHT` | ✅ | README:208 | — | `16` | 官方默认值 |
| `LK_POOL_STAGE` | ✅ | README:210 | — | `1` | 官方默认值 |
| `LK_POOL_LOG` | ✅ | README:211 | — | `100` | **新增**：每 100 decode step 打 `[pool] hit=`，用于验证 288 槽是否真命中 |
| `LK_GPU_RESIDENT_MOE_LAYERS` | ✅ | README:186 | `""` | `""` | 保留（空 = 无常驻层） |
| `LK_GPU_RESIDENT_MOE_LAYERS_SPEC` | ✅ | README:187 | `""` | `""` | 保留 |
| `LK_GPU_PREFETCH_WINDOW` | ✅ | README:198 | `1` | `1` | 保留 |
| `LK_GPU_PREFILL_MIN_TOKENS` | ✅ | README:194；`lk_moe_env.py:111` | `1024` | `1024` | 保留（档位开关，>0 才开 GPU 预填充） |
| `LK_GPU_PREFILL_ENGINE` | ✅ | README:195；`lk_moe_env.py:125` | `triton` | `triton` | 保留。官方明确 `cutlass` **非默认**：额外 bf16 舍入会漂移长上下文 logits 并翻转临界 EOS |
| `LK_GPU_PREFILL_SUB_M` | ✅ | README:197；`lk_moe_env.py:140` | `8192` | **`16384`** | **改**：README L258 配方值，与 `--chunked-prefill-size 32000` 配套 |
| `LK_GPU_RESERVE_GB` | ✅ | README:199；`kv_cache_configurator.py:241` | — | `0` | 保留语义：`_lk_moe_device_reservation_gb()` 优先用 `lk_moe.get_device_mem_usage()` 上报值并**不再重复扣**；仅当上报失败才回退读此 env。故置 0 让自动路径生效 |
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` | ✅ | `environ.py:1704` `EnvBool(False)` | `1` | `1` | 保留（engram 两表放主机内存） |
| `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK` | ✅ | 源码可查 | `0` | `0` | 保留 |
| `SGLANG_SKIP_P2P_CHECK` | ✅ | `custom_all_reduce_utils.py:430` | `1` | `1` | 保留（跳过 custom AR 的 P2P 能力探测） |
| `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` | ✅ | 模板原样 | `lo` | `lo` | 保留（单机避免 bootstrap 挑到慢网卡） |
| `GLIBC_TUNABLES` | ✅ | 模板原样 | `glibc.malloc.thp_enabled=0` | 同 | 保留 |
| `FLASHINFER_USE_CUDA_NORM` / `FLASHINFER_DISABLE_VERSION_CHECK` | ✅ | 模板原样 | `1` / `1` | 同 | 保留 |

### 2.2 lk_moe C++ 侧直读（Python 源码 grep 不到，按 README 官方表核对）

`LK_THREADS`、`LK_THREAD_BINDING`、`LK_NUMA_ILP`、`LK_NUMA_HUGE`、`LK_POWER_SAVING`、
`LK_POOL_STAGE`、`LK_HF_MBOX`、`LK_DECODE_TS`、`LK_TS_SKIPGEMM`、`LK_MBOX_STOPFILE` 等由
lk_moe 在 bootstrap 时直读（README:173 明确 "read directly by lk_moe at bootstrap"）。
本套件用到的三个（`LK_THREADS` / `LK_THREAD_BINDING` / `LK_POWER_SAVING`）均在官方表中，
拼写与 README 完全一致。

### 2.3 必须删除的（重点核对项）

| 变量 | 模板值 | SM90 上结论 | 证据 |
|---|---|---|---|
| `SGLANG_SM8X_MARLIN_GROUP32` | `1` | **删除（纯 no-op）** | `python/sglang/srt/layers/quantization/fp8.py:562-570` |
| `SGLANG_SM8X_MARLIN_GROUP32_ONLY` | `1` | **删除（纯 no-op）** | 同上 :569-571 |

源码原文（`fp8.py`）:

```python
self.use_marlin_group32 = (
    _is_cuda
    and 80 <= _marlin_group32_cap[0] * 10 + _marlin_group32_cap[1] < 89   # ← 只认 SM80..SM88
    and self.quant_config.weight_block_size is not None
    and len(self.quant_config.weight_block_size) == 2
    and self.quant_config.weight_block_size[1] == 32
    and get_bool_env_var("SGLANG_SM8X_MARLIN_GROUP32", default="true")
)
self.marlin_group32_only = self.use_marlin_group32 and get_bool_env_var(
    "SGLANG_SM8X_MARLIN_GROUP32_ONLY", default="false")
```

- H100 = capability `(9,0)` → `90`；`80 <= 90 < 89` 为 **False**
- ⇒ `use_marlin_group32` 恒 `False` ⇒ `marlin_group32_only` 恒 `False` ⇒ **两个 env 在 SM90 上完全失效**
- 注：本模型 `weight_block_size = [32, 32]`，在 SM86 上**本会**命中，所以模板在 3090 上是有效的；
  换到 H100 就自动失效了 —— 保留不会报错，但会误导人把它当作"已开启的优化"，故本套件**不导出**。
- **不要**替换成 `--moe-runner-backend marlin`：那是 SM8x 路径（模板的 dsv41 脚本本身也没设）。
  本模型量化是 `quant_method=fp8` + `expert_dtype=fp4`，MoE 计算委派给 lk_moe 的量化 kernel（README:326）。

### 2.4 已弃用、不要用的

| 变量 | 结论 | 证据 |
|---|---|---|
| `LK_POOL_LAYERS` | **RETIRED，解析器是死代码**，且不支持与 GPU prefill 按层混用 | README:209 |
| `LVLLM_*`（旧 LvLLM 栈名） | 会被自动改名成 `LK_*` 并打一行 warning；两者同设时 `LK_*` 胜 | README:143；`common.py:1554-1566` |
| `LK_POOL_WINDOW` 设非法值 | clamp 到最小 8，**永不关闭池**（刻意设计：perf 旋钮不该偷偷翻转开关） | `lk_moe_env.py:129-132` |

---

## 3. 与模板的逐项差异（`launch/sglang_serve_dsv41_tp4.sh` → 本套件）

| # | 项 | 模板 | 本套件 | 理由 |
|---|---|---|---|---|
| 1 | 日志目录 | `/tmp/opencode/serve_logs` | `${LSGLANG_ROOT}/logs` | **硬约束：禁止产物落 /tmp** |
| 2 | `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 4 个 **UUID** | 裸 bus id / 裸序号在本环境会静默回落 GPU0；UUID 是唯一可靠形式 |
| 3 | 端口 | `8070` | `18010` | 任务指定 |
| 4 | 模型路径 | `~/Models/DeepSeek-V4.1-Flash-1017` | `/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017` | 任务指定 |
| 5 | `--mem-fraction-static` | `0.80` | `0.88` | 任务指定（H100 80GB 有更大空间） |
| 6 | `--max-running-requests` | `2` | `16` | 任务指定 |
| 7 | `--context-length` | 无 | `1048576` | **新增**，任务要求 1M 上下文 |
| 8 | `--max-total-tokens` | `262144` | **不传** | 模板值会把 1M 上下文的 KV 池人为砍到 256K；该字段是硬上限，不传则 sglang 按 `mem-fraction-static` 自算。核算见第 4 节 |
| 9 | `--chunked-prefill-size` | `1024` | `32000` | 模板 1024 是 8GB 卡调参；1M 上下文用 1024 需要上千个 chunk。README:258 Super Mix Prefill 配方给 `32000` |
| 10 | `--disable-custom-all-reduce` | 有 | **删除** | 该开关 = "禁用 custom all-reduce，回退 NCCL"。模板在 SM86 混合 PCIe 上需要；本机物理卡 4-7 **彼此 NV18（18 条 NVLink）全互联且同构**（`nvidia-smi topo -m` 实测），NVLink 下 custom AR 更快。若真出现 all-reduce/P2P 报错，加回 `SERVE_EXTRA_ARGS` 即可 |
| 11 | `--speculative-attention-mode decode` | 有 | **删除** | 模板并未开 `--speculative-algorithm`，该参数属空转项；本套件同样不开推测解码 |
| 12 | `SGLANG_SM8X_MARLIN_GROUP32` / `_ONLY` | `1` / `1` | **删除** | SM90 上恒失效，见 2.3 |
| 13 | `LK_POOL_SLOTS` | `16` | `288` | 任务要求：每层 288 个 GPU 专家槽。显存核算见第 4 节 |
| 14 | `LK_GPU_PREFILL_SUB_M` | `8192` | `16384` | README:258 配方值，与新的 chunked-prefill-size 配套 |
| 15 | `LK_THREADS` | `48` | **`24`** | 见下 |
| 16 | `LK_THREAD_BINDING` | `NUMA_NODE` | **`CPU_CORE`** | 见下 |
| 17 | `LK_POOL_LOG` | 未设（=0 静默） | `100` | 新增：池命中率是 288 槽是否有效的唯一观测手段 |
| 18 | `LK_GPU_RESERVE_GB` | 未设 | `0` | 显式写出，语义见 2.1 |
| 19 | 其余 | `LK_MOE_HYBRID_ENABLED` / `LK_NUMA_INTERLEAVE` / `LK_POWER_SAVING` / `LK_GPU_PREFETCH_WINDOW` / `LK_GPU_PREFILL_MIN_TOKENS` / `LK_GPU_PREFILL_ENGINE` / `LK_GPU_RESIDENT_MOE_LAYERS(_SPEC)` / `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` / `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK` / `SGLANG_SKIP_P2P_CHECK` / `GLIBC_TUNABLES` / `GLOO_*` / `FLASHINFER_*` / `--swa-full-tokens-ratio` / `--disable-shared-experts-fusion` / `--reasoning-parser` / `--tool-call-parser` / `--trust-remote-code` / `--served-model-name` | 原样保留 | — |

### 3.1 `LK_THREADS`: 基线 `48` → 本套件 `24`（**有证据的偏离，请 lead 确认**）

- README:173 `LK_THREADS` 类型列写的是 **`cores/GPU`**
- README:268 机器级说明: **"HT on -> physical cores / GPUs"**
- 本机实测（`lscpu`）: `Core(s) per socket: 48`，`Socket(s): 2` → **96 物理核**；`Thread(s) per core: 2` → 192 逻辑核
- ⇒ `96 / 4 = 24`
- 模板的 `48` 来自 SM86 参考机（该机核数不同，不是本机），照抄会 2× 超订
- 另: 4 张卡**全在 NUMA node1**，而 node1 只有 **48 个物理核**；`48 × 4 = 192` 线程压在 48 核上与 `CPU_CORE` 独占绑定语义直接冲突
- **回退方式**: `LK_THREADS=48 ./serve_dsv41_tp4_h100.sh`（或改 conf），一行即可

### 3.2 `LK_THREAD_BINDING`: 基线 `NUMA_NODE` → 本套件 `CPU_CORE`（**同上，请 lead 确认**）

- README:175 `CPU_CORE` 是**默认值**，且 mode-2/4（本场景：CPU 混合）调优配方 README:255 明确推荐 `CPU_CORE`
- 模板的 `NUMA_NODE` 也用在内置 quick-start（README:98）里，属合法取值，不是错误
- 本套件选 `CPU_CORE` 的唯一理由：4 卡同节点时 `NUMA_NODE` 无法给每线程独占核
- **回退方式**: 改 conf 的 `LK_THREAD_BINDING="NUMA_NODE"`

> 以上 15/16 两项**行为差异未被任何测试覆盖**（服务尚未启动过）。若 lead 倾向保守复现模板，
> 直接改 conf 两行即可，脚本已做成一键开关。

---

## 4. 显存核算

### 4.1 权重切分（模型自带 config.json 实测值）

| 字段 | 值 |
|---|---|
| `hidden_size` | 5120 |
| `moe_intermediate_size` | 2304 |
| `n_routed_experts` | 384 |
| `num_experts_per_tok` | 6 |
| `num_hidden_layers` | 40 |
| `quantization_config` | `{"quant_method":"fp8","weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}` |
| `engram_layer_ids` | `[1, 14]` |
| `engram_num_embeddings` | `[384006168, 384016682]` |
| `engram_head_dim` | 256 |
| `num_nextn_predict_layers` | 3（`dspark_target_layer_ids=[37,38,39]`） |
| `max_position_embeddings` | 1048576 |

**单个路由专家**: `3 × 2304 × 5120 = 35,389,440` 参数 × 0.5 B（fp4）= **16.875 MiB**
（+ ue8m0 块缩放：`(2304/32)×(5120/32)=11520` B/矩阵 × 3 ≈ 0.033 MiB，可忽略）

**交叉验证**（用于确认 fp4 假设）:
- 全部路由专家 = `384 × 40 × 16.875 MiB` = **253.1 GiB**
- engram 两表 ≈ `(384006168 + 384016682) × 256 × 1 B` ≈ **183 GiB**（1 B/元素）
- 注意力/dense/shared/embedding ≈ 39 GiB
- 合计 ≈ 475 GiB，与权重目录实测 `510,311,045,870 字节 = 475.26 GiB` 吻合 ✅

### 4.2 `LK_POOL_SLOTS=288` 的显存代价

```
池占用 = 288 槽 × 40 层 × 16.875 MiB = 194,400 MiB ≈ 189.8 GiB (4 卡合计，≈47.45 GiB/卡)
```

| 项 | 数值 |
|---|---|
| 4 卡总显存 | `4 × 81,559 MiB = 326,236 MiB` = 318.6 GiB |
| `--mem-fraction-static 0.88` 预算 | `326,236 × 0.88 = 287,088 MiB` ≈ **280.4 GiB** |
| 池占用（288 槽） | **189.8 GiB**（占预算 **67.7%**） |
| engram 两表 | **183 GiB → 主机内存**（`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`） |
| 主机内存余量 | node0 free ≈ 704 GB + node1 free ≈ 390 GB ≈ **1094 GB**，够（`LK_NUMA_INTERLEAVE=1` 跨节点交织） |
| 剩余给 KV + 非 MoE 权重 | ≈ 90 GiB |

> **288/384 = 75% 常驻**。也就是说池要吃掉 2/3 的显存预算来缓存 3/4 的专家。
> 这是用户明确要求的配置，本套件照做并如实标出代价。

### 4.3 ⚠ 本方案最大不确定点（必须在启动时验证）

上面 189.8 GiB 的推算建立在**"TP4 下每卡只持 1/4 个专家"**这个假设上。
若 lk_moe 的槽语义是"**每卡持完整专家**"，则需要 `4 × 189.8 = 759 GiB` → 4×80GB 物理不可能，启动必然 OOM。

- **判据 1**: 启动日志里 lk_moe 的 load 行（README:262 指出这是判断 hybrid 是否生效的官方依据）
- **判据 2**: `LK_POOL_LOG=100` 打开的 `[pool] ... hit=` 行（README:263）
- **判据 3**: 起来后 `scripts/status.sh` 的"4 卡合计已用"
- **降档预案**（脚本已内置提示）:

| `LK_POOL_SLOTS` | 池 ≈ GiB（合计） | 常驻比例 |
|---|---|---|
| `0` | 0（池关闭，纯 CPU 混合） | 0% |
| `72` | ≈ 47.5 | 18.75% |
| `144` | ≈ 94.9 | 37.5% |
| `288` | ≈ 189.8 | 75% |

建议启动顺序：**`0` 起一次拿基线 → `72` → `144` → `288`**，每档看显存余量与 `hit=`。
（改一行 conf 即可，脚本幂等。）

### 4.4 `--max-total-tokens` 核算（模板 `262144` 偏小）

官方 help（`arg_groups/fields/schedule.py:39`）:

> "The maximum number of tokens in the memory pool. **If not specified, it will be automatically
> calculated based on the memory usage fraction.** This option is typically used for development
> and debugging purposes."

即该字段是**硬上限**，不是目标值。`--context-length 1048576` 配 `--max-total-tokens 262144`
会直接把 1M 上下文的 KV 池砍到 256K，**自相矛盾**。

与 lk_moe 的协作（`mem_cache/kv_cache_configurator.py:224-250`）:

```python
def _lk_moe_device_reservation_gb() -> float:
    if not is_lk_moe_feature_enabled(): return 0.0
    report = getattr(lk_moe, "get_device_mem_usage", None)
    if report is not None: reported_gb = float(report()) / (1 << 30)
    if reported_gb > 0.0:
        logger.info("lk_moe static device arena: %.2f GB already resident "
                    "(allocated in MOE ctors, excluded from free memory); "
                    "no additional KV reservation.")
        return 0.0
    gb = float(get_str_env_var("LK_GPU_RESERVE_GB", "0") or 0)
    ...
```

⇒ lk_moe 在 **MOE 构造函数**里就把池显存分配掉，KV 定容时会把它从 free 里排除，**不会重复扣**。
所以 `KV 预算 = mem_fraction_static × 总显存 − 非MoE权重 − lk_moe池`。

**KV 每 token 估算**（MLA，`num_key_value_heads=1` 共享）:

```
(head_dim 512 + qk_rope_head_dim 64) = 576 B/层/token   (fp8 KV)
× 40 层 = 22.5 KiB/token    ← 未计 compress_ratios 的压缩，属上界
1 个 1M 上下文请求 ≈ 22.5 GiB
```

- 想让**单个请求吃满 1M** → `--max-total-tokens` 下限 ≈ `1,048,576 + 余量` ≈ **1,200,000**
- 但 `--max-running-requests 16` 若 16 路全部吃满 1M 需要 `16 × 22.5 = 360 GiB`，物理不可能；
  池会自动限制并发长度，这是正常行为
- **本套件结论：`MAX_TOTAL_TOKENS=""`（不传）**，让 sglang 自算。
  需要硬编码时把 conf 里 `MAX_TOTAL_TOKENS` 填成整数即可（脚本会追加 `--max-total-tokens <n>`）。
  注意该字段带 `human_readable_int` 解析器，`1M` 这种写法也合法。

---

## 5. 启动前还缺什么（checklist）

| # | 缺口 | 状态 | 说明 |
|---|---|---|---|
| 1 | `Lsglang/env/bin/sglang` 入口 | ⏳ **缺** | env-builder 安装中（`logs/03_pip_install.log`）。装好后本脚本 preflight 才能过 |
| 2 | `lk_moe==2.7.1` 可 import | ⏳ 待验 | 必须 `python -c "import lk_moe; print(lk_moe.get_device_mem_usage())"` 能跑，否则 4.3 的自动扣减路径退化到 `LK_GPU_RESERVE_GB` |
| 3 | `sglang serve --help` 二次核对 | ⏳ 待做 | 环境就绪后跑一次，确认第 1 节表格与实际 `--help` 一致（本次是源码核对，同源但非同一产物） |
| 4 | CUDA 13.2.1 前提 | ⚠ 待查 | README:58 明确前置 "CUDA 13.2.1 (driver + toolkit) - matches torch==2.14.1"。宿主驱动 `595.45.04`；**`nvcc` 在 PATH 中未找到**（`which nvcc` 空）。若需源码编译才用得上 nvcc；用预编译 wheel 则只需驱动够。请 env-builder 确认 torch 的 CUDA build |
| 5 | lk_moe 槽语义（4.3） | ⚠ **高风险** | 只能在启动时通过日志判定；已备降档预案 |
| 6 | SM90 未被官方测试过 | ⚠ 中风险 | README:299 支持矩阵里 DeepSeek-V4.1-Flash-1017 的 **sm90 = `ok`（supported, not yet tested）**，sm86 = ✅ tested。所有 benchmark 都是 SM86/SM120 机器。故 SM90 属首次尝试 |
| 7 | 端口 18010 | ✅ 已核 | 当前空闲 |
| 8 | 物理卡 4-7 空闲 | ✅ 已核 | 4 张卡各 `0–4 MiB` used，无计算进程；GPU0/1 各 72,041 MiB 属他人，未触碰 |
| 9 | 权重完整性 | ✅ 已核 | 目录存在，48 个 safetensors 分片齐全 |
| 10 | 主机内存 | ✅ 已核 | node0 free ≈ 704 GB + node1 free ≈ 390 GB；engram 183 GiB 可容纳 |
| 11 | NUMA 形态偏离官方建议 | ⚠ 低风险 | README:267 建议 "NUMA node 数取 GPU 数的倍数 (2/4/8)"。本机 2 节点 / 4 GPU，且 **4 张卡全在同一节点**，不满足。缓解：`LK_NUMA_INTERLEAVE=1` 让主机权重跨节点交织 |
| 12 | `/dev/shm` | ⚠ 待查 | 实测 1008 GB 总 / **已用 429 GB** / 可用 580 GB。engram 主机表若走共享内存布局（`SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT="shared"`，默认）需确认 580 GB 够 |

---

## 6. 脚本自检记录（未启动服务，全部只读）

全部脚本已做 `bash -n` 语法检查并通过；已做幂等/失败路径的端到端演练：

| 演练 | 命令 | 期望 | 实测 |
|---|---|---|---|
| preflight 全量 + 打印命令 | `serve_dsv41_tp4_h100.sh --dry-run --allow-missing-sglang` | exit 0，preflight 全绿 | ✅ exit 0 |
| 缺少 sglang 入口时拒启 | `serve_dsv41_tp4_h100.sh` | exit 1 | ✅ exit 1 |
| 未运行时停止 | `stop_dsv41_tp4_h100.sh` | exit 0，无操作 | ✅ exit 0 |
| 未运行时探活 | `health.sh` | exit 4 | ✅ exit 4 |
| 未运行时状态 | `status.sh --brief` | exit 4 | ✅ exit 4 |
| 配置快照落盘 | 上述 dry-run | `configs/*.effective.<ts>.conf` | ✅ 已生成 |

preflight 实测输出（含 UUID→物理序号逐张断言）:

```
1) 端口 18010            [ OK ]   端口 18010 空闲
2) 依赖文件              [WARN]   sglang 入口缺失 (--allow-missing-sglang 忽略)
                         [ OK ]   权重目录存在 / 48 个 safetensors 分片
3) GPU 绑定 (UUID)       [ OK ]   4 个 UUID 逐一 -> 物理 GPU4..7 (与期望一致)
                         [ OK ]   禁止卡 物理 GPU0 / GPU1 未出现在可见列表
4) 目标卡显存占用        [ OK ]   GPU4..7 各 0 MiB — 空闲
5) CPU / NUMA 自检       [ OK ]   逻辑 CPU 192, 物理核 96
                         [ OK ]   LK_THREADS=24 等于 README 公式值 (96 ÷ 4)
                         [ OK ]   LK_MOE_HYBRID_ENABLED=1 / LK_NUMA_INTERLEAVE=1
                         [ OK ]   未设置 SM8x MARLIN_GROUP32*
                         [WARN]   LK_POOL_SLOTS=288 → 池显存约 189 GiB 合计 (见 4.3 不确定性)
```

要执行的完整命令（`--dry-run` 输出，**尚未执行**）:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-7561b28f-b049-bbdb-31f8-16efeb06663d,GPU-fb59db94-0e4e-4c4f-8ae5-4d0fdd1e0f0b,GPU-3c5d7267-445c-4679-bce9-503bedd0c920,GPU-3c0f2856-a6e3-b9d5-fd10-a08f55579fff \
LK_MOE_HYBRID_ENABLED=1 LK_THREADS=24 LK_THREAD_BINDING=CPU_CORE \
LK_NUMA_INTERLEAVE=1 LK_POWER_SAVING=1 OMP_NUM_THREADS=1 \
LK_POOL_SLOTS=288 LK_POOL_WINDOW=32 LK_POOL_INFLIGHT=16 LK_POOL_STAGE=1 LK_POOL_LOG=100 \
LK_GPU_RESIDENT_MOE_LAYERS= LK_GPU_RESIDENT_MOE_LAYERS_SPEC= LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 LK_GPU_PREFILL_ENGINE=triton LK_GPU_PREFILL_SUB_M=16384 \
LK_GPU_RESERVE_GB=0 SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 SGLANG_SKIP_P2P_CHECK=1 \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
FLASHINFER_USE_CUDA_NORM=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
/mnt/SSD01/models/.llm-env/Lsglang/env/bin/sglang serve \
  --model /mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017 \
  --tensor-parallel-size 4 --context-length 1048576 \
  --max-running-requests 16 --mem-fraction-static 0.88 \
  --chunked-prefill-size 32000 --swa-full-tokens-ratio 0.1 \
  --disable-shared-experts-fusion --reasoning-parser auto --tool-call-parser auto \
  --trust-remote-code --enable-decoder-swa-bounded-replay \
  --host 0.0.0.0 --port 18010 \
  --served-model-name DeepSeek-V4.1-Flash-1017
```

（`# (不设) SGLANG_SM8X_MARLIN_GROUP32 / _ONLY` —— 已刻意排除，见 2.3）
