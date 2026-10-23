# HiCache 预备报告（DeepSeek-V4.1-Flash-1017 · TP4 × 4×H100 SM90）

> 性质：**只读调研 + 配置草案 + 预算核算 + 验证方案**。本轮**未**启动/重启任何服务，**未**修改 `configs/`、`scripts/`、`site-packages/`，**未**占用 GPU（仅 `nvidia-smi -L` 查询）。
> 代码口径：本轮服务运行用的解释器是 `env/bin/python`（无 `PYTHONPATH`、无 `.pth` 指向源码树），即**已安装副本** `env/lib/python3.1/site-packages/sglang/`；下文的 file:line 一律给**源码树** `/mnt/SSD01/models/.llm-env/Lsglang/lsglang/python/sglang/…`（=`git HEAD 60f665852403`），凡本文引用到的文件均已逐字节比对与 site-packages 副本**一致**（例：`srt/arg_groups/fields/memory.py`、`srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`）。为行文简洁，所有路径以 `sglang/…` 起头，实际前缀为 `/mnt/SSD01/models/.llm-env/Lsglang/lsglang/python/sglang/`。
> 现场事实来源：服务 pid 16080；日志 `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261023_081016.log`（下称 **L**）；配置 `/mnt/SSD01/models/.llm-env/Lsglang/configs/dsv41_tp4_h100.conf`（下称 **C**）；进程环境 `/proc/16080/environ`、命令行 `/proc/16080/cmdline`。

---

## 0. 本轮实测量（原文摘录）

| 项 | 值 | 证据 |
|---|---|---|
| 生效命令行 | `--tensor-parallel-size 4 --mem-fraction-static 0.88 --max-running-requests 16 --context-length 1048576 --chunked-prefill-size 8192 --moe-runner-backend flashinfer_mxfp4 --max-total-tokens 4194304 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --speculative-attention-mode decode --cuda-graph-max-bs-decode 16 --cuda-graph-max-bs-prefill 16 --enable-decoder-swa-bounded-replay …` | `/proc/16080/cmdline` |
| 进程环境 | `LK_GPU_RESIDENT_MOE_LAYERS=0-19`、`LK_POOL_SLOTS=128`、`LK_NUMA_INTERLEAVE=1`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`CUDA_LAUNCH_BLOCKING=1`、`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`、`SGLANG_DSV41_{SKIP_CAPTURE_PREWARM,AUTOTUNE_DUMMY_DECODE,SWA_PAD_SENTINEL,POST_CONVERT_RELEASE,DUMP_FLASHMLA_ARGS}=1` | `/proc/16080/environ` |
| KV 预算/rank | `bytes_per_full_token=1670.75`、`available_bytes=10.03 GB`、`swa_fixed=1.34 GB`、`c128_state_fixed=0.00`、`c2_state_fixed=0.00`、`swa_ring_fixed=0.00`、`c4_state_fixed=0.00`、`full_token=5585408` | L:774,776,783,787 |
| 池尺寸（cap 前/后各一次） | `full=5585408, swa=59904, c4=1396352, c128=43636` → `full=4194304, swa=59904, c4=1048576, c128=32768, c4_state=0, c128_state=0` | L:775,778,782 |
| SWA 定容模式 | `mode=cap, swa_tokens=59904, request_cap+headroom=59904, prefix_tails=64` | L:772 |
| KV 池构造 | `Initialize DeepSeekV4TokenToKVPool with max_num_reqs=16 swa_size=59904 c4_size=1048576 c4_logical_size=1048576 c128_size=32768 c4_state_pool_size=0 c128_state_pool_size=0 unified=False unified_fp8=False`；`DSV4 SWA storage: worker=target, storage=paged, paged_allocator=True` | L:792,796 |
| 收尾 | `max_total_num_tokens=4194304, …, available_gpu_mem=7.04 GB` | L:960 |
| 专家池 | `[pool] slab: 11.21 GiB (20 layers x 573.75 MiB)` ×4 rank | L:586,590,594,598 |
| 主机内存 | `MemTotal 2015.7 GiB`、`MemFree 1000 GiB`、`MemAvailable 1153 GiB`、`Cached 781 GiB`、`Shmem 622.8 GiB`、`Swap 0` | `/proc/meminfo` |
| NUMA | node0 size 1007.7 GiB / free 515.4 GiB；node1 size 1007.9 GiB / free 484.7 GiB；距离 21 | `numactl --hardware` |
| cgroup | `memory.max = max`（无上限）、无 `memory.high`；可见 `…/app.slice/dsh.service` 目录存在 | `/sys/fs/cgroup/user.slice/user-1084.slice/user@1084.service/app.slice/dsh.service/memory.max` |
| 4 个 scheduler RSS | 228.12 / 228.15 / 228.22 / 228.24 GiB | `ps -eo pid,rss,comm` |
| dspark draft 架构（决定 HiCache draft 打包模式） | `Draft checkpoint bundles a DSpark head; loading draft arch DeepseekV4ForCausalLMDSpark.`（×4 rank） | L:165,166,173,177；改写处 `configs/model_config.py:940-955` |
| **解析后 server_args（日志原文, L:80）** | `page_size=256`、`c128_page_size=16`、`swa_full_tokens_ratio=0.8`（`_swa_full_tokens_ratio_explicitly_set=False` ⇒ cap 模式）、`attention_backend='dsv4'`、`dsv4_attn_backend='flashmla'`、`enable_encoder_swa_bounded_replay=False` / `enable_decoder_swa_bounded_replay=True`、`enable_hierarchical_cache=False`、**`hicache_ratio=2.0` 与 `hicache_host_memory_fraction=0.8` 同时存在**、`enable_metrics=False`、`cuda_graph_config={'decode':{'backend':'full','bs':[1,2,3,4,5,6,7,8,10,12,14,16]},'prefill':{'backend':'disabled'}}` | L:80 |
| 被引用外部既有核算 | 主机常驻 = 专家 275.77 GiB + engram 188.83 GiB = **464.60 GiB**；node1 曾测空闲 401.77 GiB；每 full-token 1630 B（=652×1 + 326×3）；真正落全量 KV 的层 = {2,8,14,20} | `/mnt/SSD01/models/.llm-env/Lsglang/docs/budget-lsglang.md:19,21,183-184,378-403` |

---

## 1. HiCache 的开关与语义（本版本 sglang）

字段定义：`sglang/srt/arg_groups/fields/memory.py`。归一化/校验：`sglang/srt/arg_groups/hicache_hook.py`、`sglang/srt/arg_groups/kv_cache_hook.py`。

### 1.1 参数表（名称 / 取值 / 默认 / 单位与分母 / 定义处 / 实现处）

| CLI 参数 | 取值 | 默认 | **单位与分母** | 定义 file:line | 语义实现 file:line |
|---|---|---|---|---|---|
| `--enable-hierarchical-cache`（别名 `--enable-hicache`） | bool | `False` | — | `arg_groups/fields/memory.py:98-101` | `mem_cache/kv_cache_builder.py:290-425` |
| `--hicache-host-memory-mode` | `cache` / `buffer_only` | `cache` | — | `memory.py:102-108` | `hicache_hook.py:206-255` |
| `--hicache-ratio` | float | `None`（运行时置 `2.0`；buffer_only `1.2`；decode 回退档 `0.2`） | **无量纲；= 该 rank host KV 池 token 容量 ÷ 该 rank device KV 池 token 容量**（`self.size = int(device_capacity_tokens * host_to_device_ratio)`） | `memory.py:109-112` | `hicache_hook.py:100-103`；`pool_host/base.py:216`；`kv_cache_builder.py:226-242`；DSV4 专用页数公式 `hybrid_cache/hybrid_pool_assembler.py:590-619` |
| `--hicache-size` | int | `0`（不启用） | **GB = 1e9 字节；per rank（每 rank）** —— `int(host_size * 1e9 // size_per_token)`；PP 组内取 token 容量最小值同步 | `memory.py:113-116` | `pool_host/base.py:111-150,211-219`；`host_memory.py:136-147`（错误文案原文 "set --hicache-size explicitly (GB per rank)"） |
| `--hicache-host-memory-fraction` | float ∈ (0,1] | `0.8` | **本机全体 rank 的 host 池合计**占"可见主机内存（受 cgroup `memory.max`/`memory.high` 约束，先扣 10 GiB 保留）"的比例；**仅当 ratio 与 size 都未显式给出时生效** | `memory.py:117-120` | `hicache_hook.py:104-105`（显式 ratio/size ⇒ 置 `None`）；`hicache_auto_size.py:82-125`；`pool_host/base.py:28,84-108` |
| `--hicache-write-policy` | `write_back` / `write_through` / `write_through_selective` | `write_through` | — | `memory.py:121-127` | `hybrid_cache/hybrid_cache_controller.py`（write_policy 参数 `hybrid_pool_assembler.py:1171`） |
| `--hicache-io-backend` | `direct` / `kernel` / `kernel_ascend` | `kernel` | — | `memory.py:128-134` | layout/io 自动改写 `hicache_hook.py:153-176` |
| `--hicache-mem-layout` | `layer_first` / `page_first` / `page_first_direct` / `page_first_kv_split` / `page_head` | `page_first` | — | `memory.py:135-147` | 同上；各 host 池按 layout 分配 `mem_cache/memory_pool_host.py:270-301` |
| `--hicache-storage-backend` | `file/sim/mooncake/npu_memcache/hf3fs/nixl/aibrix/dynamic/eic/simm/mori/shm/tensorcast/seaweedfs` | `None` | L3；缺省即"只做 L1(device)+L2(host)" | `memory.py:148-169` | `mem_cache/hicache_storage.py` |
| `--hicache-storage-prefetch-policy` | `best_effort` / `wait_complete` / `timeout` | `timeout` | — | `memory.py:170-176` | — |
| `--hicache-storage-backend-extra-config` | JSON / `@file` | `None` | — | `memory.py:177-180` | — |
| `--hicache-storage-prefetch-retry-poll-interval` | int | `8` | 调度 pass 数 | `memory.py:181-194` | `managers/scheduler.py:3340-3360` |
| `--hicache-storage-prefetch-retry-max-attempts` | int | `8` | 次数 | `memory.py:195-204` | 同上 |

### 1.2 互斥与自动改写（硬校验）

| 规则 | 证据 |
|---|---|
| `--enable-hierarchical-cache` 与 `--disable-radix-cache` 互斥 | `arg_groups/kv_cache_hook.py:393-397` |
| 与 `--enable-lmcache`、`--enable-unified-cache-external-linker` 互斥 | `kv_cache_hook.py:434-441`；`hicache_hook.py:31-42` |
| 与 `--enable-kv-cache-sharding` 互斥 | `arg_groups/kv_shard_hook.py:123-128` |
| 与 `--enable-int8-mamba-checkpoint` 互斥 | `arg_groups/mamba_hook.py:83-88` |
| `--enable-unified-memory` + EAGLE/EAGLE3 + HiCache 被 assert 拒绝 | `kv_cache_hook.py:671-679` |
| SM100 原生 NVFP4 + trtllm_mha + HiCache 被拒 | `kv_cache_hook.py:257-263` |
| `page_first` + `io=direct` ⇒ 自动改 `page_first_direct`；`page_first_direct` + `io=kernel` ⇒ 自动改 `io=direct` | `hicache_hook.py:153-176` |
| DCP>1 时：L3(storage) 不支持；spec 只允许 DSPARK；仅 MLA 宿主池 | `hicache_hook.py:110-150` |
| decode 侧 `--disaggregation-decode-enable-radix-cache` + SWA 模型不支持 HiCache | `mem_cache/kv_cache_builder.py:263-271` |

### 1.3 **DeepSeek-V4 专属硬约束（本轮关键）**

| # | 事实 | 证据（原文） |
|---|---|---|
| D1 | **DSV4 拒绝 `--hicache-size`** | `hybrid_cache/hybrid_pool_assembler.py:603-607`：`raise ValueError("DeepSeek V4 HiCache currently does not support --hicache-size; use --hicache-ratio instead.")` |
| D2 | **必须显式给 `--hicache-ratio`**：只给 `--enable-hierarchical-cache` 时，`hicache_ratio` 被默认成 2.0，而 `hicache_host_memory_fraction` 仍为 **0.8**（置 `None` 的条件看的是用户**显式**输入）⇒ 进入自动定容路径，而自动定容的 `_estimate_hicache_bytes` 只接受 `MHATokenToKVPool / MLATokenToKVPool / SWAKVPool / HybridLinearKVPool / MiniMaxSparseKVPool`；`DeepSeekV4TokenToKVPool` 的基类是 `BaseSWAKVPool`（不是 `SWAKVPool`）⇒ 抛 `ValueError: HiCache auto-sizing does not support DeepSeekV4TokenToKVPool; set --hicache-ratio or --hicache-size explicitly.` | `hicache_hook.py:100-105`；`hicache_auto_size.py:35-41,63-79,90-95`；`mem_cache/deepseek_v4_memory_pool.py:930`（`class DeepSeekV4TokenToKVPool(BaseSWAKVPool)`）vs `mem_cache/swa_memory_pool.py:21`（`class SWAKVPool(BaseSWAKVPool)`）；调用点 `kv_cache_builder.py:420-425`。**运行期直接佐证**：本轮（未开 HiCache）日志 L:80 的解析后参数里 `hicache_ratio=2.0` 与 `hicache_host_memory_fraction=0.8` **同时存在**，正是"默认 ratio 已填、fraction 未被置 None"的状态 |
| D3 | DSV4 的 **FULL 锚池是纯逻辑池，不占主机字节** | `mem_cache/memory_pool_host.py:61-93`（`LogicalHostPool`：`self.kv_buffer = None`，`size_per_token = 0`）；装配处 `hybrid_pool_assembler.py:947-959` |
| D4 | DSV4 host 池由 6 类池组成：SWA、C4(及其 indexer/rope)、C128(及其 rope)、C4_STATE、C4_INDEXER_STATE，**外加按 `compress_ratios` 展开的 C1/C2 低比池 + 其 indexer** | `hybrid_pool_assembler.py:879-1181`（SWA 961-990、C4 992-1048、C4 state 1050-1090、C128 1092-1153、**C1/C2 恒被追加** 1155-1159 + 定义 748-824） |
| D5 | host 页数公式（每 rank，按该 rank 自己的 device 页数） | `hybrid_pool_assembler.py:590-619`：`device_full_pages=ceil(size_full/page_size)`；`full_host_pages=int(device_full_pages*ratio)`；`device_swa_pages=ceil(kvcache.swa_size/swa_page_size)`；`swa_host_pages=int(device_swa_pages*ratio)`；`c128_host_pages=full_host_pages`（GPU 无 `c128_attn_allocator`，NPU 才改写） |
| D6 | 每个 host 池字节 = `layer_num × num_host_pages × item_bytes`（**不是**乘 KV token 数；`item_bytes` 是设备行的原始字节宽度） | `memory_pool_host.py:251`；`pool_host/base.py:239`（`requested_bytes = self.size * self.size_per_token`） |
| D7 | 单池准入检查用**每 rank**可用内存，且**不与其它池累计**（`host_memory_budget_bytes()` 无入参即重新采样 psutil）；自动定容路径才有 `host_memory_budget_scope` 的"一次性账面" | `memory_pool_host.py:252`；`pool_host/base.py:58-65,84-108,240-247` |
| D8 | SM120 的 direct-SWA 布局被 DSV4 HiCache 显式拒绝（SM90 不涉及） | `hybrid_pool_assembler.py:633-643` |
| D9 | indexer 页尺寸：SM90 ⇒ **64**（该 flag 要求 `sm//10==10` 或 120/121） | `mem_cache/deepseek_v4_memory_pool.py:519-526`；`layers/deep_gemm_wrapper/configurer.py:69-83` |

---

## 2. "预算 128 GB + 乘数 6" 落到参数上

### 2.1 换算定律（每 rank）

由 D5/D6：**host 字节 ≈ ratio × (该 rank 被镜像的 device 池字节和)**。对本部署逐池代入（**实测 `page_size=256`**（L:80）与 **`swa_page_size = get_schedule().page_size = 256`**（`kv_cache_configurator.py:1796-1798`）、SWA 为 cap 模式（`_swa_full_tokens_ratio_explicitly_set=False`，L:80）、`unified=False`，`bytes_per_page_padded = ceil_div(页token×584,576)×576`（`deepseek_v4_memory_pool.py:294-307`）；C1/C2 的层与池尺寸 `deepseek_v4_memory_pool.py:1507-1548,1563-1591`；层归属 `budget-lsglang.md:378-403` + `config.json compress_ratios/kv_source_layer_ids`；indexer 页尺寸 64（D9））：

| 池 | 层数 | device 页数(整份) | item_bytes | device 字节/rank | host 字节/rank |
|---|---|---|---|---|---|
| FULL 锚（LogicalHostPool） | — | `ratio×16384` 页(256t) | 0 | 0 | **0**（D3） |
| C1 KV（ratio=1，owner 层 20） | 1 | 16384 × 1 | 149760 (= 256t×585) | 2.454 GB | `ratio × 2.454 GB` |
| C2 KV（ratio=2，owner 层 2/8/14） | 3 | 16384 × 3（128t 页） | 74880 (= 128t×585) | 3.681 GB | `ratio × 3.681 GB` |
| C1 indexer（fp4，68 B/token） | 1 | 16384 × 1（每 256t 页 4 个 64t index 页） | 17408 | 0.285 GB | `ratio × 0.285 GB` |
| C2 indexer | 3 | 16384 × 3（每 256t 页 2 个 64t index 页） | 8704 | 0.428 GB | `ratio × 0.428 GB` |
| SWA（40 target + 1 dspark draft 层） | 41 | 234 页(256t) | 149760 | 1.437 GB | `ratio × 1.437 GB` |
| C4 / C128 / 其 indexer / rope / state | **0** | — | — | 0 | 0（本模型 `compress_ratios` 无 4/128；`c4_state=c128_state=0`，见 L:775,778） |
| **合计（被镜像 device 池）** | | | | **≈ 8.284 GB / rank = 7.715 GiB / rank** | **ratio × 8.284 GB / rank** |

> 该合计对"全局 page_size 取 128 还是 256"以及 "swa_page_size 取 128 还是 256"**不敏感**：页数 × item_bytes 的乘积不变（例如 SWA：`468×74880 = 234×149760`，C1 KV：`32768×74880 = 16384×149760`）。因此上表与最终合计只依赖 `full_token=4194304`、`swa=59904`、`bytes_per_token=584`、层数 {1,3,41}。

主机总量 = 4 × 每 rank（KV 不按 TP 切分，`budget-lsglang.md:37,204,449`；本版本同口径 `deepseek_v4_memory_pool.py:273-315` 无 tp 除法）。

### 2.2 与"128 GB ÷ 4 rank = 32 GB/rank ⇒ ratio≈6"是否自洽

| 口径 | 数 | 结论 |
|---|---|---|
| ratio=6 的每 rank device 镜像需求 | `32 GB ÷ 6 = 5.33 GB/rank` | **实测被镜像 device 池 = 8.284 GB/rank**（7.71 GiB/rank），比 5.33 GB 大 **1.55×** |
| ratio=6 的实际主机占用 | `6 × 8.284 = 49.7 GB/rank` ⇒ **198.8 GB（185.1 GiB）合计** | 与"约 128 GB"差 **+70.8 GB（+55%）** |
| 命中"合计 ≈128 GB"的 ratio | `128 ÷ (4 × 8.284) = 3.86`（每 rank 32.0 GB） | — |
| 命中"合计 ≈128 GiB"的 ratio | `137.4 ÷ (4 × 8.284) = 4.15` | — |

**判定**："每 rank 128/4=32 GB + 乘数 6" 的两个前提里，**乘数 6 与"每 rank 32 GB"在同一 device 口径下不自洽**；若二者都必须成立，则要求每 rank 被镜像 device 池 = 5.33 GB，而实测为 8.284 GB。反过来说：**128 GB 对应 ratio≈3.9，ratio=6 对应 ≈199 GB**。

> 注：lead 假设里的"device KV 5.3 GiB/rank"未在本轮现场找到对应量。可对照的实测口径有：被镜像池 8.284 GB(7.71 GiB)/rank；压缩 KV 系数 `1630 B/full-token × 4194304 = 6.837 GB`(6.37 GiB)/rank；KV 预算 `available_bytes = 10.03 GB`(L:774)；SWA 池 1.437 GB/rank。

### 2.3 ratio → 主机占用对照表（每 rank / 4 rank 合计，十进制 GB）

| ratio | 每 rank host | 4 rank 合计 | 备注 |
|---|---|---|---|
| 2 | 16.6 GB | 66.3 GB | 默认值（`hicache_hook.py:100-103` 给 cache 模式的默认） |
| 3 | 24.9 GB | 99.4 GB | |
| **3.86** | **32.0 GB** | **127.9 GB** | 对应"约 128 GB" |
| 4 | 33.1 GB | 132.5 GB | |
| **6** | **49.7 GB** | **198.8 GB** | 对应"乘数 6" |
| 8 | 66.3 GB | 265.1 GB | 单池仍 < 256 GiB 注册分块阈值 |

### 2.4 主机侧附加上限（含 pinned/注册开销）

| 项 | 量 | 证据 |
|---|---|---|
| host 池本体 | `ratio × 8.284 GB/rank`（pinned） | 2.1 |
| 注册方式 | CUDA 默认走 `alloc_with_host_register`：先分配普通张量再 `cudaHostRegister`（**整块页锁定，不可回收**）；分块阈值 `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB` 默认 **256**（ratio=6 时各池均 < 256 GiB，故单次注册） | `mem_cache/pool_host/common.py:143-199,424-431`；`srt/environ.py:796` |
| 元数据（已验证的一项） | FULL 锚池 `free_slots = torch.arange(size, int64)` ⇒ `ratio × 4194304 token × 8 B`（ratio=6 ⇒ **201.3 MB/rank**） | `memory_pool_host.py:96-101`；`chunk 装配 hybrid_pool_assembler.py:947-949` |
| 元数据（未逐项核对） | `DeepSeekV4PagedHostPool.clear()` 的 `mem_state/slot_used/free_slots`，量级 ≪ 池本体 | `memory_pool_host.py:425-433` |
| 元数据（记账表） | `device_ptrs` 每池一个小 CUDA 表 | `memory_pool_host.py:314-327` |
| **主机占用上限（ratio=6，含元数据）** | **≈ 198.8 GB + 4×0.20 GB ≈ 199.6 GB（185.9 GiB）** | 合计 |

### 2.5 device（HBM）侧额外开销（同一个 HiCache 开关会额外吃显存）

| 池 | staging 页容量 | 层数 | item_bytes | device 字节/rank |
|---|---|---|---|---|
| SWA | min(234,64)=64 | 41 | 149760 | 393.0 MB |
| C1 KV | 64 | 1 | 149760 | 9.6 MB |
| C2 KV | 64 | 3 | 74880 | 14.4 MB |
| C1/C2 indexer | 64 | 1 / 3 | 17408 / 8704 | 1.1 / 1.7 MB |
| **合计** | | | | **≈ 419.8 MB / rank（0.391 GiB/rank；4 rank ≈ 1.68 GB / 1.56 GiB）** |

- 触发条件：`layout == "page_first"` **且** 平台具备 `hicache_device_kernels`（SM90/CUDA 满足）；默认 layout 就是 `page_first` ⇒ **默认即分配**。
- 分配时机在 KV 定容**之后** ⇒ 吃的是 `--mem-fraction-static 0.88` 的余量（L:960 `available_gpu_mem=7.04 GB`），**不从 KV 池里切**。
- 证据：`memory_pool_host.py:333-349`、`pool_host/base.py:30`。若改用 `--hicache-mem-layout page_first_direct` + `--hicache-io-backend direct`（自动改写规则 `hicache_hook.py:153-176`）则不建该 staging，但 io 后端语义随之改变（未实测，见 §7）。

### 2.6 推荐参数组合（写"每 rank X GB / ratio=Y"）

> 以下为**草案**，尚未应用、未验证。

| 方案 | ratio | 每 rank host | 4 rank 合计 | 目标 |
|---|---|---|---|---|
| **A（对齐"约 128 GB"）** | `--hicache-ratio 3.86` | 32.0 GB/rank | 127.9 GB | 命中用户给的预算 |
| **B（对齐"乘数 6"）** | `--hicache-ratio 6` | 49.7 GB/rank | 198.8 GB | 命中用户给的乘数（超预算 55%） |
| C（默认） | 不传（=2.0） | 16.6 GB/rank | 66.3 GB | 最小风险档 |

两种方案的**其余参数取默认**（`write_through` / `page_first` / `kernel` / 无 L3）：`--enable-hierarchical-cache --hicache-ratio <Y>`。**两者都必须显式传 ratio（D2）**。

---

## 3. 兼容性逐条判定

| # | 主题 | 判定 | 证据与理由 |
|---|---|---|---|
| 1 | **MLA / DeepSeek-V4.1 + flashmla 稀疏 decode** | **有专门支持（代码级）**；本部署仍有两个硬约束 | 本 fork 有 DSV4 专属 HiCache 栈：`build_deepseek_v4_hicache_stack`（`hybrid_pool_assembler.py:879-1181`）、host 池类 `DeepSeekV4PagedHostPool`/`DeepSeekV4StateHostPool`（`memory_pool_host.py:210,721`）、页数公式（`hybrid_pool_assembler.py:590-619`）、low-ratio 镜像（748-824）与 sidecar 挂载表（1816-1841）；attention 文件 `layers/attention/deepseek_v4_backend.py`、arg 钩子 `arg_groups/deepseek_v4_hook.py` 中 `hicache|hierarchical` **0 命中** ⇒ 不要求特定 attention 后端、不要求特定 layout。**约束**：D1（不能用 `--hicache-size`）、D2（必须显式 ratio）。DCP>1 才有 MLA 专属限制（`hicache_hook.py:139-144`），本部署 DCP=1。 |
| 2 | **CUDA 图（本轮实测 decode 12 桶；prefill 图关闭）+ `--enable-decoder-swa-bounded-replay`** | **无显式不兼容证据；未实机验证** | 全树检索：HiCache 开启时**没有任何代码路径关闭图捕获**（`srt/model_executor/` 下承载图逻辑的文件中 `hicache|hierarchical` 0 命中）；`managers/scheduler.py:4304-4308` 只有一句 TODO `# todo (zhiqiang): disable cuda graph execution if hicache loading triggered`（**未实现**）；H↔D 传输由调度线程在 forward 之前发起，且 `cache_controller.load_fence_stream / write_fence_stream = tp_worker.model_runner.forward_stream`（`scheduler.py:613-626`），即传输与图重放**同流有序**；`--enable-decoder-swa-bounded-replay` 走的是 `enable_decoder_swa_bounded_replay` 特性（`fields/exec_.py:140`；`models/deepseek_v4.py:4016`；`deepseek_v4_backend.py:1330,1775-1790,2886,3148`），与 `pool_configurator.py:1175` 用的 `enable_encoder_swa_bounded_replay` **是两个字段** —— 本部署的 SWA 池是 paged（L:796），说明 target 侧未走 encoder-replay 请求窗。结论仅到"无代码级冲突"，**本组合（decode 12 桶 + bounded replay + HiCache）无任何日志/测试证据**。 |
| 3 | **lk_moe 专家池（`LK_POOL_SLOTS` / `LK_GPU_RESIDENT_MOE_LAYERS`）** | **无代码级交互/冲突** | lk_moe 只经 `LK_*` 环境变量与专家层接口生效；sglang 侧 HiCache 代码与 `LK_*` 无交叉调用（`srt/` 内 `hicache` 与 `lk_moe` 无同文件命中）；host-专家母本 275.77 GiB 与 HiCache host 池是**两笔独立的主机占用**（`budget-lsglang.md:180-184`）。唯一交互在资源：§2.5 的 **0.391 GiB/卡** device staging 要从 0.88 的余量里出，与 slab/resident 竞争的是同一块 slack。 |
| 4 | **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** | **HiCache L1/L2 路径无已知冲突；本机叠加未实测** | 全树 `expandable_segments` 的冲突记录只在两处：`disaggregation/mooncake/utils.py:30-61`（PD/Mooncake 传输显式要求关闭，本部署无 PD、无 storage backend）、`weight_cache/daemon.py:264,416-424`（权重缓存 daemon，未启用）。HiCache 的 host 分配走 `cudaHostRegister`（`pool_host/common.py:143-199`），不经 torch 的 device 缓存分配器；device 侧 staging 走普通 `torch.empty`（`memory_pool_host.py:345`），与 expseg 兼容。**注意**：expseg 是本轮"试验A"新增项（C:202），HiCache 与它的叠加**没有实测记录**。 |
| 5 | **dspark（spec decode）** | **有专门支持（PACKED 模式）** | `speculative/base_spec_worker.py:37-53`：`_can_pack_hicache_mtp` 对 `is_dspark() and hf_config.architectures[0] == "DeepseekV4ForCausalLMDSpark"` 返回 `True`；`_build_hicache_draft_plan` 在 HiCache 开启时把 draft 池交给 target（PACKED，`:260-315`），draft 的 SWA buffer 被并入 SWA host 池层数（`hybrid_pool_assembler.py:918-930`）——**本轮实测 draft 架构确为 `DeepseekV4ForCausalLMDSpark`（L:165,166,173,177；改写处 `configs/model_config.py:940-955`），故走 PACKED 分支，SWA host 池层数 = 40 + draft 层**；预算侧也为此预留：`bytes_per_full_token ×= (40+1)/40`（`pool_configurator.py:1223-1230`）—— 实测 `1630 × 1.025 = 1670.75`（L:774）**逐位吻合**。`hicache_hook.py:123-128` 在 DCP>1 时只允许 DSPARK，反向印证 DSPARK 是 HiCache 的一等路径。 |
| 6 | **是否需要额外显存** | **需要，≈0.391 GiB/rank（419.8 MB）**；不占 KV 预算，吃 0.88 的 slack | §2.5；`memory_pool_host.py:333-349`。另加 `staging_buffer` 之外的每池 `device_ptrs` 小表。 |

---

## 4. 主机内存可行性核算

### 4.1 账（含 HiCache@ratio6 与 @ratio3.86）

| 项 | GiB | 说明/证据 |
|---|---|---|
| 必须常驻 host：专家母本 | 275.77 | `budget-lsglang.md:182` |
| 必须常驻 host：engram 两表 | 188.83 | `budget-lsglang.md:183`；`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`（/proc environ） |
| **既有常驻小计** | **464.60（498.86 GB）** | `budget-lsglang.md:184` |
| HiCache @ratio=6 | **185.1（198.8 GB）** | §2.1/2.4 |
| HiCache @ratio=3.86 | **119.1（127.9 GB）** | §2.1 |
| HiCache@r6 + 既有常驻 | **649.7 GiB** | 464.60 + 185.1 |
| 4 个 scheduler RSS（现状，未含 HiCache） | 228.1~228.2 × 4 | `ps`（RSS 含共享页，非可加 PSS） |
| 全机 | 2015.7 GiB 总 / MemFree 1000 GiB / **MemAvailable 1153 GiB** | `/proc/meminfo` |
| node0 | 1007.7 GiB 总 / free 515.4 GiB | `numactl --hardware` |
| node1 | 1007.9 GiB 总 / free 484.7 GiB | 同上 |
| cgroup 约束 | `memory.max = max` ⇒ **无 cgroup 上限**；`host_memory_budget_bytes()` 退化为 `psutil MemAvailable − 10 GiB`，再 `//4 rank` = **285.8 GiB/rank** 的单池准入上限 | `/sys/fs/cgroup/.../memory.max`；`mem_cache/host_memory.py:41-161`；`pool_host/base.py:84-108` |
| 若 cgroup 上限可读时本应打印的日志行 | `HiCache memory headroom: host %.1f GiB, cgroup %.1f GiB` —— **本机不会出现**（headroom=None） | `host_memory.py:155-160` |

### 4.2 是否会被 NUMA 或容量卡住

- 需放：`464.60 + 185.1 = 649.7 GiB`（ratio=6）。`LK_NUMA_INTERLEAVE=1` 下每 node 摊到 **324.9 GiB**，**小于**任一 node 的 1007.9 GiB；也小于 node1 曾测空闲 401.77 GiB（`budget-lsglang.md:188`）。
- **当前**机器可直接看到的余量：`MemAvailable = 1153 GiB`、node1 free 484.7 GiB、node0 free 515.4 GiB。ratio=6 的 185.1 GiB 在**总量**与**单 node**两个口径下都放得下。
- **唯一的结构性风险**：HiCache host 池由 `cudaHostRegister` **页锁定、不可回收**（`pool_host/common.py:143-199`），本机 `Swap = 0`；它会把可回收的 page cache 余量（现 786 GiB buff/cache，其中 NFS checkpoint 475 GiB 回读依赖页缓存）再压掉 185.1 GiB。
- 关于 `LK_NUMA_INTERLEAVE`：既有结论是"必需 1"（`budget-lsglang.md:192`、C:119）。**没有发现任何理由需要为 HiCache 调小或关闭它**；反向需求也没有证据——HiCache host 池是进程内匿名内存，`numactl --interleave=all`（`lk_moe_env.py:103-105`，见 `budget-lsglang.md:192`）若对该进程生效，则它同样会被 interleave（**未实测**，见 §7-6）。

### 4.3 替代预算（只列数字）

| 目标主机总量 | 所需 ratio | 每 rank host |
|---|---|---|
| 64 GB | 1.93 | 16.0 GB |
| 96 GB | 2.90 | 24.0 GB |
| **128 GB** | **3.86** | 32.0 GB |
| 160 GB | 4.83 | 40.0 GB |
| **≈199 GB（ratio=6 的实际值）** | **6.00** | 49.7 GB |
| 256 GB | 7.73 | 64.0 GB |

（口径：`总量 = 4 × ratio × 8.284 GB`，十进制 GB；每单位 ratio = 33.14 GB 总量。）

---

## 5. 验证方案（只读、可执行）

> 前提：任何启用动作都必须在**另一次**启动里做（本轮不动在跑的服务）。以下命令只读。

### 5.1 启动期"生效判据串"

以日志为准（每个 TP rank 各一次）。**判定 HiCache 已生效 = 同时满足 ①②③**：

① **host 池实体分配行（DSV4 专属，`memory_pool_host.py:303-312`）**：
```
Allocating <X> GB host memory for V4 paged pool '<名字>' (layers=<L>, pages=<P>, item_bytes=<B>, layout=<LAYOUT>).
```
预期出现的池与量（ratio=R）：

| 日志中的 `<名字>` | layers | pages | item_bytes | 该行 X（GB）≈ |
|---|---|---|---|---|
| `DEEPSEEK_V4_C1` | 1 | `R×16384` | 149760 | `R×2.454` |
| `DEEPSEEK_V4_C2` | 3 | `R×16384` | 74880 | `R×3.681` |
| `DEEPSEEK_V4_C1_INDEXER`（或 `_PAYLOAD`/`_SCALE` 两行，取决于 indexer 布局） | 1 | `R×16384` | ~17408 | `R×0.285` |
| `DEEPSEEK_V4_C2_INDEXER` | 3 | `R×16384` | ~8704 | `R×0.428` |
| `SWA` | 41 | `R×234` | 149760 | `R×1.437` |

**不应出现**：`DEEPSEEK_V4_C4*`、`DEEPSEEK_V4_C128*`（本模型无 ratio-4/128 层）、`DEEPSEEK_V4_C4_STATE`、`DEEPSEEK_V4_C128_STATE`（`c4_state=c128_state=0`）。

② **无失败/降级行**：
- `Not enough host memory for V4 paged pool …`（`memory_pool_host.py:254-258`）
- `Not enough host memory available. Requesting … Please reduce the size of the hierarchical cache.`（`pool_host/base.py:241-247`）
- `HiCache <pool> host pool (<n> tokens) is smaller than the device pool …`（`pool_host/base.py:229-236`）
- `HiCache auto-sizing does not support DeepSeekV4TokenToKVPool …`（`hicache_auto_size.py:69-72`）—— 出现即说明 ratio 没传对（D2）

③ **进程主机占用上升**（只读、无需进服务）：
```bash
# 4 个 scheduler 的 RSS 增量应 ≈ 4 × ratio × 8.284 GB + 元数据
ps -eo pid,ppid,rss,comm | awk '$4 ~ /^sglang::schedul/ {printf "%s %.1f GiB\n", $1, $3/1048576}'
grep -E "MemFree|MemAvailable" /proc/meminfo            # MemFree 应下降 ≈ 总量
numactl --hardware | grep -E "node (0|1) (size|free)"   # interleave 下两边各降 ≈ 总量/2
```

④ **辅助行**（可能不出现，不构成失败）：
- `HiCache memory headroom: host … GiB, cgroup … GiB` —— 仅当 cgroup 限额可读（本机 `memory.max=max`，**不会出现**；`host_memory.py:143-161`）
- `HiCache staging prepared before KV sizing: … MiB, … layers` —— 仅 MHA 池路径（`kv_cache_builder.py:71-106`），DSV4 不走此处

### 5.2 运行期指标（Prometheus `/metrics`）

> **前置条件（实测）**：本部署 `enable_metrics=False`（L:80），而 `/metrics` 端点由该开关门控（`entrypoints/http_server.py:305,2670`）⇒ **当前口径下 `/metrics` 不可用**，要看这些指标必须**另加** `--enable-metrics`（这本身是一次配置变更）。不开 metrics 时，运行期判据改用 §5.3 的响应字段（`prompt_tokens_details.cached_tokens`）与 `/get_server_info`。

| 指标 | 含义 | 定义处 |
|---|---|---|
| `sglang:hicache_host_used_tokens` | host 池已用 token | `observability/metrics_collector.py:750-755`；采集 `managers/scheduler_components/metrics_reporter.py:1210-1222` |
| `sglang:hicache_host_total_tokens` | host 池总容量（token） | 同上 `:756-761` |
| `sglang:hicache_backup_tokens_total` / `hicache_backup_bytes_total` / `hicache_backup_duration_seconds` | D→H 备份量/字节/耗时（含 write_back 的秒级窗口） | `metrics_collector.py:2497-2540` |
| `sglang:hicache_dropped_tokens_total`（含 `reason`/`pool` 标签） | 因 host 池满等原因丢弃 | `metrics_collector.py:2541-2580` |
| `sglang:hicache_backup_dropped_tokens_total`、`sglang:hicache_prefetch_aux_alloc_failed_tokens_total` | L3 相关（无 storage backend 时不产生） | `metrics_collector.py:2211,2221` |
| `sglang:cache_hit_rate` | radix 前缀命中率（含 host 命中） | `metrics_collector.py:380-382` |

> 注意：`_log_hicache_stats` 要求 `tree_cache.token_to_kv_pool_host` 或 `full_kv_pool_host` 非空（`metrics_reporter.py:1213-1222`，内含 `assert host_pool is not None`）；DSV4 栈通过 `entry_map` 安装 `ComponentType.FULL → ("full_kv_pool_host","_full_kv_pool_host")` 别名（`hybrid_pool_assembler.py:1697`）。该别名在 DSV4 树上是否真的可见**未实机确证**（§7-3）。

### 5.3 能压出主机缓存收益的 workload

- 结构：**多条请求共享超长前缀**，且**第一轮与第二轮之间让 device KV 被挤出去**（HiCache 只在 radix 命中失败且 host 命中成功时才走 L2 回填）。
- 建议（贴合本部署上限：`--max-running-requests 16`、`ctx 1048576`、4M token KV）：
  1. 预热：发 1 条长前缀（建议 512K–600K token，占满单请求窗口）请求 A，完成。
  2. 挤出：发 12–16 条**不同前缀**的长请求把 device KV 顶掉（总 token 逼近 4194304）。
  3. 复测：重发请求 A（或与其共享 512K 前缀的变体）。
  4. 判据：第二轮 A 的 `prompt_tokens_details.cached_tokens` 应 ≈ 共享前缀长度（**该判据不需要 metrics**）；若同时加了 `--enable-metrics`，另看 `sglang:cache_hit_rate` 上升、`sglang:hicache_host_used_tokens` 在步骤 1–3 期间持续 > 0 且接近前缀长度；TTFT 相比第一轮显著下降（跨 device 驱逐）。
- 反例（压不出收益）：单请求反复发同一 prompt（device 侧本就命中，不进 L2）、或前缀长度 < `page_size(256)`。

### 5.4 一键只读巡检脚本片段（不改任何文件）

```bash
L=/mnt/SSD01/models/.llm-env/Lsglang/logs/<新日志>.log
grep -nE "Allocating .* host memory for V4 paged pool" "$L"          # ①
grep -nE "Not enough host memory|smaller than the device pool|auto-sizing does not support" "$L"   # ② 应为空
curl -s http://127.0.0.1:<port>/metrics | grep -E "hicache|cache_hit_rate"   # 仅在另行开启 --enable-metrics 后可用
ps -eo pid,ppid,rss,comm | awk '$4 ~ /^sglang::schedul/ {printf "%s %.1f GiB\n", $1, $3/1048576}'
free -g; numactl --hardware | grep -E "node (0|1) (size|free)"
```

---

## 6. 回滚（回到当前配置）

HiCache 只经 CLI 开关生效，**回滚 = 删除下面这些行**；不需要动 env、不需要动 conf 之外任何东西。

要删/注释的（按 §8 草案）：

| 行 | 内容 | 说明 |
|---|---|---|
| 1 | `--enable-hierarchical-cache` | 总开关（`memory.py:98-101`） |
| 2 | `--hicache-ratio <R>` | DSV4 必填项（D1/D2）；删掉即彻底关闭 |
| 3 | 任何其它 `--hicache-*` | 若草案里被展开成显式值（write-policy / mem-layout / io-backend / storage-*）一律删除；**注意**：若显式写了 `--hicache-mem-layout page_first_direct`，删除后即回到默认 `page_first` |

- 若曾为调试加过 `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB`，一并删除（默认 256 与当前环境等价，`environ.py:796`）。
- 回滚后判据：日志不再出现 `Allocating … host memory for V4 paged pool`；`/metrics` 不再有 `sglang:hicache_*`；4 个 scheduler RSS 回落 ≈ ratio×8.284 GB；`max_total_num_tokens` 仍为 4194304（KV 定容不受 HiCache 影响，因为 host 池不占 KV 预算）。
- 回滚**不需要**重启以外的任何操作；但**必须**与"是否保留 expseg/其它试验开关"分开记录（C:202 的 expseg 与 HiCache 是两件独立的改动）。

---

## 7. 未确认项清单

1. **DSV4 + HiCache + 6 图桶 + `--enable-decoder-swa-bounded-replay` 的实机组合**：全树无该组合的测试或日志；§3-2 只能到"无代码级冲突"。**无法判定**。
2. **`SGLANG_DSV41_SWA_PAD_SENTINEL` / `SGLANG_DSV41_DUMP_FLASHMLA_ARGS` / `CUDA_LAUNCH_BLOCKING=1` 与 HiCache 的叠加**：无源码交集，也无实测。**无法判定**。
3. **运行期指标可用性**：`_log_hicache_stats` 需要 tree_cache 暴露 `token_to_kv_pool_host`/`full_kv_pool_host`（`metrics_reporter.py:1213-1222` 内含 assert）。DSV4 的别名安装路径存在（`hybrid_pool_assembler.py:1697`），但**未确认**该别名在 DSV4 树对象上可读；若不可读，`--enable-metrics` 打开时该 assert 可能触发。**无法判定**。
4. **`hicache_host_used_tokens` 的 token 口径**：FULL 锚池 `logical_size = R×4194304`（`memory_pool_host.py:522-530`、`hybrid_pool_assembler.py:947-949`），而实体字节在 C1/C2/SWA sidecar 上；该指标在 DSV4 上代表"逻辑锚/实体池"哪一侧，**未实机确证**。
5. **`--hicache-ratio` 在 4 个 rank 上的一致性**：DSV4 页数按各 rank 自己的 device 页数算（D5），本部署 4 卡同构（L:774-795 四 rank 数字全同）⇒ 应当一致；若某卡 `available_bytes` 不同，host 池会出现 rank 间不等（**未验证**）。
6. **`numactl --interleave=all` 是否覆盖 `cudaHostRegister` 的匿名页**：**未验证**。若覆盖，ratio=6 时每 node 摊 ≈92.6 GiB；若不覆盖（首触策略落单 node），单 node 需求 = 185.1 GiB < node1 现空闲 484.7 GiB，仍可容纳。
7. **`page_first_direct` + `direct` io 在 DSV4 上是否可用**（省掉 §2.5 的 0.391 GiB/rank device staging 的唯一路径）：未实测，**无法判定**；且该布局要求 `_require_packed_device_rows` 检查通过（`pool_host/base.py:332-339`），DSV4 的 row stride 特性未核对。
8. **启动耗时影响**：`cudaHostRegister` 对 4×(ratio×8.284 GB) 的注册耗时、以及 `clear()` 里 `torch.arange(size, int64)`（ratio=6 时 2.01e8 元素/rank）的初始化耗时**未实测**；`STARTUP_TIMEOUT_S=3600`（C:471）是否仍够，**未验证**。
9. **`--hicache-write-policy` 取值对 TTFT 的实测影响**：默认 `write_through` 与 `write_back` 的取舍只有代码语义（`memory.py:121-127`；`serving_hook.py:491-505` 对 optimistic prefill 的限制），本部署未启用 optimistic prefill，**无实测**。

---

## 8. 配置草案 diff（**仅文本，未应用**；目标文件 `/mnt/SSD01/models/.llm-env/Lsglang/configs/dsv41_tp4_h100.conf`）

```diff
--- configs/dsv41_tp4_h100.conf	(2026-10-23 08:05 现状, 未修改)
+++ configs/dsv41_tp4_h100.conf	(HiCache 草案, 未应用)
@@ 环境变量区 —— 锚点 C:249 (SGLANG_DSV41_POST_CONVERT_RELEASE 之后)
 SGLANG_DSV41_POST_CONVERT_RELEASE="1"
+
+# ---- HiCache 草案 (未应用) ----
+# host 池经 cudaHostRegister 页锁定/不可回收 (mem_cache/pool_host/common.py:143-199, 424-431)
+# 本模型换算 (每 rank): host GB ≈ ratio × 8.284 GB; 4 rank 合计 = ×4
+#   ratio=3.86 → 32.0 GB/rank, 合计 127.9 GB   ← 对齐"约 128 GB"
+#   ratio=6    → 49.7 GB/rank, 合计 198.8 GB   ← 对齐"乘数 6"
+# DSV4 硬约束: 禁用 --hicache-size (hybrid_pool_assembler.py:603-607); 必须显式 --hicache-ratio
+#             (否则 auto-size 对 DeepSeekV4TokenToKVPool 抛错, hicache_auto_size.py:69-72)
+# 额外 device 开销: write-back staging ≈0.391 GiB/rank (memory_pool_host.py:333-349)
+# SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB="256"   # 默认 256 (srt/environ.py:796), 与本环境等价, 无需写
+# ---- 草案结束 ----

@@ SERVE_COMMON_ARGS —— 锚点 C:440 (--enable-decoder-swa-bounded-replay 之后)
   --enable-decoder-swa-bounded-replay
+  # ---- HiCache 草案 (未应用): 三选一 ----
+  # 方案A (对齐 128 GB):
+  #   --enable-hierarchical-cache
+  #   --hicache-ratio 3.86
+  # 方案B (对齐乘数 6):
+  #   --enable-hierarchical-cache
+  #   --hicache-ratio 6
+  # 方案C (默认, 最小风险):
+  #   --enable-hierarchical-cache
+  #   --hicache-ratio 2
+  # 以下三项取默认值, 一般不必显式写; 写出来只为冻结语义:
+  #   --hicache-write-policy write_through     # 默认 write_through (fields/memory.py:121-127)
+  #   --hicache-mem-layout page_first          # 默认 page_first (fields/memory.py:135-147)
+  #   --hicache-io-backend kernel              # 默认 kernel (fields/memory.py:128-134)
+  # 不引入 L3: 不写 --hicache-storage-backend (默认 None, fields/memory.py:148-169)
+  # ---- 草案结束 ----
   --disable-shared-experts-fusion
```

回滚 = 删除上表中"方案"两行与任何展开的 `--hicache-*` 行（§6）。
