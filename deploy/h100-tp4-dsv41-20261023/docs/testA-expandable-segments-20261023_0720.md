# 试验 A：`expandable_segments` 单一变量对照

观测时间：2026-10-23 06:51 – 07:17（Asia/Shanghai）

## 0. 对象与可控条件

| 项 | 基线 | 本轮（试验 A） |
| --- | --- | --- |
| serve 日志 | `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261023_025933.log`（1,563,926 B） | `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261023_065049.log`（40,688 B @06:51，观测结束时 853 行） |
| 启动器 stdout | —（无对应 launch-*.out 保留） | `/mnt/SSD01/models/.llm-env/Lsglang/logs/launch-expseg-20261023_0650.out` |
| 观测用基线显存记录 | `/mnt/SSD01/models/.llm-env/Lsglang/docs/pristine-run-20261023_0324.md` 第 493–495 行 | 本报告第 6 节 |
| 权重 | `/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017` | 同 |
| 物理卡 | GPU4/5/6/7（UUID 固定） | 同 |

本轮 rank 进程命令行原文（`ps` 取值）：

```
/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python /mnt/SSD01/models/.llm-env/Lsglang/env/bin/sglang serve --model /mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017 --tensor-parallel-size 4 --mem-fraction-static 0.88 --max-running-requests 16 --context-length 1048576 --chunked-prefill-size 8192 --moe-runner-backend flashinfer_mxfp4 --disable-flashinfer-autotune --enable-decoder-swa-bounded-replay --disable-shared-experts-fusion --reasoning-parser auto --tool-call-parser auto --trust-remote-code --max-total-tokens 4194304 --cuda-graph-max-bs-decode 16 --cuda-graph-max-bs-prefill 16 --host 0.0.0.0 --port 18010 --served-model-name DeepSeek-V4.1-Flash-1017
```

`server_args` 逐字段比对（脚本解析两份日志中的 `server_args={...}` 字面量，17 个关键字段 + `cuda_graph_config` 全量）：

| 字段 | 基线 | 本轮 |
| --- | --- | --- |
| tp_size / mem_fraction_static | 4 / 0.88 | 4 / 0.88 |
| max_running_requests / context_length | 16 / 1048576 | 16 / 1048576 |
| chunked_prefill_size / max_total_tokens | 8192 / 4194304 | 8192 / 4194304 |
| moe_runner_backend / attention_backend | flashinfer_mxfp4 / dsv4 | flashinfer_mxfp4 / dsv4 |
| kv_cache_dtype | fp8_e4m3 | fp8_e4m3 |
| disable_shared_experts_fusion | True | True |
| enable_decoder_swa_bounded_replay | True | True |
| cuda_graph_max_bs_decode / prefill | 16 / 16 | 16 / 16 |
| ep_size / port | 1 / 18010 | 1 / 18010 |
| weight_loader_drop_cache_after_load | False | False |
| disable_flashinfer_autotune | True | True |
| **差异字段数** | — | **0** |

环境变量差异（读自 `/proc/<pid>/environ`，本轮 PID 55964 / 56896 / 56897 / 56898 / 56899）：

| 变量 | 基线 | 本轮 |
| --- | --- | --- |
| `LK_GPU_RESIDENT_MOE_LAYERS` | `0-19` | `0-19` |
| `LK_POOL_SLOTS` | `128` | `128` |
| `PYTORCH_CUDA_ALLOC_CONF` | 未设置 | `expandable_segments:True` |

`CUDA_VISIBLE_DEVICES=GPU-7561b28f-b049-bbdb-31f8-16efeb06663d,GPU-fb59db94-0e4e-4c4f-8ae5-4d0fdd1e0f0b,GPU-3c5d7267-445c-4679-bce9-503bedd0c920,GPU-3c0f2856-a6e3-b9d5-fd10-a08f55579fff`（4 个 rank 进程与父进程一致，映射 GPU4/5/6/7）。

启动器回显：

```
 专家方案     : resident=0-19 (W4A8) + pool=128 槽 (W4A16 等价)
```

本轮 serve 日志头部（静态文本）第 52 行：

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 1. 基线 vs 本轮 逐项对照表

| # | 判据 | 基线（025933） | 本轮（065049） | 差值 |
| --- | --- | --- | --- | --- |
| 1 | `Load weight begin` avail mem（TP0/TP1/TP2/TP3, GB） | 77.41 / 77.36 / 77.36 / 77.60 | 77.40 / 77.35 / 77.35 / 77.58 | −0.01 / −0.01 / −0.01 / −0.02 |
| 1 | `Load weight end` **mem usage**（TP0/TP1/TP2/TP3, GB） | 63.65 / 63.70 / 63.70 / 63.46 | **55.84 / 55.88 / 55.88 / 55.65** | **−7.81 / −7.82 / −7.82 / −7.81** |
| 1 | `Load weight end` **avail mem**（TP0/TP1/TP2/TP3, GB） | 13.76 / 13.67 / 13.67 / 14.14 | **21.56 / 21.47 / 21.47 / 21.94** | **+7.80 / +7.80 / +7.80 / +7.80** |
| 1 | `Load weight end` elapsed（s） | 573.11 / 573.06 / 573.01 / 572.90 | 361.89 / 361.60 / 361.71 / 361.63 | −211.22 / −211.46 / −211.30 / −211.27 |
| 2 | `max_total_num_tokens` | **2,380,800** | **4,194,304** | **+1,813,504（+76.2%）** |
| 2 | `available_gpu_mem` | 7.14 GB | 9.94 GB | +2.80 GB |
| 2 | `ValueError … minimum viable` | 0 行 | 0 行 | 0 |
| 2 | `max_total_tokens=4194304 is larger than the profiled value …`（请求 4M 被钳制至 profiled 值） | **4 行**（行 699 / 702 / 705 / 706，4 个 rank 各 1 行） | **0 行** | **−4** |
| 3 | `[pool] slab:` | `11.21 GiB (20 layers x 573.75 MiB)` × 4 | `11.21 GiB (20 layers x 573.75 MiB)` × 4 | 0 |
| 3 | `lk_moe static device arena:` | `11.21 GB` × 4 | `11.21 GB` × 4 | 0 |
| 4 | `Capture target decode CUDA graph begin` avail mem（GB） | 10.34 / 10.24 / 10.24 / 10.71 | 13.13 / 13.04 / 13.04 / 13.51 | +2.79 / +2.80 / +2.80 / +2.80 |
| 4 | 捕获桶数 | `bs=[1, 2, 4, 8, 12, 16]`（6 桶） | `bs=[1, 2, 4, 8, 12, 16]`（6 桶） | 0 |
| 4 | 捕获耗时（s） | 29.38 / 29.38 / 29.38 / 29.39 | 28.36 / 28.36 / 28.36 / 28.35 | −1.02 / −1.02 / −1.02 / −1.04 |
| 4 | 捕获 mem usage（GB，TP0/其余） | 3.19 / 0.34 / 0.34 / 0.34 | 3.19 / 0.33 / 0.33 / 0.33 | 0 / −0.01 / −0.01 / −0.01 |
| 4 | 捕获报错 / invalidated / 降级 | 无 | **无** | 0 |
| 5 | `/health` | 200 OK（03:17:59） | 200 OK（07:05:18） | — |
| 5 | `max_running_requests` / `context_len` | 16 / 1048576 | 16 / 1048576 | 0 |
| 5 | 端口 18010 | LISTEN | LISTEN（`sglang`, pid=55964, fd=74） | — |
| 6 | 显存稳态（GPU4/5/6/7, MiB） | 73957 / 71124 / 71124 / 70644 | 71059 / 68226 / 68226 / 67746 | **−2898 / −2898 / −2898 / −2898** |
| 7 | 崩溃/异常类关键词计数（排除头部 1–63 行） | 全 0 | **全 0** | 0 |
| 7 | `expandable` / `segment` 相关 warning/error | — | **0 行**（仅头部第 52 行回显） | 0 |
| 8 | 短请求可用性 | — | HTTP 200，32 completion tokens，0.906 s | — |
| 8 | 首条 `[pool] dstep=100` `win hit` / `cum hit` | 93.6% / 93.6%（h=42610 m=2936 p=755） | 91.0% / 91.0%（h=41431 m=4115 p=623） | −2.6 / −2.6 |
| — | `The server is fired up and ready to roll!` | 03:17:57 | 07:05:15 | — |

### 图 1：Load weight 阶段分配器内存（基线 vs 本轮）

![Experiment A load-weight allocator memory](</mnt/SSD01/models/.llm-env/Lsglang/docs/testA-load-mem-compare-20261023_0720.png>)

### 图 2：稳态设备显存与 KV 定容结果

![Experiment A steady-state VRAM and KV sizing](</mnt/SSD01/models/.llm-env/Lsglang/docs/testA-vram-kv-compare-20261023_0720.png>)

## 2. 判据 1：`Load weight begin` / `Load weight end` 四行

### 2.1 本轮原文（日志行号：begin 186/190/191/194，end 670/673/676/679）

```
186:[2026-10-23 06:51:49 TP0] Load weight begin. avail mem=77.40 GB
190:[2026-10-23 06:51:49 TP2] Load weight begin. avail mem=77.35 GB
191:[2026-10-23 06:51:49 TP1] Load weight begin. avail mem=77.35 GB
194:[2026-10-23 06:51:49 TP3] Load weight begin. avail mem=77.58 GB
670:[2026-10-23 06:57:51 TP1] Load weight end. elapsed=361.60 s, type=DeepseekV4ForCausalLM, quant=fp8, avail mem=21.47 GB, mem usage=55.88 GB.
673:[2026-10-23 06:57:51 TP3] Load weight end. elapsed=361.63 s, type=DeepseekV4ForCausalLM, quant=fp8, avail mem=21.94 GB, mem usage=55.65 GB.
676:[2026-10-23 06:57:51 TP2] Load weight end. elapsed=361.71 s, type=DeepseekV4ForCausalLM, quant=fp8, avail mem=21.47 GB, mem usage=55.88 GB.
679:[2026-10-23 06:57:51 TP0] Load weight end. elapsed=361.89 s, type=DeepseekV4ForCausalLM, quant=fp8, avail mem=21.56 GB, mem usage=55.84 GB.
```

### 2.2 基线原文（日志行号：begin 185/192/195/189，end 669/672/675/678）

```
185:[2026-10-23 03:00:40 TP0] Load weight begin. avail mem=77.41 GB
192:[2026-10-23 03:00:40 TP1] Load weight begin. avail mem=77.36 GB
195:[2026-10-23 03:00:40 TP2] Load weight begin. avail mem=77.36 GB
189:[2026-10-23 03:00:40 TP3] Load weight begin. avail mem=77.60 GB
669:[2026-10-23 03:10:13 TP3] Load weight end. elapsed=572.90 s, type=DeepseekV4ForCausalLM, quant=fp8, avail mem=14.14 GB, mem usage=63.46 GB.
672:[2026-10-23 03:10:13 TP0] Load weight end. elapsed=573.11 s, type=DeepseekV4ForCausalLM, quant=fp8, avail mem=13.76 GB, mem usage=63.65 GB.
675:[2026-10-23 03:10:13 TP2] Load weight end. elapsed=573.01 s, type=DeepseekV4ForCausalLM, quant=fp8, avail mem=13.67 GB, mem usage=63.70 GB.
678:[2026-10-23 03:10:13 TP1] Load weight end. elapsed=573.06 s, type=DeepseekV4ForCausalLM, quant=fp8, avail mem=13.67 GB, mem usage=63.70 GB.
```

## 3. 判据 2：KV 定容

### 3.1 本轮原文

```
703:[2026-10-23 06:57:54 TP0] DSV4 pool sizes: full=4194304, swa=53504, c4=1048576, c128=32768, c4_state=0, c128_state=0
707:[2026-10-23 06:57:54 TP3] DSV4 pool sizes: full=4194304, swa=53504, c4=1048576, c128=32768, c4_state=0, c128_state=0
710:[2026-10-23 06:57:54 TP2] DSV4 pool sizes: full=4194304, swa=53504, c4=1048576, c128=32768, c4_state=0, c128_state=0
711:[2026-10-23 06:57:54 TP1] DSV4 pool sizes: full=4194304, swa=53504, c4=1048576, c128=32768, c4_state=0, c128_state=0
804:[2026-10-23 07:04:48 TP0] max_total_num_tokens=4194304, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=16, context_len=1048576, available_gpu_mem=9.94 GB
```

定容前的容量测算（本轮，`available_bytes` 基线 4.78 GB → 本轮 12.06 GB）：

```
694:[2026-10-23 06:57:54 TP0] DSV4 memory calculation: unified=False, unified_fp8=False, dspark_draft_bf16=False, bytes_per_full_token=1630.00, available_bytes=12.06 GB, c128_state_fixed=0.00 GB, c2_state_fixed=0.00 GB, swa_fixed=1.17 GB, swa_ring_fixed=0.00 GB, c4_state_fixed=0.00 GB, full_token=7177216
697:[2026-10-23 06:57:54 TP3] DSV4 memory calculation: unified=False, unified_fp8=False, dspark_draft_bf16=False, bytes_per_full_token=1630.00, available_bytes=12.06 GB, c128_state_fixed=0.00 GB, c2_state_fixed=0.00 GB, swa_fixed=1.17 GB, swa_ring_fixed=0.00 GB, c4_state_fixed=0.00 GB, full_token=7177216
701:[2026-10-23 06:57:54 TP2] DSV4 memory calculation: unified=False, unified_fp8=False, dspark_draft_bf16=False, bytes_per_full_token=1630.00, available_bytes=12.06 GB, c128_state_fixed=0.00 GB, c2_state_fixed=0.00 GB, swa_fixed=1.17 GB, swa_ring_fixed=0.00 GB, c4_state_fixed=0.00 GB, full_token=7177216
702:[2026-10-23 06:57:54 TP1] DSV4 memory calculation: unified=False, unified_fp8=False, dspark_draft_bf16=False, bytes_per_full_token=1630.00, available_bytes=12.06 GB, c128_state_fixed=0.00 GB, c2_state_fixed=0.00 GB, swa_fixed=1.17 GB, swa_ring_fixed=0.00 GB, c4_state_fixed=0.00 GB, full_token=7177216
```

`sizing` 阶段（第 695/699/704/706 行）先给出 `full=7177216, swa=53504, c4=1794304, c128=56072`，随后 4 个 rank 落定值为 `full=4194304`（受 `--max-total-tokens 4194304` 上限约束）。

### 3.2 基线原文

```
693:[2026-10-23 03:10:16 TP0] DSV4 memory calculation: unified=False, unified_fp8=False, dspark_draft_bf16=False, bytes_per_full_token=1630.00, available_bytes=4.78 GB, c128_state_fixed=0.00 GB, c2_state_fixed=0.00 GB, swa_fixed=1.17 GB, swa_ring_fixed=0.00 GB, c4_state_fixed=0.00 GB, full_token=2380800
695:[2026-10-23 03:10:16 TP3] DSV4 memory calculation: unified=False, unified_fp8=False, dspark_draft_bf16=False, bytes_per_full_token=1630.00, available_bytes=4.78 GB, c128_state_fixed=0.00 GB, c2_state_fixed=0.00 GB, swa_fixed=1.17 GB, swa_ring_fixed=0.00 GB, c4_state_fixed=0.00 GB, full_token=2380800
694:[2026-10-23 03:10:16 TP0] DSV4 pool sizes: full=2380800, swa=53504, c4=595200, c128=18600, c4_state=0, c128_state=0
698:[2026-10-23 03:10:16 TP3] DSV4 pool sizes: full=2380800, swa=53504, c4=595200, c128=18600, c4_state=0, c128_state=0
703:[2026-10-23 03:10:16 TP2] DSV4 pool sizes: full=2380800, swa=53504, c4=595200, c128=18600, c4_state=0, c128_state=0
704:[2026-10-23 03:10:16 TP1] DSV4 pool sizes: full=2380800, swa=53504, c4=595200, c128=18600, c4_state=0, c128_state=0
797:[2026-10-23 03:17:22 TP0] max_total_num_tokens=2380800, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=16, context_len=1048576, available_gpu_mem=7.14 GB
```

### 3.3 `max_total_tokens` 钳制信息

基线全文件命中 4 行，4 个 rank 各 1 行：

```
699:[2026-10-23 03:10:16 TP0] max_total_tokens=4194304 is larger than the profiled value 2380800. Use the profiled value instead.
702:[2026-10-23 03:10:16 TP3] max_total_tokens=4194304 is larger than the profiled value 2380800. Use the profiled value instead.
705:[2026-10-23 03:10:16 TP2] max_total_tokens=4194304 is larger than the profiled value 2380800. Use the profiled value instead.
706:[2026-10-23 03:10:16 TP1] max_total_tokens=4194304 is larger than the profiled value 2380800. Use the profiled value instead.
```

基线钳制链：请求 `4194304` → profiled `2380800` → 落定 `max_total_num_tokens=2380800`。

本轮该字符串全文件命中 0 行。本轮 profiled 值（3.1 节第 695/699/704/706 行）为 `full=7177216`，高于请求值 `4194304`；落定 `max_total_num_tokens=4194304`（第 804 行）。

`ValueError` / `minimum viable` 在两份日志全文件中均命中 0 行（无排除）。

## 4. 判据 3：`[pool] slab:` 与 `lk_moe static device arena:`

### 4.1 本轮原文（slab 行号 577/581/585/589；arena 行号 687/689/690/691）

```
575:Backend_NUMA init suscess, numa nodes: 1 cors:96 cpus:192 max_threads: 24
576:[pump] ON pool=ON slots=128 window=32 inflight=16 log=100
577:[pool] slab: 11.21 GiB (20 layers x 573.75 MiB)
578:[pool] H2D staging ON: ring=8 slab=4.70MB
581:[pool] slab: 11.21 GiB (20 layers x 573.75 MiB)
585:[pool] slab: 11.21 GiB (20 layers x 573.75 MiB)
589:[pool] slab: 11.21 GiB (20 layers x 573.75 MiB)
687:[2026-10-23 06:57:54 TP0] lk_moe static device arena: 11.21 GB already resident (allocated in MOE ctors, excluded from free memory); no additional KV reservation.
689:[2026-10-23 06:57:54 TP3] lk_moe static device arena: 11.21 GB already resident (allocated in MOE ctors, excluded from free memory); no additional KV reservation.
690:[2026-10-23 06:57:54 TP2] lk_moe static device arena: 11.21 GB already resident (allocated in MOE ctors, excluded from free memory); no additional KV reservation.
691:[2026-10-23 06:57:54 TP1] lk_moe static device arena: 11.21 GB already resident (allocated in MOE ctors, excluded from free memory); no additional KV reservation.
```

### 4.2 基线原文（slab 行号 576/580/584/589；arena 行号 685/688/689/690）

```
574:Backend_NUMA init suscess, numa nodes: 1 cors:96 cpus:192 max_threads: 24
575:[pump] ON pool=ON slots=128 window=32 inflight=16 log=100
576:[pool] slab: 11.21 GiB (20 layers x 573.75 MiB)
577:[pool] H2D staging ON: ring=8 slab=4.70MB
580:[pool] slab: 11.21 GiB (20 layers x 573.75 MiB)
584:[pool] slab: 11.21 GiB (20 layers x 573.75 MiB)
589:[pool] slab: 11.21 GiB (20 layers x 573.75 MiB)
685:[2026-10-23 03:10:16 TP0] lk_moe static device arena: 11.21 GB already resident (allocated in MOE ctors, excluded from free memory); no additional KV reservation.
688:[2026-10-23 03:10:16 TP3] lk_moe static device arena: 11.21 GB already resident (allocated in MOE ctors, excluded from free memory); no additional KV reservation.
689:[2026-10-23 03:10:16 TP2] lk_moe static device arena: 11.21 GB already resident (allocated in MOE ctors, excluded from free memory); no additional KV reservation.
690:[2026-10-23 03:10:16 TP1] lk_moe static device arena: 11.21 GB already resident (allocated in MOE ctors, excluded from free memory); no additional KV reservation.
```

## 5. 判据 4：decode CUDA 图捕获

### 5.1 本轮原文（begin 行号 790–793；end 行号 799/803/805/806）

```
790:[2026-10-23 07:04:20 TP2] Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, avail mem=13.04 GB
791:[2026-10-23 07:04:20 TP0] Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, avail mem=13.13 GB
792:[2026-10-23 07:04:20 TP1] Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, avail mem=13.04 GB
793:[2026-10-23 07:04:20 TP3] Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, avail mem=13.51 GB
799:[2026-10-23 07:04:48 TP3] Capture target decode CUDA graph end. bs=[1, 2, 4, 8, 12, 16], elapsed=28.35 s, mem usage=0.33 GB, avail mem=13.17 GB.
803:[2026-10-23 07:04:48 TP0] Capture target decode CUDA graph end. bs=[1, 2, 4, 8, 12, 16], elapsed=28.36 s, mem usage=3.19 GB, avail mem=9.94 GB.
805:[2026-10-23 07:04:48 TP2] Capture target decode CUDA graph end. bs=[1, 2, 4, 8, 12, 16], elapsed=28.36 s, mem usage=0.33 GB, avail mem=12.70 GB.
806:[2026-10-23 07:04:48 TP1] Capture target decode CUDA graph end. bs=[1, 2, 4, 8, 12, 16], elapsed=28.36 s, mem usage=0.33 GB, avail mem=12.70 GB.
```

### 5.2 基线原文（begin 行号 785–788；end 行号 794–796/807）

```
785:[2026-10-23 03:16:53 TP2] Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, avail mem=10.24 GB
786:[2026-10-23 03:16:53 TP0] Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, avail mem=10.34 GB
787:[2026-10-23 03:16:53 TP3] Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, avail mem=10.71 GB
788:[2026-10-23 03:16:53 TP1] Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, avail mem=10.24 GB
794:[2026-10-23 03:17:22 TP1] Capture target decode CUDA graph end. bs=[1, 2, 4, 8, 12, 16], elapsed=29.38 s, mem usage=0.34 GB, avail mem=9.91 GB.
795:[2026-10-23 03:17:22 TP0] Capture target decode CUDA graph end. bs=[1, 2, 4, 8, 12, 16], elapsed=29.38 s, mem usage=3.19 GB, avail mem=7.14 GB.
796:[2026-10-23 03:17:22 TP2] Capture target decode CUDA graph end. bs=[1, 2, 4, 8, 12, 16], elapsed=29.38 s, mem usage=0.34 GB, avail mem=9.91 GB.
807:[2026-10-23 03:17:22 TP3] Capture target decode CUDA graph end. bs=[1, 2, 4, 8, 12, 16], elapsed=29.39 s, mem usage=0.34 GB, avail mem=10.38 GB.
```

捕获完成后运行时确认图生效（本轮，行号取自观测结束时日志）：

```
[2026-10-23 07:06:48 TP0] Decode batch, #running-req: 1, #full token: 256, full token usage: 0.00, #swa token: 256, swa token usage: 0.00, cuda graph: True, gen throughput (token/s): 35.54, #queue-req: 0
```

`Capture` 与 `graph` 相关行中未见 `error` / `invalidated` / `fallback` / `disable` 字样；`disable_cuda_graph`、`debug_cuda_graph` 均为默认值（见第 0 节 `server_args` 比对，两份日志 `cuda_graph_config` 全量相等）。

## 6. 判据 5：`/health`、端口、回显

| 项 | 本轮取值 | 基线取值 |
| --- | --- | --- |
| `curl /health` | `HTTP 200` | `HTTP 200`（日志 03:17:59） |
| 端口监听 | `LISTEN 0 2048 0.0.0.0:18010 users:(("sglang",pid=55964,fd=74))` | `0.0.0.0:18010` |
| 就绪行 | `[2026-10-23 07:05:15] The server is fired up and ready to roll!` | `[2026-10-23 03:17:57] The server is fired up and ready to roll!` |
| `max_running_requests` | 16 | 16 |
| `context_len` | 1048576 | 1048576 |

启动器自检结论原文（`/mnt/SSD01/models/.llm-env/Lsglang/logs/launch-expseg-20261023_0650.out`）：

```
 自检结果: 通过 ✅  (EXPECT='[pool] slab:' 命中 4 行; W4A8 禁止串 0 行)
```

启动器轮询期间 `/health` 返回 503 直至就绪（本轮 830 行前均为 `503 Service Unavailable`，就绪后为 `200 OK`），与基线同型（基线 816–831 行同为 503，837 行起 200）。

## 7. 判据 6：显存快照（GPU4/5/6/7，MiB）

本轮按 `nvidia-smi --query-gpu=memory.used -i 4,5,6,7` 采集：

| 时刻 | GPU4 | GPU5 | GPU6 | GPU7 |
| --- | --- | --- | --- | --- |
| 06:52:09 | 42426 | 42522 | 42522 | 42042 |
| 06:55:09 | 42672 | 42768 | 42768 | 42288 |
| 06:58:10 | 67600 | 67692 | 67692 | 67212 |
| 07:01:10 | 67602 | 67694 | 67694 | 67214 |
| 07:04:10 | 67602 | 67696 | 67696 | 67216 |
| 07:07:10 | 71059 | 68226 | 68226 | 67746 |
| 07:10:10 | 71059 | 68226 | 68226 | 67746 |
| 07:13:10 | 71059 | 68226 | 68226 | 67746 |
| 07:16:11 | 71059 | 68226 | 68226 | 67746 |
| 07:15:37 / 07:16:17 / 07:16:58 | 71059 | 68226 | 68226 | 67746 |

本轮稳态值（07:07–07:17 内 8 次采样恒定）：**71059 / 68226 / 68226 / 67746 MiB**。

基线稳态值（来自 `/mnt/SSD01/models/.llm-env/Lsglang/docs/pristine-run-20261023_0324.md` 第 493–495 行，03:20:09 / 03:20:36 / 03:24:05 三次采样恒定）：

| 时刻 | GPU4 | GPU5 | GPU6 | GPU7 |
| --- | --- | --- | --- | --- |
| 03:20:09 | 73957 | 71124 | 71124 | 70644 |
| 03:20:36 | 73957 | 71124 | 71124 | 70644 |
| 03:24:05 | 73957 | 71124 | 71124 | 70644 |

差值（本轮 − 基线）：−2898 / −2898 / −2898 / −2898 MiB。

采集期间 GPU0/1/2/3 未被本服务占用（可见列表已排除，见第 0 节 `CUDA_VISIBLE_DEVICES`）；同一时刻 GPU0 = 72041 MiB、GPU1 = 72041 MiB、GPU2 = 67090 MiB、GPU3 = 0 MiB。

## 8. 判据 7：崩溃/异常计数

检索方式：`tail -n +64 <日志> | grep -c -- "<关键词>"`（排除脚本头 1–63 行静态文本）。

| 关键词 | 本轮（065049） | 基线（025933） |
| --- | --- | --- |
| `SEGV` | 0 | 0 |
| `OOM` | 0 | 0 |
| `illegal` | 0 | 0 |
| `Traceback` | 0 | 0 |
| `CUDA error` | 0 | 0 |
| `core dumped` | 0 | 0 |
| `ValueError` | 0 | 0 |
| `AssertionError` | 0 | 0 |
| `expandable` | 0 | 0 |
| `segment` | 0 | 0 |

计数口径说明：大小写不敏感检索 `OOM` 得 8 次命中，全部来自 `DSV4 SWA sizing:` 行中的子串 `headroom`（本轮共 8 行含 `headroom`）；大小写敏感检索 `OOM` 为 0 次。本项目采用大小写敏感口径。

`expandable` / `segment` 全文（含头部 1–63 行）检索结果，仅为头部静态回显 1 行：

```
52:export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

启动器 stdout 中 `expandable` / `segment` / `warning` / `error` 匹配 0 行。

torch 分配器层面未见任何 `expandable_segments` 相关 warning 或 error（例如 `expandable_segments not supported on this platform`、`expandable_segments:True cannot be used with ...` 等均为 0 行）。

观测窗口内另存在一次被中止的启动尝试，非本试验对象，记录如下：

| 文件 | 状态 |
| --- | --- |
| `/mnt/SSD01/models/.llm-env/Lsglang/logs/launch-expseg-20261023_0648.out` | 113 B，内容：`.../configs/dsv41_tp4_h100.conf: line 142: syntax error near unexpected token `('` |
| `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261023_064934.log` | 38,020 B，尾部停在 `Memory baseline taken. mem usage=0.00 GB`（06:50:38），随后 `resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown` |

以上两份文件早于本轮 serve 日志（06:51:49 起），不含 `Load weight end`。

## 9. 判据 8：短请求与首条 `[pool] dstep`

请求（`POST http://127.0.0.1:18010/v1/chat/completions`，`max_tokens=32`，`temperature=0`）：

```
{"id":"5f2fa0b962454376845bcea5cf599582","object":"chat.completion","created":1792710392,"model":"DeepSeek-V4.1-Flash-1017","choices":[{"index":0,"message":{"role":"assistant","content":"KV缓存是在自回归生成时把已计算的Key/Value张量存下来复用，避免每生成一个新token都重算历史token的注意力键","reasoning_content":null},"logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":11,"total_tokens":43,"completion_tokens":32,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default","weight_versions":[{"version":"default","start":0,"end":32}]}}
```

耗时 `real 0m0.906s`，HTTP 200。

为触发 `log=100` 的计数器，另发 4 个同型 32-token 请求（均 HTTP 200），累计解码步数达 100。

首条 `[pool] dstep=100`（本轮行号 839）：

```
[pool] dstep=100 win hit=91.0% nr=2.8% psl=0.31 link=0.00GB/s gather=3us copyiss=788.8us que=222 hf_hook=0us hf_dec=0us hf_hook_max=0us hf_dec_max=274us | cum hit=91.0% h=41431 m=4115 p=623 B=2683791360 resident=623 nr=1167 vf=0 oom=0 stalls=0 sk=1586 stale=0 tag=0 pubbad=0 publag=2507 cmiss=0 cdup=0
```

基线首条 `[pool] dstep=100`（基线行号 884）：

```
[pool] dstep=100 win hit=93.6% nr=1.6% psl=0.38 link=0.00GB/s gather=4us copyiss=1463.8us que=42 hf_hook=0us hf_dec=18us hf_hook_max=0us hf_dec_max=35849us | cum hit=93.6% h=42610 m=2936 p=755 B=3468718080 resident=755 nr=686 vf=0 oom=0 stalls=0 sk=491 stale=0 tag=0 pubbad=0 publag=1284 cmiss=0 cdup=0
```

| 字段 | 基线 | 本轮 |
| --- | --- | --- |
| `win hit` / `cum hit` | 93.6% / 93.6% | 91.0% / 91.0% |
| `nr` | 1.6% | 2.8% |
| `psl` | 0.38 | 0.31 |
| `copyiss` | 1463.8us | 788.8us |
| `que` | 42 | 222 |
| `h` / `m` / `p` | 42610 / 2936 / 755 | 41431 / 4115 / 623 |
| `resident` / `nr`(计数) | 755 / 686 | 623 / 1167 |
| `vf` / `oom` / `stalls` | 0 / 0 / 0 | 0 / 0 / 0 |

## 10. 观测窗口内使用的命令

```
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 4,5,6,7
curl -s -o /dev/null -w "HTTP %{http_code}\n" --max-time 20 http://127.0.0.1:18010/health
curl -s --max-time 120 http://127.0.0.1:18010/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"..."}],"max_tokens":32,"temperature":0}'
tr '\0' '\n' < /proc/<pid>/environ | grep -E "PYTORCH_CUDA_ALLOC_CONF|CUDA_VISIBLE_DEVICES|LK_GPU_RESIDENT_MOE_LAYERS|LK_POOL_SLOTS"
```

## 11. 产物清单

| 文件 | 说明 |
| --- | --- |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/testA-expandable-segments-20261023_0720.md` | 本报告 |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/testA-load-mem-compare-20261023_0720.png` | 图 1：Load weight mem usage / avail mem 对照 |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/testA-vram-kv-compare-20261023_0720.png` | 图 2：稳态显存 / KV 定容对照 |
