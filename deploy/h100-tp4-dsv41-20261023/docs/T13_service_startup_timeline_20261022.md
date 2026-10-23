# T13 服务启动与长跑监控 · 时间线 (2026-10-22)

监控对象: DeepSeek-V4.1-Flash-1017, lk_moe 混合 (前 20 层 resident W4A8 + 后 20 层专家池),
端口 18010, 物理卡 4-7 (UUID 固定), 启动脚本
`/mnt/SSD01/models/.llm-env/Lsglang/scripts/serve_dsv41_tp4_h100.sh`。

监控者: teammate `svc-monitor`。日志目录 `/mnt/SSD01/models/.llm-env/Lsglang/logs/`。
本文件只记录观测到的事实 (时间戳 / 原始日志串 / 数字 / 路径), 不含判断。

---

## 0. 实例清单

| # | 日志文件 (绝对路径) | 拉起时间 | sglang PID | 结束时间 | 结束方式 |
| --- | --- | --- | --- | --- | --- |
| 1 | /mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261022_134044.log | 13:40:58 | 14319 | 13:41:46 | Python 异常 (模型构造期) |
| 2 | /mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261022_134318.log | 13:43:37 | 23880 | 13:58:30 | CUDA OOM (W4A8 权重重排) |
| 3 | /mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261022_140449.log | 14:05:08 | 25085 | 14:14:42 | CUDA OOM (W4A8 权重重排) |
| 4 | /mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261022_142033.log | 14:20:45 | 27144 | 14:58:31 (树拆除 14:58:48) | SIGSEGV (cuda graph capture 期) |

`logs/serve_dsv41_tp4_h100.log` 是指向最新一次日志的软链。PID 文件:
`/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100.pid`。

---

## 1. 实例 #1 · 134044 (13:40:58 → 13:41:46, 存活 48 s)

- 13:41:11 `Setting KV cache dtype to fp8_e4m3 for DeepseekV4ForCausalLM.`
- 13:41:17 `server_args={...}`
- 13:41:19 / 13:41:22 `Ignore import error when loading sglang.srt.multimodal.processors.inkling: No module named 'torchaudio'`
  (同类行也出现在 midashenglm / torchaudio 相关 import)
- 13:41:44 TP0-TP3 各自打印同一条 torchaudio import 忽略行
- 13:41:46 `[TP1] Scheduler hit an exception: Traceback (most recent call last):`
  栈 (前 12 行):
  `scheduler.py:6256 run_scheduler_process → scheduler.py:570 __init__ → scheduler.py:1117 init_model_worker
  → scheduler.py:1026 init_tp_model_worker → tp_worker.py:318 __init__ → tp_worker.py:480 _init_model_runner
  → ModelRunner(...)`
- 日志停在 13:41:46, 进程退出, 端口 18010 无监听。

---

## 2. 实例 #2 · 134318 (13:43:37 → 13:58:30, 存活 14 m 53 s)

配置: `--tensor-parallel-size 4 --mem-fraction-static 0.88 --max-running-requests 16
--context-length 1048576 --chunked-prefill-size 8192`; `LK_POOL_SLOTS=192`,
`LK_GPU_RESIDENT_MOE_LAYERS=0-19`; 无 `--ep-size`。

阶段行:

- 13:48:48 `[TP2] [prewarm] MHC load-time prewarm begin (not is_nextn)`
- 13:49:48 `[TP1] engram host table layer 14: layout=shared, 24171 MiB resident, 0 MiB in huge pages (0%)`
- 13:51:40 ~ 13:55:05 多条 `Required memory for warmup: <0.13-1.05>GB, Available memory: 37.03643798828125GB`
- 13:55:05 `[TP0] MoE vision-gate prewarm: 1 router spec(s) warmed`
- 13:55:05 `[TP0] DeepSeek V4 MHC prenorm prewarm: 22 n_splits buckets`
- 13:58:30 `[TP0] Preparing DSv4 MXFP4 experts for FlashInfer SM90 CUTLASS W4A8 (layer: model.layers.0.mlp.experts)...`
  同一秒 OOM:

```
[rank1]:[W1022 13:58:30.980884404 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 1 while trying to allocate 20132659200 bytes (free: 8293515264, total: 85019590656).
[rank1]:[W1022 13:58:30.009182698 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 1 while trying to allocate 20132659200 bytes (free: 12227772416, total: 85019590656).
```

- 共 5 条 `memory allocation failed with OOM` (device 0/1/2 都出现, 均为 20132659200 B = 18.75 GiB)
- 栈尾: `flashinfer/fused_moe/prepare.py:241, in _process_humming_mxfp4_w4a8_payload`
  → `rewritten = lut[delta, fp4_codes.to(torch.long)]`
- 完整报错串: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.75 GiB.
  GPU 2 has a total capacity of 79.18 GiB of which 11.39 GiB is free.
  Including non-PyTorch memory, this process has 67.41 GiB memory in use.`
- 终止: `13:58:30 Received sigquit from a child process.` /
  `kill_process_tree called: parent_pid=23880, include_parent=True, pid=23880`

崩溃后显存残留: 13:59 物理卡 6 被僵尸进程 PID 24504 (`[sglang::schedul] <defunct>`, 状态 `Zl`)
占 69030 MiB; 约 1 分钟后回收, 14:00 四卡恢复 0 MiB。

---

## 3. 实例 #3 · 140449 (14:05:08 → 14:14:42, 存活 9 m 34 s)

配置增量: `--ep-size 4 --attn-dp-size 1`, `LK_POOL_SLOTS=48`, 其余同 #2。

- 日志前缀出现 EP 位号: `TP0 EP0` / `TP1 EP1` / `TP2 EP2` / `TP3 EP3`
- 14:06:58 `Finished streaming dequant fp8 wo_a`
- 14:12:26 `engram host table: dropped the page cache of 48 checkpoint files (475 GiB) after faulting the shared table`
- 14:12:43 `engram host table layer 1: ... 24171 MiB resident`
- 14:13:01 `[prewarm] block-fp8 linear prewarm: capture_bs=[1, 2, 4, 8, 12, 16] block=0 M buckets=[1, 2, 4, 8, 12, 16]`
- 14:13:30-14:13:44 `Required memory for warmup: ...` (同 #2 的数值序列)
- 14:14:35 `[prewarm] block-fp8 linear prewarm done: warmed 210 module(s), 7 distinct (N,K) shape(s)`
- 14:14:42 OOM:

```
[rank3]:[W1022 14:14:42.578265606 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 3 while trying to allocate 18119393280 bytes (free: 13731430400, total: 85019590656).
[rank3]:[W1022 14:14:42.613061857 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 3 while trying to allocate 18119393280 bytes (free: 17271422976, total: 85019590656).
```

- 峰值 18119393280 B = 16.88 GiB (对比 #2 的 18.75 GiB); 栈尾同 #2:
  `flashinfer/fused_moe/prepare.py:241 _process_humming_mxfp4_w4a8_payload → lut[delta, fp4_codes.to(torch.long)]`;
  报错串 `GPU 3 ... 16.09 GiB is free ... this process has 62.71 GiB memory in use`
- 本实例**没有**打出运行时 `CUTLASS W4A8 (layer: ...)` 行 (grep `CUTLASS W4A8` 的唯一命中来自脚本 banner 的自检断言文本)
- `Padding MXFP4 experts from intermediate` 出现 0 次
- 终止: `14:14:42 Received sigquit from a child process.` / `kill_process_tree called: parent_pid=25085`

---

## 4. 实例 #4 · 142033 (14:20:45 → 14:58:31, 存活 37 m 46 s)

配置: 与 #3 相同 argv (`--ep-size 4 --attn-dp-size 1 --chunked-prefill-size 8192`,
`LK_POOL_SLOTS=48`), 另加环境补丁:

- banner 第 11 行: `[ENV-PATCH] flashinfer/fused_moe/prepare.py: _process_humming_mxfp4_w4a8_payload
  已改为 int32+按专家分块 (bit-equivalent, .orig 已备份, 回滚见
  /mnt/SSD01/models/.llm-env/humming/patches/rollback_humming_w4a8_prepare_patch.sh)`
- `flashinfer/fused_moe/prepare.py` mtime `2026-10-22 14:18:29`

### 4.1 阶段行 (摘录)

| 时刻 | 行 |
| --- | --- |
| 14:22:41 | `Finished streaming dequant fp8 wo_a` |
| 14:27:54 | `engram host table: dropped the page cache of 48 checkpoint files (475 GiB)` |
| 14:28:10 | `engram host table layer 1: ... 24170 MiB resident` |
| 14:28:27 | `engram host table layer 14: ... 24171 MiB resident` |
| 14:28:28 | `[prewarm] block-fp8 linear prewarm: capture_bs=[1, 2, 4, 8, 12, 16] ...` (4 卡同时) |
| 14:29:10 | `Required memory for warmup: 1.0454940795898438GB, Available memory: 37.03643798828125GB` |
| 14:30:05 | `Initialized lk_moe with 96 experts for layer 0 [GPU]` (TP0-TP3 各一条) |
| 14:30:05 | `Preparing DSv4 MXFP4 experts for FlashInfer SM90 CUTLASS W4A8 (layer: model.layers.0.mlp.experts)...` |
| 14:47:20 | `Initialized lk_moe with 96 experts for layer 1..8 [GPU]` (每秒多层) |
| 14:48:45 | 4 卡 `Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0.` |
| 14:48:49 | 4 卡 `Memory pool end. avail mem=` 10.53 / 11.00 / 10.62 / 10.53 GB (TP2/TP3/TP0/TP1) |
| 14:52:22 | `[attn-prewarm] rank=0..3/4 runner=EagerRunner is_spec=False block=0 capture_bs=[1, 2, 4, 8, 12, 16]` |
| 14:52:22 | `[attn-prewarm] rank=N buffers ok (input_ids=16 seq_lens=16, per_req_tokens=1)` |
| 14:52:22 | `[sc-dbg] disabled reason=not an EXTEND forward rows=None inner=8192` (4 卡) |
| 14:55:48 / 14:56:18 | `[TensorRT-LLM][INFO] Set logger level to INFO` (共 4 条) |
| 14:58:31 | `Fatal Python error: Segmentation fault` |

### 4.2 判据串核对

- `CUTLASS W4A8 (layer: ...)`: **20 行**, 覆盖 `model.layers.0` ~ `model.layers.19` 的全部 `.mlp.experts`
- `CUTLASS W4A16 (layer: ...)`: **0 行**
- `Padding MXFP4 experts from intermediate`: **0 次**
- `[pump] ON pool=ON slots=48 window=32 inflight=16 log=100`
- `[pool] slab: 16.81 GiB (20 layers x 860.62 MiB)`
  (860.62 MiB / 48 槽 = 17.93 MiB/槽/层; 脚本预检公式 line 587 用 4,700,160 B/槽 并额外 `/4`,
   对 48 槽给出 1.05 GiB/卡, 实测为该值的 16 倍)
- `[pool] staging ring: 1.68 GiB (96 rows, 3 sets, window 1)` (96 rows 与 ep4 的每 rank 96 experts 一致)
- `[pool] ... hit=` 命中率行: **未出现**
- sglang memory breakdown (weights / KV cache / cuda graph 分项) 与 `max_total_num_tokens`: **未出现**
- `/health`: 全程 HTTP 000 (无监听)

### 4.3 首处 fatal

```
Fatal Python error: Segmentation fault        (第 765 行)
...
Fatal Python error: Segmentation fault        (第 821 行)
```

崩溃线程栈 (自下而上):

```
multiprocessing/spawn.py:122 spawn_main → process.py:314 _bootstrap → process.py:108 run
→ scheduler.py:6256 run_scheduler_process → scheduler.py:570 __init__
→ scheduler.py:1133 init_model_worker → scheduler.py:1111 init_all_cuda_graphs
→ tp_worker.py:428 init_cuda_graphs → model_runner.py:1257 init_cuda_graphs
→ model_runner_components/cuda_graph_setup.py:392 capture_cuda_graphs
→ runner/eager_runner.py:152 __init__ → runner/base_runner.py:278 warmup
→ models/deepseek_v4.py:5556 prepare_before_cuda_graph_capture
→ runner/base_runner.py:770 _dummy_run
→ models/deepseek_v2.py:1104 forward → deepseek_v2.py:1425 forward_normal
→ fused_moe_triton/layer.py:1641 forward → layer.py:1662 forward_impl
→ layer.py:1753 run_moe_core → layer.py:2647 _cpu_prefill   ← 段错误落点
```

主进程侧异常:

```
sglang/srt/entrypoints/engine.py:1997 _wait_for_scheduler_ready → connection.py:250 recv ...
EOFError
→ http_server.py:2951 launch_server → engine.py:1339 _launch_subprocesses
→ engine.py:990 wait_for_ready → engine.py:1999 _raise _scheduler_died_error
RuntimeError: Rank 0 scheduler died during initialization (exit code: -11).
```

同时出现 (14:58:31) `[rank1]/[rank3] TCPStore.cpp:125 [c10d] recvValue failed ... Connection was likely closed`
与 `ProcessGroupNCCL.cpp:1893 ... TCPStore check for dump key "exception_dump" failed (store unavailable)`。

### 4.4 显存与进程

- 崩溃前 (14:55) 四卡 `71050 / 71144 / 71144 / 70664 MiB` (物理卡 4/5/6/7)
- 14:59:49 四卡 `4 MiB / 4 MiB / 4 MiB / 4 MiB`
- 14:59: 仍可见 `sglang::schedul` 进程 2 个, 主进程 PID 27144 状态 `Ssl`
- 监控期间 GPU0/1 未被触碰, 恒为 `70496 MiB`(其他用户 vLLM::Worker_TP0/TP1)

---

## 5. 监控期间的采集方式

每回合前台循环 (`for i in $(seq 1 N); do ...; sleep 30; done`), 采集项:

- `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -i 4,5,6,7` (物理卡 4-7)
- 新日志文件发现: `ls -1t /mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_*.log | head -1`
- 增量行读取: 记录上次读到的行号, 只 grep 新增区间 (避免重复)
- 端口/健康: `curl -s localhost:18010/health` 与 `/v1/models`
- 进程存活: `ps -p <PID> -o stat=` / `pgrep -af "[s]glang"`
- 判据串 grep 集合: `Preparing DSv4` / `CUTLASS W4A8 (layer:` / `W4A16` / `Padding MXFP4 experts` /
  `[pool]` / `slab` / `staging ring` / `[pump]` / `max_total_num_tokens` / `Memory pool` /
  `memory allocation failed with OOM` / `Traceback` / `capture` / `sigquit` / `Fatal Python error`

## 6. 未完成的采集项

- `max_total_num_tokens` (KV 池容量) —— 四个实例均未走到该方法打印即退出
- sglang memory breakdown (weights / KV cache / cuda graph 三项显存分项)
- `[pool] ... hit=` 命中率行 (LK_POOL_LOG=100)
- `/health` 通过时刻与端到端启动总耗时 (未就绪)
- `/v1/models` 与 `scripts/health.sh` 最小 chat 推理 (服务未就绪)
