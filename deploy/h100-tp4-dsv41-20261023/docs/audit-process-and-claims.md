# 部署过程审计报告: DSV4.1-Flash-1017 · Lsglang · 物理卡 4-7

| 项 | 值 |
| --- | --- |
| 审计对象 | 2026-10-22 13:24 — 20:21 的 16 次 `sglang serve` 实例(端口 18010) |
| 审计窗口 | 首个启动 13:40:55 → 最后实例结束 20:20:35(缺口 18:22:35→19:52:53) |
| 方式 | **全程只读**: 日志逐行解析 + 源码/配置文本读取 + 二进制反汇编复核 `cuobjdump` + 算术复算。未启动任何服务、未占用 GPU、未改 site-packages、未写 `/tmp` |
| 本报告产物 | 本文件(唯一产物);解析脚本与中间输出在 `/data3/shared/wangyf/llm-serving/audit_scratch/` |
| 明确未做 | 未读 `/mnt/SSD01/models/.llm-env/deprecated/` 与任何 `*_DEPRECATED*` 路径的内容(仅 `stat` 目录元数据);未使用 nvidia-smi 之外的任何 GPU 工具;未跑 compute-sanitizer / core dump |

## 0. 证据基

| 证据源 | 绝对路径 | 本报告用途 |
| --- | --- | --- |
| 启动日志(16 个) | `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261022_<HHMMSS>.log` | 时间线、阶段标志、fatal 原文、环境与命令行 |
| 有效配置快照(逐次) | `/mnt/SSD01/models/.llm-env/Lsglang/configs/dsv41_tp4_h100.effective.<ts>.conf` | 每次实际导出的环境变量与完整命令行 |
| 启动脚本 / 主配置 | `/mnt/SSD01/models/.llm-env/Lsglang/scripts/serve_dsv41_tp4_h100.sh`、`/mnt/SSD01/models/.llm-env/Lsglang/configs/dsv41_tp4_h100.conf` | 约束条款、参数拼装 |
| 被 import 的代码 | `/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/...` | 崩溃点与其依赖 |
| pin 源码副本 | `/mnt/SSD01/models/.llm-env/Lsglang/lsglang`(tag `lsglang-v1.6.1-2-g60f6658524`, 工作树干净) | 与 site-packages 未打补丁文件 **逐字节一致**(下表) |
| 二进制反汇编原始输出 | `/mnt/SSD01/models/.llm-env/Lsglang/logs/bincheck/`(`01_list_elf.txt`、`02_symbols_sm90a.txt`、`03_sass_{13,16,19,25}.txt`、`xelf/`) | 复核 FlashMLA 结论 |
| FlashMLA pin 源码 | `/mnt/SSD01/models/.llm-env/FlashMLA-3620b63`(`git rev-parse HEAD` = `3620b63fc4ebe33950c50e1b7701b6607f44150e`) | 复核地址算式 |
| 补丁与备份链 | `/mnt/SSD01/models/.llm-env/humming/patches/`(`.orig/.orig2/.orig3/.orig4/.orig5/.patched*`) | 判断"哪一版代码在跑" |

site-packages 与 pin 源码一致性抽查(md5,本轮实测):

| 文件 | site-packages | pin 源码 | 判定 |
| --- | --- | --- | --- |
| `srt/mem_cache/allocator/swa.py` | `ca2d6f3e…` | `ca2d6f3e…` | 相同 |
| `srt/mem_cache/memory_pool.py` | `cfd2a2bd…` | `cfd2a2bd…` | 相同 |
| `srt/mem_cache/dsv41_request_window.py` | `1902cce8…` | `1902cce8…` | 相同 |
| `srt/mem_cache/deepseek_v4_memory_pool.py` | `ced08655…` | `ced08655…` | 相同 |
| `srt/model_executor/runner/decode_cuda_graph_runner.py` | `6f5edecb…` | `6f5edecb…` | 相同 |
| `srt/model_executor/cuda_graph_buffer_registry.py` | `05e280e1…` | `05e280e1…` | 相同 |
| `srt/layers/moe/fused_moe_triton/layer.py` | `d3dc47e3…`(含 15:08 补丁) | `50664c5c…` | **不同**(补丁链) |

---

## 1. 启动与 GPU 工时账

### 1.1 阶段标志行的定义

下表所有阶段均取自日志中的真实字符串;`n` = 该标志行在单实例日志中的出现次数(4 = 4 个 TP/EP rank 各一行)。

| 阶段标志行(原文子串) | 含义 | 备注 |
| --- | --- | --- |
| `Load weight begin` | 开始加载权重 | 每实例恒 4 行 |
| `Finished streaming dequant fp8 wo_a` | 权重 dequant 流式完成 | 每实例恒 4 行 |
| `Preparing DSv4 MXFP4 experts for FlashInfer SM90 CUTLASS W4A8` / `W4A8 (layer:` | 进入逐层 W4A8 预处理 | 每实例最多 20 层 |
| `Memory pool end` | KV/SWA 池构造完成 | 每实例 4 行 |
| `[attn-prewarm]`、`[prewarm-skip]`、`[autotune-decode]` | 捕获前 eager 预热 / 跳过预热 / autotune dummy 标记 | 取决于开关 |
| `Super-chunk prefill wrapper active: CAP=…, inner=…` | SuperChunk 注意力 wrapper 被安装 | 见 §2.4 |
| `Capture target decode CUDA graph begin` / `Capturing batches` | 进入 decode CUDA 图捕获 | 仅 4 个实例到达 |
| `[flashmla-args] layer_id=…` | 本轮自建 dump 助手输出 | 仅 1 个实例到达 |
| `max_total_num_tokens`、`Uvicorn running`、`The server is fired up` | 服务就绪 | **全 16 个实例中命中 0 次** |

### 1.2 逐实例时间线(表 A)

时间 = 脚本头 `时间` 字段 → 日志中最后一条带时间戳的行;`mtime` 为日志文件落盘时间(用于"被中断"实例)。占卡 = 每一次启动的 `effective.conf` 中 `CUDA_VISIBLE_DEVICES` 恒为 4 个 UUID,经 `nvidia-smi -L` 映射为 **物理 GPU4/5/6/7**(16/16 实例完全一致)。

| # | 实例 | 启动 | 结束 | mtime | 存活 | 最远阶段(时间戳) | 首个 fatal / 结束方式(一行原文) | 占卡 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 134044 | 13:40:55 | 13:41:46 | 13:41:46 | 51 s | 权重加载(`Load weight begin` 13:41:45) | `TypeError: '>' not supported between instances of 'int' and 'NoneType'`(13:41:46, `layer.py:1947`) | 4-7 |
| 2 | 134318 | 13:43:34 | 13:58:30 | 13:58:30 | 14 m 56 s | W4A8 预处理(首个 `W4A8 (layer:` 13:58:30) | `torch.OutOfMemoryError: … Tried to allocate 18.75 GiB`(`prepare.py:241`) | 4-7 |
| 3 | 140449 | 14:05:05 | 14:14:42 | 14:14:42 | 9 m 37 s | dequant 完成(14:06:57),未进 W4A8 | `torch.OutOfMemoryError: … Tried to allocate 16.88 GiB`(`prepare.py:241`) | 4-7 |
| 4 | 142033 | 14:20:42 | 14:58:48 | 14:58:59 | 38 m 06 s | `[attn-prewarm]` 14:52:22(捕获前预热) | `Fatal Python error: Segmentation fault` → 栈顶 `fused_moe_triton/layer.py:2647 in _cpu_prefill` | 4-7 |
| 5 | 150833 | 15:08:48 | 15:23:56 | 15:23:56 | 15 m 08 s | `[attn-prewarm]` 15:18:03 | `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered`(15:23:56) | 4-7 |
| 6 | 153158 | 15:32:06 | 15:48:23 | 15:48:28 | 16 m 17 s | `[attn-prewarm]` 15:41:49 | SEGV → 栈顶 `flashinfer/norm/__init__.py:189 in _rmsnorm_impl` | 4-7 |
| 7 | 155228 | 15:52:37 | 16:08:25 | 16:08:35 | 15 m 48 s | `[attn-prewarm]` 16:01:51 | SEGV → `sgl_kernel/flash_mla.py:303 _flash_mla_with_kvcache_sched_meta`(经 `super_chunk_backend.py:501`) | 4-7 |
| 8 | 161136 | 16:11:47 | 16:21:01 | 16:21:35 | 9 m 14 s | `DeepGEMM JIT Pre-Compile` 16:20:17–16:21:01 | **被中断**(无 fatal 行;末行为 `resource_tracker` 退出告警) | 4-7 |
| 9 | 162421 | 16:24:29 | 16:40:46 | 16:41:14 | 16 m 17 s | `[attn-prewarm]` 16:34:10 | SEGV → `flash_mla.py:303 _flash_mla_with_kvcache_sched_meta` | 4-7 |
| 10 | 164716 | 16:47:25 | 17:03:26 | 17:04:12 | 16 m 01 s | `[prewarm-skip]` 16:56:51 + `Memory pool end` | SEGV → `fused_moe_triton/layer.py:2664 in _cpu_prefill`(autotune dummy) | 4-7 |
| 11 | 170709 | 17:07:20 | 17:22:57 | 17:23:36 | 15 m 37 s | `[attn-prewarm]`/`[autotune-decode]` 17:16:25 | SEGV → `flash_mla.py:303 _flash_mla_with_kvcache_sched_meta` | 4-7 |
| 12 | 172558 | 17:26:13 | 17:41:10 | 17:41:11 | 14 m 57 s | **图捕获**(`Capture target decode CUDA graph begin` 17:35:50;`Capturing batches`) | SEGV → `flash_mla.py:303 _flash_mla_with_kvcache_sched_meta` | 4-7 |
| 13 | 174609 | 17:46:21 | 18:01:13 | 18:01:40 | 14 m 52 s | **图捕获**(17:56:08 起) | SEGV → `sglang/kernels/ops/layernorm/mhc.py:2796 in hc_combine`(图捕获内) | 4-7 |
| 14 | 180710 | 18:07:19 | 18:22:35 | 18:22:45 | 15 m 16 s | **图捕获**(`Capturing batches` 18:19:02) | SEGV → `flash_mla.py:303 _flash_mla_with_kvcache_sched_meta`(图捕获内) | 4-7 |
| 15 | 195241 | 19:52:53 | 20:07:04 | 20:07:04 | 14 m 11 s | **图捕获 + 参数 dump**(`Capturing batches` 20:05:27;`[flashmla-args] layer_id=1` 20:06:48) | `NotImplementedError: "min_all_cuda" not implemented for 'Float8_e4m3fn'`(20:07:04, `deepseek_v4_backend.py:4256 in _r`) | 4-7 |
| 16 | 201810 | 20:18:29 | 20:20:35 | 20:21:52 | 2 m 06 s | 权重加载完成(`Finished streaming dequant` 20:20:35) | **被中断**(无 fatal 行) | 4-7 |

### 1.3 工时汇总

| 量 | 值 | 依据 |
| --- | --- | --- |
| 实例数 | **16** | `logs/serve_dsv41_tp4_h100_20261022_*.log` 计数(不含 13:26/13:28 两次 `health_*.log`,其内容为"端口 18010 无监听",未起进程、未占卡) |
| 总进程存活时长 | 13 754 s = **3.821 h** | 表 A 逐行求和 |
| **总 GPU 占用(4 卡)** | **15.28 卡·小时** | 3.821 h × 4 卡(每实例恒定占用 GPU4-7) |
| 观测窗口 | 6.661 h(13:40:55 → 20:20:35) | 同上 |
| 窗口内 4 卡占用率 | 57.4 % | 15.28 / (4 × 6.661) |
| 服务就绪成功次数 | **0**(`max_total_num_tokens` / `/health` 从未出现) | 全 16 个日志 grep 计数 = 0 |

### 1.4 白跑分类核算(逐条给证据)

| 档 | 判据 | 实例 | 卡·小时 | 占总 GPU 工时 |
| --- | --- | --- | --- | --- |
| **T1 可归因于本方明确缺陷** | 见 §2.1–§2.3 | 1、2、3、15 | **2.640** | **17.3 %** |
| T2 被更优方案取代而中止 | 实例 8 的 `chunked-prefill-size=16384` 假设,在运行中被"备选A"取代(后续实例头部原文:`(上一条已被备选A取代并回退为 8192)`) | 8 | 0.616 | 4.0 % |
| T3 自述"预期无效"的单变量排除轮 | 实例 13 头部原文:`新增 --swa-full-tokens-ratio 0.1(对齐厂商模板; 排除 SWA 池尺寸因素,预期无效作单变量排除)` | 13 | 0.991 | 6.5 % |
| T4 外部中断(原因不在日志内) | 实例 16:无 fatal、无 SIGTERM 记录,日志止于权重 dequant 完成 | 16 | 0.140 | 0.9 % |
| 其余(SEGV / 非法访存的定位轮) | §2.4 逐实例判定 | 4、5、6、7、9、10、11、12、14 | 10.898 | 71.3 % |

合计 15.285 卡·小时(与 §1.3 的 15.282 差 0.003,来自逐行四舍五入)。

**T1 三条缺陷的独立证据(逐条)**

| 缺陷 | 证据 | 实例 | 卡·小时 |
| --- | --- | --- | --- |
| 启动命令漏传 `--chunked-prefill-size` | 实例 1 完整命令行(`effective.20261022_134044.conf` 第 50 行、日志第 44 行)无该旗标;实例 2 起出现 `--chunked-prefill-size 8192` | 1 | 0.057 |
| 自建 instrumentation 未做 dtype 校验 | 实例 15 崩溃于 `deepseek_v4_backend.py:4256` 的 `int(t.min())`,该行属 19:51 应用的 `sglang-dsv41-dump-flashmla-args.patch` | 15 | 0.946 |
| W4A8 预处理 int64 索引 + 全张量一次性 gather | 实例 2/3 崩溃点同为 `flashinfer/fused_moe/prepare.py:241` `rewritten = lut[delta, fp4_codes.to(torch.long)]`(备份 `prepare.py.orig:240-241` 原文一致) | 2、3 | 1.637 |

---

## 2. "本可被静态检查拦下"清单

判定口径: **可拦下** = 启动前存在纯静态手段(读源码/读配置/算术核算)足以避免这次运行;否则记 **不可拦下** 或 **无法判定**(证据不足)。

### 2.1 ① 参数依赖缺失:实例 1 未传 `--chunked-prefill-size` → **可拦下**

| 事实依据 | 内容 |
| --- | --- |
| 崩溃点 | 日志 13:41:46 TP1 traceback 末帧:`File ".../sglang/srt/layers/moe/fused_moe_triton/layer.py", line 1947, in get_cpu_prefill_group_max_len` → `TypeError: '>' not supported between instances of 'int' and 'NoneType'` |
| 该行代码 | `layer.py:1945-1947`(pin 源码与当时 site-packages 一致):`max_num_batched_tokens = get_server_args().chunked_prefill_size` / `group_batch_size = min(4096, max(max_num_batched_tokens, get_super_chunk_size()))` |
| 为什么是 None | `srt/arg_groups/fields/schedule.py:52` 声明字段默认值 `chunked_prefill_size: Optional[int] = None`;`srt/runtime_context.py:1281` 的 `get_server_args()` 返回 `_CONTEXT.server_args`,而 `srt/server_args.py:625-634` 的原文写它是 **"the operator's raw input"**(原始输入,非解析后取值) |
| 对照组 | 同实例日志的 `server_args=` dump 打印的是**解析后**取值(`entrypoints/engine.py:170-171` `server_args.resolved_dict()`),显示 `'chunked_prefill_size': 8192` —— 即"日志显示 8192、代码读到 None"这一分歧本身可由两条源码路径静态解释 |
| 生效确认 | 实例 2 起命令行含 `--chunked-prefill-size 8192`,该 TypeError 不再出现,失败点前移到 `prepare.py:241` |
| 本方文档覆盖情况 | `docs/param-check.md:33`、`:142`、`:354` 已经登记过该旗标(并给出 `arg_groups/fields/schedule.py:52` 出处),但登记为"调度参数";`get_cpu_prefill_group_max_len` 这一消费者与其 raw/resolved 语义未在这一检查中登记 |

**结论:可拦下。** 触发条件是"旗标未出现在命令行"(`effective.20261022_134044.conf:50`),可静态判定;消费者的 None 不可容忍性在 `layer.py:1945-1947` 单文件内即可读出。

### 2.2 ② 显存算术:实例 2、3 的 OOM → **可拦下**

| 事实依据 | 内容 |
| --- | --- |
| 崩溃点(两次相同) | 实例 2 日志 13:58:30 TP1:`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.75 GiB … free: 11.39 GiB`;实例 3 日志 14:14:42 TP3:`… 16.88 GiB … free: 16.09 GiB`;两处 traceback 末帧均为 `flashinfer/fused_moe/prepare.py:241, in _process_humming_mxfp4_w4a8_payload` |
| 该行原文(当时在跑的版本) | `/mnt/SSD01/models/.llm-env/humming/patches/prepare.py.orig:240-241`:`delta = delta_scale_offsets.repeat_interleave(32, dim=-1).to(torch.long)` / `rewritten = lut[delta, fp4_codes.to(torch.long)]` —— **索引张量被升到 int64** |
| 算术复算 | 18.75 GiB = 20 132 659 200 B = 384 experts × 1280 rows × 5120 K × **8 B(int64)** —— 逐项相乘完全吻合;补丁自带注释亦给出同一算式(`sglang-dsv41-humming…` 见 `flashinfer-0.7.2-humming-w4a8-int32-chunk.patch` 的 `num_experts * rows * K * 8 B … (18.75 GiB at 384 experts / 1280 rows / K = 5120)`) |
| 静态可行性 | 该算式只依赖模型/权重形状(384/1280/5120)与源码里写死的 `.to(torch.long)`,不需要运行;`--mem-fraction-static 0.88` 与当时 `free` 量级同样可静态估算 |
| 本方修复(证明可静态处理) | 14:05:52 生成、14:06:12 应用的 `flashinfer-0.7.2-humming-w4a8-int32-chunk.patch` 把两处改为 `torch.int32` 并按 expert 分块;实例 4 起 `W4A8 (layer:` 完成 20 层,该 OOM 不再出现 |
| 注意 | 实例 3 的变量是 `LK_POOL_SLOTS 192→48`(头部 `方案` 行),它只减少了 KV 占用,未触及权重预处理峰值;两次请求量下降(18.75→16.88 GiB)但均失败 |

**结论:可拦下(两条记录同因,合计 1.637 卡·小时)。**

### 2.3 ④ 自建 instrumentation 的 dtype 校验:实例 15 → **可拦下**

| 事实依据 | 内容 |
| --- | --- |
| 崩溃原文 | 日志 20:07:04 TP3:`NotImplementedError: "min_all_cuda" not implemented for 'Float8_e4m3fn'`;栈:`deepseek_v4_backend.py:4281 in _forward_attention` → `:4256 in _r` → `f"min={int(t.min())} max={int(t.max())}"` |
| 代码来源 | 本方 19:51 应用的 `sglang-dsv41-dump-flashmla-args.patch`(门为 `SGLANG_DSV41_DUMP_FLASHMLA_ARGS`);实例 15 的 `effective.…195241.conf:36` 导出该门 =1 |
| 目标张量 dtype 可静态确定 | 同一日志 13:41 起即打印 `Setting KV cache dtype to fp8_e4m3 for DeepseekV4ForCausalLM`;dump 行的 `extra_k_cache` 参数在 `:4029` 取自 KV 池 extra buffer(该 buffer 即 fp8) |
| 后果 | 该轮的目的是 dump 参数以定位 SEGV;helper 在到达 `flash_mla_with_kvcache` 之前抛错(`Capture cuda graph failed: "min_all_cuda" not implemented`),本轮信息未获得 |
| 后续修复(证明可静态处理) | 20:17:24 重写的同一补丁加入 dtype 白名单 + `.to(torch.float32)` 兜底 + `try/except` 降级(`n/a(Exception)`);当前 site-packages = `…patched5`(md5 `3c8184a6…`),与本轮实测一致 |

**结论:可拦下。** 门后面的 helper 对"被观察张量可能是 float8"这一事实,在写代码时即可从 dtype 声明与池类型读出;`torch` 对 float8 无 `min/max` 归约是运行前可知的库事实(补丁重写版自己的注释亦如此写)。

### 2.4 ③ 分支可达性 / SEGV 家族(实例 4-7、9-14)→ **不可拦下或无法判定**

先给一条本轮实测的**分支可达性**证据:

| 事实 | 证据 |
| --- | --- |
| SuperChunk 注意力 wrapper 的安装门 | `srt/model_executor/model_runner_components/attention_backend_setup.py:130-141`:`get_super_chunk_size() > 0 and (get_schedule().chunked_prefill_size or 0) > 0 and get_super_chunk_size() > get_schedule().chunked_prefill_size and not model_runner.is_draft_worker` |
| `get_super_chunk_size()` 的来源 | `srt/utils/lk_moe_env.py:133-136`:档位关闭时返回 0,否则返回 `LK_GPU_PREFILL_SUB_M` |
| 运行时实证(wrapper 装/不装) | 实例 4/5/6/7 日志各有 4 行 `Super-chunk prefill wrapper active: CAP=16384, inner=8192`;实例 8(该轮把 chunked 改 16384)与实例 9-16 该行命中 **0 次** |
| 对应的环境开关 | 实例 ≤7:`LK_GPU_PREFILL_MIN_TOKENS=1024`;实例 ≥9:`LK_GPU_PREFILL_MIN_TOKENS=0` |

| 实例 | 崩溃点(栈顶,首行原文见 §1.2) | 当轮声明的新变量(日志头部 `ENV-CHANGE`/`ENV-PATCH`) | 静态可拦判定 |
| --- | --- | --- | --- |
| 4 | `fused_moe_triton/layer.py:2647 _cpu_prefill`,经 `base_runner.py:758 run_once`(捕获前预热) | `prepare.py` int32 分块补丁(wrapper 仍装) | **可拦下的部分是 §2.2 已计;该次 SEGV 本身不可拦下**:崩溃在"捕获前预热 dummy 前向"里,当时没有能定位该内存错误的静态结论 |
| 5 | `RuntimeError: Triton Error [CUDA]: an illegal memory access`(无 Python 层栈顶可读) | 头部**只声明**了 prepare.py 补丁;但相对实例 4 实际有两处**未在头部登记**的变化:`LK_GPU_PREFILL_MIN_TOKENS 1024→1`(`effective.…150833.conf:16`)、E2 补丁(`_eager_decode_prewarm` 作用域,15:08:17 落盘、15:08:36 应用,见 §4.4 F8)。`FLASHINFER_USE_CUDA_NORM` 仍为 1 | **无法判定**(内核级非法访存,日志未给 Python 栈;无静态证据指向具体代码;该轮非单变量) |
| 6 | `flashinfer/norm/__init__.py:189 _rmsnorm_impl` → `sgl_kernel/elementwise.py:120` → `layernorm.py:624` | 无新声明(与实例 5 同配置) | **无法判定** |
| 7 | `sgl_kernel/flash_mla.py:303 _flash_mla_with_kvcache_sched_meta`,经 `super_chunk_backend.py:501` | `FLASHINFER_USE_CUDA_NORM 1→0`,目的写明"norm 由 CUDA-JIT 切到 CuTe-DSL" | **不可拦下**(切 norm 实现对实例 6 的栈顶是合理假设;崩溃点被移动到 flash_mla,说明该假设部分成立) |
| 9 | `flash_mla.py:303 _flash_mla_with_kvcache_sched_meta`(wrapper 已不装) | `LK_GPU_PREFILL_MIN_TOKENS 1024→0`(关 GPU-prefill 档) | **不可拦下**:该轮把"wrapper 安装"这一变量消掉,是排除法必需的一步;wrapper 关闭后仍在同一函数崩溃,属新证据 |
| 10 | `fused_moe_triton/layer.py:2664 _cpu_prefill`(autotune dummy) | `SGLANG_DSV41_SKIP_CAPTURE_PREWARM=1` | **不可拦下** |
| 11 | `flash_mla.py:303`(autotune 段) | `SGLANG_DSV41_AUTOTUNE_DUMMY_DECODE=1` | **不可拦下** |
| 12 | `flash_mla.py:303`(图捕获内) | `--disable-flashinfer-autotune` | **不可拦下**(实例 10/11 连续两次崩在 autotune dummy,该轮移除 autotune 是必要排除) |
| 13 | `sglang/kernels/ops/layernorm/mhc.py:2796 hc_combine`(图捕获内) | `--swa-full-tokens-ratio 0.1`,头部自述"预期无效作单变量排除" | **不可拦下**;但见 §4.4:该轮声明的新代码(E 补丁)**未生效** |
| 14 | `flash_mla.py:303`(图捕获内) | 命令行与实例 13 相同;`effective.…180710.conf:36` 新增 `SGLANG_DSV41_SWA_PAD_SENTINEL=1`(日志头部 `ENV-CHANGE` 未登记该项) | **无法判定** |

**结论:SEGV/非法访存家族 9 轮(10.898 卡·小时)按现有证据记 不可拦下/无法判定。** 依据是同批次的两份静态文档最终都把该路径判为"索引合法、二进制与源码一致"(`docs/index-producer-trace.md:387` 的 X 行与 `docs/flashmla-sm90-index-contract.md:516`),即静态手段**没有**定位到该内存错误;这些轮次属单变量排除,崩溃点为逐轮观测所得。

### 2.5 其它两轮的判定

| 实例 | 静态问题 | 判定与依据 |
| --- | --- | --- |
| 8 | 把 `--chunked-prefill-size` 提到 16384 以使 wrapper 安装门为假 | **无法判定**:按 `attention_backend_setup.py:130-141`,`16384 > 16384` 为假,该轮的静态推理成立(日志亦无 wrapper 激活行);它是与"备选A"并列的一条可选项,只是被备选A取代。9 m 14 s 中大部分耗在 `DeepGEMM JIT Pre-Compile`(16:20:17–16:21:01),该阶段日志自带提示"typically 10-20 mins" |
| 13 | `--swa-full-tokens-ratio 0.1`,头部自述"预期无效" | **不可拦下但可核验为自愿排除轮**:该轮在启动时即被标注为预期无效(头部原文),属主动单变量排除;另见 §4.4 的未生效事实 |
| 16 | 无 | **无法判定**:日志无 fatal、无 SIGTERM 记录,pid 文件记录的 `PID=27993` / `LAUNCHER_PID=26267` 在审计时已不存在 |

### 2.6 清单汇总

| 判定 | 条目数 | 实例 | 卡·小时 | 占比 |
| --- | --- | --- | --- | --- |
| **可拦下** | **4 条记录 / 3 类缺陷** | 1、2、3、15 | 2.640 | 17.3 % |
| 不可拦下 | 6 轮(每轮都在排除一个已由前一轮证据支持的变量) | 7、9、10、11、12、13 | 6.236 | 40.8 % |
| 无法判定 | 6 轮(SEGV/非法访存的定位证据不足,或结束方式无日志) | 4、5、6、8、14、16 | 6.409 | 41.9 % |
| 合计 | 16 轮 | — | 15.285 | 100 % |

---

## 3. 两份技术文档核心结论的独立复核

复核方法:**不看论证过程,只看原始证据**。凡文档引用的文件,本轮均独立打开并逐行核对;凡引用的二进制事实,本轮均用同一输入文件重算。

### 3.1 `docs/index-producer-trace.md`(387 行,2026-10-22 19:54)

| # | 文档结论(位置) | 本轮独立复核 | 判定 |
| --- | --- | --- | --- |
| 1 | `full_to_swa` 两处构造均非 `torch.empty`;A 路径前 `size+page_size` 行恒 0、末 1 行为 -1(`:19`、`:119-122`) | `srt/mem_cache/allocator/swa.py:101-112` 原文逐行一致(注释 `# Trailing -1: a last_loc of -1 (no prefix) indexes it…` + `torch.cat([torch.zeros(size + self.page_size, int64), torch.tensor([-1])])`);B 路径 `:750-755` 为 `torch.arange(size_swa + page_size)` + 末 `-1`,一致 | **支持** |
| 2 | `req_to_token` 是 `torch.zeros`(int32),行 0 为 padding 保留行(`:20`、`:178-181`) | `srt/mem_cache/memory_pool.py:398-407` 原文一致:`# +1 padding row at index 0: cuda-graph padded batches default req_pool_indices to 0…`、`self._alloc_size = size + 1`、`torch.zeros((self._alloc_size, max_context_len), dtype=torch.int32)`、`free_slots = list(range(1, self._alloc_size))` | **支持** |
| 3 | capture 期 `indices` 值域只可能是 {合法 loc, -1}(`:317-327`、`:387`) | 机制层成立:内核 `kernels/ops/attention/dsv4_attn_metadata_kernels.py:591-596` 的 `valid` 掩码 + `tl.where(valid, swa, -1)`;`window_layout`(`mem_cache/dsv41_request_window.py:91`)`torch.where(valid, …, -1)`;辅助表零初始化(第 1、2 条)排除"未初始化垃圾" | **支持(限值域上界)** |
| 4 | 活跃分支是 ① request-window,`BuildCausalSwaPageIndices` **不在活跃路径**(`:257-279`、`:377`) | **与运行日志相反**:16/16 实例的解析后 flag 为 `'enable_encoder_swa_bounded_replay': False`(仅 `enable_decoder_swa_bounded_replay: True`);`mem_cache/deepseek_v4_memory_pool.py:1114-1141` 只在 `encoder_replay` 为真且非 draft 时构造 `request_window`;`:1220-1227` 据此打印 `storage=`;实际日志为 `DSV4 SWA storage: worker=target, storage=paged, paged_allocator=True`(**11 个**到达池构造的实例全部如此:142033/150833/153158/155228/162421/164716/170709/172558/174609/180710/195241)⇒ `request_window is None` ⇒ 活跃分支是 **③** | **不支持** |
| 5 | 分支 ① 下的具体值(每行恰 1 个有效索引、`lengths ∈ {0,1}`、`extra_indices = -1`)(`:365-373`) | 该推导以第 4 条为前提;分支①在本部署未激活,故这些"具体值"对本部署**不成立** | **不支持** |
| 6 | 三条互斥分支的结构与判据行(`:263-265`) | `deepseek_v4_backend.py` 中确实存在三分支:`if self.token_to_kv_pool.request_window is not None:` → `window_layout(...)` + `_pad_last_dim(request_layout.indices)`;`elif dspark_block_size is not None:` → `get_dspark_swa_page_indices/dspark_swa_buffers`;`else:` → `BuildCausalSwaPageIndices.execute(...)` | **支持** |
| 7 | 内核语义(①`other=-1` 只在 valid=False 生效;②第二次 load 无上界约束)(`:84-87`) | `dsv4_attn_metadata_kernels.py:578-596` 原文一致(见 §3.2 表) | **支持** |
| 8 | `req_to_token` 中"确有 `-1` 的行"(`:229`) | 该行只给了写者清单(`:186-193`),未给出任何"写入 -1"的具体代码行;本轮亦未在写者中定位到写 -1 的语句 | **证据不足** |
| 9 | `full_to_swa[-1]` 不属越界,依赖表尾哨兵(`:22`、`:150`) | 表尾确有 `-1` 元素(`swa.py:103-112`);torch 参考实现 `dsv4_attn_metadata_kernels.py:549-550` 以 `-1` 索引 | **支持** |

### 3.2 `docs/flashmla-sm90-index-contract.md`(516 行,2026-10-22 19:52)

复核对象:同一份原始输出 `/mnt/SSD01/models/.llm-env/Lsglang/logs/bincheck/`(其中 `xelf/` 171 个已抽取 cubin)。本轮重算结果与文档表格**逐格吻合**:

| 文档结论 | 文档位置 | 本轮重算(命令 → 结果) | 判定 |
| --- | --- | --- | --- |
| cubin 清单 57 `sm_90a` / 57 `sm_100` / 57 `sm_103a`,普通 `sm_90` **0** | `:372-377` | `grep -oE "sm_[0-9]+[af]?" 01_list_elf.txt \| sort \| uniq -c` → `57 sm_90a` / `57 sm_100` / `57 sm_103a`;`sm_90` 精确匹配 = 0 | **支持** |
| 承载函数恰 6 个 sm_90a cubin,每个 1 个 function,且字节数与实例化文件一一对应 | `:381-390` | 6 个符号命中(见 §3.3);`xelf/…abi3.{13,16,19,22,25,28}.sm_90a.cubin` 字节 = 117 712 / 98 000 / 80 848 / 75 472 / 80 208 / 74 832,与文档表格 6/6 相同;`cuobjdump -sass …abi3.13…cubin` 的 `Function :` 计数 = 1;`cuobjdump -symbols` 中 `STT_FUNC` 恰 1 个,其名前缀含 `_v4_persistent_h64_cu_` ⇒ 对应 `v4_persistent_h64.cu`,与文档一致 | **支持** |
| `num_blocks` 在 SASS 中出现 **0** 次;`extra_num_blocks` 0 次;`page_block_size` 1 次;`topk` 3 次 | `:449-460` | `grep -c num_blocks 03_sass_13.txt` → 0(16/19/25 与 `02_symbols_sm90a.txt` 同样为 0);参数偏移直方图 `c[0x0][0x230]`=0、`c[0x0][0x280]`=0、`c[0x0][0x234]`=1、`c[0x0][0x238]`=3 | **支持** |
| sm_90a 专属指令计数 | `:476-485` | `grep -oF` 计数:`HGMMA`=40、`UTMALDG`=16、`UTMASTG`=2、`UBLKCP`=16、`SETMAXREG`=3、`SETMAXNREG`=0、`WGMMA`=0、`LDGSTS`=0、`CP.ASYNC.BULK`=0 | **支持** |
| 地址算式逐条(`t == -1` 页 0 分支、u32 除/模软件序列、除零保护、`576`/`8` 常量、两项地址与源码对齐) | `:411-470` | 文档引用的 9 个 SASS 地址锚点全部存在且指令文本一致:`/*ac20*/ ISETP.GE.AND P0, PT, R18, UR13, PT`、`/*ac40*/ SEL R31, R12, 0xffffffff, !P0`、`/*ac50*/ ISETP.NE.AND P0, PT, R31, -0x1, PT`、`/*ae10*/ @P0 IMAD.MOV.U32 R22, RZ, RZ, R18`、`/*ab50*/ LDC R14, c[0x0][0x234]`、`/*ac70*/ ISETP.NE.U32.OR P4, PT, R14, RZ, !P0`、`/*ae90*/ IMAD.WIDE.U32 R22, R22, UR12, RZ`、`/*aef0*/ IMAD.WIDE R20, R14, 0x240, R22`、`/*b070*/ IMAD.WIDE R22, R29, 0x240, R22` | **支持** |
| 二进制与 pin 源码在地址算式上一致 | `:514-516` | SASS 侧如上;**源码侧**本轮亦核对:`FlashMLA-3620b63/csrc/kernels/sm90/decode/sparse/splitkv_mla.cuh:533-539` 原文为 `if (rel_block_idx*TOPK_BLOCK_SIZE + … >= topk_length) { token_index = -1; }` 与 `block_index = token_index == -1 ? 0 : (int)((uint32_t)token_index/(uint32_t)page_block_size);` / `rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;`(并含那条 `UINT_MAX%page_block_size` 注释);两处预取 `__ldg(gIndices + …)` 分别在 `:475`(含 `round*NUM_TOKENS_PER_ROUND`)与 `:520`(含 `(block_idx+1)*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2)`),与文档引用的 `:474-476`/`:519-520` 及"`+0x80` = 32 个 int32 = TOPK_BLOCK_SIZE/2"的对应关系一致 | **支持** |
| 源码是否用 `sm_90a` 宏二次门控:0 命中;唯一 arch 门是 `__CUDA_ARCH__ == 900` | `:487-502` | `grep -rn "__CUDA_ARCH_FEAT_SM90_ALL\|SM90_ALL" FlashMLA-3620b63/csrc` → **0 命中**;`:89` 起的 `#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || …` 原文一致 | **支持** |
| 实例化集合:pin 源码 6 个 `.cu`,与 wheel 的 6 个实例化一一对应;`Decode_Sm90_Impl` 特性表无 V41 格式 | `:504-510` | `…/csrc/kernels/sm90/decode/sparse/instantiations/` 恰 6 个文件(v4/v32/v32_no_rope × h64/h128);`csrc/api/sparse_decode.cpp:54-67` 的 `DECLARE_SUPPORTED_FEATURES` 列出 `V32_KVCACHE_FORMAT`、`V32_NO_ROPE_KVCACHE_FORMAT`、`V4_KVCACHE_FORMAT` 等,无 V41 项 | **支持** |
| pin 与本地路径一致 | `:12-14` | `git -C FlashMLA-3620b63 rev-parse HEAD` = `3620b63fc4ebe33950c50e1b7701b6607f44150e`,log = `misc: set v41 format as defaultx (#15)` | **支持** |
| 参数结构体字段偏移映射(`num_blocks` +0x20 / `page_block_size` +0x24 / `topk` +0x28 …) | `:403-409`、`:455-458` | 偏移的**计数**已复算一致(见上);偏移到字段**名字**的映射本轮未独立定位到结构体定义 | **证据不足(仅命名映射)** |
| 结论句:"二进制 `sparse_decode_fwd` 与 pin 源码在地址算式与关键指令上一致" | `:516` | 上述各条均在 SASS+源码两侧成立 | **支持** |

### 3.3 第三问:两条互斥结论(`cubin 是 sm_90` vs `是 sm_90a`)的判定

**判定:`cubin 是 sm_90` 那一版是取样/度量缺陷造成的;正确的是 `sm_90a`。**

| 事实依据 | 内容 |
| --- | --- |
| 缺陷机制(文档自述) | `docs/index-producer-trace.md:248-251` 的口径更正框原文:早期版本"取样用了**有损正则** `grep -oE "sm_[0-9]+"`,它会吃掉架构后缀字母(`sm_90a` → `sm_90`)";并注明该错误"以及据此发出的报文" |
| 本轮独立复现(同一输入文件) | `cd logs/bincheck && grep -oE "sm_[0-9]+" 01_list_elf.txt \| sort \| uniq -c` → `57 sm_90` / `57 sm_100` / `57 sm_103`(**正好复现错误结论**);`grep -oE "sm_[0-9]+[af]?" 01_list_elf.txt \| sort \| uniq -c` → `57 sm_90a` / `57 sm_100` / `57 sm_103a` |
| 原始文件本身 | `01_list_elf.txt` 第 1-12 行逐行给出 `flashmla_ops.abi3.<n>.sm_90a.cubin`,即文件名自带后缀,阅知即知 |
| 另一条同源污染的候选 | `docs/lsglang-report.md:394` 记录 lk_moe wheel 的 nvcc 参数含 `-arch sm_80 / sm_90 / sm_100 / sm_120`(来自 `strings`/元数据,不是 cubin 清单);该处出现裸 `sm_90` 是字符串事实,与 cubin 清单无关 |
| 旁证(本轮实测) | `cuobjdump --list-elf` 于**另一个**二进制 `sgl_kernel/flash_ops.abi3.so` → `372 sm_80` / `372 sm_86` / `372 sm_90a`(**无裸 sm_90**)。即两份 `.so` 都不含裸 `sm_90` cubin,裸 `sm_90` 只能由字符串/正则处理产生 |
| 留下的痕迹 | 错误版本只存在于**被就地覆盖的那一行**与"发出的报文";文件系统内没有该错误版本的独立副本(更正框是该错误的唯一文件级记录) |

### 3.4 过度断言清单(把"未证实"写成"已证实"、相关性写成因果、静态推断写成实测)

| # | 位置 | 原文要点 | 问题 | 证据 |
| --- | --- | --- | --- | --- |
| O1 | `docs/index-producer-trace.md:8` | 表头声明"**纯静态阅读**。未占 GPU、未启动服务、未跑任何复现" | 该文 §7-§8 的关键前提是"本配置 ⇒ 活跃分支是①",而该前提与运行日志相反;纯静态推理被当作"已定事实"发布 | 见 §3.1 第 4 条:日志 `DSV4 SWA storage: … storage=paged`;`'enable_encoder_swa_bounded_replay': False` |
| O2 | `docs/index-producer-trace.md:229` | "`req_to_token` 中**确有** `-1` 的行(写入方见 §3.2)" | 写成既有事实,但只列了写者清单,没给写 -1 的代码行;本轮亦未定位到 | 该行自引用的 §3.2 只有文件清单(`:186-193`) |
| O3 | `docs/index-producer-trace.md:257` | 小节标题"**关键发现**:用户分析的 kernel 在本配置下**不在活跃路径上**" | 与运行日志直接冲突;且该发现本身恰使该文 §1 的逐行内核分析失去对象 | 同上;另见 `mem_cache/deepseek_v4_memory_pool.py:1220-1227` 的 storage 判定 |
| O4 | `docs/index-producer-trace.md:365-373` | 以"输入 req=0、pos≈0"推出 capture 期"**每行恰好 1 个 True**"等具体值 | 由静态推断给出"确定值",但前提分支未激活 | 分支依据同 O1 |
| O5 | `docs/flashmla-sm90-index-contract.md:6` | "本文全部结论来自**静态源码阅读**,不涉及 GPU 运行" | §14 的全部结论来自 **wheel 二进制反汇编**(`cuobjdump`),不是源码阅读;虽然仍是静态、零 GPU,但对象类别不同 | `:354-357` 明写对象是 `flash_mla_ops.abi3.so` 与 `cuobjdump` |
| O6 | `docs/flashmla-sm90-index-contract.md:230` | "E 补丁(哨兵门)把 pad 行改成 `page=-1, length=0` …(`SGLANG_DSV41_SWA_PAD_SENTINEL=1`)" | 写成现存事实;但该补丁在本文写作前**已被回退**,site-packages 内已无此代码 | `patches/…deepseek_v4_backend.py.orig5`(19:51:00 备份)与当前 site-packages 的 `grep -c SWA_PAD_SENTINEL` 均为 **0**;`patched4` 为 2 |
| O7 | `docs/flashmla-sm90-index-contract.md:262` | "此时若 `params.extra_indices == nullptr` … ⇒ 该分支下**会发生非法访存**" | 机制性叙述以条件句给出,但句子本身是断言式;文档自身在 §11.3 承认需运行期观测 | 同处括号内已注明"该条件的完整可证伪形式…见 §13.3",属自我限定,程度轻 |
| O8 | `docs/index-producer-trace.md:248-251` | 更正框把根因归为"有损正则" | 该归因**有独立证据支持**(本轮已复现);但错误结论"据此发出的报文"的原文在文件系统内不可检索,无法逐字比对 | §3.3 表 |

### 3.5 可复核性缺陷(结论未必错,但**引用的行号对不上现存文件**)

| 缺陷 | 证据 |
| --- | --- |
| `docs/index-producer-trace.md` 与 `docs/flashmla-sm90-index-contract.md` 对 `deepseek_v4_backend.py` 的引用行号,属于**已被回退的中间版本** | 文档写于 19:52-19:54。当前 site-packages(5 180 行)无 E 补丁;`orig5`(19:51:00 备份,5 104 行)同样无 E 补丁;`patched4`(17:45:08,5 127 行)含 E 补丁。索引文 `:4837`(分支)= `orig5` 的 `4814` + 23 行(正是 E 补丁的行数);flashmla 文 `:16` 的 `4267-4282` = 崩溃栈里的 `4244` + 23。即两文的行号系统性对应 **patched4**,而该版本在文档写作前已不存在于 site-packages |
| 崩溃栈行号与文档行号相差 23 行 | 实例 9/11/12 的栈为 `deepseek_v4_backend.py:4244 in _forward_attention`;两文引用同一处为 `4267` |
| 补丁链的静默丢失 | `patched4`(含 E 补丁)存在,`orig5`(19:51 备份)不含 ⇒ 期间执行了 E 补丁回退(`patches/rollback_sglang_dsv41_swa_pad_sentinel.sh` 的机制是 `cp -p "$f.orig4" "$f"` 并删除 `orig4`);但 E 补丁的 gate 变量仍留在配置里(见 §4.4) |

---

## 4. 纪律遵守情况

### 4.1 是否存在"未先做静态检查即启动"

按时间线列出每次启动前**最近一次**静态产物与该次启动自身的静态声明(头部 `ENV-CHANGE` / `[判据]` 行):

| 实例 | 启动 | 启动前最近的静态产物(文件 mtime) | 该次启动自带的静态声明 |
| --- | --- | --- | --- |
| 1 | 13:40:55 | `docs/param-check.md` 13:28、`docs/budget-lsglang.md` 13:36:46 | 头部有"就绪后自检断言"行;无参数依赖核算 |
| 2 | 13:43:34 | `docs/env-report.md` 13:42:04 | 方案行 `pool=192 槽` |
| 3 | 14:05:05 | `logs/OOM-evidence-20261022_135830.md` 14:02:58 | 方案行 `pool=48 槽`、`--ep-size 4` |
| 4 | 14:20:42 | `logs/OOM-evidence-20261022_141442-ep4.md` 14:19:30、prepare 补丁 14:05:52 | `[ENV-PATCH]` prepare.py int32 分块 |
| 5 | 15:08:48 | `logs/SEGV-evidence-20261022_145831.md` 15:06:56、E2 补丁脚本 15:08:36 | 头部仅 `[ENV-PATCH]` prepare.py;E2 补丁与 `LK_GPU_PREFILL_MIN_TOKENS 1024→1` **未登记**(见 §4.4 F8) |
| 6 | 15:32:06 | `logs/SEGV-evidence-20261022_154735.md` … 该文件 15:50:30 属**实例 6 之后**;实例 6 前最近一次为实例 5 的证据文件 15:06:56 | 无新声明 |
| 7 | 15:52:37 | `logs/SEGV-evidence-20261022_154735.md` 15:50:30 | `[ENV-CHANGE] FLASHINFER_USE_CUDA_NORM 1→0` |
| 8 | 16:11:47 | (无新文件) | `[ENV-CHANGE] chunked-prefill-size 8192→16384` + 目的(附源码行 `attention_backend_setup.py:131-140`) |
| 9 | 16:24:29 | `logs/SEGV-evidence-20261022_163410.md` 16:43:48 属之后;最近为 15:50:30 | `LK_GPU_PREFILL_MIN_TOKENS 1024→0` + 目的 |
| 10 | 16:47:25 | `logs/SEGV-evidence-20261022_163410.md` 16:43:48 | `SGLANG_DSV41_SKIP_CAPTURE_PREWARM=1` + `[判据]` 生效凭据行 |
| 11 | 17:07:20 | `patches/apply_sglang_dsv41_autotune_dummy_decode.sh` 17:05:56 | `SGLANG_DSV41_AUTOTUNE_DUMMY_DECODE=1` + 两条 `[判据]` |
| 12 | 17:26:13 | `patches/base_runner.py.patched3` 17:05:44 | `--disable-flashinfer-autotune` + 复查 `[判据]` |
| 13 | 17:46:21 | `patches/sglang-dsv41-swa-pad-sentinel.patch` 17:45:08(应用脚本 17:45:21) | `--swa-full-tokens-ratio 0.1`(自述预期无效);**E 补丁的 gate 未导出**(见 §4.4) |
| 14 | 18:07:19 | 同实例 13 的补丁链 | 命令行同实例 13;`effective` 中新增 gate(头部未登记) |
| 15 | 19:52:53 | `patches/apply_sglang_dsv41_dump_flashmla_args.sh` 19:51:09;`docs/flashmla-sm90-index-contract.md` 19:52:39 | 无 `[判据]` 行 |
| 16 | 20:18:29 | `patches/…deepseek_v4_backend.py.patched5` 20:17:24、`humming/docs/SEGV-145831-analysis.md` 20:17:45 | 无 `[判据]` 行 |

事实性结论(不下判断):

* 16/16 实例的日志头部都带**至少一条**启动意图声明(`ENV-CHANGE`/`ENV-PATCH`/`方案`/`就绪后自检断言`);实例 10、11、12 带明确 `[判据] 生效凭据` 行。
* 实例 1 启动前 12 分钟内存在两份静态文档(`param-check.md`、`budget-lsglang.md`),但其中没有覆盖 §2.1 的消费者语义。
* 实例 2、3 启动前的静态产物是**前一次 OOM 的证据整理**(`OOM-evidence-*.md`),不是显存算术核算;两条 OOM 的算式在本轮已可复算(§2.2)。
* 实例 8、13 的静态声明中**给出了具体源码位置**(`attention_backend_setup.py:131-140`、SWA 池尺寸),属"先读码后改参"的形式。

### 4.2 `/tmp` 残留(约定见 `scripts/serve_dsv41_tp4_h100.sh:15`、`:171`、`configs/dsv41_tp4_h100.conf:61`:"任何产物都不写 /tmp")

审计窗口(13:24–20:22)内、所有者为 `wangyf` 且在 `/tmp` 下的残留:

| 路径 | 类型/大小 | 时间 | 生成者(可核) |
| --- | --- | --- | --- |
| `/tmp/sglang_temp_file_1792669994.{548663_6861398,5522234_7048746,5534184_4673751,5545504_5902109}.sh` | 4 × 94 B,权限 **0777** | 19:53:14 | `sglang/srt/utils/numa_utils.py:102` 的 `f"/tmp/sglang_temp_file_{time.time()}_{random…}.sh"`(内容 = `exec numactl --interleave=all …/python "$@"`);时间戳对应实例 15 启动(19:52:53) |
| `/tmp/sglang_temp_file_1792671528.{976249_598250,9793606_6977246,9803429_9576123,981273_4267527}.sh` | 4 × 94 B,0777 | 20:18:48 | 同上;对应实例 16(20:18:29) |
| `/tmp/tmpynm1bdmu/`、`/tmp/tmpq3l3gxec/`、`/tmp/tmph0tcum64/`、`/tmp/tmp_pfsj8ni/`、`/tmp/tmpdck4f0l8/`、`/tmp/tmphjkpsl3a/` | 目录(空),0700 | 20:18:40–20:19:02 | 实例 16 进程树 |
| `/tmp/tmpkvdwnmlh`、`/tmp/tmptlxmx2nq`、`/tmp/tmp7hmn53mz` | socket | 20:18:55–20:19:04 | 实例 16 进程树 |
| `/tmp/tmp92bgsaqr`、`/tmp/tmpg4nz1hzw` | 0 B | 20:18:48 | 实例 16 进程树 |
| `/tmp/tmpxft_0000885a_00000000-{3_cuda.fatbin.c,4_cuda.module_id,5_cuda.cpp4.ii,6_cuda.cudafe1.*,8_cuda.cubin}` | CUDA JIT 中间件(最大 4.3 MB) | 20:06:57–20:07:01 | 实例 15 的 warmup 期间 nvcc |
| `/tmp/ccfBkdkh.s` | 450 560 B | 20:07:04 | 同上 |
| `/tmp/.err_11777` | 0 B | 20:13:35 | 未定位到产生者 |

事实性说明:以上均由**运行时/第三方代码路径**(`numa_utils.py` 的 numactl 包裹脚本、CUDA JIT、torch 临时目录)写入,不是启动脚本自身的产物目录逻辑;项目脚本/配置中的产物目录为 `logs/`、`configs/*.effective.<ts>.conf`(见 `effective.20261022_134044.conf` 头部与 `:171` 注释)。`/tmp` 下另有其他用户(`chensm`、`wuw`)与本项目无关的文件,未计入。

### 4.3 `deprecated/` 与 `*_DEPRECATED*` 的读取痕迹

| 检查 | 结果 |
| --- | --- |
| 目录元数据(仅 `stat`,未读内容) | `/mnt/SSD01/models/.llm-env/deprecated/`:mtime `12:51:28.828`、atime `12:51:31.568`(相差 3 s = 创建时刻),mode 775;`/data3/shared/wangyf/llm-serving/_DEPRECATED-coldhot-2026-10-22`:mtime `12:38:05.651`、atime `12:38:05.644` |
| 两个目录的 atime 均早于部署窗口 | 窗口起点 13:24(环境安装)/13:40:55(首次启动)⇒ **未发现**窗口内访问痕迹 |
| 文本层痕迹 | `logs/`、`scripts/`、`configs/` 内 `grep -l "deprecated/"` = **0 个文件**;`docs/` 内 3 处"deprecated"字样为合规声明或 Python 弃用警告文案(`docs/index-producer-trace.md:8`、`docs/budget-lsglang.md:8`、`docs/correctness-gate-protocol.md:170`) |
| 方法学限制 | atime 在 `relatime/nodiratime` 语义下不可靠(仅 mtime 之后首次访问才更新一次);因此本项只能给出"未发现痕迹",不能给出"确证未读" |

### 4.4 其它可核事实(与声明/现状不一致处)

| # | 事实 | 证据 |
| --- | --- | --- |
| F1 | 实例 13 运行在"E 补丁**已应用但 gate 未打开**"的状态:`SGLANG_DSV41_SWA_PAD_SENTINEL` 不在 `effective.20261022_174609.conf` 的任何一行;实例 14 的 `effective.20261022_180710.conf:36` 才导出 =1 | `diff` 两份 effective:唯一差异为第 36 行(实例 14 多出 `export SGLANG_DSV41_SWA_PAD_SENTINEL=1`) |
| F2 | 实例 14 是该 gate 唯一真正生效的一次:日志中 `SWA/indexer padding rows now carry sentinel` 出现 **124** 次;实例 15、16 各 **0** 次(其 effective 仍导出 =1) | `grep -c` 逐实例计数 |
| F3 | 实例 15、16 运行时,E 补丁代码已不在 site-packages,但配置仍在导出其 gate | `patches/…backend.py.orig5`(19:51:00)`SENTINEL=0`、`patched5`(20:17:24)`SENTINEL=0`、当前 site-packages `SENTINEL=0`;而 `configs/dsv41_tp4_h100.conf:234` 为 `SGLANG_DSV41_SWA_PAD_SENTINEL="1"` |
| F4 | E 补丁**回退**这一步没有出现在任何日志头部的 `ENV-PATCH/ENV-CHANGE` 行里(15/16 实例头部均无相关登记) | 逐实例头部抽取,仅 15/16 的 effective 里可见该变量,头部声明无 |
| F5 | 实例 14 的 E 补丁代码注释本身写着 "…is NOT yet established -- **keep the gate off unless a numerical A/B (see docs) confirms it**",而该 gate 在实例 14/15/16 的配置中为 1 | `patches/sglang-dsv41-swa-pad-sentinel.patch` 注释;`effective` 第 36 行 |
| F6 | 5 处 site-packages 补丁的备份链现状:`.orig`(flashinfer `prepare.py`、`fused_moe_triton/layer.py`、`deepseek_v4.py`)、`.orig2`/`.orig3`(`deepseek_v4.py`、`base_runner.py`)、`.orig4`/`.patched4`(backend+indexer,E 补丁)、`.orig5`/`.patched5`(backend,dump 补丁) | `ls -la` 结果;`orig4` 未被删除(与 E 回退脚本"删除 orig4"的预期不同),说明 E 的回退路径与脚本描述并非同一路径 |
| F7 | 实例 16 的 pid 文件记录 `PID=27993`、`LAUNCHER_PID=26267`,审计时两者均不存在;`logs/serve_dsv41_tp4_h100.log` 是指向 `…_201810.log` 的符号链接 | `cat logs/serve_dsv41_tp4_h100.pid`;`ps` |
| F8 | 实例 5 相对实例 4 的两处变化**未进入头部变更清单**:① `LK_GPU_PREFILL_MIN_TOKENS 1024→1`;② E2 补丁(`_eager_decode_prewarm` 作用域)当时已装在 site-packages。实例 5 的头部只有一行 `[ENV-PATCH] prepare.py…`;`[ENV-PATCH-2]` 首次出现在实例 6 的头部 | ① `diff effective.…142033.conf effective.…150833.conf` 的唯一实质差异行是 `export LK_GPU_PREFILL_MIN_TOKENS=1024 → =1`;② `patches/…fused_moe_triton__layer.py.patched` 的 md5 `d3dc47e3…` 与当前 site-packages 同文件 md5 **完全相同** ⇒ 该补丁自 15:08:17 起持续在装,实例 5(15:08:48)必然带着它启动 |
| F9 | 实例 6 的头部把 E2 补丁记为 `[ENV-PATCH-2]`,与 F8 的时间差为 23 分钟(≈1 个实例),即"安装时点"与"声明时点"不同步 | 实例 5/6 头部原文对比 |

---

## 5. 本轮未取得证据的事项(不计入结论)

1. 实例 5 的非法访存(instance-level IMA)未在日志中留下 Python 层栈顶;该轮的具体内核与参数不可定位。
2. 实例 8、16 的"被中断"由谁发起、以何信号发起:日志无 SIGTERM/SIGINT 记录,进程与 launcher 均已回收。
3. `docs/index-producer-trace.md:229` 所称"`req_to_token` 确有 -1 的行"的写入点,本轮未在写者集合中定位。
4. `docs/flashmla-sm90-index-contract.md:403-409` 中"参数偏移 → 结构体字段名"的映射,本轮只复算了偏移计数,未定位 `SparseAttnDecodeParams` 定义以核对字段名。
5. `/tmp/.err_11777` 的产生者。
6. 各轮 SEGV 的物理根因(内存错误本体):两份静态文档的最终结论都指向"索引合法 + 二进制与源码一致",即静态手段未能定位;本轮同样未定位。

---

### 附:本报告全部数字的可复算输入

| 数字 | 复算方式 |
| --- | --- |
| 16 实例 / 存活时长 / 卡·小时 | `logs/serve_dsv41_tp4_h100_20261022_*.log` 的头部 `时间` 与全文件时间戳极值(脚本 `/data3/shared/wangyf/llm-serving/audit_scratch/dur.py`) |
| 阶段标志、fatal 原文 | 同上日志逐行 grep(脚本 `.../parse_logs.py`、`.../fatal.py`,输出 `.../parsed.json`、`.../fatal.txt`、`.../durations.txt`) |
| 18.75 GiB / 16.88 GiB | 384×1280×5120×8 B = 20 132 659 200 B = 18.75 GiB;两数均取自 OOM 原文 |
| 57/57/57 与 `sm_90`=0 | `grep -oE "sm_[0-9]+[af]?" logs/bincheck/01_list_elf.txt \| sort \| uniq -c` |
| `num_blocks`=0、指令计数、偏移直方图 | `grep -oF` / `grep -c` on `logs/bincheck/03_sass_13.txt` |
| 6 个稀疏 decode cubin 与字节数 | `logs/bincheck/xelf/*.sm_90a.cubin` 的 `stat -c%s` 与 `cuobjdump -symbols/-sass` |
