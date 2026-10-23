# 核查报告：`LK_VERIFY_RESIDENT=1` 下的 CPU MoE 路径 与 dspark 的 Markov/TP 同步

核查人：子代理（纯只读，未改任何文件、未重启服务、未使用 GPU）
核查日期：2026-10-23
核查对象（**唯一运行代码**）：

- sglang（安装版，含 .py 源码）：`/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/`
- lk_moe 2.7.1（**闭源** wheel：`METADATA:5 License: Proprietary`、`:9-10 License-File: LICENSE / THIRD_PARTY_LICENSES`）：`/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/lk_moe/`

证据分档（全文每一处结论都标注其中一档）：

- **【已证实-代码】**：安装版 `.py` 源码行，或安装版 wheel 内随包的 `METADATA`/pybind docstring，可直接引用并复核。
- **【已证实-推导】**：由已证实的代码语义 + 算术/一致性必然性推出（推导前提会写明）。
- **【推测】**：只有字符串/符号名/命名语义作为线索，闭源内部无法确证。
- **【需探针】**：只能实测，代码里读不出数值。

本次核对所用的启动配置（来自 `docs/run-dspark-20261023_0850.md:75`，与任务背景一致）：
`--tensor-parallel-size 4`、`--max-running-requests 16`、`--cuda-graph-max-bs-decode 16`（**解码 CUDA 图开启**）、
`--speculative-algorithm DSPARK --speculative-dspark-block-size 5 --speculative-attention-mode decode`；
env：`LK_MOE_HYBRID_ENABLED=1`、`LK_GPU_RESIDENT_MOE_LAYERS="0-19"`、`LK_POOL_SLOTS=128`、`LK_VERIFY_RESIDENT=1`。

模型几何（`/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017/config.json` → `text_config`）：
`num_hidden_layers=40`、`hidden_size=5120`、`moe_intermediate_size=2304`、`n_routed_experts=384`、
`num_experts_per_tok=6`（= MoE `top_k`）、`dspark_block_size=5`、`dspark_markov_rank=256`、`dspark_num_experts_per_tok=3`。【已证实-代码（配置文件）】

---

## 0. 先给两条"计数前提"（后面两节都用到）

**(0-a) 池层数量 = 20 层/rank。【已证实-代码】**
`is_lk_moe_gpu_resident_layer()`（`sglang/srt/utils/lk_moe_env.py:218-247`）把 `LK_GPU_RESIDENT_MOE_LAYERS` 解析成层号集合，
`is_lk_pool_layer()`（同文件 `:283-288`）对 `model_type=="main"` 且**非**驻留层返回池成员；
`self.pool_decode = is_lk_pool_decode_enabled() and self.is_pool_layer`（`sglang/srt/layers/moe/fused_moe_triton/layer.py:602`）。
40 层 - 驻留 0..19 = **层 20..39，共 20 个池层**。实测日志侧一致：`docs/mem-delta-ep1-vs-ep4.md:165` 记 `[CPU]/[GPU]` 各 80 行 = 每 rank 20+20。

**(0-b) 枚举的"每步"= 一次 decode 前向的 verify 步。** 单流 dspark 时该前向的 MoE 行数 `bs = 请求数 × verify_num_draft_tokens = 1 × 6 = 6` 行。【已证实-代码（行数公式，见 1.1）】

---

# 核查项 1：`LK_VERIFY_RESIDENT=1` 是否跳过了"后 20 层的 native CPU MoE"

## 1.1 `lk_moe.cpu_decode(...)` 是否无条件调用？参数各是什么？`covered` 只 gate 了什么？

**结论：无条件调用。【已证实-代码】**

`_cpu_decode()` 全文 `.../sglang/srt/layers/moe/fused_moe_triton/layer.py:2589-2635`：

```python
2589  def _cpu_decode(self, hidden_states, topk_weights, topk_ids):
2590      bs = hidden_states.size(0)
2592      draft_tokens = self.speculative_num_draft_tokens or 1
2593      mult = draft_tokens if draft_tokens > 1 else 1
2594      rows_per_token = mult if (mult > 1 and bs % mult == 0) else 1
2595      real_tokens = bs // rows_per_token
2597      covered = (self._pool_layer_info is not None
2598               and real_tokens <= FusedMoE._pool_decode_cap)
2600      if covered:
2601          self.lk_moe.pool_gpu_report(True)
2602          self._pool_decode_launch(hidden_states, topk_ids, topk_weights,
2603                                real_tokens=real_tokens,
2604                                rows_per_token=rows_per_token)
2605      stream_ptr = torch.cuda.current_stream().cuda_stream
2606      self.lk_moe.cpu_decode(                  # ← 不在任何 if 内，无条件执行
2607          stream_ptr, bs, self.top_k,
2610          hidden_states.data_ptr(), topk_ids.data_ptr(), topk_weights.data_ptr(),
2613          FusedMoE.output_gpu.data_ptr(),
2614          real_tokens=real_tokens, rows_per_token=rows_per_token)
2618      if covered:
2619          torch.cuda.current_stream().wait_stream(FusedMoE.pool_stream)
2620          if rows_per_token == 1:                       # 普通 decode（无 draft 行）
2621              FusedMoE.output_gpu[:real_tokens].add_(FusedMoE.pool_gout[:real_tokens])
2623          elif FusedMoE._pool_verify_rows_mult > 1:     # dspark verify（本配置走这条）
2624              n_rows = real_tokens * rows_per_token
2625              FusedMoE.output_gpu[:n_rows].add_(FusedMoE.pool_gout[:n_rows])
2632      output = FusedMoE.output_gpu[:hidden_states.size(0)]
2635      return output.to(hidden_states.dtype)             # 每层每步一次 fp32→bf16 拷贝
```

- `covered` **只** gate 三件事：`pool_gpu_report(True)`、`_pool_decode_launch(...)`（GPU 半）、以及 `wait_stream + add_`（合并）。**它没有 gate `cpu_decode`。**【已证实-代码】

**参数含义**（真实签名取自安装版 `.so` 内 pybind docstring，用
`CUDA_VISIBLE_DEVICES="" .../env/bin/python -c "import lk_moe;print(lk_moe.MOE_FP8.cpu_decode.__doc__)"` 读出；
`MOE_FP8/NVFP4/MXFP4/BF16/WNA16` 六个类的签名一致）：

```
cpu_decode(self, stream, batch_size, k, input_gpu, expert_ids_gpu, weights_gpu, output_gpu,
           real_tokens=-1, rows_per_token=1) -> None            # 【已证实-代码（docstring）】
```

| 实参 | 形参 | 含义 | 出处 |
|---|---|---|---|
| `stream_ptr` | `stream` | 调用方当前 CUDA stream 的裸指针（GPU 侧 kernel 排到这条流） | layer.py:2605-2607 |
| `bs` | `batch_size` | 本层 MoE 输入的行数 = **请求数 × verify_num_draft_tokens**（dspark 单流 = 6） | layer.py:2590 |
| `self.top_k` | `k` | 路由 top-k（本模型 = 6；`cfg.top_k = self.top_k`） | layer.py:2609, 2091 |
| `hidden_states.data_ptr()` | `input_gpu` | 隐状态（设备指针） | layer.py:2610 |
| `topk_ids.data_ptr()` | `expert_ids_gpu` | 每行 k 个专家 id（设备指针） | layer.py:2611 |
| `topk_weights.data_ptr()` | `weights_gpu` | 每行 k 个权重（设备指针） | layer.py:2612 |
| `FusedMoE.output_gpu.data_ptr()` | `output_gpu` | **CPU 半写入的部分和缓冲**；GPU 半随后 `add_` 到同一缓冲 | layer.py:2613, 2618-2625 |
| `real_tokens` | `real_tokens` | 请求数（= 行数/每请求行数），池侧 reduce 的"token 数" | layer.py:2595, 2514-2530 |
| `rows_per_token` | `rows_per_token` | 每请求的行数（dspark = 6） | layer.py:2594 |

`speculative_num_draft_tokens` 来源：`layer.py:576 self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens`；
`output_gpu` 是类属性、`float32`、形状 `(max_num_seqs, hidden_size)`，`max_num_seqs = _get_max_num_seqs()`（`layer.py:581, 2369-2381, 2388-2393`）= `max_running_requests × (1+speculative_num_draft_tokens)` 或 `× (1+dspark_block_size)` → 本部署（`max_running_requests=16`）为 **96 或 112 行**（取哪条分支取决于 `server_args.speculative_num_draft_tokens` 是否为空）。`_pool_decode_cap = max(1, self.max_running_requests) = 16`（`layer.py:2450`）→ `covered` 的实际判据是"**请求数 ≤ 16**"。【已证实-代码】

VR 在 Python 侧的两处落点：
`layer.py:2451-2458`
```python
2451  # LK_VERIFY_RESIDENT: verify batches cover the DRAFT rows' resident
2452  # experts on the GPU too (row-major slot space of bs*k), so the CPU
2453  # side computes only misses on every row, not just real rows.
2454  rpt0 = self.speculative_num_draft_tokens or 1
2455  mult = rpt0 if (get_verify_resident() and rpt0 > 1) else 1
2457  FusedMoE._pool_verify_rows_mult = max(FusedMoE._pool_verify_rows_mult, mult)
```
`lk_moe_env.py:143-146`
```python
143  def get_verify_resident() -> int:
144      # VR: draft-row pool-resident experts computed GPU-side; has an acceptance
145      # fidelity tradeoff -> python-owned cfg value (LK_* env is the script UI).
146      return get_int_env_var("LK_VERIFY_RESIDENT", 0)
```
→ VR=1 的作用是**把 GPU 热的覆盖范围从"真实行"扩到"草稿行"**，即 GPU 半覆盖 `bs×k` 整个行主序 slot 空间（`layer.py:2526-2531` 注释），而不是让 CPU 半免做。【已证实-代码】

## 1.2 CPU 侧实际算什么：只是"未命中专家"，还是所有行的路由/合并/累加也在 CPU？

**结论（分三段，档位不同）**

1. **路由（topk）不在 CPU。**【已证实-代码】`cpu_decode` 收的是 `topk_ids/topk_weights` 的**设备指针**（`layer.py:2611-2612`），topk 由上游模型 router/`topk_output` 在 GPU 上算完（`run_moe_core(dispatch_output)` 的入参即含 `dispatch_output.topk_output.topk_ids/weights`，`layer.py:1725-1743`）。
2. **CPU 半算"非驻留（miss）专家"，GPU 半算"池驻留专家"，两边各出一份部分和，最后相加。**【已证实-代码 + 推导】
   - GPU 半的 kernel 明确只吃驻留 slot：`reduce_down` 传 `RESIDENT_ONLY=1`（`layer.py:2581-2584`），slot 表 `-1` 表示非驻留（`layer.py:2466, 2549-2551`）。
   - CPU 半写入的 `output_gpu` 与 GPU 半的 `pool_gout` 由 `add_` 合并（`layer.py:2618-2625`）→ 两者必须是**不相交的部分和**，否则重复计数。
   - 随包 METADATA 的权威描述：`.../lk_moe-2.7.1.dist-info/METADATA:25`
     "A background DMA pump keeps hot experts resident in VRAM … **Pool-resident experts are computed on the GPU while the rest stay on the CPU** … the split is decided by one shared snapshot word, **so every expert is computed exactly once**."
   - 闭源侧 pybind docstring：`pool_gpu_report` = "python acks whether the GPU side actually launches for this layer (**CPU-side skip is armed only after a True; default False = CPU-only**)"；Python 在 `covered` 分支里每步 `pool_gpu_report(True)`（`layer.py:2601`），初始化时已先 `pool_gpu_report(False)`（`layer.py:2418`）再 `True`（`layer.py:2487`）。
   → "CPU-side skip" 指**跳过 GPU 已覆盖的那部分专家**，不是跳过整个 MoE。
   - 一致性计数器佐证：`pool_stats()` 暴露 `cover_miss` / `cover_dup`（`.so` 字符串常量，`_lk_moe_C_avx512_amx.so` `strings` 行 31123-31124；`pool_stats.__doc__` = "async expert pool counters (hits/misses/pumps/link bytes)"），`[pool]` 行里打印为 `cmiss=` / `cdup=`（同文件格式串，`strings` 行 31149）。这组计数显然就是"每个专家恰好一次"的守门计数。【**推测**（名字与打印位推断其语义）/ 已证实-代码（字段确实存在）】
3. **CPU 是否对"所有行"都做累加/写回？** —— **推导上必须写满 [0,bs) 每一行，但是否逐行遍历 k 个 slot 无法从安装版确证。**【已证实-推导 / 部分推测】
   - 推导前提：`FusedMoE.output_gpu` 是**跨步复用的持久缓冲**（`layer.py:2388-2393` 只创建一次），而 `add_` 每步在其上做 `+=`（`2620-2625`）。若某行 CPU 这步不写，上一 step 的残值会被 GPU 半 `add_` 到本步结果里 → **CPU 半每步必须对 [0,bs) 的每一行写出一个部分和**（哪怕某行零 miss，也要写 0）。【已证实-推导】
   - 路由之外，CPU 半对"它负责的那些专家"必然要做加权累加（combine）：这是产出 `output_gpu` 部分和的算术必要条件；但"逐 slot 遍历并跳过驻留专家"还是"只取 miss 列表"，安装版看不到（闭源，`_lk_moe_C_avx512_amx.so`）。**【推测】**
   - 有一条来自闭源侧的旁证：`LK_VERIFY_SKIP_MISS` = "**Skip verification for rows whose experts missed residency**"（`METADATA:93`；Python 侧注释 "verify rows drop CPU-side misses (**distribution declaration, not bitwise**)"，`lk_moe_env.py:149-151`）。只有 CPU 半按行/按 miss 决策，这个开关才有意义。本配置**未设**该 env → 取默认 0 → **没有丢 miss**。【已证实-代码】
4. **明确边界**：`.so` 里的 CPU 计算核心是闭源的（`_lk_moe_C_avx512_amx.so`，主机侧无内部符号：`nm --defined-only` 返回 0 条，`nm -DC` 只剩导入/导出表）。因此"CPU 侧内部是否真的只算 miss 的 GEMV"在**安装版内无法逐指令确证**，只能由上面 2/3 的接口语义+计数器+随包文档判定。**【闭源无法确证 → 标注为推测级】**

## 1.3 每步的同步与合并开销

**（a）`wait_stream` 次数：每个池层每次前向各 1 次 → 20 次/步。【已证实-代码】**
- 池→主：`torch.cuda.current_stream().wait_stream(FusedMoE.pool_stream)`（`layer.py:2619`），在 `if covered:` 内。
- 主→池：`side_stream.wait_stream(torch.cuda.current_stream())`（`layer.py:2534`，进 `_pool_decode_launch`）。
- 另有 1 次跨流进度记录：`self.lk_moe.pool_record_aux(FusedMoE.pool_stream.cuda_stream)`（`layer.py:2587`，docstring "mark decode gemv stream progress for worker-copy ordering"）。
- 20 个池层 × 每层一次 = **20 次池→主 wait、20 次主→池 wait、20 次 record/步**（前提：这 20 层都 `covered`；若有层在 `_init_pool_decode` 里被判 `pool_decode_on but lk_moe declined`（`layer.py:2425-2429`）则 `_pool_layer_info` 保持 `None`、该层不 covered）。
- **重放期语义（重要）**：`run_moe_core` 只在 `torch.cuda.is_current_stream_capturing() or get_is_capture_mode()` 时走 `_cpu_decode`（`layer.py:1738-1743`），即 **Python 侧只在捕获期执行**；图重放期这些 `wait_stream/add_` 是**已捕获的图节点**，逐层执行（仍是 20 次/步）。重放期谁触发 CPU 半计算不在 Python 侧可见：`LK_HF_MBOX`（默认 1）= "**Mailbox early-start of the host-side decode** … saves the ~50 µs graph host-node wake-up latency"（`METADATA:94`；Python 只透传 `cfg.hf_mbox`，`layer.py:2084`），且 `.so` 内含**设备函数** `host_func_mbox::doorbell(const unsigned short*, const int*, const float*, int,int,int,int,int,int, unsigned long long*, MailboxControl*)`（`_lk_moe_C_avx512_amx.so` 符号/`strings` 行 31478）+ 字符串 `[mbox] more than %d MOE instances -> mailbox OFF for the rest`（同文件 31479）。**【推测】**：具体挂钩方式（host node vs 设备门铃+主机线程+设备自旋确认）闭源不可见；可确证的只有"存在该机制且默认开启"。

**（b）每步每层的 GPU 派发/合并量（可静态计数）。【已证实-代码（计数）+ 需探针（耗时）】**
`covered` 时每池层每步至少 10 次派发：

| 项 | 位置 | 规模（单流 1 请求、bs=6、H=5120、k=6） |
|---|---|---|
| `gather_rows`（仅 rows_per_token>1 且非 all_rows 时；本配置走 all_rows 分支不执行） | layer.py:2541-2545 | — |
| `slot_table_fill` | 2549-2551 | 1 个 program |
| `gout_part.zero_()` | 2552 | 96×5120 fp32（`cap*mult` 行，`layer.py:2464-2465`） |
| `moe_gateup` + `reduce_gate` | 2553-2568 | n_slots = bs×k = **36** 个 slot × I=2304 |
| `moe_down` + `reduce_down` | 2569-2584 | 36 slot → H=5120 |
| `pool_record_aux`（事件记录） | 2587 | 1 |
| `add_` | 2619-2625 | **n_rows×H = 6×5120 = 30 720 个 float32 = 122 880 B** |
| `output.to(hidden_states.dtype)` | 2632-2635 | 同 6×5120 fp32→bf16 拷贝 |

→ 单流一步：合并类（`add_`）流量 ≈ 0.12 MB/层 × 20 = **2.4 MB/步**；若打满 16 请求（bs=96 行）则 ≈ 1.97 MB/层 → **39 MB/步**。
（`dtype` 事实：`FusedMoE.output_gpu`/`pool_gout` 均为 `torch.float32`，`layer.py:2388-2393`、`2464-2465`。）**【已证实-代码】**

## 1.4 `copyiss ≈ 1.0–1.3 ms` 测的是哪一段？

**产出位置：不在任何 Python 文件里，而在闭源 `.so` 的 `[pool]` 统计行里。【已证实-代码】**

- 格式串（`.so` 内，`strings -a lk_moe/_lk_moe_C_avx512_amx.so` 第 31149 行）：
  `[pool] dstep=%ld win hit=%.1f%% nr=%.1f%% psl=%.2f link=%.2fGB/s gather=%.0fus copyiss=%.1fus que=%ld hf_hook=%.0fus hf_dec=%.0fus hf_hook_max=%.0fus hf_dec_max=%.0fus | cum hit=%.1f%% h=%ld m=%ld p=%ld B=%ld resident=%ld nr=%ld vf=%ld oom=%ld stalls=%ld sk=%ld stale=%ld tag=%ld pubbad=%ld publag=%ld cmiss=%ld cdup=%ld`
  同一二进制中**没有**第二个 `copyiss` 出现（5 个架构变体 `.so` 全量 `grep -l copyiss` 只命中这 5 个 `.so`，`sglang/` 下 **0 命中**）。→ **该字段的计时点、归一化方式、定义全部在闭源侧。**
- 使它出现的唯一 Python 通路：`cfg.pool_log = get_pool_log()`（`layer.py:2107`）← `LK_POOL_LOG`，注释 "dstep interval of the C++ `[pool]` stats line (0 = silent)"（`lk_moe_env.py:159-161`）；随包 METADATA 亦记为 "Diagnostic: **windowed** `[pool]` stats line every N decode steps"（`METADATA:76`）。
- 同一行里可确证语义的字段只有：`h/m/p` ↔ `pool_stats()` 的 `hits/misses/pumps`；`vf/nr/stale/tag/pubbad/publag/cmiss/cdup` ↔ `.so` 中 `verify_fail/not_ready/stale_slot/tag_bad/publish_bad/publish_lag/cover_miss/cover_dup`（`strings` 行 31112-31125）。**`copyiss` / `psl` / `gather` / `hf_hook` / `hf_dec` 不在 `pool_stats()` 的键里，也无任何文档定义**。

**因此，任务里的"`copyiss` 测的是哪一段"无法用安装版代码判定。**
可判定的只有"它**不可能**是哪个"：

- 它是 **pool/pump 子系统**的窗口统计（格式串与 `[pool] H2D staging ON`(31147)、`[pump] ON pool=… slots=… window=… inflight=… log=…`(31151)、`pool_stats` 同处一块字符串区），即统计的是**泵/拷贝/发布侧**的量，而非调用线程里 `cpu_decode` 的 CPU 计算时长。
- 依据：`[pool]` 行中没有任何字段来自 CPU MoE 计算路径（`cpu_decode` 无任何向 Python 暴露的计时接口；主机侧 Python 只透传 cfg）。`cpu_decode` 的耗时若不另设探针，**不会**出现在这一行里。**【已证实-代码（字段清单）/ 推测（"不代表 CPU MoE 总时长"这一否定结论）】**
- 另需注意窗口归一化未知：文档里同一运行的连续多行 `copyiss` 可为 933 µs 与 8828 µs（`docs/run-dspark-20261023_0850.md:460-463`），而 `hf_dec`/`hf_dec_max` 成对出现说明这一行里既有均值也有峰值字段；`copyiss` 无 `_max` 姊妹字段，**"是均值还是窗口累加"不能从代码确定** → 直接拿 "1.0–1.3 ms" 去减总耗时会引入口径错误。**【需探针】**

---

# 核查项 2：TP4 下 5 次顺序 Markov 采样 / 每次 `DSPARK_DRAFT` broadcast

## 2.1 Markov 采样循环的定位、次数、每步迭代内的 GPU 操作

**循环本体：【已证实-代码】**`.../sglang/srt/speculative/dspark_components/dspark_draft.py:239-280`

```python
239  def draft_markov_block(*, markov_head, base_logits, anchor_tokens, draft_hidden,
250                         step_temperatures, noise, tp_sync, logits_out=None, greedy_draft=False):
253      bs, gamma = base_logits.shape[:2]
260      state = markov_head.init_state(draft_hidden)
261      prev_tokens = anchor_tokens.long()
262      draft_tokens = []
263      for step in range(gamma):                 # ← 循环次数 = gamma
264          prev_tokens, state = _draft_step(..., base_logits[:, step], ..., step=step, ...)
275          prev_tokens = tp_sync.sync(SpecTpSyncSite.DSPARK_DRAFT, prev_tokens)   # ← 循环体内
276          draft_tokens.append(prev_tokens)
279      noise.offset.add_(1)
280      return torch.stack(draft_tokens, dim=1), logits_out
```

- **次数 = `gamma`，且 `gamma = speculative_num_draft_tokens - 1`**：
  `dspark_config.py:51-58`（`dspark_gamma_from_num_draft_tokens(n) = n - 1`）与 `:130-134`（`DSparkRuntimeConfig(gamma=gamma, verify_num_draft_tokens=gamma+1, …)`）。
  本配置未给 `--speculative-num-draft-tokens`，走 `dspark_block_size=5` → **gamma=5、verify_num_draft_tokens=6**。**5 次，与 block_size/gamma 相等。【已证实-代码】**
- **顺序性**：第 `step+1` 次迭代的输入 `prev_tokens` 正是第 `step` 次的输出（`:264/:275/:276`），数据依赖 → 无法并行。**【已证实-代码】**

**每次迭代内的 GPU 操作**（`_draft_step`，`dspark_draft.py:187-236`）：

1. `markov_head.step_bias(prev_tokens, …)`（`:210-212`）→ 本模型用 `DSparkV4MarkovHead`（`srt/models/deepseek_v4_dspark.py:413+`）：
   - embedding gather：`get_prev_embeddings()` = `markov_w1(prev_tokens)`（`deepseek_v4_dspark.py:493-494`，`VocabParallelEmbedding`）；
   - **本地 GEMV**：`F.linear(latent, markov_w2.weight[org_vocab_start:org_vocab_end])`（`deepseek_v4_dspark.py:507-525`）——注意是 **vocab 分片后的本地宽度**。
2. `MarkovSampleStep.execute(...)`（`dspark_draft.py:223-235` → `kernels/ops/speculative/dspark/markov_sample.py:302-351`）：
   - `_launch_partial`（triton，板形 `(rows, parts)`）→ 加 bias、按 step dtype 取整、可选存 step logits、Gumbel-max（Philox 10 轮，`markov_sample.py:21-24`）；
   - 因为本模型 `vocab_shard` 非空（见 2.2），走 `markov_sample.py:323-329` 分支：**再 1 次 `_launch_partial` + 1 次 `gather_logits(local)`（TP all-gather）**；
   - `_markov_sample_finish_kernel[(rows,)]`（`:342-350`）跨分片/跨 rank 取全局 argmax。
3. `tp_sync.sync(SpecTpSyncSite.DSPARK_DRAFT, prev_tokens)`（`:275`）→ **1 次 broadcast**（见 2.2）。
4. 循环外：`torch.stack`（`:278`，1 次拷贝）+ `noise.offset.add_(1)`（`:279`）。

**本配置确实走这条采样路径（不是被折成 greedy 快路径）。【已证实-代码】**
- `_draft_step` 的融合贪心快路径要求 `logits_out is None and len(prev_tokens) == 1`（`:199-209`，`MarkovGreedyStep`，见 `kernels/…/dspark_draft_model.py:16-27`）；而 `greedy_draft` 由全局 env 决定：`SGLANG_DSPARK_USE_GREEDY_DRAFT = EnvBool(False)`（`environ.py:563`），`self._greedy_draft = envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.get()`（`dspark_worker_v2.py:323`）→ **默认 False（采样）**，`DsparkDraftSampler.__call__` 于是传 `logits_out=self._step_logits(...)`（`dspark_draft_sampler.py:115-118`）→ 快路径不成立，5 次都走 `MarkovSampleStep`。
- 另外 `_draft_step` 里 logits 保留宽度 `step_logits_width()` 在 sharded 时是**整个 vocab**（`dspark_draft.py:182-184`）。

## 2.2 是否存在 `DSPARK_DRAFT` 集合通信？每次 verify 步的通信次数

**存在，且是 TP broadcast。【已证实-代码】**

- `spec_tp_sync.py:28 DSPARK_DRAFT = 17`（枚举成员）；`:66-81` 把旧的 `dspark-draft-*`/`2..6` 全部 **retired 并归并到 `DSPARK_DRAFT`**（注释："Retired handles of the per-path DSpark draft sites, merged into DSPARK_DRAFT."）。
- `SpecTpSync.sync()`（`spec_tp_sync.py:135-138`）：`if site in self._sites: self._tp_group.broadcast(values, src=0)`；`_sites` 在 `:124-125` 由 `SGLANG_SPEC_TP_SYNC` 解析，**`tp_group.world_size > 1` 才非空**。
  默认 `SGLANG_SPEC_TP_SYNC = EnvStr("all")`（`environ.py:1499`）→ TP4 下 **所有 site 开启**。
- `_tp_group = SpecTpSync(get_dp_tp_group())`（`dspark_worker_v2.py:237`）；`GroupCoordinator.broadcast`（`srt/distributed/parallel_state.py:1560-1577`）优先走 PyNccl（`pynccl_comm.broadcast`），否则回退 `torch.distributed.broadcast`。
- 通信量级：`prev_tokens = anchor_tokens.long()`（`dspark_draft.py:261`）→ 形状 **[bs] 的 int64**；单流 bs=1 → **8 字节的 broadcast**，纯时延主导。**【已证实-代码】**

**每次 verify 步的通信次数（推导式）**

设 `W = (world_size > 1 且 site 开启)`（本部署 W = 真），`G = gamma = 5`：

| 站点 | 次数/步 | 条件与出处 |
|---|---|---|
| `DSPARK_DRAFT` | **G = 5** | `dspark_draft.py:275`，在 `for step in range(gamma)` 内；折叠路径下由 draft 图捕获，eager 路径下直发 |
| 草稿 Markov 的 vocab all-gather（**不是 broadcast**：`group.all_gather(dim=0)` 或 NVLink push/pull） | **G = 5** | `markov_sample.py:323-329`（`gather_logits`）+ `vocab_gather.py:41-51`（`NcclVocabGather.gather_stacked → group.all_gather(local, dim=0)`）；分片由 `SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD = EnvBool(True)`（`environ.py:568`）与 `configure_tp_shard()`（`deepseek_v4_dspark.py:437-491`，调用点 `:826/:838`）建立。**实测日志确认本部署走了分片**：`docs/run-dspark-20261023_0850.md:128` = `DSpark markov_w2 vocab gather: NcclVocabGather`（即 NCCL ring，非 NVLink 多播面）。 |
| accept 合并 `build_accept_outs` | **3**（`correct_len`、`bonus`、`cap_trim_lens`） | `dspark_verify.py:910-912` 三连 `tp_sync.sync(sync_site, …)`；site = `DSPARK_ACCEPT_GRAPH`（图内，`:802`）或 `DSPARK_ACCEPT_GREEDY/_SAMPLE`（eager，`:188-202`） |
| `DSPARK_PLAN` | **0 或 1** | `dspark_planner.py:539-544`；前置 `_schedule_verify_lens` 在 `self._budget_planner is None or confidence is None or budget is None` 时 `return None`（`:537-538`）→ 不产生通信 |
| `DSPARK_TARGET` | **decode 步 0 次** | 全仓唯一调用点在 `_forward_prefill()`（`dspark_worker_v2.py:664`），不在 `_forward_decode()`（`:804-1019`）路径上 |

**合计（本配置，单流）：每 verify 步 ≈ 5 + 3 = 8 次 TP broadcast（+ 可选 1 次 PLAN = 9），另加 5 次 vocab all-gather。**
**频率**：这些是**每 verify 步**一次（不是每层一次）——`draft_markov_block` 在 draft 模型前向的尾部跑一次（每步一次提案），`build_accept_outs` 每步调一次；`DSPARK_DRAFT` 在**同一步内**出现 5 次（gamma 次）。全仓 `SpecTpSyncSite.*` 调用点清单见 `dspark_draft.py:275`、`dspark_planner.py:544`、`dspark_verify.py:189/191/802`、`dspark_worker_v2.py:498/664`。【已证实-代码】

## 2.3 这些同步会进 CUDA 图吗？图内 broadcast 是否合法？

**（a）进图：默认会。【已证实-代码】**
- `SGLANG_DSPARK_FOLDED_PROPOSAL = EnvBool(True)`（`environ.py:564`）；`init_cuda_graphs()` 在折叠开启时构建采样器并挂成 draft 图的尾钩（`dspark_worker_v2.py:519-525`：`self._draft_sampler = self._maybe_build_draft_sampler()` → `capture_tail_hooks.append(make_draft_sampler_capture_hook(...))`）。
- 尾钩在**捕获期**真正调用采样器：`draft_worker_common.py:148-159`（`draft_sampler(out.hidden_states, forward_batch.input_ids)`，捕获循环见 `srt/model_executor/runner/decode_cuda_graph_runner.py:1333-1334`）。
- 采样器本身就是"eager 路径 markov block 作为 draft cuda graph 的尾巴"：`dspark_draft_sampler.py:21-22`（类 docstring，原话）→ `__call__` 里调用 `draft_markov_block(...)`（`:107-119`），即 **2.1/2.2 的 5 次 broadcast + 5 次 all-gather 都发生在捕获内**。
- target verify 侧的汇总只在别处（`layer.py:2589` 与本核查无关）；accept 的 3 次 broadcast 在图内由 `DsparkVerifyEpilogue._accept` 完成（`dspark_verify.py:793-805`，并在 `:780` 注释 "Before the TP broadcast in build_accept_outs: every rank commits rank 0's draw."）。
- 重放期 Python 不执行，故这些广播**以图节点形式每步执行**；"5 次/步"是图内容决定的，不是运行期 Python 决定。**【已证实-代码/推导】**

**（b）合法性：本构建显式在捕获上下文里启用 PyNccl，且把 NCCL 调用当作图可捕获。【已证实-代码】**
- 解码图捕获走 `graph_capture(stream=…)`：`decode_cuda_graph_runner.py:1168-1182`（`from sglang.srt.distributed.parallel_state import graph_capture`，`:42`）；
  `parallel_state.py:631-697` 的 `graph_capture()` 在 `:682` 做 `pynccl_comm.change_state(enable=True)`（注释块 `:660-676` 给出 "图内启用 PyNccl / 图外启用 torch.distributed" 的选择表），`change_state` 实现见 `device_communicators/pynccl.py:409-421`。
- `PyNcclCommunicator` 默认 `self.disabled = True`，注释明确 "to use it, use under `with obj.change_state(enable=True)`, usually when we are using CUDA graph"（`pynccl.py:134-137`）；
  `broadcast()` 直接发 `ncclBroadcast(sendbuff, recvbuff, numel, dtype, src, comm, cudaStream_t(stream.cuda_stream))`（`pynccl.py:371-395`）——单次 NCCL 调用、跑在当前（捕获）流上。
- 同类机制在仓库里被显式称为可捕获：`pynccl.py:325` "…single NCCL operation, which is **CUDA-graph-capturable**"；`vocab_gather.py:55` 有 "# CUDA-graph capture." 的实现注释；NVLink 分支甚至专门处理捕获下的对称内存别名（`vocab_gather.py:75-152`）。
- **因此：图内 broadcast/all-gather 在本构建中是"被设计为合法"的（没有看到"被迫走 eager"的分支）。**【已证实-代码】
- 边界（无法判定项）：安装版里**看不到** NCCL 通讯器创建时的 graph 使用模式（`pynccl.py:112-124`：只有对称内存开启时才设 `ncclConfig_t.graphUsageMode = 1`；本部署是否走该分支、以及 NCCL 版本对 8 字节 broadcast 的图内捕获细节，代码里读不出）→ **"每个节点是否真的零同步、有无隐式 device 同步"记为无法判定。**【需探针】

## 2.4 与无 dspark 的对照：这些路径是否完全不执行？

**是，全部不执行。**【已证实-代码】

- worker 选择：`SpeculativeAlgorithm.create_worker()`（`srt/speculative/spec_info.py:302`）先 `assert not self.is_none()`（`:307-309`），只有 `is_dspark()`（`:323`）才返回 `DSparkWorkerV2`（`:328`）→ 未开 dspark 时**不构造 spec worker**，`DSparkWorkerV2` 整条链路不存在。
- 于是被整体跳过的调用（按步序）：
  1. `DSparkWorkerV2.forward_batch_generation` → `_forward_decode`（`dspark_worker_v2.py:576-609, 804+`）；
  2. `DraftBlockProposer.propose` / `_run_forward`（`dspark_draft.py:369-452, 480-589`）——包括**草稿模型前向与草稿 CUDA 图**；
  3. `DsparkDraftSampler.__call__`（`dspark_draft_sampler.py:92-128`，仅在 `SGLANG_DSPARK_FOLDED_PROPOSAL` 且可入图时）；其 eager 孪生 `sample_draft_block`（`dspark_draft.py:283-316`）；
  4. **`draft_markov_block` 的 gamma 次循环及其内的全部 GPU op**（`dspark_draft.py:239-280`）——即 5 次 `MarkovSampleStep.execute`（含 5 次 vocab all-gather）；
  5. **5 次 `DSPARK_DRAFT` broadcast**（`dspark_draft.py:275`）+ 3 次 accept broadcast（`dspark_verify.py:910-912`）+ 可选 `DSPARK_PLAN`（`dspark_planner.py:544`）；
  6. verify 窗口与 target verify 图内的 accept 尾处理（`DsparkVerifyEpilogue._accept`，`dspark_verify.py:768-810`）；
  7. `SpecTpSync` 本身只在 spec worker 里构造（`dspark_worker_v2.py:237`）→ 无 dspark 时 `SpecTpSync.sync` 完全不被调用。
- **不算 dspark 专属、基线里也存在的部分**（避免把归因搞错）：`_cpu_decode` 与池合并仍在每步执行；只是 draft 行倍数消失——无 dspark 时 `bs = 请求数`，`mult=6`，`bs % mult != 0`（单流 bs=1）→ `rows_per_token = 1`（`layer.py:2592-2595`），合并分支变为 `output_gpu[:real_tokens].add_(pool_gout[:real_tokens])`（`:2620-2622`），每层 slot 数由 `bs×k = 36` 降为 `1×k = 6`。**20 次 `wait_stream`/`add_` 在 12.66 ms 的基线里同样存在。**【已证实-代码】

---

# 3. 结论分档汇总

| # | 结论 | 档位 |
|---|---|---|
| 1.1-a | `lk_moe.cpu_decode(...)` **无条件**调用；`covered` 只 gate `pool_gpu_report(True)`/`_pool_decode_launch`/`wait_stream+add_` | 已证实-代码（layer.py:2600-2630） |
| 1.1-b | 参数语义：`bs`=行数(=请求×6)、`k`=top_k=6、`real_tokens`=请求数、`rows_per_token`=6、`output_gpu`=CPU 半部分和缓冲 | 已证实-代码（pybind docstring + 调用点） |
| 1.1-c | `covered` ⟺ 该层池几何就绪 且 请求数 ≤ `max_running_requests`(16) | 已证实-代码（2597-2598, 2450） |
| 1.2-a | 路由不在 CPU（topk 由 GPU 侧 router 出，CPU 只收设备指针） | 已证实-代码 |
| 1.2-b | CPU 算"非驻留(未命中)专家"部分和，GPU 算驻留专家部分和，两边相加 = 每个专家恰好一次 | 已证实-代码（METADATA:25 + 2534-2584 + add_ 语义 + `cover_miss/cover_dup` 计数） |
| 1.2-c | CPU 每步必须为 [0,bs) 每行写出部分和（含零 miss 行） | 已证实-推导（持久缓冲 + `add_` 复用） |
| 1.2-d | CPU 是否逐 slot 遍历/仅走 miss 列表；miss 的 GEMV 内部实现 | **闭源无法确证**（推测级） |
| 1.3-a | 每池层每步各 1 次池→主 wait、1 次主→池 wait、1 次 event 记录；20 层 ⇒ 20/步 | 已证实-代码（2534, 2587, 2619；层数见 0-a） |
| 1.3-b | 重放期这些是图节点；CPU 半由 mailbox 机制触发（`LK_HF_MBOX=1`，设备函数 `host_func_mbox::doorbell` 存在） | 已证实-代码（存在性）/ **推测**（具体挂钩） |
| 1.3-c | `add_` 规模：n_rows×5120 fp32（单流 6 行 = 122 880 B/层；满 16 请求 = 1.97 MB/层） | 已证实-代码 |
| 1.4-a | `copyiss` 只在闭源 `.so` 的 `[pool]` 窗口行里产出，Python 侧 0 命中；其定义/归一化不可从安装版取得 | 已证实-代码（产出位置）/ **需探针**（定义） |
| 1.4-b | 该行不含任何"CPU MoE 计算时长"字段 ⇒ `copyiss` 不能当作 `cpu_decode` 整体耗时 | 已证实-代码（字段清单）/ 推测（否定结论） |
| 2.1-a | Markov 循环 `for step in range(gamma)`，gamma = block_size = 5；顺序依赖 | 已证实-代码（dspark_draft.py:263-276; dspark_config.py:51-58,130-134） |
| 2.1-b | 每步迭代 = 本地 GEMV + 2 次 partial kernel + 1 次 vocab all-gather + 1 次 finish kernel + 1 次 broadcast | 已证实-代码（markov_sample.py:302-351; deepseek_v4_dspark.py:507-525） |
| 2.1-c | 本部署走采样分支（非贪心快路径） | 已证实-代码（environ.py:563 默认 False） |
| 2.2-a | `DSPARK_DRAFT` 是 TP broadcast，每步 **5 次**（在循环体内） | 已证实-代码（dspark_draft.py:275; spec_tp_sync.py:135-138） |
| 2.2-b | 每步另有 5 次 vocab all-gather（NCCL，实测日志确认分片开启） | 已证实-代码 + 实测日志（run-dspark:128） |
| 2.2-c | accept 每次 3 次 broadcast；PLAN 0/1 次；TARGET 在 decode 0 次 | 已证实-代码（dspark_verify.py:910-912; dspark_planner.py:537-544; dspark_worker_v2.py:664 仅在 prefill） |
| 2.2-d | 合计 ≈ 8（+1）次 broadcast/verify 步，另 5 次 all-gather | 已证实-推导 |
| 2.3-a | 折叠默认开启 ⇒ 上述 Markov 循环与 5 次 broadcast/all-gather 在 draft 图**捕获内**发生 | 已证实-代码（environ.py:564; dspark_worker_v2.py:519-525; draft_worker_common.py:148-159） |
| 2.3-b | 图内集合通信被显式支持：`graph_capture()` 内 `pynccl.change_state(enable=True)`，`ncclBroadcast` 跑捕获流 | 已证实-代码（parallel_state.py:682; pynccl.py:134-137,371-395） |
| 2.3-c | 图内是否零隐藏同步/是否所有节点都真正可捕获 | **无法判定 → 需探针** |
| 2.4 | 无 dspark 时该链路整体不构造：markov/全部 broadcast/verify epilogue 都不执行；但池层 `wait_stream+add_` 仍在 | 已证实-代码（spec_info.py:302-328 等） |

---

# 4. 若要量化以下各项，需要什么探针

前提说明：**图重放期 Python 不执行**，故任何"在图内插入 Python 计时"的想法都不成立；能进图内的只有 kernel/事件节点。

| 项 | 需要的计时点或计数器 | 是否可在 CUDA 图内安全采集 |
|---|---|---|
| 每层每步 `wait_stream` 的真实 stall（20 次/步合计） | nsys/`CUPTI` 的 stream timeline：主 stream 在 `add_` 前后的空闲区间；或临时改跑 eager（关 CUDA 图）后用 `torch.cuda.Event` 夹住 2534/2619 两处 | 图内 **不建议**用 CUDA Event（capture 期 record/elapsed 语义受限）；**nsys replay 解析**是安全路径；eager 下 event 安全 |
| `add_` + `.to(bf16)` 合并开销（2.4 MB/步，单流） | 4 个 ncu metric（`dram__bytes`、`sm__cycles`）或 nsys 里按 kernel 名归并（`elementwise_kernel`/`vectorized_elementwise`） | 图内可被 nsys/ncu 采集（profiling 会打断图执行，需重放窗口）；图内不能加 host 计时 |
| `lk_moe.cpu_decode` 的 CPU 时长（含 miss 的 GEMV + 每行累加） | host 侧计时点只能落在闭源内部 → 需 lk_moe 提供（现无接口）；退而求其次：`perf record` 采样 4 个 rank 进程的宿主线程（`.so` 主机符号已 strip，只能落到 `.so` 边界）；或对比实验（`LK_POOL_SLOTS=0` 纯 CPU vs 有池） | 与图无关（host 代码），图内外都安全；但精度只到"整段 host 时间"，分不出 routing/combine/GEMV |
| CPU 半"只算 miss"是否成立（每个专家恰好一次） | 现成计数器：`lk_moe.pool_stats()` → `cover_miss`/`cover_dup`（另 `hits/misses/pumps/stalls/verify_fail/not_ready/…`），对照 `[pool]` 行的 `cmiss/cdup` | **安全**：纯主机侧读计数，不需要在图上挂任何东西（但要保证读取发生在图外、且窗口足够长） |
| 每步 `DSPARK_DRAFT` broadcast 实际次数与耗时（×5） | 次数：给 `SpecTpSync.sync` 加计数只能统计**捕获期**调用（=图内容），统计不到重放次数；重放期需 **nsys 的 NCCL kernel 时间线 / NCCL_DEBUG=INFO 计数**（或 NCCL profiler plugin） | 张量仅 8 B（单流），图内采集靠外部 profiler 安全；**图内插 host 计时代码不可行** |
| 每步 5 次 vocab all-gather 的耗时 | nsys 中 `ncclAllGather`/`AllGather` kernel（`NcclVocabGather` → `group.all_gather(dim=0)`），按 step 聚合；注意本部署实测为 NCCL 路径而非 NVLink | 同上（外部 profiler 安全，图内 host 计时不可行） |
| accept 3 次 broadcast（`correct_len/bonus/cap_trim_lens`） | 同 NCCL 时间线；若要区分"图内 accept"与"eager accept"，用 `SGLANG_DSPARK_FOLDED_PROPOSAL=0` 或关图做对照 | 同上 |
| `DSPARK_PLAN` 是否每步发生 | Python 侧计数最省事（该调用在**图外**，`dspark_planner.py:544` 每步都会执行 Python）→ 直接在 `SpecTpSync.sync` 里计数即可，精确 | 图外的 Python 计数**安全**（该站点不进图） |
| `copyiss` 到底测哪一段 / 是均值还是窗口和 | 闭源内部计时点（不可得）→ 替代：`LK_POOL_LOG` 打点行 + nsys 同窗口的池拷贝/host 侧活动对比；或向 lk_moe 提供方要字段定义 | 行本身是 host 打印，读取安全；**定义不可从安装版获得** |
| `hf_hook` / `hf_dec`（mailbox 唤醒 + host decode） | 已在 `[pool]` 行内（闭源计时点）；若要拆到每次 wake：`LK_HF_MBOX=0` 做 A/B，或 nsys 看 host↔device 门铃附近的时间线 | 主机侧打印，读取安全 |
| dspark 是否真在 5×markov 上花掉大部分时间 | 需要"draft 图节点级"分解：nsys 对 draft 图 replay 的 kernel/collective 时间线；或 `SGLANG_DSPARK_FOLDED_PROPOSAL=0` + 关草稿图，走 eager 后用 `profile_range`/CUDA event 逐段计时（`markov_sample.py` 的 launch 是可分离的 eager 路径） | eager 对照路径下 event **安全**；图内只能靠外部 profiler |
| 30 ms/token（每 verify 步 ≈55.6 ms）如何分解 | 需要端到端 step 计时（已有）+ nsys 全帧分解：target verify 图 / draft 图 / CPU 半 / 8 次 broadcast / 5 次 all-gather 六段 | 全帧分解用外部 profiler 安全；图内插桩不可行 |
