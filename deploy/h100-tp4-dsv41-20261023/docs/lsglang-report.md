# Lsglang (guqiong96/Lsglang) 只读评估报告

- 分析对象: `/mnt/SSD01/models/.llm-env/Lsglang/lsglang`
- 目标模型: `/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017`
- 目标硬件: 4×H100 80GB HBM3 (SM90, capability 9.0), 驱动 595.45.04, CUDA toolkit `/usr/local/cuda-13.2`
- 分析方式: 纯只读 (读仓库源码 + git 历史 + 解包 lk_moe wheel 读源码/字符串), **未安装依赖、未编译、未占用 GPU、未修改仓库源码**
- 证据目录: `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/` (lk_moe-2.7.1 wheel 与解包内容 `whl271/`)

> 本报告只陈述可从仓库/产物中直接核验的事实与算式, 不下"哪套栈更好"的判断。

---

## 0. 一页摘要

| 项 | 事实 |
| --- | --- |
| 仓库定位 | sglang 的 **fork**, 在上游 sglang 之上只叠加 lk_moe 混合 MoE 集成层 |
| 版本 | `python/pyproject.toml` `version = "1.6.1"`; 最新 tag `lsglang-v1.6.1` (2026-10-21); HEAD `60f6658524` = `lsglang-v1.6.1-2-g60f6658524` (2026-10-22 09:21) |
| 上游基线 | sglang main @ `06342acad2` (2026-10-21) |
| lk_moe 钉版 | **`lk_moe==2.7.1`** (`python/pyproject.toml`) |
| 预编译 wheel | **有**: PyPI 上有 `lsglang` (纯 Python, `py3-none-any`) 到 1.6.1; `lk_moe` 有 cp312 manylinux wheel; `sglang-kernel` 有 abi3 wheel。清华镜像均可下载 |
| **每层 N 个 GPU 专家缓存槽** | **存在**。开关 = `LK_POOL_SLOTS`, 语义 = **每个 MoE 层的常驻专家槽数** (注意口径差异, 见 §2.1) |
| 动态换入换出 | **存在**。后台 DMA pump, **2nd-touch window LRU**, 窗口由 `LK_POOL_WINDOW` (默认 32) 控制 |
| DeepSeek-V4.1-Flash-1017 | README 支持表**显式列出**该 checkpoint; 架构 `DeepseekV41ForCausalLM` → 运行时重写为 `DeepseekV4ForCausalLM` |
| engram 放主机内存 | **有**: `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` (默认 `False`) |
| 1M 上下文 | `max_position_embeddings=1048576`, rope `yarn` factor 16 → 运行时改用 `rope_type="deepseek_yarn"`; KV dtype 默认 `fp8_e4m3` |
| dspark | **支持** (`SpeculativeAlgorithm.DSPARK`); 但 dsv41 的 dspark 启动脚本已于 `76bd58545f` 删除 |
| SM90 专家 fp4 后端 | 默认 `flashinfer_mxfp4` → `Mxfp4FlashinferCutlassMoEMethod` = **W4A16**; 可选 `--flashinfer-mxfp4-moe-precision=fp8` 切到 **W4A8 (Humming 式)** |
| 池内专家的 GPU 内核 | **闭源 lk_moe** (Triton kernel 随 wheel 发布; C++ 池状态机在 `_lk_moe_C*.so`), sglang 侧只做 launch 编排 |
| 主要风险 | SM90 在支持表里只标 "ok" (**未实测**); 池内专家 kernel 不可被 sglang 侧替换; 288 槽 × 40 层 ≈ 50.4 GiB/卡 的显存账未经验证 |

---

## 1. 这是什么? 与 SGLang 上游、lk_moe 的关系

### 1.1 定位 (README 原文)

`README.md:3-9`:

> Lsglang is a special extension of sglang that adds **CPU-GPU hybrid (MOE) inference** on top of the latest sglang release version, fully compatible with stock sglang.
> The actual hybrid inference engine is **lk_moe**, sglang/vllm only provide the "GPU path", lk_moe provides the "hybrid path". Lsglang is the concrete integration case of lk_moe into sglang.

`README.md:11-13` 声明发布策略:

> Lsglang version updates are released **in sync with sglang releases** ... We do **not** pile on extra features; unless a necessary bug-fix patch is required, the diff against upstream stays minimal (just the lk_moe layer).

集成面清单见 `docs/integration.md` (含 "Case study — Lsglang (sglang) file-by-file" 表格): 改动集中在 `pyproject.toml`、`srt/utils/common.py`、`srt/layers/moe/fused_moe_triton/layer.py`、`srt/layers/quantization/*`、`srt/model_loader/loader.py`、`srt/utils/numa_utils.py`。

### 1.2 版本与 tag

| 项 | 值 | 证据 |
| --- | --- | --- |
| `pyproject.toml` version | `1.6.1` | `python/pyproject.toml`:7 |
| HEAD | `60f6658524032f0ce35a2a2756371dcd0291445f` (2026-10-22 09:21:07 +0800) | `git log -1` |
| HEAD subject | `fix(lk_moe): default LK_GPU_PREFILL_ENGINE back to triton` | `git log -1` |
| `git describe` | `lsglang-v1.6.1-2-g60f6658524` | `git describe --tags` |
| 最新 release tag | `lsglang-v1.6.1` (2026-10-21); 上一个 `lsglang-v1.6.0` (2026-10-15) | `git for-each-ref --sort=-creatordate` |
| branch | `main`, 19016 commits | `git rev-parse --abbrev-ref HEAD` / `git rev-list --count HEAD` |
| 工作树 | 干净 (`git status --short` 无输出) | |
| 远端 | `git@github.com:guqiong96/Lsglang.git` | `git remote -v` |

`RELEASE_NOTES.md:3-4` (v1.6.1): Base = upstream sglang main @ `06342acad2` (2026-10-21, 自 v1.6.0 基线 `583fada954` 起 1387 个 commit); lk_moe v2.7.1。

### 1.3 wheel 可获得性 (实测清华镜像)

| 包 | 镜像状态 | 说明 |
| --- | --- | --- |
| `lsglang` | **有**, 到 `1.6.1-py3-none-any.whl`; 历史版本 1.4.0 / 1.4.1 / 1.4.2 / 1.4.3 / 1.4.4 / 1.4.6-1.4.9 / 1.4.12-1.4.15 / 1.5.0 / 1.5.2 / 1.5.3 / 1.5.5 / 1.5.6 / 1.6.0 / **1.6.1** | **纯 Python wheel** (`py3-none-any`), 只含集成层 |
| `lk_moe` | **有**, 1.1.0 → **2.7.1** (共 30+ 版本, 全部 `cp312-cp312-manylinux_2_34_x86_64.whl`) | 引擎本体, **License: Proprietary** (`lk_moe-2.7.1.dist-info/METADATA`) |
| `sglang-kernel` | **有**, 到 `0.4.10-cp310-abi3-manylinux2014_x86_64.whl` | 与 pin 一致 (`sglang-kernel==0.4.10`) |
| `flashinfer-python` | 需自验 | pin 为 `flashinfer_python[cu13]==0.7.2` |
| `flash-attn-4` | 镜像 index 可达 | pin 为 `flash-attn-4>=4.0.0b18` |

注意: `lsglang` wheel 是纯 Python, 因此**不能靠 wheel 省掉编译** —— 重活在本仓库的 C++/CUDA (sglang 侧) 与 `sglang-kernel` / `flashinfer` / `lk_moe` 等二进制依赖。

---

## 2. 【最关键】每层 N 个 GPU 专家缓存槽 + 动态换入换出

**结论: 存在, 就是 `LK_POOL_SLOTS`(GPU Pool / expert cache)。**

### 2.1 `LK_POOL_SLOTS` 精确语义

| 维度 | 事实 | 证据 |
| --- | --- | --- |
| 开关 | 值本身就是开关, `0`/未设 = 关 (所有 `LK_POOL_*` 失效) | `README.md:205` (配置参数表 GPU Pool 段); `python/sglang/srt/utils/lk_moe_env.py:281-285` (`get_pool_slots()` 默认 0; `is_lk_pool_enabled()` = master 且 slots>0) |
| 单位 | **每个 MoE 层的常驻专家槽数** (不是全模型总数, 不是覆盖率) | lk_moe METADATA "Dual-knob rule" 表: "resident expert slots per MOE layer; **VRAM cost = slots × layers × expert bytes**" |
| per-layer 还是 global | **容量按层计, 每层统一 N 个槽**; 成员(哪些层参与)是**全局**的 = 所有"非 GPU 常驻"层 | `README.md:209` `LK_POOL_LAYERS` 标注 RETIRED: "pool membership is global (`LK_POOL_SLOTS` > 0); the parser is dead code" |
| 默认值 | `0` (关) | `lk_moe_env.py:282` |
| 取值范围 | 整数; 无上限校验, 实际有效上限受 `num_local_experts` 约束 (见 §2.4) | 二进制无 clamp 字符串, `[pool] slab:` 直接按 N 分配 |
| 作用范围 | **主模型层 (`model_type == "main"`) 才建池**, draft/MTP/DSpark 层不建真实池 (退化为 prefill shell) | `lk_moe_env.py:287-292` `is_lk_pool_layer()`: `if model_type != "main": return False`; `layer.py:2057` `cfg.pool_cache = self.is_pool_layer` + 注释 "pool_cache says whether this instance's slots are real (main) or a pure prefill shell with an all-zero snapshot (draft)" |
| 与 GPU 常驻层互斥 | 是。`LK_GPU_RESIDENT_MOE_LAYERS` 列出的层永远不建池 | `lk_moe_env.py:290-291` (`is_lk_moe_gpu_resident_layer` 为真则 `is_lk_pool_layer` 返回 False); `README.md:158` "resident layers ... (complementary: a resident layer never pools)" |
| 容量的**最强证据** | 启动日志格式串 `[pool] slab: %.2f GiB (%d layers x %.2f MiB)` —— 总分配 = 层数 × 每层字节 | lk_moe `_lk_moe_C*.so` 字符串 |
| 槽位索引空间 | **按 `(layer, expert)` 二维**: "resident slot of **(layer, expert)** or -1" | lk_moe `.so` 绑定说明串; py 绑定 `pool_slot(layer, expert)` / `pool_ready_slot(layer, expert)` / `pool_snapshot_slot(layer, expert)` |

**关键推论 (对"288 每层"的理解)**: 用户要的"每层 288 槽"与 `LK_POOL_SLOTS=288` **语义一致**; 但**不是** `288 × 40 = 11520` 个共享槽池 —— 而是**每层各自 288 个槽, 全模型实际是 40 个独立的 288 槽缓存**, 显存按 40×288 计。

代码侧构造路径 (sglang 只填配置, 不实现池):

- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:2046-2110` `_fill_lk_cfg()`:
  - `:2058-2059` 注释 "Slab sizing: **one contiguous pool allocation per weight type**. Members = non-resident layers; the pool tier off means zero rows."
  - `:2060` `cfg.pool_member_layers = 0`
  - `:2061-2072` 若池开启: 读 text sub-config 的 `num_hidden_layers`, `cfg.pool_member_layers = sum(1 for i in range(n_layers) if not is_lk_moe_gpu_resident_layer(i))`
  - `:2081` `cfg.pool_slots = get_pool_slots()`
  - `:2080` `cfg.pool_decode`; `:2105-2107` `cfg.pool_window` / `pool_inflight` / `pool_log`
- `layer.py:2151-2152` 等 5 处 `self.lk_moe_config = lk_moe.MOEConfigV2(); self._fill_lk_cfg(self.lk_moe_config)` → 交给 lk_moe 的 `MOE_FP8` / `MOE_MXFP4` / `MOE_NVFP4` / `MOE_WNA16` / `MOE_BF16` 等类
- 池成员语义再确认: `docs/integration.md`: "Pool identity is resolved here too, as config fields only (`layer_id`, `gpu_pool`, `route_probe`) — the engine reads no pool-membership env vars"

### 2.2 动态性: 槽位如何填充/驱逐 (冷热判定)

lk_moe METADATA 原文:

> **GPU Pool (expert cache)**: A background DMA pump keeps hot experts resident in VRAM (**2nd-touch window LRU**, off the critical path). Pool-resident experts are computed on the GPU while the rest stay on the CPU — both in decode and prefill — **the split is decided by one shared snapshot word, so every expert is computed exactly once**. Works for both NVFP4 and MXFP4 experts. **Fully fail-closed: an empty or disabled pool is bit-identical to the plain hybrid path.**

| 参数 | 默认 | 语义 | 证据 |
| --- | --- | --- | --- |
| `LK_POOL_WINDOW` | `32` | **预期复用窗口(步)**: 被驱逐的专家只有在窗口内再次出现才会被泵回; 非法值 clamp 到最小 **8**, **永不关闭功能** | `lk_moe_env.py:128-132` `return max(8, get_int_env_var("LK_POOL_WINDOW", 32))` + 注释 "Tuned default (hit ~81%)" |
| `LK_POOL_INFLIGHT` | `16` | 在飞 DMA 作业数 / event-ring 深度; 背压上限 = 8× | `lk_moe_env.py:142-145` |
| `LK_POOL_STAGE` | `1` | 合并式 H2D staging ring (先在 NUMA 片段内 gather, 再每数组一次 bulk copy); **由库直接读 env** | `README.md` PLE/NVMe 段前后 "Read directly by the library: ... the direct-read knobs (`LK_POOL_STAGE`, ...)" |
| `LK_POOL_DECODE` | 跟随 pool | decode 的 GPU 半边; 未设时跟随 `LK_POOL_SLOTS>0`, 显式 `1`/`0` 覆盖 | `lk_moe_env.py:297-301` |
| `LK_POOL_DET_SPLIT` | `0` | 诊断: 确定性切分 —— 先 publish claimed 批次, 让 GPU 半边在该批次的 copy fence 上流式等待 | `lk_moe_env.py` + README 诊断表 |
| `LK_POOL_SPLIT_PROBE` | `0` | 诊断: 打印冻结切分 (claim/ready/snapshot 计数), 仅 ≤32 routed id 的批 | `layer.py:2493-2508` |
| `LK_POOL_LOG` | `0` | 诊断: 每 N 个 decode step 打一行 `[pool]` 统计 | `lk_moe_env.py:163-165` |
| `LK_ROUTE_PROBE` | off | **热专家统计/离线标定入口**: "per-step `[RP]` routing traces for offline calibration (requires the pool on)" | README 诊断表 |

**统计与热专家判定实现位置**: 在 lk_moe 内 (闭源), 而非 sglang:
- C++ 侧类 `PoolPump` (DMA worker), 源文件标记 `moe_v2_expert_pool.cu` / `moe_v2_gpu_borrow.cu` / `moe_v2_gpu_forward_kernels.cu` / `moe_v2_cpu_decode.cu` / `moe_v2_gpu_metadata.cu` / `moe_v2_gpu_memory.cu` / `moe_v2_gpu_prefill_cutlass.cu`
- 绑定: `pool_gpu_report` ("THIS instance's pool geometry (empty dict if pool off / draft)")、`pool_gpu_live`、`pool_layer_info` ("per-layer pool geometry for the GPU side (empty dict if absent)")、`pool_freeze_view`、`pool_verify_sample`、`pool_capture_pause/resume`、`pool_wait_fence` / `pool_wait_all_fences`
- 运行期统计行格式串 (含命中/驱逐/停滞计数):
  `[pool] dstep=%ld win hit=%.1f%% nr=%.1f%% psl=%.2f link=%.2fGB/s gather=%.0fus copyiss=%.1fus que=%ld ... | cum hit=%.1f%% h=%ld m=%ld p=%ld B=%ld resident=%ld nr=%ld vf=%ld oom=%ld stalls=%ld sk=%ld stale=%ld tag=%ld pubbad=%ld publag=%ld cmiss=%ld cdup=%ld`
- 另有 `async expert pool counters (hits/misses/pumps/link bytes)` 与 `byte-check resident slots against the host weights (drains the pump first)` (自检绑定)

sglang 侧参与的部分 (开源可见):
- `layer.py:2417-2488`: 每个池层登记 `self._pool_layer_info` (含 `H, I, k, scale_group, packed_k_per_iter, weight_addr, scale_addr, slot_bytes, slot_scale_bytes, format, snapshot_addr, ...`), 分配 `FusedMoE.pool_stream`, 计算 `FusedMoE._pool_decode_cap`
- `layer.py:2489-2587` `_pool_decode_launch()`: 在私有 stream 上 launch Triton kernel, 最后 `self.lk_moe.pool_record_aux(pool_stream.cuda_stream)` 记录 fence
- `layer.py:2589-2620` `_cpu_decode()`: `real_tokens <= _pool_decode_cap` 时走 `_pool_decode_launch()` + `self.lk_moe.cpu_decode(...)` 两条半边同时算
- `lk_moe_env.py:54-65` `park_lk_moe_expert_pool()`: CUDA-graph capture/replay 前后暂停/恢复池的 DMA pump (否则 `cuGraphLaunch` 与 in-flight copy 在打补丁的 P2P 驱动上死锁)

### 2.3 显存代价公式与 288 槽的账

**公式 (权威来源, lk_moe METADATA)**: `VRAM = slots × layers × expert bytes`
启动日志形式: `[pool] slab: %.2f GiB (%d layers x %.2f MiB)`

**错位点 (必须澄清)**: `expert bytes` 是**单卡本地的专家字节** (已按 `moe_tp_size` 切宽、按 `moe_ep_size` 切专家数), 不是全模型未切分的专家字节。

本模型几何 (来自 `/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017/config.json` 与 `model.safetensors.index.json` 实测):

| 项 | 值 |
| --- | --- |
| 层数 | 40 (**全部 40 层都是 MoE 层**, index 中每层均有 `ffn.experts.0..383`) |
| `hidden_size` | 5120 |
| `moe_intermediate_size` | 2304 |
| `n_routed_experts` | 384 (`expert id range 0..383, count 384`) |
| `n_shared_experts` | 1 (启动脚本用 `--disable-shared-experts-fusion` 关掉融合) |
| `num_experts_per_tok` | 6 |
| 专家量化 | `quantization_config`: `quant_method=fp8`, **`expert_dtype=fp4`**, `weight_block_size=[32,32]`, `scale_fmt=ue8m0` |
| 专家权重命名 | `layers.N.ffn.experts.M.{w1,w3,w2}.{weight,scale}` → MXFP4: fp4 e2m1 权重 + e8m0 块尺度 |
| 总权重 | 476 GiB |
| `--tensor-parallel-size 4` 且**不加** `--ep-size` 时 | `ep_size` 默认 = **1** (`python/sglang/srt/arg_groups/fields/parallel.py:269-275`) → `moe_ep_size=1`, `moe_tp_size = tp_size // moe_ep_size // moe_dp_size = 4` (`runtime_context.py:151-156`) → 每卡 **384 个本地专家**, 每个专家宽度切为 `2304//4 = 576` (`layer.py:453-454`) |

**单槽字节 (TP4 / ep_size=1, 每卡)**:

```
权重 w13 = 2 × 576 × 5120 = 5,898,240 参数
权重 w2  = 5120 × 576     = 2,949,120 参数
合计 8,847,360 参数 × 0.5 B(4bit)            = 4,423,680 B = 4.2188 MiB
MXFP4 尺度(group 32 沿 K): w13 [1152,160]×1B + w2 [5120,18]×1B
                                            =   276,480 B = 0.2637 MiB
单槽合计                                     ≈ 4,700,160 B = 4.4824 MiB
```

交叉校验 (未切分的整专家, 用于对照): `(2×2304×5120 + 5120×2304)×0.5 + 尺度(1,105,920 B) = 17.93 MiB`/专家 → ×384 = **6.72 GiB/层**, 与任务下达时的口径一致。

**288 槽的账 (每卡)**:

| N (`LK_POOL_SLOTS`) | 每层 | × 40 层 | 占 0.88×80 GiB = 70.4 GiB 的比例 |
| --- | --- | --- | --- |
| 64 (README 建议示例) | 287 MiB | **11.2 GiB** | 16% |
| 96 | 430 MiB | 16.8 GiB | 24% |
| 128 | 574 MiB | 22.4 GiB | 32% |
| 192 | 861 MiB | 33.6 GiB | 48% |
| **288 (用户目标)** | **1.261 GiB** | **50.4 GiB** | **72%** |
| 384 (全常驻) | 1.681 GiB | 67.2 GiB | 95% |

**结论性事实**: 288 槽/层需要每卡 **约 50.4 GiB** 专供池槽, 剩余 70.4 − 50.4 = **20.0 GiB/卡**需同时容纳: 本卡注意力/dense 权重、KV cache (1M 上下文 × fp8_e4m3)、激活与 CUDA graph 缓冲。是否相容取决于 KV cache 目标 token 数与 `--mem-fraction-static` 的取法, **本报告不做可行性判断, 只给算式**。

**显存预算/校验发生在哪里 (关键代码位置)**:

- 池 slab 由 **lk_moe 在 MOE 构造期直接 `cudaMalloc`** (日志串: `[pool] slab cudaMalloc(%.2f GiB) failed (%s): engine falls back to its own arena`), 因此对 sglang 而言**已从 free 显存中扣除**
- sglang 侧唯一的接口是 `python/sglang/srt/mem_cache/kv_cache_configurator.py:217-246` `_lk_moe_device_reservation_gb()`:
  - 先 `lk_moe.get_device_mem_usage()` 取 lk_moe 常驻字节; 若 > 0 → 返回 0.0 (不再额外预留 KV), 日志 "lk_moe static device arena: %.2f GB already resident (allocated in MOE ctors, excluded from free memory); no additional KV reservation."
  - 报告为 0 时才读 `LK_GPU_RESERVE_GB` 作 fallback
- `lk_moe.get_device_mem_usage()` 的口径 (METADATA 原文): "persistent device bytes owned by lk_moe: **pool rows (or the one per-type slab)**, the cutlass prefill arena, and the Triton staging ring (`window × E × per-expert bytes`)。Borrowed cutlass blocks are aliases inside the slab and are not counted twice"
- `--mem-fraction-static` 的定义: `(model weights + KV cache pool) / GPU memory capacity` (`python/sglang/srt/arg_groups/memory_hook.py:60-61`)

### 2.4 槽数的**有效上限**(288 是否可达)

| 场景 | `moe_ep_size` | 每卡本地专家数 | 每槽字节(每卡) | 288 槽是否有意义 |
| --- | --- | --- | --- | --- |
| `--tensor-parallel-size 4`(**默认, 不加 --ep-size**) | 1 | **384** | 4.48 MiB | **有** (288 / 384 = 75%) |
| `--tensor-parallel-size 4 --ep-size 4` | 4 | **96** | 17.93 MiB | **无** (288 > 96, 超过该层专家总数) |

- `ep_size` 默认值 = 1: `python/sglang/srt/arg_groups/fields/parallel.py:269-275`
- `num_local_experts = _num_global_routed // storage_ep_size` (`storage_ep_size = moe_ep_size`): `layer.py:431-443`
- `intermediate_size_per_partition = intermediate_size // moe_tp_size`: `layer.py:453-454`
- 旁证: `mxfp4_flashinfer_cutlass_moe.py:71-79` 注释把 "2304 / TP=4 = 576" 作为 `intermediate_size_per_partition` 的例子, 即默认 TP4 下 `moe_tp_size=4` (ep_size=1)
- 仓库自带的 DSV4.1 TP4 精度测试**显式**用 EP4: `python/sglang/test/dsv41_flash_common.py:22` `TP4_EP4_ARGS = ["--attn-dp-size", "1", "--ep-size", "4"]` (用于 `test/registered/e2e/models/test_deepseek_v41_flash_gb300_tp4ep4.py`)。**即仓库作者在同为 TP4 的参考配置里用的是 EP4, 与"每层 288 槽"目标冲突, 这是必须在启动前明确的一个选择。**

### 2.5 是否有预热 / 统计相关参数

| 参数 | 默认 | 用途 | 证据 |
| --- | --- | --- | --- |
| `LK_ROUTE_PROBE` | off | 每步 `[RP]` 路由轨迹, **用于离线标定热专家分布** (需池开启) | README 诊断表 |
| `LK_POOL_LOG` | 0 | 窗口化 `[pool]` 统计行 (hit / oom / stalls / stale ...) | `lk_moe_env.py:163-165` |
| `LK_POOL_SPLIT_PROBE` | 0 | 每池层的冻结切分计数 | `layer.py:2493-2508` |
| `pool_verify_sample(layer, max_experts)` | — | 按需的**池字节校验绑定** (对照 NUMA 源逐字节比较); METADATA 明确 "not an env knob" | lk_moe METADATA + `.so` 串 |
| 池预热 | — | 未见独立的"预热"env; 预热由背景 DMA pump 持续进行 (第二次 touch 才驻留) | METADATA "2nd-touch window LRU" |

### 2.6 其它与 engram / SM90 / SuperMix / DSPARK 有关的 LK_* 全清单

`LK_*` 的唯一真源是 `python/sglang/srt/utils/lk_moe_env.py` (305 行, 文件头注释: "Everything LK_* lives here so the knobs have a single source of truth")。

**核心 / CPU / NUMA / 电源**

| 变量 | 默认 | 含义 | file:line |
| --- | --- | --- | --- |
| `LK_MOE_HYBRID_ENABLED` | `0` | **总开关**; 关 = 完全不 import lk_moe, 走原版 sglang 位对位 | `lk_moe_env.py:100-101` |
| `LK_NUMA_INTERLEAVE` | `true` (fork 默认开) | 主机专家权重跨 NUMA 交错 (pump 的每节点 copy 流必需) | `lk_moe_env.py:103-105` |
| `LK_NUMA_ILP` | `0` | NUMA 分配器 ILP 模式 (库直读) | METADATA; README |
| `LK_NUMA_HUGE` | `0` | 2M THP 对齐主机 slab (库直读) | METADATA |
| `LK_THREADS` | cores/GPU | 主机 decode 线程数 (库 bootstrap 直读) | METADATA |
| `LK_THREAD_BINDING` | `CPU_CORE` | `CPU_CORE` / `NUMA_NODE` | METADATA |
| `LK_POWER_SAVING` | `0` | CPU 省电模式 | METADATA |
| `LK_HF_MBOX` | `1` | 主机侧 decode 的 mailbox 提前启动 (省 ~50 µs graph 唤醒延迟) | `lk_moe_env.py:157-160` |
| `LK_MOE_LAYERWISE_LOAD` | off | 逐层加载 | `lk_moe_env.py:303-304` |
| `LK_MBOX_STOPFILE` | — | 有序停机 stop-file (库直读) | METADATA |
| `LK_DECODE_TS` / `LK_TS_SKIPGEMM` | — | decode 时间戳轨迹 / skip-the-gemm A/B (库直读) | METADATA |

**GPU 常驻层**

| 变量 | 默认 | 含义 | file:line |
| --- | --- | --- | --- |
| `LK_GPU_RESIDENT_MOE_LAYERS` | 未设 | 常驻 VRAM 的 MoE 层清单, 语法 `0`, `0-1`, `0,9` | `lk_moe_env.py:210-212`, `:222-252` |
| `LK_GPU_RESIDENT_MOE_LAYERS_SPEC` | 未设 | draft 模型层 (`_SPEC` 优先) | `lk_moe_env.py:197-201` |
| `LK_GPU_RESIDENT_MOE_LAYERS_MTP` / `_DSPARK` | 跟随 `_SPEC` | 分角色覆盖 | `lk_moe_env.py:202-209` |
| `LK_GPU_RESERVE_GB` | `0` | framework 侧 fallback: 为 lk_moe staging 从 KV 预算里预留 GB (仅当 `get_device_mem_usage()` 报 0 时生效) | `kv_cache_configurator.py:241-246` |
| `LK_MOE_LAYERWISE_LOAD` | off | 见上 | |

**Super Mix Prefill / GPU prefill**

| 变量 | 默认 | 含义 | file:line |
| --- | --- | --- | --- |
| `LK_GPU_PREFILL_MIN_TOKENS` | `0`(关) | **双开关**: >0 既开 tier 又设 token 批门槛 | `lk_moe_env.py:107-111` |
| `LK_GPU_PREFILL_ENGINE` | **`triton`** (HEAD 已改回) | `triton` = staging ring + Triton group-GEMM (fp32 激活, 与 CPU 参考一致); `cutlass` = 恢复的 C++ grouped-GEMM (bf16 中间量, 长上下文 logits 漂移、翻转边界 EOS, 故 2026-10-22 回退) | `lk_moe_env.py:113-127` + `git log -1` |
| `LK_GPU_PREFILL_WAVE_EXPERTS` | `-1` 自动 | 仅 cutlass; `-1` 自动 / `0` 单发 / `>0` 显式上限 (clamp 到 E) | `lk_moe_env.py:180-187` |
| `LK_GPU_PREFILL_SUB_M` | `0` | super-chunk staging 大小 (token); `0` = 按 chunk | `lk_moe_env.py:136-140` |
| `LK_GPU_PREFETCH_WINDOW` | `1` | **staging ring 深度**(非跨层预取); 仅 triton; `2` = ping-pong 需快 PCIe; ring 显存 = `window × E × per-expert bytes` | `lk_moe_env.py:171-178` |
| `LK_PP_DEBUG` | off | prefill staging tier 追踪 | `lk_moe_env.py:168-169` |

**验证 / dspark 相关**

| 变量 | 默认 | 含义 | file:line |
| --- | --- | --- | --- |
| `LK_VERIFY_RESIDENT` | `0` | verify 批覆盖 draft 行的池常驻专家 (有精度取舍) | `lk_moe_env.py:147-150` |
| `LK_VERIFY_SKIP_MISS` | `0` | 跳过未驻留行的 verify | `lk_moe_env.py:153-155` |
| `LK_SPEC_PROBE` | off | CUDA-event 计时的 draft/verify 分段 | README 诊断表 |
| `LK_DISABLE_QSA_SHARE` | off | draft worker 关闭 QSA share (A/B 隔离) | README 诊断表 |

**engram / PLE / NVMe**

| 变量 | 默认 | 含义 | file:line |
| --- | --- | --- | --- |
| `LK_ENGRAM_NVME` | `0` | DSV4.1 Engram 走 NVMe 精确查找 (**表不进主机内存, 也不进 HBM**) | `python/sglang/srt/utils/common.py:1536` |
| `LK_ENGRAM_NVME_CACHE_MB` | `1024` | Engram NVMe store cache | `common.py:1541` |
| `LK_ENGRAM_NVME_STAGING_MB` | `128` | Engram NVMe staging | `common.py:1545` |
| `LK_NVME_DIRECT` | `0` | 行填充改用 O_DIRECT + pinned 直映 cache (仅在功能后端开启时有效) | `common.py:1528` |
| `LK_NVME_CACHE_MB` | `1024` | pinned 直映行 cache | `common.py:1551` |
| `LK_PLE_BLOCK_MB` | 未设 | PLE prefill conv 临时量预算 (仅 Qwen3.8) | `python/sglang/srt/models/qwen4_exp.py:659-661` |
| `LK_PLE_PROBE` / `LK_PLE_PROBE_DUMP` | off | PLE 行局部性/流量探针 | `lk_moe_env.py:214-220` |

**旧名兼容 (LVLLM_* → LK_*)**: `common.py:1557-1566` 的 `_LK_ENV_RENAMES` 表在启动时自动改名并打一行警告 (`LK_*` 优先)。覆盖 `LVLLM_MOE_NUMA_ENABLED`、`LVLLM_ENABLE_NUMA_INTERLEAVE`、`LVLLM_GPU_PREFILL_MIN_BATCH_SIZE`、`LVLLM_GPU_PREFETCH_WINDOW`、`LVLLM_GPU_RESIDENT_MOE_LAYERS{,_SPEC,_MTP,_DSPARK}`、`LVLLM_ENABLE_MOE_LAYERWISE_LOAD`、`LVLLM_GPU_RESERVE_GB`。

**与 SM90 / Marlin 相关的 `SGLANG_*` (非 LK_ 前缀)**

| 变量 | 默认 | 含义 | file:line |
| --- | --- | --- | --- |
| `SGLANG_SM8X_MARLIN_GROUP32` | `true` | SM8x 稠密 block-FP8 group-32 Marlin W8A16 decode | `python/sglang/srt/layers/quantization/fp8.py:566` |
| `SGLANG_SM8X_MARLIN_GROUP32_ONLY` | `false` | 仅用该 offload 路径 | `fp8.py:569`, `:905` |
| `SGLANG_SM8X_MARLIN_GROUP32_MAX_M` | `16` | 该路径的最大 M | `fp8.py:576` |
| `SGLANG_DSV4_FP4_DEQUANT` | `false` | 为 `true` 时把 fp4 专家反量化成 fp8 并改走 FP8 LK MoE 路径 | `python/sglang/srt/environ.py:1645`; `fp8.py:424-425`, `:2032` |
| `SGLANG_DSV4_FP4_EXPERTS` | 未设 | 手工指定 fp4 专家布局 (未设时自动探测) | `python/sglang/srt/configs/model_config.py:656-667` |
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` | `false` | **engram 放主机内存** | `environ.py:1704` |
| `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT` | `"shared"` | `shared` = 整个 TP 组一份、每 rank 都映射、无查找 all-reduce (需同 PID namespace); `per_rank` = 每 rank 一份匿名映射只放自己的行, 需 all-reduce, 是唯一能拿大页的布局 | `environ.py:1711` |
| `SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN` | `true` | 加载时 `cudaHostRegister` (~40 s / 24 GiB base page); `false` 保持 pageable, 每次查找多一次 host page walk | `environ.py:1715` |
| `SGLANG_DSV4_KV_LAYOUT` | — | DSV4 KV 布局 (`auto` / `v41`) | `deepseek_v4_hook.py:32-33` |

---

## 3. DeepSeek-V4.1 支持

### 3.1 架构注册

| 项 | 事实 | file:line |
| --- | --- | --- |
| 权重里的架构名 | `architectures = ["DeepseekV41ForCausalLM"]`, `model_type = "deepseek_v41"`, `text_config.model_type = "deepseek_v41_text"` | checkpoint `config.json` |
| 运行时类名 | `DeepseekV4ForCausalLM` (非 V41) | `python/sglang/srt/models/deepseek_v4.py:4743` |
| 名称重写 | `configs/deepseek_v41.py:38-39`: `if values.get("architectures") == ["DeepseekV41ForCausalLM"]: values["architectures"] = ["DeepseekV4ForCausalLM"]` | |
| 配置归一化 | `configs/deepseek_v41.py:19-40` `normalize_deepseek_v41_config()`: 把 `text_config` 拍平到顶层, 把 `vision_config` 字段映射成 `vision_*` | |
| 配置类族 | `DeepseekV41Config` / `DeepseekV41TextConfig` / `DeepseekV41VisionConfig` | `configs/deepseek_v41.py:42-89` |
| 运行时配置 | `python/sglang/srt/configs/deepseek_v4.py` (`DeepSeekV4Config`) | |
| 模型 hook / override | `arg_groups/model_hook.py:490-491` → `elif model_arch in ["DeepseekV4ForCausalLM"]`; `arg_groups/model_overrides/deepseek_v4.py:16` `@_register_for("DeepseekV4ForCausalLM")` | |
| 多模态 | `deepseek_v41_vit` / `deepseek_v41_image_processing` (vision 层) | `deepseek_v4.py:177-180` |
| 上游 parity fixture | `rust/sglang-processor/tests/fixtures/parity/deepseek-v4.1-flash.json` 记录 `deepseek-ai/DeepSeek-V4.1-Flash` rev `dba1be0a...` | |
| README 支持表 | 显式列出 `DeepSeek-V4.1-Flash-1017`, Launch 列指向 `launch/sglang_serve_dsv41_tp2_3090.sh`, NVMe 列指向 `launch/sglang_serve_dsv41_tp2_3090_nvme.sh` | `README.md` 支持表 |

### 3.2 engram 实现

| 项 | 事实 | file:line |
| --- | --- | --- |
| 配置字段 | `engram_layer_ids`, `engram_num_embeddings`, `engram_max_ngram_size`, `engram_vocab_size`, `engram_n_heads`, `engram_head_dim`, `engram_pad_token_id`, `engram_compressed_vocab_size` 全部在配置类里 | `configs/deepseek_v41.py:58-69` |
| 本 checkpoint 的值 | `engram_layer_ids=[1,14]`, `engram_num_embeddings=[384006168, 384016682]`, `engram_head_dim=256`, `engram_n_heads=8`, `engram_max_ngram_size=4`, `engram_vocab_size=16000000`, `engram_compressed_vocab_size=99092` | checkpoint `config.json` |
| 权重 key | index 实测: 仅 `layers.1.*` 与 `layers.14.*` 含 `engram.*` | `model.safetensors.index.json` |
| 实现文件 | `python/sglang/srt/layers/engram.py` (含 `EngramEmbedding`, `shard_range`, `_HostTable`)、`python/sglang/srt/layers/engram_nvme.py` | |
| 内核 | `python/sglang/kernels/ops/embeddings/{engram_gate,engram_gather,engram_hash}.py` | |
| 管理 | `python/sglang/srt/model_executor/model_runner_components/ngram_embedding_manager.py` | |

### 3.3 engram 放 CPU 内存 —— **有**

- 开关: `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` (`environ.py:1704`, `EnvBool(False)`), 注释原文: "DeepSeek-V4.1 engram host table: **keep the tables in host memory** (layout below) and gather rows from the GPU instead of sharding them over HBM."
- 实现: `engram.py:868-880` `EngramEmbedding` docstring: "Rows are sharded over the TP group in device memory ...; with `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` they **live in host memory instead**, as one shared copy or one shard per rank (see `_HostTable`)."
- 分支: `engram.py:928-938`
  - NVMe 优先: `is_lk_engram_nvme_enabled()` → `NvmeEngramStore` (表留在 safetensors 分片里, pread)
  - `elif envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get(): self._init_host_table(...)`
  - `else:` 普通 HBM 分片
- `_HostTable` (`engram.py:641-676`): `shared` 布局 = `memfd` + `mmap(MAP_SHARED)`, 整个 TP 组共享一份; `per_rank` = `mmap(MAP_PRIVATE|MAP_ANONYMOUS)`, 每 rank 只放自己的行
- 内存量核算 (`engram.py:948-962` `_init_host_table`): `w_bytes = n × dim` (fp8 e4m3), `s_bytes = n × (dim // FP8_BLOCK_SIZE)`, `FP8_BLOCK_SIZE = 128` (来自 `sglang.kernels.ops.attention.dsv4.torch_quant`) → 每行 `256 + 2 = 258 B`

  | 层 | rows | 字节 |
  | --- | --- | --- |
  | 1 | 384,006,168 | 99,073,591,344 B ≈ 92.27 GiB |
  | 14 | 384,016,682 | 99,076,303,956 B ≈ 92.27 GiB |
  | 合计 | 768,022,850 | 198,149,895,300 B ≈ **184.5 GiB** |
- 固定 pin: `SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN` 默认 `true`; 注释给出量化代价 "**~40 s per 24 GiB** of base pages" → 184.5 GiB 约需 5 分钟额外启动时间
- 另一条路 (不进主机内存): `LK_ENGRAM_NVME=1` — `engram_nvme.py:1-12` "**Nothing here touches CUDA from the host side** ... no table copy exists anywhere (**RAM peak at load is zero**)". 与 host-table **互斥**: "NVMe set means host-table mode is rejected; unset falls through to host table when requested."
- NVMe 路径的启动脚本: `launch/sglang_serve_dsv41_tp2_3090_nvme.sh` (其中 `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0`)

### 3.4 1M 上下文

| 项 | 事实 | file:line |
| --- | --- | --- |
| `max_position_embeddings` | `1048576` (=1M) | checkpoint `config.json` |
| rope scaling | `{"rope_type":"yarn","factor":16,"beta_fast":32,"beta_slow":1,"original_max_position_embeddings":65536}` | checkpoint `config.json` |
| 配置字段 | `rope_scaling: Dict[str, float]` | `configs/deepseek_v4.py:86` |
| 运行时处理 | 读取 `get_rope_config(config)`; 当条件满足时把 `rope_type` 覆写为 **`deepseek_yarn`** 并传给 rotary embedding | `models/deepseek_v4.py:1284-1286`, `:1467-1476` |
| yarn 工具 | `layers/rotary_embedding/yarn.py` (`yarn_find_correction_range`, `yarn_get_mscale_simple`, `yarn_linear_ramp_mask`) | `layers/rotary_embedding/__init__.py:13-21` |
| 上下文长度参数 | `--context-length` 存在 (`arg_groups/fields/model.py:104`) | |
| **KV cache dtype** | **没有 `fp8_ds_mla`** (该名字在本仓库全库 0 命中); DSV4 把 `kv_cache_dtype` 默认改成 **`fp8_e4m3`**, 且只允许 `{"fp8_e4m3","bfloat16"}` (NPU 强制 bfloat16), 其他值直接 `assert` 失败 | `arg_groups/overrides.py:905-926` `_deepseek_v4_kv_cache_dtype()` |
| 注意力后端 | `attention_backend = "dsv4"` (override); 另有 `_default_dsv4_attention_backend()`: 仅当 `model_type=="deepseek_v41"` 且 `is_sm100_supported()` 且 `kv_cache_dtype=="fp8_e4m3"` 且非 disaggregation 且 layout 允许时才返回 `flashmla_mega`, **否则返回 `flashmla`** → **SM90 落在 `flashmla`** | `model_overrides/deepseek_v4.py:21`; `deepseek_v4_hook.py:29-43` |
| page size | DSV4 强制 `page_size = 256` (NPU 为 128) | `model_overrides/deepseek_v4.py:55-66` |
| SWA | `--swa-full-tokens-ratio` 在 dsv41 下**不被自动设 0.1** (条件里显式排除 `model_type == "deepseek_v41"`), 需手工给 | `model_overrides/deepseek_v4.py:44-50` |

### 3.5 dspark 投机解码

| 项 | 事实 | file:line |
| --- | --- | --- |
| 算法枚举 | `SpeculativeAlgorithm.DSPARK` | `python/sglang/srt/speculative/spec_info.py:41` |
| 判定 | `is_dspark()` / 归入 `is_dflash()` 家族 | `spec_info.py:123-127`, `:213-218` |
| 反解路径 | `dspark` 使用非 `FutureMap` 的反解 (与 `STANDALONE`/`DFLASH` 同族) | `spec_utils.py:343-346` |
| TP 同步点 | `SpecTpSyncSite.DSPARK_MEM/PLAN/ACCEPT_*/TARGET/DRAFT` | `speculative/spec_tp_sync.py:21-74` |
| draft 模型 | `python/sglang/srt/models/deepseek_v4_dspark.py` → `DeepseekV4ForCausalLMDSpark` | `test/registered/unit/model_loader/test_runai_model_streamer_loader.py:25` |
| 接受率调试 | `SGLANG_L2_PROBE_ACCEPT` (dspark accept-rate-0 三角定位) | `environ.py:1719` |
| 与池的交互 | `LK_VERIFY_RESIDENT` / `LK_VERIFY_SKIP_MISS`; `LK_GPU_RESIDENT_MOE_LAYERS_DSPARK` | 见 §2.6 |
| **当前启动方式** | dsv41 的 dspark 启动脚本**已被删除**: `76bd58545f chore(dsv4.1): drop obsolete dspark TP2/TP4 launch scripts` (删除 `launch/sglang_serve_dsv41_dspark_tp2_3090.sh` 与 `launch/sglang_serve_dsv41_dspark_tp4.sh`)。HEAD 的 `launch/` 里与 dspark 相关的只剩 `sglang_serve_0731_dspark_tp2_3090.sh` | `git log --all --name-only -- 'launch/*dspark*'`, `ls launch/` |
| 历史版本片段 (仅供参数参考, **已废弃**) | 被删的 `launch/sglang_serve_dsv41_dspark_tp4.sh` 用 `--speculative-algorithm DSPARK --speculative-dspark-block-size 5 --cuda-graph-backend-prefill disabled --disable-flashinfer-autotune` | `git show 6fe1c01eac:launch/sglang_serve_dsv41_dspark_tp4.sh` |

已知问题 (RELEASE_NOTES v1.6.1 "Known Issues"): `reasoning_content` 在带 `tool_calls` 的投机解码响应里被丢弃; 仍有 3 个投机 kernel 在首个请求时 JIT。

---

## 4. SM90 (H100, capability 9.0) 支持

### 4.1 官方立场

- README 支持表: `DeepSeek-V4.1-Flash-1017` 行的 sm90 列标 **`ok`**, 而 sm86 标 **✅ (tested)**; 图例: "✅ tested · `ok` supported, not yet tested"
- 硬件前提: "x86 CPUs with AVX2+ and Nvidia GPUs with **sm80+**" (`README.md` Note 1)
- lk_moe METADATA: "Prerequisite: **CUDA 13.2.1** (driver + toolkit, **sm80+**)"
- RELEASE_NOTES v1.6.1 "Fixes" 提及异构 TP4 与 SM8x 的确定性工作, 未提 SM90 专项验证
- **全部带实测吞吐的 launch 脚本与 benchmark 都是 SM86(3090)/SM120(5060Ti), 无 SM90 实跑记录**

### 4.2 二进制层证据 (lk_moe 2.7.1 wheel)

| 证据 | 内容 |
| --- | --- |
| nvcc 编译参数 | `-arch sm_80`、`-arch sm_90`、`-arch sm_100`、`-arch sm_120` 均出现; 工具链 `release 13.2, V13.2.78.Build cuda_13.2.r13.2/compiler.37668154_0` |
| cub 模板实例化 | `cub38_V_300_200_SM_750_800_890_900_1000_1200` ← **含 900 = SM90** |
| 池内核源文件标记 | `moe_v2_expert_pool.cu`、`moe_v2_gpu_borrow.cu`、`moe_v2_gpu_forward_kernels.cu`、`moe_v2_cpu_decode.cu`、`moe_v2_gpu_prefill_cutlass.cu`、`moe_v2_gpu_metadata.cu`、`moe_v2_gpu_memory.cu` |
| 显式版本门 | 字符串 `GroupedGemmRunner: unsupported SM version %d` (存在运行时 SM 检查, 但未见 SM90 被排除的清单) |

### 4.3 fp8 权重 + fp4 专家的 MoE kernel 路径 (问题 1/2/3 的正式答案)

**选择链 (两条, 顺序执行)**

1) **模型 override 强制 backend** — `python/sglang/srt/arg_groups/model_overrides/deepseek_v4.py:71-89`:

```python
if cfg.moe_runner_backend == "auto":
    ...
    elif (cfg.device == "cuda" and not is_hip
          and cfg.moe_a2a_backend == "none"
          and not envs.SGLANG_DSV4_FP4_DEQUANT.get()
          and model_config.is_fp4_experts
          and (is_sm80_supported() or is_sm90 or is_sm100 or is_sm120)):
        overrides["moe_runner_backend"] = "flashinfer_mxfp4"
```

2) **量化方法分派** — `python/sglang/srt/layers/quantization/fp8.py:459-495` (`is_fp4_experts` 分支):

| 条件 | 返回 | file:line |
| --- | --- | --- |
| `get_moe_runner_backend().is_marlin()` | `Mxfp4MarlinMoEMethod` | `fp8.py:444-449` |
| `is_humming()` | `Mxfp4HummingMoEMethod` | `fp8.py:451-457` |
| `is_flashinfer_mxfp4()` 且 **SM8x** | `Mxfp4MarlinMoEMethod` | `fp8.py:468-478` |
| `is_flashinfer_mxfp4()` 且 **`is_sm90 or is_sm120`** | **`Mxfp4FlashinferCutlassMoEMethod`** | `fp8.py:481-490` |
| `is_flashinfer_mxfp4()` 其余 (SM100 等) | `Mxfp4FlashinferTrtllmMoEMethod` | `fp8.py:491-495` |
| 都不匹配 (`auto` 且非上述) | `Fp8MoEMethod` (纯 FP8) | `fp8.py:496` |

注释直接给出硬件映射 (`fp8.py:486`): **"SM100 uses TRT-LLM; SM90 uses W4A16 and SM120 uses MXFP8xMXFP4."**
SM8x 注释 (`fp8.py:472-477`): **"Marlin W4A16 is the only on-GPU MXFP4 kernel for major 8"** (FlashInfer FP4 模块只发 90/100/103/107/110/120 构建)。

**因此在 SM90 上, 默认答案是: FlashInfer CUTLASS MXFP4 MoE, W4A16 —— 不是 Marlin。** Marlin 只在 `--moe-runner-backend marlin` 显式指定, 或 SM8x 回退时被选中。

`mxfp4_flashinfer_cutlass_moe.py` 头部与类文档给出定论:

- `:1-4` docstring: "``Fp8Config`` selects this backend for **SM90 and SM120**; SM100 uses the TRT-LLM implementation."
- `:36` class docstring: "FlashInfer MXFP4 MoE: **W4A16/W4A8 on SM90** and W4A8 on SM120."

### 4.4 `--moe-runner-backend` 合法值全集

**CLI choices (`python/sglang/srt/arg_groups/choices.py:136-156`, 18 项)**:
`auto`, `deep_gemm`, `triton`, `triton_kernel`, `flashinfer_trtllm`, `experimental_sgl_trtllm`, `flashinfer_trtllm_routed`, `flashinfer_cutlass`, `flashinfer_mxfp4`, `flashinfer_cutedsl`, `flashinfer_megamoe`, `cutlass`, `aiter`, `marlin`, `humming`, `experimental_sgl_marlin`, `hpc_ops`, `megamoe`, `intel_xpu`

(其中 `hpc_ops` 的注释: "FP8 MoE on Hopper (**SM90**) only"; `megamoe` 在 CLI 表里, 而 Python 枚举 `layers/moe/utils.py:183-202` 里对应位置是 `ascend` —— **两处清单不完全一致**, 以 CLI 的 `MOE_RUNNER_BACKEND_CHOICES` 为准。)

参数定义: `arg_groups/fields/exec_.py:840-847`, `choices=MOE_RUNNER_BACKEND_CHOICES`, **默认 `"auto"`**。

SM90 可用性 (基于代码分派逻辑):

| backend | SM90 | 依据 |
| --- | --- | --- |
| `flashinfer_mxfp4` | **可用 (默认)** | `fp8.py:481-490` 显式收 SM90; 反证 `fp8.py:461-467` 只把 SM8x 排除 |
| `marlin` | 可用 (W4A16) | `fp8.py:444-449`, 无 arch 门 |
| `humming` | 可用 | `mxfp4_humming_moe.py:22-40`, 文件内无 arch 门; 依赖 `humming-kernels[cu13]==0.1.12` |
| `flashinfer_cutlass` | 可用 | 通用 FlashInfer CUTLASS 路径 |
| `deep_gemm` | 可用 | SM90 是 DeepGEMM 的原生目标 |
| `triton` / `triton_kernel` | 可用 | 通用 |
| `flashinfer_trtllm*` | 不被 MXFP4 分支选中 | `fp8.py:491-495` 只在非 SM90/SM120 走 TRT-LLM |
| `hpc_ops` | 注释写明 Hopper(SM90) only | `choices.py:153` |

### 4.5 SM90 是否有 MXFP4 × FP8 (W4A8) MoE —— **有**

不是新的 backend, 而是一个**精度开关**:

| 项 | 值 | file:line |
| --- | --- | --- |
| CLI | `--flashinfer-mxfp4-moe-precision` ∈ `{"default","bf16","fp8"}`, 默认 `"default"` | `arg_groups/fields/exec_.py:848-853` |
| help 原文 | "Choose the computation precision of flashinfer mxfp4 moe. **On SM90, `fp8` selects the Humming-style MXFP4-weight x FP8-activation path introduced by FlashInfer #3738 and requires FlashInfer >= 0.6.18.**" | 同上 |
| 生效逻辑 | `precision = get_exec().moe.flashinfer_mxfp4_moe_precision`; `self._use_sm90_humming = not is_sm120 and precision == "fp8"` | `mxfp4_flashinfer_cutlass_moe.py:44-48` |
| 标签输出 | `"W4A8" if self._use_sm90_humming or self._use_mxfp8_act_scaling else "W4A16"` | `mxfp4_flashinfer_cutlass_moe.py:164` |
| 依赖满足性 | 本仓库 pin `flashinfer_python[cu13]==0.7.2` ≥ 0.6.18 ✓ | `python/pyproject.toml` |

另有一条**独立**实现: `--moe-runner-backend humming` → `Mxfp4HummingMoEMethod` (`layers/quantization/mxfp4_humming_moe.py:22-40`), 文档原文: "Used for DeepSeek-V4 FP8 checkpoints when `--moe-runner-backend humming` is selected together with `SGLANG_DSV4_FP4_EXPERTS=1`"; 权重处理走 `prepare_humming_moe_layer` (`python/sglang/srt/layers/quantization/humming_utils.py`)。

### 4.6 池内专家的 GPU decode 跑在哪个 kernel

**答案: lk_moe 的闭源实现, 不是 sglang 开源代码。**

| 层 | 位置 | 内容 |
| --- | --- | --- |
| sglang 侧 (开源, 只做编排) | `layer.py:113-118` | `import lk_moe` + `from lk_moe._hybrid_gemv_kernel import mux_*...` (失败时 `logger.error("Failed to import lk_moe module or LK_MOE_HYBRID_ENABLED is not set to 1, ...")`) |
| sglang 侧 | `layer.py:2489-2587` `_pool_decode_launch()` | 在 `FusedMoE.pool_stream` 上 launch: `slot_table_fill`、`moe_gateup`、`reduce_gate`、`moe_down`、`reduce_down`、`gather_rows`; 槽地址/字节数来自 `geo["weight_addr"]`、`geo["scale_addr"]`、`geo["slot_bytes"]`、`geo["format"]` |
| sglang 侧 | `layer.py:2589-2620` `_cpu_decode()` | 判据 `real_tokens <= FusedMoE._pool_decode_cap` → 同时调 `_pool_decode_launch()` 与 `self.lk_moe.cpu_decode(...)`; 结束 `self.lk_moe.pool_record_aux(...)` 打 fence |
| **lk_moe wheel (Triton, 随包发布但同样属闭源)** | `lk_moe/_hybrid_gemv_kernel.py:1-11` | docstring: "**GPU-side GEMV for the super-expert pool**: computes the routed experts whose slot is resident+landed this step; the CPU side skips exactly those (same frozen snapshot word, see **csrc/cuda/moe_v2_expert_pool.cuh**)。... Quant formats / activations / slot layout: single source is **csrc/cuda/moe_v2_gpu_pool_kernels.cuh**. Addresses arrive as i64 (UVA > 2 GiB)." |
| 同上 | `lk_moe/_pool_kernel_common.py:1-6` | "MOE V2 - Expert-pool GPU-side shared decoders and activation, format-driven (mirrors moe_v2_activate.h / csrc/cuda/moe_v2_gpu_pool_kernels.cuh)" |
| 同上 | `lk_moe/prefill_resident.py:1-9` | "GPU side of a split prefill: **pool-resident expert slots via the group-GEMM kernels**, paired with cpu_prefill(pool_skip=True) over the misses" |
| 同上 | `lk_moe/prefill_staging.py:1-9` | staging ring 版 group-GEMM |
| **C++ 池状态机** | `_lk_moe_C*.so` | 类 `PoolPump`; 绑定 `pool_layer_info` / `pool_gpu_report` / `pool_gpu_live` / `pool_free/slot/ready_slot/snapshot_slot` / `pool_freeze_view` / `pool_verify_sample` / `pool_capture_pause` / `pool_capture_resume` / `pool_wait_fence` / `pool_wait_all_fences` / `pool_record_main` / `pool_record_aux` / `pool_set_det_split` |

**关键区分 (对"能否替换 W4A16"这一问题)**:

- **GPU 常驻层** (`is_gpu_resident_layer`) 走 `self.quant_method.apply(...)` (`layer.py:1730-1734`) → 即 §4.3 的 FlashInfer CUTLASS MXFP4 **W4A16/W4A8** 路径 (SM90)。
- **池内专家** 走 lk_moe 自己的 Triton GEMV/group-GEMM (`layer.py:2489-2587`), 格式由 `geo["format"]` 驱动。二者是**不同的 kernel**。
- 因此: 若要把池内专家也算成 W4A16, **必须在 lk_moe 内部改** (`moe_v2_gpu_pool_kernels.cuh` / wheel 内的 Triton kernel); sglang 侧无可替换的接缝。sglang 侧唯一能影响池内核的旋钮是 `format` 一类几何字段, 由 lk_moe 根据读到的量化格式自行决定。

**lk_moe 版本**: `python/pyproject.toml` → **`lk_moe==2.7.1`** (与 `RELEASE_NOTES.md:4` "lk_moe v2.7.1" 一致); wheel METADATA `Name: lk_moe / Version: 2.7.1 / License: Proprietary / Requires-Python: >=3.10,<3.14`。

**池对量化格式的支持面** (METADATA 原文): "Works for both **NVFP4 and MXFP4** experts." 本模型 `expert_dtype=fp4` 经 `model_hook.py:541-545` 注释确认为 **MXFP4 (E2M1)** 原生布局: "FP4 experts: keep the checkpoint's native **MXFP4 (E2M1)** layout and let LK MoE (**MOE_MXFP4**, dispatch<80>/<86>) dequant E2M1 -> bf16 internally during compute. This is the same path SM120 uses; `SGLANG_DSV4_FP4_DEQUANT` stays False so the weights are NOT dequantized to fp8 (which would route the layer to the FP8 LK MoE path instead)."

lk_moe 类选择 (`layer.py:2327-2360` `_process_mxfp4()`): `MOE_MXFP4` (bf16 激活) / `MOE_MXFP4_FP16`。`_do_process_weights_after_loading()` 的分派条件在 `layer.py:1978-2007`。

---

## 5. 安装方式与依赖钉、与本机环境的适配性

### 5.1 安装步骤 (README `:56-82`)

前置: **CUDA 13.2.1 (driver + toolkit)**, 与 `torch==2.14.1` 的 CUDA 构建匹配。

```bash
conda create -n Lsglang python==3.12.11 && conda activate Lsglang
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
sudo apt-get install libnuma-dev      # 或 sudo dnf install numactl-devel
pip install lsglang                   # 或源码构建
```

源码构建 (`README.md:69-82`):

```bash
git clone https://github.com/guqiong96/Lsglang.git && cd Lsglang
pip install -U setuptools wheel scikit-build-core cmake
pip install torchaudio triton torchvision torch==2.14.1
pip install grpcio-tools wheel-stub
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release \
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
pip install -e "python" --no-build-isolation -vvv
```

注意 (`README.md` Note 2): **首次启动很慢, 因为 JIT 编译** (会看到 `cicc`/`cc1plus`/`ptxas`/`nvcc` 进程); 第二次启动复用 JIT cache。

### 5.2 关键依赖钉 (`python/pyproject.toml`, `[project].dependencies`)

| 包 | 钉版 | 备注 |
| --- | --- | --- |
| 包名/版本 | `name = "lsglang"`, `version = "1.6.1"` | `requires-python = ">=3.11"` |
| `torch` | `==2.14.1` | build-system 里也钉 `torch==2.14.1` |
| `torchaudio` | `==2.11.0` | |
| `transformers` | `==5.19.0` | |
| **`lk_moe`** | **`==2.7.1`** | **Proprietary license** |
| `flashinfer_python[cu13]` | `==0.7.2` | 注释: "keep it aligned with jit-cache version in Dockerfile" |
| `sglang-kernel` | `==0.4.10` | 二进制 |
| `sgl-deep-gemm` | `==0.2.2` | |
| `sgl-deep-ep` | `==0.1.2.post1` | |
| `nvidia-cutlass-dsl[cu13]` | `==4.8.0` | |
| `humming-kernels[cu13]` | `==0.1.12` | 对应 §4.5 的 humming 路径 |
| `flash-attn-4` | `>=4.0.0b18` | |
| `cuda-python` | `>=13.0` | |
| `compressed-tensors` | `==0.18.0` | |
| `tilelang` | `==0.1.12` | |
| `nvidia-mathdx` | `==25.6.0` | |
| `nvshmem4py-cu13` | 未钉 | |
| `tokenspeed_mla` | `==0.1.8` | |
| `torch_memory_saver` | `>=0.0.9.post1` | |
| `numba` / `sentencepiece` / `tokenizers` / `xgrammar` 等 | `0.65.1` / `0.2.1` / `0.23.2` / `0.2.7` | |

**没有 `ktransformers`** 依赖 (全库未出现)。

### 5.3 与本机环境的适配性

| 本机事实 (实测) | 仓库要求 | 适配 |
| --- | --- | --- |
| 驱动 `595.45.04`; `nvidia-smi` 报 8×H100 80GB HBM3, `compute_cap = 9.0` | CUDA 13.2.x 驱动; sm80+ | **满足** (SM90 在支持面内, 但 README 标 "ok / 未实测") |
| `/usr/local/cuda-13.2/bin/nvcc` → `release 13.2, V13.2.51` (另有 `/usr/local/cuda-12.6`) | CUDA 13.2.1 toolkit | **满足** (13.2.51 vs 13.2.1: 需自行确认是否等价于文档要求的小版本) |
| 系统 `python3` = 3.10.12 | `requires-python >= 3.11`; README 用 3.12.11; `lk_moe` wheel 只有 **cp312** | **需新建 py3.12 环境**; 系统解释器不可用 |
| conda 可用: `/data3/users/wangyf/miniconda3`, `/home/wangyf/miniconda3` | conda 安装流程 | 可用 |
| 无 root | README 的 `sudo apt-get install libnuma-dev` / `sudo dnf install numactl-devel` **不可执行** | **风险**: `lk_moe.libs/libnuma-14905a75.so.1.0.0` 已随 wheel 打包 (自带 libnuma), 可能绕开系统包; 需实测确认 |
| pypi 走清华镜像 | 需要 `lsglang` / `lk_moe` / `sglang-kernel` / `flashinfer_python` 等 | **满足**: 已实测镜像上 `lsglang` 到 1.6.1、`lk_moe` 到 2.7.1、`sglang-kernel` 到 0.4.10 均可下载 |
| github 仅 ssh 可用 | `git clone ...` | 仓库已 ssh clone 成功 (HEAD `60f6658524`) |
| 192 核 / 2015 GiB RAM | `LK_THREADS = 物理核/GPU` | 满足; 但 `LK_THREAD_BINDING`/NUMA 拓扑需按实际核对 |
| `/dev/shm` | 未在本报告中探测 | 待核 (engram `shared` 布局要求 TP 组共享 PID namespace) |

**`lk_moe` 的 libnuma 说明**: wheel 内含 `lk_moe.libs/libnuma-14905a75.so.1.0.0` (`_lk_moe_C*.so` 运行时依赖), 这是"无 root 装 libnuma-dev"问题的一个可能缓解点, 但同时 `_init_host_table` 的 `shared` 布局用 `memfd` + `mmap(MAP_SHARED)`, 需要 TP 组各 rank 共享 PID namespace。

---

## 6. 启动入口与 4 卡 TP4

### 6.1 入口

| 形式 | 位置 |
| --- | --- |
| CLI 入口 | `sglang serve ...` (README 与全部 launch 脚本都用这个) |
| launch 脚本目录 | `launch/` (16 个脚本) |
| 上游 CLI | 与 sglang 一致; lk_moe 全部通过**环境变量**注入, 不用 CLI |
| lk_moe 侧 CLI | 只有 `--ple-offload-embedding` / `--ple-offload-backend {pinned,nvme}` 是 CLI |

`launch/` 现有脚本 (HEAD):

```
sglang_serve_0731_dspark_tp2_3090.sh   sglang_serve_dsv41_tp2_3090.sh
sglang_serve_0731_tp2_3090.sh          sglang_serve_dsv41_tp2_3090_nvme.sh
sglang_serve_dsv41_tp4.sh              sglang_serve_glm53_tp2_3090.sh
sglang_serve_glm53_tp4_3090.sh         sglang_serve_qwen36_mtp_tp2_3090.sh
sglang_serve_qwen36_mtp_tp2_5060ti.sh  sglang_serve_qwen36_tp2_3090.sh
sglang_serve_qwen38_mtp_tp2_3090.sh    sglang_serve_qwen38_mtp_tp2_5060ti.sh
sglang_serve_qwen38_mtp_tp4.sh         sglang_serve_qwen38_mtp_tp4_nvme.sh
sglang_serve_qwen38_tp2_3090.sh        sglang_serve_qwen38_tp2_5060ti.sh
```

### 6.2 当前最相关的模板: `launch/sglang_serve_dsv41_tp4.sh`

**这是 HEAD 里唯一现成的 DSV4.1 + TP4 脚本** (注释头写 "2× RTX 3090 (SM86), 2x RTX 5060Ti, TP4", 即异构 TP4 场景)。全文关键部分:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=NUMA_NODE \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_ENGINE=triton \
LK_GPU_PREFILL_SUB_M=8192 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_GPU_RESIDENT_MOE_LAYERS_SPEC="" \
LK_POOL_SLOTS=16 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
SGLANG_SKIP_P2P_CHECK=1 \
SGLANG_SM8X_MARLIN_GROUP32=1 \
SGLANG_SM8X_MARLIN_GROUP32_ONLY=1 \
FLASHINFER_USE_CUDA_NORM=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
sglang serve \
  --model ~/Models/DeepSeek-V4.1-Flash-1017 \
  --served-model-name DeepSeek-V4.1-Flash-1017 \
  --host 0.0.0.0 --port 8070 --trust-remote-code \
  --enable-decoder-swa-bounded-replay \
  --tensor-parallel-size 4 \
  --max-running-requests 2 \
  --reasoning-parser auto --tool-call-parser auto \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --chunked-prefill-size 1024 \
  --max-total-tokens 262144 \
  --mem-fraction-static 0.80 \
  --swa-full-tokens-ratio 0.1 \
  --speculative-attention-mode decode
```

**该模板与本目标的差异 (需改写)**:

| 模板值 | 原因 | 目标值 |
| --- | --- | --- |
| 日志落 `/tmp/opencode/serve_logs/...` | **违反"产物不得放 /tmp"** | 改为本仓库 logs 目录 |
| `LK_POOL_SLOTS=16` | 针对 24 GB 卡的保守值 | 目标 288 (需重算显存, 见 §2.3) |
| `--max-running-requests 2` | 3090/5060Ti 小显存 | 目标 16 |
| `--mem-fraction-static 0.80` | | 目标 0.88 |
| `--max-total-tokens 262144` | 256K, 非 1M | 1M 上下文需重算 KV 预算 |
| `CUDA_VISIBLE_DEVICES=0,1,2,3` | 裸 bus id, **本环境实测无效会静默回落** | 必须用 **UUID** 形式 (本机物理卡 4-7) |
| `SGLANG_SM8X_MARLIN_GROUP32*` | 纯 SM8x 优化 (`fp8.py:566-576`) | SM90 上无意义 |
| `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0` | 异构卡用 | 同构 4×H100 可去掉 |
| `--disable-custom-all-reduce` | 弱 P2P 主机用 | H100 同机 4 卡通常有 NVLink |

**4 卡 TP4 的关键待决点**:

1. **是否加 `--ep-size 4`** —— 决定每卡本地专家数 (384 vs 96), 进而决定 `LK_POOL_SLOTS=288` 是否有意义 (§2.4)。仓库自带的 DSV4.1 TP4 精度测试用的是 EP4 (`python/sglang/test/dsv41_flash_common.py:22`)。
2. **`LK_GPU_RESIDENT_MOE_LAYERS`** —— 模板给空串 (= 无 GPU 常驻层 = 全部 40 层都建池)。若给非空, 池成员数下降, 槽显存按层数线性下降。
3. **engram 放主机内存** 与 `LK_POOL_SLOTS` 同时开启是允许的 (互不排斥: 一个在 host, 一个在 VRAM), 但要算总账。
4. **卡固定方式** —— 本环境必须用 `nvidia-smi -L` 得到的 GPU UUID。
5. NVMe 变体模板见 `launch/sglang_serve_dsv41_tp2_3090_nvme.sh` (含 `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0` + `LK_ENGRAM_NVME=1`)。

---

## 7. 需求 (a)–(e) 逐条映射

| # | 需求 | Lsglang 对应参数 | 状态 | 证据 / 备注 |
| --- | --- | --- | --- | --- |
| **(a)** | `gpu_memory_utilization = 0.88` | `--mem-fraction-static 0.88` | **已有** | 定义 = `(model weights + KV cache pool) / GPU memory capacity` (`arg_groups/memory_hook.py:60-61`); 模板用 `0.80`。注意 lk_moe 的池 slab 由库在 MOE 构造期 `cudaMalloc`, 已从 free 显存扣除 (`kv_cache_configurator.py:217-246`) |
| **(b)** | 最大并发 16 | `--max-running-requests 16` (+ `--cuda-graph-max-bs*` 需配套) | **已有** | 模板用 `2`; 精度测试用 `32` (`python/sglang/test/dsv41_flash_common.py:40`)。注意投机解码会把它重置为 48, 除非显式指定 (`arg_groups/speculative_hook.py:441` 等 4 处) |
| **(c)** | 1M 上下文 | `--context-length 1048576` + `--max-total-tokens <按 KV 预算>` | **已有** | `max_position_embeddings=1048576`, rope `yarn` factor16/orig 65536 → 运行时 `rope_type="deepseek_yarn"` (`models/deepseek_v4.py:1467-1476`); KV dtype 默认 `fp8_e4m3` (`overrides.py:905-926`); **无 `fp8_ds_mla`** (vLLM 专有名字, 本仓库 0 命中); page_size 强制 256; `--swa-full-tokens-ratio` 需手工给 (dsv41 不自动设) |
| **(d)** | engram 表放主机内存 | `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`<br>+ `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared` (默认)<br>+ `SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN=1` (默认) | **已有** | 开关默认 `False` (`environ.py:1704`); 实现 `engram.py:641-676` `_HostTable` + `:928-962`; 两条 engram 表合计 **≈184.5 GiB** 主机内存 (258 B/行 × 768,022,850 行); pin 代价 "~40 s per 24 GiB of base pages"; 互斥替代路径 `LK_ENGRAM_NVME=1` (表完全不进内存) |
| **(e)** | **每层 GPU 侧专家缓存槽 = 288, 槽位可换入换出, 冷热动态判定** | `LK_MOE_HYBRID_ENABLED=1`<br>+ **`LK_POOL_SLOTS=288`**<br>+ `LK_POOL_DECODE=1` (默认跟随)<br>+ `LK_POOL_WINDOW=32` (热窗口)<br>+ `LK_GPU_RESIDENT_MOE_LAYERS=""` (让全部 40 层都建池)<br>+ **不加 `--ep-size`** (保持 ep_size=1, 每卡 384 本地专家) | **已有 (机制存在), 但显存与 EP 口径待验证** | 语义 = 每 MoE 层常驻专家槽数, 成员全局 (`README.md:209`, `lk_moe_env.py:281-292`, `layer.py:2046-2081`); 动态 = 后台 DMA pump + **2nd-touch window LRU** (lk_moe METADATA); 显存 = `slots × layers × 每槽字节` = 288 × 40 × 4.4824 MiB ≈ **50.4 GiB/卡**; 若改用 `--ep-size 4` 则每卡仅 96 本地专家, 288 超出层专家总数 → 无意义 (`fields/parallel.py:269-275`, `layer.py:431-454`) |

### 7.1 与 LvLLM 的事实差异 (只陈述差异, 不作优劣判断)

| 维度 | LvLLM (guqiong96/Lvllm, v2.7 手写前: vLLM + lk_moe) | Lsglang (guqiong96/Lsglang, sglang + lk_moe) |
| --- | --- | --- |
| 上游框架 | vLLM fork | sglang fork (上游 main `06342acad2`) |
| 版本命名 | `lvllm-v2.x` | `lsglang-v1.x` (当前 1.6.1) |
| lk_moe 钉版 | 旧分支钉 `lk_moe==2.4.2` | **`lk_moe==2.7.1`** |
| 环境变量前缀 | `LVLLM_*` (旧) | **`LK_*`** (统一; `LVLLM_*` 仅作自动改名兼容, `common.py:1557-1566`) |
| 专家驻留控制粒度 | 按**层**清单 (`LVLLM_GPU_RESIDENT_MOE_LAYERS` 语义在源码里与 env 名疑似相反, 需读码判定) | 三种角色并存: 按**层**常驻 (`LK_GPU_RESIDENT_MOE_LAYERS`) + 按**每层槽数**缓存 (`LK_POOL_SLOTS`) + 纯 CPU 层 |
| **每层 N 个专家缓存槽** | 本次评估范围内**未在本报告核实** | **有**: `LK_POOL_SLOTS` (每层槽数) |
| GPU prefill 引擎选择 | `LVLLM_GPU_PREFETCH_WINDOW` 等 | `LK_GPU_PREFILL_ENGINE=triton\|cutlass` + `LK_GPU_PREFILL_MIN_TOKENS` + `LK_GPU_PREFILL_SUB_M` (Super Mix Prefill) |
| NVMe 权重 offload | 本报告未核实 | 有 (Engram / PLE), `LK_ENGRAM_NVME` / `--ple-offload-backend nvme` |
| 释放说明里的已知缺口 | — | RELEASE_NOTES v1.6.1 Known Issues 原文: "**lvllm decode GPU half is not yet ported**" |
| 预编译 wheel | `lvllm-2.5.2-cp312-cp312-manylinux_2_34_x86_64.whl` (含二进制) | `lsglang-1.6.1-py3-none-any.whl` (**纯 Python**, 库主体靠 `lk_moe` / `sglang-kernel` / `flashinfer` 等二进制依赖) |
| 源码安装前置 | 需要 GCC >= 13 | 需要 CUDA 13.2.1 + `torch==2.14.1` + Python 3.12 |
| 投机解码 | dspark | dspark (`SpeculativeAlgorithm.DSPARK`); 但 dsv41 的 dspark 启动脚本在 `76bd58545f` 被删除 |
| 与 engram 主机表的关系 | 同一 lk_moe 生态 | 同一 lk_moe 生态, `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` |

---

## 8. 风险清单 (分级 + 验证方式)

| 级别 | 风险 | 依据 | 验证方式 |
| --- | --- | --- | --- |
| **高** | **SM90 未实测**。README 支持表里 `DeepSeek-V4.1-Flash-1017` 的 sm90 只标 `ok`, 唯一 ✅ 是 sm86; 全部 launch 脚本/benchmark 都是 SM86/SM120 | `README.md` 支持表 + 图例 | 首启看 lk_moe 加载行与 `[pool]` 行; 小批 prefill + decode 数值比对 |
| **高** | **`LK_POOL_SLOTS=288` 与 `--ep-size` 口径冲突**: 默认 `ep_size=1` 时每卡 384 本地专家 (288 可行); 仓库自带的 DSV4.1 TP4 精度测试用 `--ep-size 4` → 每卡 96 专家 (288 不可行) | `fields/parallel.py:269-275`; `layer.py:431-454`; `python/sglang/test/dsv41_flash_common.py:22`; `mxfp4_flashinfer_cutlass_moe.py:71-79` | 启动日志里 `Initialized lk_moe with <N> experts for layer <L>` (`layer.py:2011`, `:2022`) 会打印每层本地专家数 |
| **高** | **288 槽 × 40 层 ≈ 50.4 GiB/卡** 专供池槽, 剩余 ≈20 GiB/卡 要放注意力权重 + 1M 上下文 KV(fp8_e4m3) + 激活 + CUDA graph; 相容性未经验证 | §2.3 算式 + lk_moe METADATA VRAM 公式 + `kv_cache_configurator.py:217-246` | 首启看 `[pool] slab: X GiB (40 layers x Y MiB)` 与 `--mem-fraction-static` 下的 KV 报告行 |
| **高** | **池内专家的 GPU decode kernel 不可替换** —— 实现于 lk_moe (闭源 wheel 的 Triton kernel + `_lk_moe_C*.so`), sglang 侧只做 launch; 池内专家**不走** FlashInfer W4A16 | `layer.py:113-118`, `:2489-2587`; `_hybrid_gemv_kernel.py:1-11` | `LK_POOL_LOG=100` 观察 `[pool]` 行; 代码层面已确证 |
| **中** | **lk_moe 为专有许可闭源** (`License: Proprietary`), 池的关键路径 (slot 布局、LRU、pump) 无法审计或修改 | wheel METADATA | 法务/许可确认 |
| **中** | **无 root**: README 要求 `sudo apt-get install libnuma-dev`。wheel 内自带 `lk_moe.libs/libnuma-*.so.1.0.0` 可能绕开, 但未验证 | `README.md:79-80`; wheel 文件列表 | 建环境时 `import lk_moe` 并观察 NUMA 相关启动日志 |
| **中** | **CUDA 小版本**: 文档要求 `CUDA 13.2.1`, 本机 nvcc 为 `13.2 V13.2.51`, lk_moe wheel 由 `13.2 V13.2.78` 编译 | `nvcc --version`; wheel 字符串 | 首启确认无 `cudaErrorCallRequiresNewerDriver` / JIT 失败 |
| **中** | **首次启动 JIT 编译很慢** (明确写在 README Note 2) | `README.md` Note 2 | 记录首启耗时; 二次启动对比 |
| **中** | engram host table **pin 代价**: "~40 s per 24 GiB" → 184.5 GiB 约 5 分钟额外启动时间; `shared` 布局要求 TP 组各 rank 共享 PID namespace | `environ.py:1706-1715` | 首启计时 |
| **中** | **已知质量问题** (RELEASE_NOTES v1.6.1 Known Issues): `DeepSeek-V4.1 det512 dynamic split is not yet reproducible`; `lvllm decode GPU half is not yet ported`; `reasoning_content` 在带 `tool_calls` 的投机响应里被丢弃; 3 个投机 kernel 首请求 JIT | `RELEASE_NOTES.md` Known Issues | 按其描述针对性复现 |
| **中** | `LK_POOL_SLOTS` **无上限校验**: 若误设超过每层本地专家数, 只会白白多占显存 (未见 clamp) | `lk_moe_env.py:281-285`; `.so` 字符串中无 clamp 提示 | 首启 `[pool] slab:` 行数字对齐 |
| **低** | 依赖可获得性: 清华镜像已实测有 `lsglang`(到 1.6.1) / `lk_moe`(到 2.7.1) / `sglang-kernel`(到 0.4.10) | 镜像 index 抓取 | 装机时 `pip download` 预检 |
| **低** | `--swa-full-tokens-ratio` 在 dsv41 下**不会自动设** (条件显式排除 `deepseek_v41`) | `model_overrides/deepseek_v4.py:44-50` | 启动日志 / 参数显式给值 |
| **低** | 文档与代码的**默认值漂移**: lk_moe wheel METADATA 写 `LK_GPU_PREFILL_ENGINE` 默认 `cutlass`, 而 HEAD 代码默认 `triton` (HOTFIX 2026-10-22) | METADATA vs `lk_moe_env.py:125` vs `git log -1` | 以仓库代码为准 |

---

## 9. 证据复现命令

```bash
# --- git 元数据 ---
cd /mnt/SSD01/models/.llm-env/Lsglang/lsglang
git log -1 --format='%H%n%ci%n%s'; git describe --tags; git tag --sort=-creatordate | head
git for-each-ref --sort=-creatordate --format='%(refname:short) %(creatordate:short)' refs/tags | head -12

# --- README / RELEASE_NOTES ---
sed -n '1,25p;27,52p;141,246p;289,300p' README.md
sed -n '1,10p' RELEASE_NOTES.md

# --- LK_* 全清单真源 ---
sed -n '1,305p' python/sglang/srt/utils/lk_moe_env.py
sed -n '1515,1570p' python/sglang/srt/utils/common.py

# --- 池配置构造 (per-layer, 全局成员) ---
sed -n '2046,2112p' python/sglang/srt/layers/moe/fused_moe_triton/layer.py
sed -n '2417,2500p' python/sglang/srt/layers/moe/fused_moe_triton/layer.py
sed -n '2489,2587p' python/sglang/srt/layers/moe/fused_moe_triton/layer.py

# --- SM90 fp4 专家 MoE 后端选择 ---
sed -n '68,90p' python/sglang/srt/arg_groups/model_overrides/deepseek_v4.py
sed -n '420,497p' python/sglang/srt/layers/quantization/fp8.py
sed -n '1,50p;164,166p' python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py
sed -n '136,165p' python/sglang/srt/arg_groups/choices.py
sed -n '840,857p' python/sglang/srt/arg_groups/fields/exec_.py

# --- engram 主机表 ---
sed -n '1700,1722p' python/sglang/srt/environ.py
sed -n '641,676p;860,965p' python/sglang/srt/layers/engram.py
sed -n '1,18p' python/sglang/srt/layers/engram_nvme.py

# --- KV dtype / attention backend / rope ---
sed -n '905,927p' python/sglang/srt/arg_groups/overrides.py
sed -n '28,42p' python/sglang/srt/arg_groups/deepseek_v4_hook.py
sed -n '1460,1480p' python/sglang/srt/models/deepseek_v4.py

# --- ep_size 默认 / 专家切分 ---
sed -n '266,276p' python/sglang/srt/arg_groups/fields/parallel.py
sed -n '417,455p' python/sglang/srt/layers/moe/fused_moe_triton/layer.py
sed -n '146,158p' python/sglang/srt/runtime_context.py

# --- lk_moe wheel 取源 (只解包不安装) ---
curl -s https://pypi.tuna.tsinghua.edu.cn/simple/lk-moe/ -o idx.html
# 取 lk_moe-2.7.1-cp312-cp312-manylinux_2_34_x86_64.whl 的 href → urljoin → curl
unzip -q lk_moe-2.7.1.whl -d whl271
sed -n '1,128p' whl271/lk_moe-2.7.1.dist-info/METADATA
sed -n '1,11p'  whl271/lk_moe/_hybrid_gemv_kernel.py
sed -n '1,6p'   whl271/lk_moe/_pool_kernel_common.py
# 二进制导出字符串 (池 slab / 命中统计 / (layer,expert) 槽索引 / arch)
python3 - <<'EOF'
import re,glob
p=glob.glob('whl271/lk_moe/_lk_moe_C*.so')[0]; d=open(p,'rb').read()
for pat in [b'sm_90', b'pool_member_layers', b'slot_count', b'PoolPump']:
    print(pat, len(re.findall(pat, d)))
EOF

# --- 模型几何 ---
python3 -c "import json;c=json.load(open('/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017/config.json'));print(c['text_config'])"
python3 -c "
import json,re,collections
idx=json.load(open('/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017/model.safetensors.index.json'))['weight_map']
lay=collections.defaultdict(list)
for k in idx:
    m=re.match(r'layers\.(\d+)\.(.+)',k)
    if m: lay[int(m.group(1))].append(m.group(2))
print('layers',len(lay),'MoE',sum(1 for l in lay if any(x.startswith('ffn.experts') for x in lay[l])))"
```

---

## 10. 未核实项 (明确列出)

1. `LK_POOL_SLOTS` 超过每层本地专家数时的**运行时行为** (报错 / 截断 / 纯浪费显存) —— 代码与字符串中未见 clamp, 但 lk_moe 池实现闭源, 未做实验验证。
2. **池内专家的 slot 布局与 `[pf]`/`[pool]` 实测命中率** —— 需要实跑才能得到; README 只在 §2.2 的注释里给出窗口默认值对应的 "hit ~81%", 未说明硬件条件。
3. `--ep-size` 与 `LK_POOL_SLOTS` 在**同一 rank 上的乘性显存关系**是否另有额外开销 (e.g. 池 slab 的对齐/stride 填充) —— 需首启日志 `[pool] slab: %.2f GiB (%d layers x %.2f MiB)` 才能定值。
4. `/dev/shm` 容量与 `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared` 的 PID namespace 前提是否在本机满足。
5. `CUDA 13.2.51` (本机) 与文档要求的 `13.2.1` 是否等价。
6. 无 root 条件下 `libnuma-dev` 的缺失是否真的被 wheel 内 `lk_moe.libs/libnuma-*.so` 完全覆盖。
7. LvLLM 侧的对比数据 (本报告只列了从两侧仓库头/文档可直接读到的差异, 未复核 LvLLM 的每层槽能力)。
