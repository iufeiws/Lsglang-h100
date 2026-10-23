# Lsglang 显存 / KV 预算核算 — DeepSeek-V4.1-Flash-1017 on 4×H100 80GB (TP4)

- 仓库: `/mnt/SSD01/models/.llm-env/Lsglang/lsglang` (HEAD `60f6658524`, tag 描述 `lsglang-v1.6.1-2-g60f6658524`)
- lk_moe 版本: `python/pyproject.toml:82` → `lk_moe==2.7.1` (已实测导入该 wheel 并与 PyPI 索引 sha256 一致: `7c9c5c13dbf909053014f921037efc110d1dac351d521ee5863c070d507c54d5`)
- 模型: `/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017` (40 主层 + 3 MTP/dspark 层, 384 专家/层, `expert_dtype=fp4`, `compress_ratios=[0,0,2×18,1×19,0,0,0,0]`, `kv_source_layer_ids=[2,8,14,20]`, `index_source_layer_ids=[2,8,14,20,24,28,32,36]`, `sliding_window=128`, `engram_layer_ids=[1,14]`, `max_position_embeddings=1048576`)
- 硬件: 4×H100 80GB HBM3 (81559 MiB/卡, SM90), 192 核, RAM 2015 GiB; 物理卡 4-7 位于 NUMA node1 (node1 free 411410 MB = 401.77 GiB, node0 free 721858 MB = 704.94 GiB)
- 算式脚本与证据: `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/budget_math.py`
- 约束遵守: 只读分析, 未安装环境、未占用 GPU; 未读取 `deprecated/` 及任何 `*_DEPRECATED*` 路径。

---

## 0. 结论速览

| # | 结论 | 关键数字 |
|---|---|---|
| 1 | `LK_POOL_SLOTS=288` 的池 slab = **50.43 GiB/卡** (201.71 GiB/4 卡); 有效槽数上限 = 每卡本地专家数 (ep_size=1 时 384) | §1, §8.5 |
| 2 | `--mem-fraction-static 0.88` 预算 = **70.090 GiB/卡** (280.359 GiB/4 卡); 扣掉 288 槽 + 其余权重后**留给 KV 16.68 GiB/卡** | §2 |
| 3 | **修正后 (§8): 288 槽 + 4×1M 可行**, 288 槽 + 16×1M 不可行 | §8.4: `S_max(4×1M)=344`; `S_max(16×1M)=232` |
| 4 | host 必须常驻 `275.77 (全部专家) + 188.83 (engram) = 464.60 GiB = 498.86 GB` > node1 的 401.77 GiB → **必须 `LK_NUMA_INTERLEAVE=1`** | §4 |
| 5 | 推荐: `LK_POOL_SLOTS=288` + `--max-running-requests 4` + 1M + **不传 `--swa-full-tokens-ratio`** | §5, §8.3 |
| 6 | 每 full-token KV = **1630 B** (默认 cap 模式) = 1630 + 23400·r (显式传 r); 真正落全量 KV 的只有 4 层 | §8.1, §8.2 |
| 7 | CUDA graph 捕获显存吃的是 `(1−0.88)×79.647 = 9.558 GiB` 余量, **不额外扣 KV**; 超了直接 OOM (本模型无 post-capture 重算) | §9 |

---

## 1. `LK_POOL_SLOTS=288` 的显存代价 (口径 + 公式 + 代入)

### 1.1 口径: 每卡还是全模型

| 项 | 事实 | 证据 |
|---|---|---|
| MoE 权重按 TP 切分 | `self.intermediate_size_per_partition = intermediate_size // self.moe_tp_size`; 本模型 `moe_intermediate_size=2304`, TP4 → **576** | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:453-454`; 模型 `config.json:26` |
| 每卡专家**个数**不变 | 无 EP 时 `self.num_local_experts = self.num_experts` = **384** | `layer.py:441-443, 691-692` |
| 池 slab 是**每 rank 各自分配** | `cfg.pool_slots = get_pool_slots()` 与 `cfg.intermediate_size = self.intermediate_size_per_partition` 一起进了每个 rank 的 `MOEConfigV2`; lk_moe 按该 cfg 分配 | `layer.py:2046-2109` (2081, 2093), `python/sglang/srt/utils/lk_moe_env.py:281-285` |
| 池成员层 = 非驻留的**主干 MoE 层** | `cfg.pool_member_layers = sum(1 for i in range(n_layers) if not is_lk_moe_gpu_resident_layer(i))`, `n_layers = text_cfg.num_hidden_layers` = **40** | `layer.py:2061-2072` |
| MTP/dspark 层**不进池** | `is_lk_pool_layer()`: `if model_type != "main": return False`; `get_model_type_from_layer_name`: `stages.` → dspark, `mtp.`/`.mtp.` → mtp | `python/sglang/srt/utils/lk_moe_env.py:287-292, 187-195` |
| KV 池**不按 TP 切分** (每卡一份) | DSV4 KV buffer 形状 `[num_pages, page_bytes]`, 全文件无 `tp_size` 除法 | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:279-311`; `grep tp_size` 该文件 0 命中 |

**因此本报告的显存数字一律以「每卡」为准** (每卡 80 GiB × 0.88 是判据), 同时给出 4 卡合计。

### 1.2 单专家字节数 (从 checkpoint 头部实测)

| 张量 | 全 rank 形状 (packed) | 全 rank 字节 | 每卡字节 (intermediate/4) |
|---|---|---|---|
| `w1.weight` (gate) | `[2304, 2560]` I8 | 5,898,240 | `576×2560 = 1,474,560` |
| `w3.weight` (up) | `[2304, 2560]` I8 | 5,898,240 | `576×2560 = 1,474,560` |
| `w2.weight` (down) | `[5120, 1152]` I8 | 5,898,240 | `5120×288 = 1,474,560` |
| `w1/w3.scale` | `[2304, 160]` E8M0 | 368,640 ×2 | `576×160 = 92,160` ×2 |
| `w2.scale` | `[5120, 72]` E8M0 | 368,640 | `5120×18 = 92,160` |
| **合计** | | **18,800,640 B = 17.930 MiB** | **4,700,160 B = 4.482 MiB** |

证据: `layers.0.ffn.experts.0.*` 头部实测 (脚本 `_evidence/analyze_weights2.py` 的输出; 384 专家 × 18,800,640 B = 7,219,445,760 B = 一个层的专家总量, 与 `layers.0` 的 header 汇总一致)。中间维对齐: `Mxfp4MoEMethod` 另有 round-up 逻辑, 可能对尾列补齐 (见 `layer.py:478-489` 注释) → **±5% 级别的布局误差待环境确证**。

### 1.3 池 slab 公式与代入

```
slab_bytes(每卡) = pool_slots × pool_member_layers × slot_bytes(每卡)
                 = LK_POOL_SLOTS × (非驻留主干 MoE 层数) × 4,700,160 B
```

lk_moe 侧自报格式印证: `"[pool] slab: %.2f GiB (%d layers x %.2f MiB)"` (lk_moe 2.7.1 二进制字符串)。

| `LK_POOL_SLOTS` | 40 层全在池 (每卡) | 4 卡合计 |
|---|---|---|
| 16 (仓库示例脚本值) | 2.80 GiB | 11.21 GiB |
| 96 | 16.81 GiB | 67.24 GiB |
| 128 | 22.41 GiB | 89.65 GiB |
| 192 | 33.62 GiB | 134.47 GiB |
| 224 | 39.22 GiB | 156.88 GiB |
| **288** | **50.43 GiB** | **201.71 GiB** |
| 384 (=全常驻当量) | 67.24 GiB | 268.95 GiB |

单槽成本: `4,700,160 B × 40 层 = 188,006,400 B = 0.1751 GiB/卡`。

**层口径**: 只对**主干 40 个 MoE 层**收费; MTP/dspark 的 3 层 (128 专家/层) 与任意非 MoE 层 (attention/dense/engram) 不计入池 slab。若把 dspark 也做成驻留 (走 `LK_GPU_RESIDENT_MOE_LAYERS_DSPARK`/`_MTP` 或 `_SPEC`), 那是另一笔账 (§2.3)。

---

## 2. 4×H100 80GB + `--mem-fraction-static 0.88` 的可用预算

### 2.1 预算公式 (源码口径)

```
available_for_KV(每卡) = free_at_pool_init − pre_model_load_memory × (1 − mem_fraction_static) − mm_reservation − lk_moe_reservation
                        = 总显存 × mem_fraction_static − 已常驻的静态量 (权重 + lk_moe slab + …)
```
证据: `python/sglang/srt/mem_cache/kv_cache_configurator.py:2700-2770`; lk_moe 专项抵扣 `_lk_moe_device_reservation_gb()` (`kv_cache_configurator.py:218-240`, 取 `lk_moe.get_device_mem_usage()`)。

| 项 | 每卡 | 4 卡 |
|---|---|---|
| 设备总显存 | `81559 MiB = 79.647 GiB` | 318.588 GiB |
| `--mem-fraction-static 0.88` | **70.090 GiB** | **280.359 GiB** |
| (参考) 默认值 0.9 | 71.682 GiB | 286.729 GiB |

### 2.2 权重账 (checkpoint 头部实测 → 每卡)

| 项 | 全模型 | 每卡 (TP4) | 依据 |
|---|---|---|---|
| 全 checkpoint | 475.241 GiB | — | `index.json` `total_size = 510,286,023,000` |
| Engram 两表 (host pinned) | 188.833 GiB | **0 (host)** | `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` |
| 主干 40 层专家 (384/层) | 268.945 GiB | 67.236 GiB | 由 `LK_POOL_SLOTS` 决定实际落在 GPU 的部分 |
| MTP/dspark 专家 (3×128) | 6.823 GiB | 1.706 GiB | 默认非驻留 → 在 host |
| attention / dense / shared / norm / embed / head / aligner / vision / dspark head | 10.346 GiB | 2.587 GiB | TP 分片; 复制型小张量另计 ≈0.40 GiB |
| **固定项 `W_fix` (非专家, 每卡)** | | **2.987 GiB** | 2.587 + 0.40 |
| dspark 专家若驻留 | | +1.706 GiB | 可选 |

### 2.3 变体 P (纯池: `LK_POOL_SLOTS=288`, 40 层全在池)

| 项 | 每卡 GiB | 说明 |
|---|---|---|
| 池 slab (288×40) | 50.43 | §1.3 |
| `W_fix` | 2.987 | §2.2 |
| **权重合计** | **53.41** | |
| **留给 KV** | **16.68** | `70.090 − 53.41` |
| 池内专家 GPU 侧算力路径 | **lk_moe 闭源内核 (MXFP4, 等价 W4A16)** | `layer.py:2046-2109` 只做编排; 池内解码在 lk_moe 内 (`_pool_decode_launch`, `layer.py:2489+`) |

---

## 3. KV 账: 每 token 字节数与 1M × 16 的相容性

### 3.1 每 token 每层字节 (SM90 路径)

| 常量 | 值 | 证据 |
|---|---|---|
| 主干 KV 行 (V4 FlashMLA 布局: 448 fp8 NoPE + 64 bf16 RoPE + 8 scale) | **584 B/token/层** | `kv_layout.py:17,26-56` (`V4: data 576 + scale 8`); `deepseek_v4_memory_pool.py:288-303` 断言 `448 + 64*2 + 8` |
| 页对齐 (page 128, 对齐 576) | `ceil(128×584/576)×576/128 = 585.0 B/token` | `kv_layout.py:72-74`; `pool_configurator.py:1581-1583` (`page_size % 128 == 0`) |
| V4.1 fp4 压缩布局 (V41/V41_FP4) | **SM90 不可用** → 走 V4 fp8 | `deepseek_v4_memory_pool.py:199-221` (`mode=="auto" and not is_sm100` → `KVLayout.V4`) |
| 低比压缩层的 indexer (fp4) | **68 B** = `128//2 + 128//32` | `deepseek_v4_memory_pool.py:52-56`; `pool_configurator.py:1143-1147` (低比池 `force_fp4=True`) |
| 压缩层归属 | ratio-2 owner = 层 {2,8,14} (3 层); ratio-1 owner = 层 {20} (1 层) | `deepseek_v4_memory_pool.py:93-114` + 模型 `config.json:60-111` |
| SWA 槽成本 | `585 × 40 = 23,400 B` per SWA slot | `pool_configurator.py:1357-1361` (`bytes_per_swa_token = paged_kv_bytes × num_layers`; ratio-4 项为 0) |
| SWA 比例 | `--swa-full-tokens-ratio`, 默认 **0.1** | `pool_configurator.py:1021,1105-1110` |

```
bytes_per_full_token = Σ_ratio ( (584 + 68) / ratio × owner_layers(ratio) )  +  swa_full_tokens_ratio × 23,400
                     = 3 × 326  +  1 × 652  +  swa_ratio × 23,400
                     = 1630 B  +  swa_ratio × 23,400 B
```

| `--swa-full-tokens-ratio` | B/full-token | 1×1M | 4×1M | 16×1M |
|---|---|---|---|---|
| 0.1 (默认/示例脚本) | 3970 | 3.88 GiB | 15.51 GiB | **62.03 GiB** |
| 0.02 | 2098 | 2.05 GiB | 8.20 GiB | 32.78 GiB |
| 0.005 | 1747 | 1.71 GiB | 6.82 GiB | 27.30 GiB |
| 0.0025 | 1688 | 1.65 GiB | 6.60 GiB | 26.38 GiB |
| 0 (理论下限) | 1630 | 1.59 GiB | 6.37 GiB | 25.47 GiB |

> SWA 池只需覆盖"并发窗口 + 前缀尾": `slots × (window + page)`; 16 并发 + 默认 64 条 prefix tail ≈ `(16+64)×(128+128) = 20,480` 槽位 → 对 16.78M token 只需 ratio ≈ `0.0013`。因此 **swa ratio 是可压到 0.005 量级的旋钮**, 且 SGLang 在 cap 模式 (`pool_configurator.py:1301-1324, 1397-1401`) 会自动按请求上限 + headroom 计算 (`swa_cap_tokens`) 而不按比例放大。

### 3.2 单条 1M / 16 条 1M 的绝对占用

- 单条 1M: **1.59 – 3.88 GiB/卡** (随 swa ratio)
- 16 条各 1M: **25.47 – 62.03 GiB/卡** (随 swa ratio)
- KV 池每卡独立 (`§1.1`) → 4 卡合计 = ×4 (如 16×1M @0.1: 248.1 GiB)

### 3.3 池路径下的 KV 预算 (变体 P)

| swa ratio | 权重/卡 | KV 预算/卡 | 可支撑总 token | 等价组合 |
|---|---|---|---|---|
| 0.1 | 53.41 GiB | 16.68 GiB | 4.51 M | **4 × 1M** 或 16 × 282K |
| 0.005 | 53.41 GiB | 16.68 GiB | 10.25 M | **9 × 1M** 或 16 × 640K (10×1M = 17.06 GiB 已超预算) |

### 3.4 可行 / 不可行组合 (每卡 70.090 GiB) 〔口径已被 §8 取代: 见 §8.3/§8.4〕

| # | 组合 (并发 × 上下文) | 需要 KV/卡 | 权重/卡 (变体 P) | 合计 | 判定 | 算式 |
|---|---|---|---|---|---|---|
| ✅ 1 | 4 × 1M, swa=0.1 | 15.51 GiB | 53.41 GiB | 68.92 GiB | 可行 (余 1.17) | `4×1048576×3970/2³⁰` |
| ✅ 2 | 9 × 1M, swa=0.005 | 15.35 GiB | 53.41 GiB | 68.76 GiB | 可行 (余 1.33) | `9×1048576×1747/2³⁰` (10×1M = 17.06 GiB 已超 16.68 → 取 9) |
| ✅ 3 | 16 × 600K, swa=0.005 | 15.62 GiB | 53.41 GiB | 69.03 GiB | 可行 (余 1.06) | `16×600000×1747/2³⁰` |
| ✅ 4 | 16 × 1M, swa=0.005, **S=227** | 27.30 GiB | 42.74 GiB | 70.04 GiB | 可行 | `227×0.1751 + 2.987 + 27.30` |
| ❌ 1 | **16 × 1M + S=288** | 27.30 GiB (swa 0.005) 或 62.03 (0.1) | 53.41 GiB | **80.71 / 115.44 GiB** | **不可行**, 超出 10.6 / 45.4 GiB | `53.41 + 27.30 = 80.71 > 70.09` |

**288 槽与 16×1M 的缺口算式** (swa=0.005, 最有利情形): `70.090 − 2.987 − 27.30 = 39.80 GiB` 可供 slab → `S_max = 39.80 / 0.1751 = 227.3` → **S ≤ 227 槽/层**; 若保持示例脚本的 swa=0.1, `S_max = (70.090 − 2.987 − 62.03)/0.1751 = 29.0` → **S ≤ 29 槽/层**。

---

## 4. 主机内存账 (NUMA)

| 项 | GiB | GB | 说明 |
|---|---|---|---|
| 全部主干专家 (40 层 × 384) | 268.95 | 288.77 | 池只把 S/384 的一部分放 GPU, **全量仍在 host** |
| MTP/dspark 专家 (3×128) | 6.82 | 7.33 | 默认非驻留 → host |
| **专家小计** | **275.77** | **296.10** | |
| Engram 两表 (`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`) | 188.83 | 202.75 | 层 1 / 层 14 的 `engram.embed.weight+scale` |
| **必须常驻 host 合计** | **464.60** | **498.86** | |

| NUMA 节点 | 空闲 | 是否够放 498.86 GB |
|---|---|---|
| node1 (GPU 4-7 所在) | 411410 MB = **401.77 GiB / 411.41 GB** | **不够** (差 ≈ 87 GB) |
| node0 | 721858 MB = 704.94 GiB / 721.86 GB | 单独够 |
| node0+node1 | 1106.71 GiB / 1133.27 GB | 够 (余 634 GB, 其中可容纳 475 GiB checkpoint 页缓存) |

**结论**: 必须 `LK_NUMA_INTERLEAVE=1` (默认即为 `true`, `lk_moe_env.py:103-105`; 语义 = 子进程 `numactl --interleave=all`), 让 host 常驻项跨两个节点分布; 仅靠 node1 会在加载阶段 OOM 或把 checkpoint 页缓存挤到反复回读 NFS 的境地。

若走**变体 S (k=23 层驻留)**: host 专家降到 `17/40 × 268.95 = 114.30 GiB`, 加 engram = **303.13 GiB (325.6 GB)**, node1 单节点可容纳 (401.77 GiB) — 此时 interleave 变为"更稳"而非"必须"。

---

## A. 口径确认: 每卡 vs 全模型 (决定结论方向)

| 对象 | TP4 下的切分 | 每卡占用 | 全模型占用 | 证据 |
|---|---|---|---|---|
| MoE 专家权重 (池 slab / 驻留) | intermediate 维 `// tp_size` (=576), 专家个数仍 384 | **1/4** | 4 份之和 | `layer.py:453-454, 441-443` |
| 非专家权重 (attn/dense/embed…) | 标准 TP 分片 | ≈1/4 | — | §2.2 |
| **KV 池** | **不切分** (MLA latent 每卡一整份) | **1×** | ×4 冗余 | `deepseek_v4_memory_pool.py:279-311` (无 `tp_size` 除法) |
| Engram host 表 | TP(=ETP) 分片 | 1/4 (47.21 GiB) | 188.83 GiB | 表按 TP 分片 (`config.json:135-138` 表规模 × 分片) |

**判据**: 每卡 `80 GiB × 0.88 = 70.090 GiB`。由于 KV 每卡一整份, 而专家权重每卡 1/4, 结论是:**288 槽 "装得下"(50.43 < 70.09), 但装完就没有 KV 空间了** —— 不是"很宽裕", 而是"正好把 KV 挤到只剩 16.68 GiB"。

---

## B. 变体 P (纯池, 40 层全在池, `LK_POOL_SLOTS=288`) 〔KV 口径已被 §8 修正, 结论见 §8.4〕

| 指标 | 每卡 | 4 卡 |
|---|---|---|
| 池 slab | 50.43 GiB | 201.71 GiB |
| `W_fix` | 2.987 GiB | 11.95 GiB |
| **权重合计** | **53.41 GiB** | 213.66 GiB |
| **KV 预算** | **16.68 GiB** | 66.73 GiB |
| 专家 GPU 侧内核 | **lk_moe 闭源池内核 (MXFP4/W4A16 等价)**; sglang 只做编排 | — |
| 可支撑组合 (swa=0.1) | 4 × 1M; 或 16 × 282K; 或 45 × 100K | 同 (KV 每卡独立) |
| 可支撑组合 (swa=0.005) | 9 × 1M; 或 16 × 640K; 或 102 × 100K | 同 |

---

## C. 变体 R(k): k 层 GPU-resident + (40−k) 层池 (每层 288 槽) 〔resident 成本已被 §8.6 修正为 1.868 GiB/卡/层〕

驻留层成本 = **全部 384 个专家** = `1.706 GiB/卡/层` (TP4: `384×4,700,160 = 1.80 GB = 1.681 GiB`);
池层成本 = `288 × 4,700,160 = 1.353 GB = 1.261 GiB/卡/层`。

```
W(k) = 2.987 + 1.681·k + 1.261·(40 − k) = 53.41 + 0.420·k        [GiB/卡]
KV预算(k) = 70.090 − W(k)
```

| k | 权重组分 (slab / 驻留) | W/卡 | KV 预算 | swa=0.1 可支撑 | swa=0.005 可支撑 | 专家 GPU 侧内核 |
|---|---|---|---|---|---|---|
| 0 | 50.43 / 0 | 53.41 | 16.68 | 4.20 M tok (4 × 1M) | 10.25 M tok (9 × 1M) | 全部 lk_moe 闭源 (W4A16 等价) |
| 4 | 45.40 / 6.72 | 55.09 | 15.00 | 4.06 M (3 × 1M) | 9.22 M (8 × 1M) | 4 层 W4A8 (humming/flashinfer-fp8), 36 层闭源 |
| 8 | 40.36 / 13.45 | 56.78 | 13.31 | 3.60 M (3 × 1M) | 8.18 M (7 × 1M) | 8 层 W4A8 |
| 12 | 35.31 / 20.17 | 58.46 | 11.63 | 3.15 M (3 × 1M) | 7.15 M (6 × 1M) | 12 层 W4A8 |
| 16 | 30.27 / 26.89 | 60.14 | 9.95 | 2.69 M (2 × 1M) | 6.12 M (5 × 1M) | 16 层 W4A8 |
| 20 | 25.22 / 33.62 | 61.82 | 8.27 | 2.24 M (2 × 1M) | 5.08 M (4 × 1M) | 20 层 W4A8 |

**满足 KV 目标的最大 k** (变体 R 内部):

| KV 目标 | 要求 KV/卡 (swa=0.1) | 最大 k | 要求 KV/卡 (swa=0.005) | 最大 k |
|---|---|---|---|---|
| 2 × 1M | 7.76 GiB | **20** (k=20 → 8.27 ✓; k=24 未列出 → 算得 k≤20.4) | 3.41 GiB | **>16** (k=16 → 6.12 ✓, k≤26.6) |
| 4 × 1M | 15.51 GiB | **2** (k=2 → 15.84 ✓; k=4 → 15.00 ✗) | 6.82 GiB | **>20** (k=20 → 8.27 ✓, k≤24.9) |
| **16 × 1M** | 62.03 GiB | **无解** (即使 k=0 也只有 16.68) | 27.30 GiB | **无解** (即使 k=0 也只有 16.68) |

**注意方向性**: 变体 R(k) 比变体 P **更费显存** (`+0.420 GiB/卡` 每驻留一层), 因为驻留 = 384 专家, 池 = 288 槽。变体 R 的收益不在显存, 而在**算力路径**: 驻留层可以换用 `--moe-runner-backend humming` / `--flashinfer-mxfp4-moe-precision=fp8` 的 **W4A8** 路径, 而池内专家只能走 lk_moe 闭源内核。

---

## D. 变体 S (不启用池, 只做整层驻留 + CPU 混合) 〔resident 成本已被 §8.6 修正为 1.868 GiB/卡/层〕

`LK_POOL_SLOTS=0` → 无池; `LK_GPU_RESIDENT_MOE_LAYERS` 选 k 层走 sglang GPU 路径 (可用 W4A8), 其余 `40−k` 层留在 host 由 CPU 计算。

```
W(k) = 2.987 + 1.681·k     [GiB/卡]        KV预算(k) = 70.090 − W(k)
```

| k | W/卡 | KV 预算 | swa=0.1 可支撑 | swa=0.005 可支撑 |
|---|---|---|---|---|
| 0 | 2.99 | 67.10 | 17.3 × 1M → **16 × 1M 可行 (62.03)** | 39 × 1M |
| 4 | 9.71 | 60.38 | 15.6 × 1M | 35 × 1M |
| 8 | 16.43 | 53.66 | 13.8 × 1M | 31 × 1M |
| 12 | 23.16 | 46.93 | 12.1 × 1M | 27 × 1M |
| **16** | 29.88 | 40.21 | 10.4 × 1M | 23 × 1M |
| **20** | 36.60 | 33.48 | 8.6 × 1M | 19 × 1M |
| **23** | 41.65 | 28.44 | 7.3 × 1M | **16.7 × 1M → 16×1M 可行 (27.30)** |
| 24 | 43.33 | 26.76 | 6.9 × 1M | 15.7 × 1M (16×1M ✗) |
| 32 | 56.78 | 13.31 | 3.4 × 1M | 7.8 × 1M |
| 40 | 70.22 | −0.13 | 不可行 (权重已超预算) | 不可行 |

**最大可驻留层数** (`W(k) ≤ 70.090`, 即 KV = 0): `k ≤ 39.9` → **k = 39** 层 (每卡 68.55 GiB)。
**在保住 16×1M 前提下的最大 k**: swa=0.1 → **k = 3**; swa=0.005 → **k = 23** (算式: `(70.090 − 2.987 − 27.30)/1.681 = 23.7`)。

变体 S 的专家侧: k 层 = sglang GPU 路径 (**可用 humming / flashinfer-mxfp4-fp8 = W4A8**), 其余 = CPU 计算 (无 GPU 池, 延迟最差)。

---

## E. 各变体的专家 GPU 侧内核与 16×1M 支撑能力

| 变体 | 专家 GPU 侧内核 | 16×1M 是否可行 | 若不可行, 实际能支撑的组合 | 说明 |
|---|---|---|---|---|
| P (纯池 288) | **lk_moe 闭源池内核** (等价 W4A16); 无法换 humming/W4A8 | ❌ | swa=0.005: 9 × 1M / 16 × 640K; swa=0.1: 4 × 1M / 16 × 282K | 池内专家解码在 lk_moe 内, sglang 只编排 |
| R(k) 任意 k | k 层 = sglang 路径 (可 W4A8), 40−k 层 = 闭源池内核 | ❌ (任何 k) | k=0: 9 × 1M; k=8: 7 × 1M; k=16: 5 × 1M; k=20: 4 × 1M (swa=0.005) | k 越大, 闭源内核覆盖层数越少 |
| S(k) 无池 | k 层 = sglang 路径 (可 W4A8), 40−k 层 = **CPU 计算** | ✅ 当 k ≤ 3 (swa=0.1) 或 k ≤ 23 (swa=0.005) | k=23, swa=0.005: 16 × 1M; k=8: 31 × 1M | 无 GPU 池 ⇒ 非驻留层全 CPU, 吞吐/延迟代价最大 |

---

## 5. 推荐参数集与逐项取舍 〔已按 §8 修正〕

### 5.1 推荐 A(首选: 288 槽 + 0.88 + 4×1M 全满足)

| 参数 | 建议值 | 说明 |
|---|---|---|
| `LK_MOE_HYBRID_ENABLED` | `1` | 混合总开关 |
| `LK_POOL_SLOTS` | `288` | ✅ 满足; 需 `--ep-size` 保持默认 1 (每卡 384 本地专家) |
| `--mem-fraction-static` | `0.88` | ✅ 满足 (70.090 GiB/卡) |
| `--context-length` | `1048576` | ✅ 满足 |
| `--max-running-requests` | `4` | ⚠️ 16 → 4 (用户已澄清并发超卖, 只需 4×1M) |
| `--max-total-tokens` | `4194304` (=4×1M) | 池只按需分配; 预算允许 10.4M, 设低反而把差额留给 CUDA graph/激活 |
| `--swa-full-tokens-ratio` | **不要传** | 不传 = cap 模式 (SWA 固定 0.49 GiB); 传 0.1 会把 4×1M 的 KV 需求从 6.37 → 15.51 GiB |
| `LK_NUMA_INTERLEAVE` | `1` | host 常驻 464.60 GiB > node1 401.77 GiB |
| 预算校验 | 53.41 (权重) + 6.37 (KV) + 0.49 (SWA) = **60.27 GiB/卡** | 余 9.82 GiB 供图捕获/激活 (需 ≤9.558 之外即安全) |

### 5.2 推荐 B(保 16 并发 × 1M, 降槽位) 〔已修正〕

| 参数 | 建议值 | 算术原因 |
|---|---|---|
| `LK_POOL_SLOTS` | **232** (cap 模式, 默认不要传 swa 比率) | `S_max(16×1M, cap) = (70.090 − 2.987 − 26.36)/0.1751 = 232.7` |
| `--max-running-requests` | `16` | ✅ |
| `--context-length` | `1048576` | ✅ |
| `--max-total-tokens` | `16777216` | KV = 25.47 GiB (cap) + SWA 0.90 GiB |
| `--swa-full-tokens-ratio` | **不要传** | cap 模式即可; 若必须传, 用 `0.005` (则 S_max 降到 227) |

### 5.3 推荐 C(变体 S, 无池: 整层驻留; 驻留层可走 W4A8) 〔已修正: resident 成本 ×640/576〕

| 参数 | 建议值 | 算术原因 |
|---|---|---|
| `LK_POOL_SLOTS` | `0` (关闭池) | — |
| `LK_GPU_RESIDENT_MOE_LAYERS` | 保 4×1M: `0-31` (32 层); 保 16×1M: `0-20` (21 层) | `W_S(k)=2.987+1.868k`; 4×1M: `k ≤ (70.090−2.987−7.27)/1.868 = 32.0`; 16×1M: `k ≤ 21.8` |
| `--moe-runner-backend` | `humming` 或 `--flashinfer-mxfp4-moe-precision=fp8` | 驻留层走 W4A8 (池内专家做不到) |
| `--max-running-requests` | `4` (或 16) | 见上 |
| `--context-length` | `1048576` | ✅ |
| `--max-total-tokens` | 按所选 k 反推 | — |
| `--swa-full-tokens-ratio` | 不要传 | cap 模式 |

> **取舍说明 (修正后)**: 288 槽/层 = 50.43 GiB/卡, 与 4×1M (6.37+0.49) 之和 **57.29 GiB < 70.090** ⇒ **两者可并存**; 与 16×1M (25.47+0.90) 之和 **76.80 > 70.090** ⇒ 仍互斥。变体 S(k=32) 覆盖 32×384 = 12288 个"层×专家"槽 > 288×40 = 11520, 且能让 32 层走 W4A8 — 若不需要池的动态换入换出, 变体 S(k=32) 是覆盖面更大、且可用 W4A8 的方案。

### 5.4 用户原始要求逐项裁定 〔已修正〕

| 要求 | 裁定 | 算术原因 |
|---|---|---|
| `--mem-fraction-static 0.88` | ✅ 可满足 | 70.090 GiB/卡 = 280.359 GiB/4 卡 |
| 288 GPU 专家槽/层 | ✅ **可满足** | slab = 50.43 GiB/卡; `S_max = 344` (4×1M 前提) |
| 4 并发 × 1M | ✅ **可满足** | KV = 6.37 GiB + SWA 0.49 GiB = 6.86 GiB/卡 |
| 16 并发 × 1M | ❌ 与 288 槽互斥 | 需 KV 26.37 GiB; 288 槽后只剩 16.68 GiB |
| 1M 上下文 | ✅ 支持 (模型上限 1048576) | — |
| engram 放 host | ✅ 默认即 host (`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`), 188.83 GiB | §4 |
| CUDA graph | ⚠️ 需实测 | 图+激活须 ≤9.558 GiB 余量; 观测 `Capture ... mem usage=` |

---

## 6. 待环境确证清单 (不改变上述结论的量级)

| # | 项 | 方法 | 影响 |
|---|---|---|---|
| 1 | 实际 `page_size` | 启动日志 `DSV4 pool sizes:` / `DSV4 SWA sizing:` (`pool_configurator.py:1419-1428, 1545-1551, 1619-1631`) | 影响 585 vs 584 B/token (0.2%) |
| 2 | `bytes_per_full_token` 实测值 | 同上日志 (`bytes_per_full_token=`) | 校验 §3.1 的 1630 + ratio×23400 |
| 3 | 池 slab 实测 | 启动日志 `[pool] slab: %.2f GiB (%d layers x %.2f MiB)` | 校验 §1.3 的 4.482 MiB/专家/卡 |
| 4 | lk_moe 实际版本与槽字段 | `<env>/bin/python -c "import lk_moe; c=lk_moe.MOEConfigV2(); print(c.pool_slots, c.pool_member_layers)"` | 校验 2.7.1 的 38 字段 (含 `pool_slots`/`pool_member_layers`/`pool_window=32`) |
| 5 | mxfp4 SM90 是否对专家尾列补齐 | `Mxfp4MoEMethod` round-up 逻辑 (`layer.py:478-489`) | ±5% 槽位成本 |
| 6 | KV 池每卡是否真为一整份 | 启动日志 `max_total_num_tokens` + 4 卡显存分布 (`nvidia-smi`) | 若被切分, KV 余量 ×4, 结论转宽松 |

---

## 7. 证据与产物

| 编号 | 内容 | 路径 |
|---|---|---|
| E1 | 预算算式脚本 | `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/budget_math.py` |
| E2 | lk_moe 2.7.1 wheel (sha256 `7c9c5c13…54d5`, 与镜像索引一致) + 解包目录 | `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/lk_moe-2.7.1-cp312-cp312-manylinux_2_34_x86_64.whl`, `.../whl271/` |
| E3 | `MOEConfigV2` 38 字段实测 (含 `pool_slots=0`, `pool_member_layers=0`, `pool_window=32`, `pool_inflight=16`) | `PYTHONPATH=<whl271> python -c "import lk_moe; ..."` (cp312: `/data3/users/wangyf/miniconda3/envs/esmfold2/bin/python`) |
| E4 | lk_moe 池 slab 日志格式字符串 | `strings -a .../_lk_moe_C.cpython-312-x86_64-linux-gnu.so \| grep -i slab` |
| E5 | 权重头部分析 (每专家字节 / 层总量 / engram) | `/mnt/SSD01/models/.llm-env/Lvllm/docs/_evidence/analyze_weights2.py` (同一 checkpoint) |
| E6 | 硬件/NUMA | `nvidia-smi --query-gpu=...`; `numactl -H` |

---

# 8. 【修正 · 钉死 13× 分歧】KV 模型与 4×1M 重算

> §3 把 `--swa-full-tokens-ratio` 当作恒定的"每 token 乘数", 并同时列出 launcher 的"40 层 × 全 latent"口径 → 造成 13× 分歧。修正后: **默认路径每 full-token = 1630 B**; launcher 的 22.5 KiB/token 高估 ≈14×。

## 8.1 本模型真正落全量 KV 的层: **4 层** (2/8/14/20)

| 事实 | 证据 (file:line) |
|---|---|
| ratio-1/2 压缩池的 `layer_num = len(kv_source_layers)`, **不是**该 ratio 的层数 | `mem_cache/deepseek_v4_memory_pool.py:1537-1548` (`for ratio, sources in low_ratio_sources.items(): ... layer_num=len(sources)`) |
| 池大小: ratio-2 池 = `full_size // 2`, ratio-1 池 = `full_size` | 同上 (`size=self.full_size // ratio`) |
| 每个层的归属表: ratio-1/2 的**非 owner** 层不建缓冲, 映射到 source 层槽位 (`compress_layer_id = sources.index(self.source_layer_of(idx))`) | `deepseek_v4_memory_pool.py:1763-1782` (`_init_compressed_layer_mapping`) |
| owner 层: ratio-2 → {2,8,14}; ratio-1 → {20}; ratio-0 → 纯滑窗 (无全量 KV) | `deepseek_v4_memory_pool.py:93-114` + 模型 `config.json:60-111` |
| 尾部层不得自带压缩 (`late_ratios ⊆ {0,1}`), 尾层读 source 层发布的 top-k/candidates | `models/deepseek_v4.py:4041-4047, 4063-4066` |
| SWA 池覆盖全部 40 层 (`layer_num=stage_layer_num`) | `deepseek_v4_memory_pool.py:1208-1218` |
| 行宽 584 B = 448 fp8 NoPE + 64 bf16 RoPE + 8 E8M0 | `deepseek_v4_memory_pool.py:282-303`; `kernels/ops/attention/dsv4/kv_layout.py:17,26-56` |

**结论**: 36 个 ratio-1/2 层只存 128-token 滑窗 (在 SWA 池) + 读共享压缩 KV; "40 层各存全量 latent" (`576 B × 40 = 22.5 KiB/token`) 与实现不符。

## 8.2 每层每 token 字节拆解 (`compress_ratios` 的作用)

配置器公式 (`pool_configurator.py:1363-1386`): `compressed = Σ_ratio (kv_bytes + low_ratio_index_bytes)/ratio × owner_layers(ratio)`

| 层 | `compress_ratio` | KV 行宽 | 低比 indexer (fp4) | 每 full-token |
|---|---|---|---|---|
| 2, 8, 14 | 2 | 584 B/state (1 state = 2 token) | 68 B/state | `(584+68)/2 = 326` ×3 = **978 B** |
| 20 | 1 | 584 B/state (1 state = 1 token) | 68 B/state | `(584+68)/1 = 652` ×1 = **652 B** |
| 其余 36 层 (非 owner) | 1 / 2 | 共享 source 缓冲 | — | **0** |
| **压缩合计** | | | | **1630 B/full-token** |

常量: `kv_bytes = 448 + 64×2 + 8 = 584` (`pool_configurator.py:1080-1083`); 低比 indexer `128//2 + 128//32 = 68` (`deepseek_v4_memory_pool.py:52-56`; `pool_configurator.py:1143-1147` 注明低比池 `force_fp4=True`)。

## 8.3 `--swa-full-tokens-ratio` 语义 (默认 0.8 还是 0.1?)

| 问题 | 事实 | 证据 |
|---|---|---|
| 乘的是哪部分 | 乘 **SWA 池 token 数**; 单位成本 `bytes_per_swa_token = 585 B × 40 层 = 23,400 B` | `pool_configurator.py:1345-1361`, `1377-1386` |
| 23,400 B 来源 | `_get_paged_kv_bytes_per_token(0) = page_bytes(128)/128 = 74,880/128 = 585 B` (584 + 576 对齐填充) × 40 层 | `pool_configurator.py:1326-1343`; `kv_layout.py:55-56,72-74` |
| CLI 文档值 | `fallback=0.8` (仅元数据/help) | `arg_groups/fields/schedule.py:146-158` |
| 内部常量 | `DSV4_DEFAULT_SWA_FULL_TOKENS_RATIO = 0.1` (cap 模式不可用时才用) | `pool_configurator.py:1021` |
| **实际生效规则** | `_operator_swa_full_tokens_ratio()` 仅在**用户显式传参**时返回值, 否则 `None` → 走 **cap 模式**, 系数强制置 0, SWA 变**固定字节** | `pool_configurator.py:1024-1030`, `1301-1324`, `1380-1382` |
| 显式性判定 | `= (cfg.swa_full_tokens_ratio is not None)`, 在 fallback 填充前判定 | `arg_groups/kv_cache_hook.py:415-419`; `arg_utils.py:171-201` |

cap 模式公式 (`pool_configurator.py:900-945, 1291-1299, 1388-1395`):
```
swa_cap_tokens  = ceil_align( per_request×num_reqs + 2×chunked_prefill_size + page , page )
                  + (4×max_running_requests) × (window + page)
per_request     = window(128) + SGLANG_SWA_EVICTION_INTERVAL(默认128) + page + page
swa_fixed_bytes = swa_cap_tokens × 23,400 B          # 与 token 数无关
```

| `chunked_prefill_size` | `max_running_requests` | `swa_cap_tokens` | SWA 固定成本 |
|---|---|---|---|
| 8192 | 4 | 22,656 | 0.494 GiB |
| 8192 | 16 | 41,088 | 0.895 GiB |
| 16384 | 16 | 57,472 | 1.252 GiB |

显式传 `r` → 每 full-token 加 `r × 23,400 B`; 示例脚本的 `0.1` 使 KV 需求放大约 2.4× (纯浪费: SWA 只需覆盖窗口 + 前缀尾)。

## 8.4 可用公式 + 4×1M / 16×1M (每卡)

```
bytes/full-token = 1630                         (默认: cap 模式, SWA 另计固定字节)
                 = 1630 + 23400 × r             (显式 --swa-full-tokens-ratio r)

KV_budget(每卡) = 70.090 − W_fix(2.987) − pool_slab(S) − c2_state(≈0.001)   [GiB]
KV_need(每卡)   = N×C×bytes_per_full_token/2^30 + swa_fixed_bytes
```

| 目标 | cap 模式 (默认) | r=0.005 | r=0.1 (示例脚本) |
|---|---|---|---|
| 单条 1M | 1.59 GiB | 1.71 GiB | 3.88 GiB |
| **4 × 1M** | **6.37 GiB** (+0.49~1.25 SWA) | 6.82 GiB | 15.51 GiB |
| 10 × 1M | 15.92 GiB | 17.06 GiB | 38.77 GiB |
| 16 × 1M | 25.47 GiB (+0.9 SWA) | 27.30 GiB | 62.03 GiB |

**口径确认 (每卡一整份)**: DSV4 KV buffer = `[num_pages, page_bytes]`, `layer_num` 按层数计, 与 TP 无关; 全文件无 TP 除法; `assert config.num_key_value_heads == 1` → MLA latent 每卡一份 ⇒ 4 卡合计 = ×4。证据: `deepseek_v4_memory_pool.py:279-311, 1208-1218, 1526-1548`; `models/deepseek_v4.py:1169`; `grep tp_size deepseek_v4_memory_pool.py` = 0 命中; 预算按该 rank 自身空闲显存计算 (`kv_cache_configurator.py:2700-2770`)。

### 表 8-4 变体 P (S=288, 40 层全池, W=53.41 GiB/卡) — 修正后

| KV 口径 | KV 预算/卡 | 4×1M 需 | 判定 | 16×1M 需 | 判定 | 容量 |
|---|---|---|---|---|---|---|
| **cap (默认)** | 16.68 GiB | **7.27 GiB** | ✅ 余 9.41 | 26.37 GiB | ❌ | 10.4 M tok (≈9.9 × 1M) |
| r=0.005 | 16.68 GiB | 6.82 GiB | ✅ 余 9.86 | 27.30 GiB | ❌ | 10.25 M |
| r=0.1 | 16.68 GiB | 15.51 GiB | ✅ 余 1.17 | 62.03 GiB | ❌ | 4.51 M |

**结论翻转**: `S_max(4×1M, cap) = (70.090 − 2.987 − 6.86)/0.1751 = 344.1` → **S=288 与 4×1M 相容, 余量充足**; `S_max(16×1M, cap) = 232.7` → 16×1M 仍不可行 (用户已澄清并发超卖, 只需 4×1M)。

## 8.5 `LK_POOL_SLOTS` 的有效上限 = 每卡本地专家数 (compat-prober 提供, 已核对)

| 事实 | 证据 |
|---|---|
| `cfg.expert_num = self.moe_runner_config.num_local_experts` | `layers/moe/fused_moe_triton/layer.py:2090` |
| `_num_local_routed = _num_global_routed // storage_ep_size`, `storage_ep_size = moe_ep_size` | `layer.py:439-443` |
| `ep_size` 默认 **1** (`--ep` / `--expert-parallel-size`) | `arg_groups/fields/parallel.py:269-276` |
| sglang 对 `LK_POOL_SLOTS` **无上限校验**, 原样透传 | `layer.py:2081` |

⇒ **ep_size=1 (默认): 每卡 384 routed (+1 shared) → 288 槽成立**; `--ep-size 4`: 每卡 96 routed → 池槽应 ≤96 (实用 64–80), 否则池容量 3× 于本地专家数。

## 8.6 两条"待确证项"收口

| 项 | 结论 | 证据 |
|---|---|---|
| SM90 的 `dsv4_attn_backend` / KV 布局 | **`flashmla` + `KVLayout.V4` (fp8)**: `SGLANG_DSV4_KV_LAYOUT` 默认 `"v4"` (`srt/environ.py:1657`) → `select_dsv4_kv_layout()` 早退返回 V4; `flashmla_mega` (强制 v41) 需 DeepSeek-V4.1 + SM100/SM103 | `deepseek_v4_memory_pool.py:186-201`; `arg_groups/deepseek_v4_hook.py:29-43, 75-89` |
| ⇒ 预算含义 | 压缩 KV 只可能是 fp8 584 B/state; **`V41_FP4` (fp4 压缩) 不计入** | `kv_layout.py:16-33` |
| mxfp4 专家尾列补齐 | **resident 层: 补** — SM90 `flashinfer_cutlass` 路径 `_pad_intermediate = (intermediate % 128 != 0)` → 576→**640** (+11.11%), w13/w2 填 0、scale 填 1, 并改写 `intermediate_size_per_partition`; hidden 5120 已是 128 倍数不补 | `mxfp4_flashinfer_cutlass_moe.py:74-84, 150-160`; `mxfp4_flashinfer_trtllm_moe.py:54-99` |
| **池层: 不补** | "lk_moe CPU-resident layers take raw pointers + exact strides: **no padding**" (`if not cpu_resident and not is_sm90: round_up(...)`) | `mxfp4_marlin_moe.py:84-91` |
| ⇒ 对 §1.3 / §C / §D 的修正 | **池 slab 保持 4.482 MiB/专家/卡** (S=288 → 50.43 GiB/卡, 不变); **resident 层改为 `1.681 × 640/576 = 1.868 GiB/卡/层`** | 上两行 |

### 表 8-6 变体 R(k) / S(k) 修正 (resident 层 ×640/576)

`W_R(k) = 2.987 + 1.868·k + 1.261·(40−k) = 53.41 + 0.607·k` [GiB/卡]

| k | W_R(k) | KV 预算 (cap) | 4×1M 判定 (需 6.37) |
|---|---|---|---|
| 0 | 53.41 | 16.68 | ✅ |
| 8 | 58.27 | 11.82 | ✅ |
| 16 | 63.13 | 6.96 | ✅ (需 6.86, 余 0.10) |
| 20 | 65.55 | 4.54 | ❌ (需 6.86) |

`W_S(k) = 2.987 + 1.868·k` [GiB/卡] → 纯驻留上限 `k ≤ (70.090−2.987)/1.868 = 35.9` → **k=35** (KV≈0); 保 4×1M (需 7.27) 时 `k ≤ 32.0` → **k=32**; 保 16×1M (需 26.37) 时 `k ≤ 21.8` → **k=21**。

---

# 9. CUDA graph 捕获显存 (此前遗漏项)

| 项 | 事实 | 证据 |
|---|---|---|
| 图后端默认 | decode = `FULL`, prefill = `BREAKABLE` | `model_executor/cuda_graph_config.py:149-154` |
| `--cuda-graph-max-bs-decode` / `-prefill` | 默认 `None` (运行期解析; prefill 文档默认 2048) | `arg_groups/fields/exec_.py:585-589`; `arg_groups/cuda_graph_hook.py:359-360` |
| `--cuda-graph-bs-decode` / `-prefill` | 显式批大小列表, 默认 `None` | `fields/exec_.py:596-601` |
| 关闭方式 | **`--disable-cuda-graph` 是 `no_cli=True` (本 fork 敲不出来)**; 可用 `--disable-decode-cuda-graph` / `--disable-prefill-cuda-graph` / `--disable-cuda-graph-padding` / `--cuda-graph-config '{...}'` | `fields/exec_.py:610-627` |
| **图显存是否计入 `mem_fraction_static`** | **是, 但方向是"由余量承担"**: KV 预算 = `free_at_pool_init − pre_model_load_memory×(1−mem_fraction_static)` ⇒ 0.88 里已预留 `0.12 × 79.647 = 9.558 GiB` 给图捕获 + 激活 + sampling | `kv_cache_configurator.py:2700-2770` |
| post-capture KV 重算 | **本模型不生效 (双重)**: `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` 默认 **False**, 且 `post_capture_kv_sizing_planned()` 对 **MLA** 直接 `return False` (DSV4.1 是 MLA) ⇒ KV 池尺寸在捕获前锁定, 图显存只能吃余量 | `srt/environ.py:618`; `arg_groups/overrides.py:1810-1822`; `model_override_base.py:185-189` |
| 余量护栏值 | `pre_capture_activation_reserve_mb()` 对 >60 GiB 卡取 `max(…, 10 GiB)`; DSV4 非 mamba 分支不使用该下限, 可作"图+激活"上界参考 | `runtime_context.py:1789-1811`; `kv_cache_configurator.py:2731-2743` |
| 实测观测点 | 日志 `Capture <name> <backend> end. … mem usage=<X> GB, avail mem=<Y> GB.` | `model_executor/cuda_graph_setup.py:744-749, 650-656` |

**对预算的含义 (与 compat-prober 口径一致, 但不重复扣减)**: `0.88 × 79.647 = 70.090 GiB` 中 **KV = 70.090 − 权重**; 未被权重与 KV 用掉的部分正是那 **9.558 GiB 余量**, 图捕获 + 激活 + sampling 必须装进它 ⇒ §8.4 的 KV 数字**无需再减 10 GiB** (重复扣减)。风险是: 若图捕获 + 激活 > 9.558 GiB, 因本模型不会 post-capture 缩小池, 直接 **OOM**。缓解: 降 `--cuda-graph-max-bs-decode`、`--disable-prefill-cuda-graph`、或降 `--max-running-requests`。
