# 每卡显存完整清单 — DeepSeek-V4.1-Flash-1017 / Lsglang 1.6.1 / 4×H100 (TP4)

- 配置快照 (来自被审计运行的 `server_args`): `TP4 / ep_size=1 / attn_dp_size=1 / pp=1 / dp=1`, `mem_fraction_static=0.88`, `max_running_requests=16`, `max_total_tokens=4194304`, `context_length=1048576`, `chunked_prefill_size=8192`, `max_prefill_tokens=16384`, `page_size=256`, `kv_cache_dtype=fp8_e4m3`, `dsv4_attn_backend=flashmla`, `moe_runner_backend=flashinfer_mxfp4`, `flashinfer_mxfp4_moe_precision=default` (⇒ W4A16), `ep_size=1`, `moe_a2a_backend=none`, `disable_custom_all_reduce=False`, `disable_shared_experts_fusion=True`, `speculative_algorithm=None`, `cuda_graph_config.decode={backend:full,max_bs:16,bs:[1,2,4,8,12,16]}`, `cuda_graph_config.prefill={backend:disabled}`, `LK_POOL_SLOTS=192`, `LK_GPU_RESIDENT_MOE_LAYERS=0-19`, `enable_decoder_swa_bounded_replay=True`, `language_model_only=False`, `enable_multimodal=None`
- 审计锚点: 最近一次运行在 KV 定容时刻, 每卡 `空闲 8.6556 GiB` (TP0, 已含 `gc+empty_cache` 回收 0.5000 GiB), 加载起始可用 `77.36–77.60 GB(=GiB)` ⇒ **本进程占用 = 68.9 GiB/卡** (TP0 精算 `77.36 − 8.6556 = 68.70`)
- 原始日志 (只读引用):
  - `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261023_013857.log`
  - `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261023_010939.log`
  - `/mnt/SSD01/models/.llm-env/Lsglang/logs/OOM-evidence-20261022_141442-ep4.md`
  - `/mnt/SSD01/models/.llm-env/Lsglang/logs/SEGV-evidence-20261022_145831.md`
- 单位约定: 文中 `GB` 均为 SGLang 日志写法, 数值按 GiB 理解 (`get_available_gpu_memory` 返回 `bytes / 2^30`, `sglang/srt/utils/common.py:624`)
- 约束遵守: 只读分析; 未启动服务, 未占用 GPU (仅读取既有日志); 未修改任何文件; 未读取 `deprecated/` 与 `*_DEPRECATED*`。

---

## 0. 主表 (每卡字节数)

| # | 项 | GiB/卡 | 依据 | 判定 |
|---|---|---|---|---|
| A1 | resident 20 层专家 w13 (含 576→640 补齐) | 23.438 | 日志枚举 `uint8 (384,1280,2560)` ×20 | 已确认 |
| A2 | resident 20 层专家 w2 (含 K 320 补齐) | 11.719 | 日志枚举 `uint8 (384,5120,320)` ×20 | 已确认 |
| A3 | resident 20 层专家 scale (w13) | 1.465 | 日志枚举 `uint8 (384,20,40,16,16)` ×20 | 已确认 |
| A4 | resident 20 层专家 scale (w2) | 0.732 | 日志枚举 `uint8 (384,80,5,16,16)` ×20 | 已确认 |
| A | **resident 专家小计** | **37.354** | A1+A2+A3+A4 | 已确认 |
| B | **池 slab** (192 槽 × 20 层) | **16.809** | 日志 `[pool] slab: 16.81 GiB (20 layers x 860.62 MiB)`; 手算 `192×4,700,160×20 = 18,048,614,400 B` | 已确认 |
| C1 | attention (TP 分片部分: wq_b/wo_a/wo_b/压缩机/indexer) | 1.202 | 检查点头部按 `wq_b`(列并行)/`wo_a`(列并行)/`wo_b`(行并行) 分片、其余按全量计 | 推算 |
| C2 | attention (复制部分: `wq_a`/`wkv` = `ReplicatedLinear`, q_norm/kv_norm/attn_sink; **全量计, 不除 TP**) | 0.342 | 源码 `deepseek_v4.py:1208-1214` (`ReplicatedLinear`); 检查点逐张量字节 | 推算 |
| C3 | shared experts (TP 分片) | 0.330 | 检查点 `layers.*.ffn.shared_experts.*` 1.320 GiB / 4; 本运行 `disable_shared_experts_fusion=True` (独立模块) | 推算 |
| C4 | 路由 gate / 其它 ffn (TP 分片) | 0.037 | 检查点 `.ffn.*`(非 experts/shared) 0.146 GiB / 4 | 推算 |
| C5 | 其它复制项 (norm/`hc_*`/`gate.bias` 等) | 0.147 | 检查点逐张量分类 (不随 TP 切分) | 推算 |
| C6 | embed_tokens + lm_head (词表按 TP 分片, bf16) | 0.616 | 日志枚举 `bf16 (32320,5120)` ×2 张 (32320 = 129280/4) | 已确认 |
| C7 | vision tower (32 层 ViT) + aligner | 0.904 | 该运行 `model_type=deepseek_v41` + `vision_n_layers=32` + `language_model_only=False` ⇒ `self.vision=ViT(args)` 实例化 (`deepseek_v4.py:4786-4806`), 权重**不**走 skip 分支 (`:5968-5972`); ViT 内层为 `nn.Linear` (不随 TP 切分, `deepseek_v41_vit.py:143-144`) | 推算 |
| C8 | MTP/dspark 层 | 0 | `speculative_algorithm=None` ⇒ 草稿模型不实例化; 检查点 `mtp.*` 7.377 GiB 不加载 | 已确认 |
| C9 | Engram 两表 | 0 (GPU) | 走主机 pinned 表; GPU 枚举中无 engram 张量组 (主机侧 188.833 GiB 总量) | 已确认 |
| D1 | Rope 表 A: `complex64 (1048576,32)` ×2 | 0.500 | 日志枚举, `ref_by=['_lru_cache_wrapper']` | 已确认 |
| D2 | Rope 表 B: `fp32 (1048832,64)` ×2 | 0.500 | 日志枚举 | 已确认 (生成点未定位, 见 §3) |
| E | 索引 candidate 缓冲 `bf16 (2048,4096)` ×40 | 0.625 | 日志枚举; 形状与 `candidate_topk_blocks=2048` × `index_n_heads(32)×index_head_dim(128)=4096` 相符 | 已确认 (归属为推算) |
| F | 池解码辅助缓冲 (`hbuf`/`gout`/`slot_tab`/`hpart`/`gout_part`) | ≈0.010 | `fused_moe_triton/layer.py:2455-2476`, `rows = max_running_requests(16) × k(6) = 96`, `H=5120, I=576` | 推算 |
| G | lk_moe H2D staging ring | 0.0044 | 日志 `[pool] H2D staging ON: ring=8 slab=4.70MB` | 已确认 |
| H | 非 torch 基线 (CUDA context / NCCL / custom-AR / cuBLAS·cuDNN·flashinfer workspace) | 2.2–2.7 | 两次 OOM 自报: `62.71 − 60.48 = 2.23`; `67.41 − 64.71 = 2.70` (PyTorch 之外的部分) | 已确认 (总量级), 内部构成无法静态判定 |
| — | **已识别合计** | **≈59.38** | A+B+C1..C7+D1+D2+E+F+G (+H 单列) | — |
| — | **残差 (对 68.7–68.9)** | **≈9.3–9.5** | 见 §5: 扣掉实测非 torch 基线 2.2–2.7 后, 仍有 **≈6.6–7.3 GiB 的 torch 张量无法静态归属** | 无法静态判定 |

> 说明: 表中未含 KV 池、SWA 池、`req_to_token`、CUDA 图捕获显存 —— 它们在 KV 定容**之后**分配 (本次运行在定容处失败, 这部分从未分配); decode 图桶为 `bs=[1,2,4,8,12,16]`、prefill 图 `disabled`。

---

## 1. 逐项依据

| 项 | 依据 (file:line / 日志 / 算式) |
|---|---|
| A1–A4 | 日志 `SGLANG_DSV41_POST_CONVERT_RELEASE=1: largest live CUDA tensor groups (dtype, shape, GiB, ref_by)` 四行: `uint8 (384,1280,2560) 23.438` / `uint8 (384,5120,320) 11.719` / `uint8 (384,20,40,16,16) 1.465` / `uint8 (384,80,5,16,16) 0.732`。每项 ×20 层: `20×384×1280×2560 = 25,165,824,000 B = 23.438 GiB` (1280 = 2×640, 即 576→640 的 SM90 CUTLASS 补齐, `mxfp4_flashinfer_cutlass_moe.py:79,150-160`); `20×384×5120×320 = 12,582,912,000 B = 11.719 GiB`; scale 组 `20×384×1280×160 = 1,572,864,000 B` / `20×384×5120×20 = 786,432,000 B` |
| A (对照) | 未补齐口径 (576/288) 为 `20×384×4,700,160 = 36,097,228,800 B = 33.62 GiB` ⇒ **补齐使 resident 部分 +3.73 GiB/卡 (+11.11%)** |
| B | 日志 `[pool] slab: 16.81 GiB (20 layers x 860.62 MiB)`; `860.62 MiB = 192 × 4,700,160 B = 902,430,720 B` ⇒ 池层用**未补齐**行宽 (`mxfp4_marlin_moe.py:84-91`: "lk_moe CPU-resident layers take raw pointers + exact strides: no padding")。池成员 = `池计费层数 = 40 − |resident| = 20` (`fused_moe_triton/layer.py:2060-2072`) |
| A 与 B 是否重叠 | **不重叠**: resident 层不建池 (`is_lk_pool_layer` ⇒ 非 resident 才建, `lk_moe_env.py:287-292`), 池 slab 只为 20 个非 resident 层计费 (日志 "20 layers") |
| C1–C5 | 检查点头部逐张量统计 (`model.safetensors.index.json` + 各 shard 头), 按类求和后按分片规则折算: attention 全量 5.151 GiB, 其中 `wq_a`/`wkv` 为 `ReplicatedLinear` (`deepseek_v4.py:1208-1214`), `wq_b`/`wo_a`/`wo_b` 为列/行并行 (随 TP4 切分) |
| C6 | 日志枚举 `bf16 (32320,5120) 0.616` (2 张 = embed_tokens + lm_head; `32320 = 129280/4`) |
| C7 | `deepseek_v4.py:4786-4806` (实例化条件) 与 `:5968-5972` (仅当 `self.vision is None` 才 skip `vision./aligner./image_` 权重); 本运行三条件全部满足 ⇒ 加载; ViT/aligner 权重 0.904 GiB (检查点 `vision.*` 0.768 + `aligner.*` 0.137), **不随 TP 切分** (ViT 内层 `nn.Linear`) |
| C8 | 日志 `speculative_algorithm=None`; 检查点 `mtp.*` = 7.377 GiB 未被加载 (GPU 枚举中无对应大组) |
| C9 | GPU 枚举中无 engram 张量组; 表规模 `layers.1/14.engram.embed.weight+scale = 188.833 GiB` (主机 pinned) |
| D1 | `kernels/ops/attention/deepseek_v4_rope.py:27-28` `@lru_cache(2) def precompute_freqs_cis(...)`; 调用点 `deepseek_v4.py:1303-1313` (`seqlen=config.max_position_embeddings=1048576`, `dim=64` ⇒ `(1048576, 32)` complex64); `lru_cache(2)` 容纳两套 rope 参数 (`rope_theta=10000` 与 `compress_rope_theta=160000`) ⇒ 2×0.25 GiB |
| D2 | 形状 `(1048832, 64)` = `(max_position_embeddings + page_size(256), qk_rope_head_dim)`; `precompute_cos_sin_table` 用 `torch.view_as_real(...).transpose(1,2).reshape(seqlen,-1)` 生成 fp32 `(seqlen, 64)` (`deepseek_v4_rope.py:71-81`), 但其**唯一调用点**是 SM100 的 mega 路径 (`deepseek_v4.py:1589-1596`), 本运行 `dsv4_attn_backend=flashmla` ⇒ 该调用点不成立; **生成点静态未定位** (§3) |
| E | `bf16 (2048,4096) 0.625` (40 张 = 每层 1 张, 每张 16.8 MB); 形状与 `config.candidate_topk_blocks=2048` × `index_n_heads(32)×index_head_dim(128)=4096` 一致 |
| F | `fused_moe_triton/layer.py:2455-2476`: `rows = cap(16) × k(6) × 1 = 96`; `pool_hbuf(96,1152) f32 = 0.42 MB`, `pool_gout(16,5120) f32 = 0.33 MB`, `pool_hpart(NGK,96,1152) f32 ≤ 7.1 MB` (NGK 由 kernel caps 决定, 上界 16), `pool_gout_part(96,5120) f32 = 1.97 MB`, `slot_tab/gathered_ids/gathered_weights(96,) ≈ 1.2 KB` |
| G | 日志 `[pool] H2D staging ON: ring=8 slab=4.70MB`; 同一行在 4 个 rank 各出现一次 |

---

## 2. 会随"每卡本地专家数"缩放的项 (384 vs 96)

> **更正 (2026-10-23)**: 本节原先给出 "ep1→ep4 省 40.62 GiB/卡" (把 A、B 两行各自 ÷4)。**该算法错误**, 已被实测否证, 下表为更正后数值。错因: **EP 分片的是"专家", 不是"中间维"** —— 中间维 ÷TP 的因子**已经含在**前面所用的"每专家字节数"里 (ep1: 384 专家 × 4.482 MiB/专家 = 1.72 GiB/层; ep4: 96 专家 × 17.93 MiB/专家 = 1.72 GiB/层, 二者相等)。因此 ep 变化**不改变**每卡专家字节总量, 唯一可计算的不对称是 ep1 触发的 **576→640 补齐**。

| 项 | 缩放关系 | ep1 (384/卡, 本配置) | ep4 (96/卡) | 差 |
|---|---|---|---|---|
| A resident 专家 (20 层) | 不随 ep 缩放; 仅 ep1 触发补齐 | 37.354 GiB (640 补齐, 1.868 GiB/层) | 33.620 GiB (2304, 不补齐, 1.681 GiB/层) | **−3.734** |
| B 池 slab | 不随 ep 缩放 (槽数×单槽字节 之积相等) | 16.809 GiB (192 槽 × 4,482,560 B) | 16.809 GiB (48 槽 × 18,800,640 B) | **0** (逐字节相同) |
| G H2D staging ring | ∝ 单专家字节数 | 4.70 MB | 4.70 MB (分解后每专家更小) | ≈0 |
| **合计 GPU 侧 (A+B+G)** | | **54.16 GiB** | **50.43 GiB** | **−3.73 GiB** |
| **实测整卡占用 (日志 `Load weight end … mem usage=`)** | | **69.79–70.03 GiB** | **58.29–58.52 GiB** | **−11.5 GiB** ⇒ 除 −3.73 外**另有约 −7.8 GiB 未归属** |
| C1–C7 非专家权重 | 与专家数**无关** | 3.49 GiB | 3.49 GiB | 0 |
| D1/D2 rope 表 | 无关 | 1.000 GiB | 1.000 GiB | 0 |
| E candidate 缓冲 | 无关 (按 token/块) | 0.625 GiB | 0.625 GiB | 0 |
| H 非 torch 基线 | 无关 | 2.2–2.7 GiB | 2.2–2.7 GiB | 0 |
| — | **主机侧** (lk_moe CPU 母本 + engram 表) | 见 §4 | 见 §4 | ∝ 本地专家数 |

依据: `_num_local_routed = _num_global_routed // storage_ep_size`, `storage_ep_size = moe_ep_size` (`fused_moe_triton/layer.py:439-443`); `cfg.expert_num = num_local_experts` (`:2090`); 实测旁证见 `OOM-evidence-20261022_141442-ep4.md`: ep4 生效时日志为 `Initialized lk_moe with 96 experts`。

---

## 3. Rope / 频率表专题 (D1 / D2)

| 项 | 表示 | 形状 × 份数 | 每卡字节 | 持有者 | 生成点 |
|---|---|---|---|---|---|
| D1 | complex64 `freqs_cis` | `(1048576, 32)` × 2 | 0.500 GiB | `lru_cache(2)` (`ref_by=['_lru_cache_wrapper']`) + 各 rotary 模块 `register_buffer("freqs_cis", persistent=False)` (`deepseek_v4.py:1313`) = 同一对象共享 | `precompute_freqs_cis` (`kernels/ops/attention/deepseek_v4_rope.py:27-28, 66-68`), 调用 `deepseek_v4.py:1303-1313` |
| D2 | fp32 cos/sin | `(1048832, 64)` × 2 | 0.500 GiB | torch 缓存/模块持有 (日志 `ref_by=[]`) | **未定位**: 形状为 `(max_position_embeddings + page_size, rope_dim)`, 与 `precompute_cos_sin_table` 的输出格式一致, 但其唯一调用点属 SM100 mega 路径 (`deepseek_v4.py:1589-1596`); `prime_rope_cos_sin` 仅在 NPU 后端 (`hardware_backend/npu/dsv4/dsv4_rope.py:189`) ⇒ 两者均不适用于本 SM90 配置 |

**为什么同时存在两种表示 (D1+D2 = 1.0 GiB)**: D1 是 `freqs_cis` 的复数形式 (源), D2 是实部/虚部展平后的实数表 (供 rope kernel 直接寻址); 两者形状差 256 行说明 D2 另做了按页填充 (1048576 → 1048832)。

**三种压缩方案与节省 (每卡)**:

| 方案 | 做法 | 节省 | 代价/前提 |
|---|---|---|---|
| 1. 去掉 D2 的 fp32 两份 | 让消费方改用 D1 派生 (或 bf16 派生) | **0.500 GiB** | 需确认无 kernel 依赖 fp32 表; 派生表需保持 kernel 期望的 stride |
| 2. D2 降为 bf16 (保持两份) | `(1048832,64)` fp32(0.25 GiB/张) → bf16(0.125 GiB/张) | **0.250 GiB** | 需确认 rope 精度可接受 (bf16 表直接参与 cos/sin 乘) |
| 3. D1 改按需/分块生成 | 不常驻整表, 按窗口分块生成 | 上界 **0.500 GiB** (整表) | 需 kernel 支持分块寻址; 会引入生成/换页开销 |
| 1+2 同时 | | **0.750 GiB** | 三项前提叠加 |
| 1+3 同时 | | **1.000 GiB** | D1 与 D2 都不常驻 |

---

## 4. 主机侧 (不进每卡 GPU 账, 仅列出)

| 项 | 大小 | 依据 |
|---|---|---|
| lk_moe CPU 母本: 20 池层专家 (未补齐) | 20 × 384 × 4,700,160 = 33.62 GiB (全 4 rank 合计) | 池层 `device='cpu'` (`fp8.py`-等价的 CPU 路径), 日志 `Initialized lk_moe with 384 experts for layer 20..39 [CPU]` |
| 全模型专家 (若全留 host, 变体 S) | 275.77 GiB | 检查点头部 |
| Engram 两表 (pinned) | 188.833 GiB | 检查点头部 + `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` |

---

## 5. 残差与不可判定项

```
本进程占用 (每卡)            68.70–68.90 GiB      # 77.36–77.60(起始可用) − 8.6556(定容时空闲)
已识别项合计                  59.38 GiB
  ├─ resident 专家 (A)        37.354
  ├─ 池 slab (B)              16.809
  ├─ 非专家权重 (C1–C7)        3.578
  ├─ rope 表 (D1+D2)           1.000
  ├─ candidate 缓冲 (E)        0.625
  └─ 池缓冲 + staging (F+G)    0.014
残差                          9.32–9.52 GiB
  ├─ 实测非 torch 基线 (H)     2.23–2.70 GiB      # OOM 自报 torch 之外的部分
  └─ 无法归属的 torch 张量     6.6–7.3 GiB
```

**残差可见性 / 为什么静态算不出来 (逐条)**

| 类别 | 为什么不可静态判定 | 可见性 |
|---|---|---|
| torch 张量中"小于 top-8 截断"的部分 | 日志枚举器只打印 **最大 8 组** (`largest live CUDA tensor groups`, 实现位于 `mxfp4_flashinfer_cutlass_moe.py:264-280` 的 `SGLANG_DSV41_POST_CONVERT_RELEASE` 门), 每层 ~20 张权重 (总 800+ 张) 被截断; 单组 <0.5 GiB 者不显示 | 需提高打印组数 (env 门) 或改用 `torch.cuda.memory_allocated()` 分组统计 |
| 分片假设差异 | 检查点只能给出"全量/TP 分片"两种可能; 是否存在未被 TP 切分的权重 (如 `wq_a`/`wkv` 已确认为 `ReplicatedLinear`) 需逐模块核对源码 | 可静态核对 (本报告已核对 3 处) |
| lk_moe 库内 `cudaMalloc` (除 arena 外) | 库为闭源二进制, 无分配日志; `get_device_mem_usage()` 自报口径为 "pool rows + prefill arena + Triton staging ring, **只计 cudaMalloc 的字节**", 日志仅给出 `16.81 GB` (与 slab 相等) | 只能靠进程级对比 (mem_get_info − torch reserved) |
| CUDA context / NCCL / custom all-reduce / cuBLAS·cuDNN·flashinfer workspace | 由驱动与第三方库 cudaMalloc, torch 枚举器不可见 | 由 OOM 自报差值实测: 2.23 / 2.70 GiB |
| 本配置 `disable_custom_all_reduce=False` | custom all-reduce 与 NCCL 同时存在 (是否真正启用 custom AR 取决于启发式), 两者缓冲均在非 torch 桶内 | 非 torch 桶已含在 2.2–2.7 GiB 内, 未单独拆分 |
| `torch` 缓存段 (reserved-but-unallocated) | 该值在 OOM 自报中为 `63.6 MiB` (两次样本), 定容前 `gc+empty_cache` 另回收 0.5000 GiB | 已确认量级 (≤0.5 GiB), 不构成残差主因 |
| 未分配项 (KV/SWA/req_to_token/CUDA 图) | 本次运行在 KV 定容处失败 (`Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.88`, 最小可行 `0.8894`) ⇒ 这些从未分配 | 日志已确认未分配 |
