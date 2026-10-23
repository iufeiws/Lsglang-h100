# 两次启动每卡显存差对账 — DeepSeek-V4.1-Flash-1017, 4×H100 80GB, TP4

- A: `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261022_211211.log` (`--ep-size 4 --attn-dp-size 1`, `LK_POOL_SLOTS=48`; 同配置复现日志 `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261022_195241.log`, `_180710.log`)
- B: `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261022_223401.log` (无 `--ep-size`, 无 `--attn-dp-size`, `LK_POOL_SLOTS=192`)
- 运行代码 = `/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/**` (与仓库 `/mnt/SSD01/models/.llm-env/Lsglang/lsglang/python/sglang/**` 仅差 `_eager_decode_prewarm` 补丁; 下文 file:line 除注明外均指 site-packages 副本)
- 模型头: `/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017/{config.json,model.safetensors.index.json,model-0000X-of-00048.safetensors}` 的 48 个 header (96085 个张量, 与 index 完全一致)
- 只读核算: 未启动服务、未占用 GPU、未改动 site-packages/脚本/配置; 未读 `deprecated/` 与任何 `*_DEPRECATED*`

## 0. 口径

| 项 | 值 | 依据 |
| --- | --- | --- |
| 日志 `mem usage` 定义 | `load_model()` 前空闲显存 − 载入后空闲显存 (设备级 `torch.cuda.mem_get_info`, 先 `empty_cache`) | `srt/model_executor/model_runner.py:1334` (`Load weight begin`), `:1425-1440` (`mem usage = before_avail − after_avail`), `srt/utils/common.py:478-626` |
| 日志 `GB` 的实际单位 | **GiB** (`free_gpu_memory / (1 << 30)`) | `srt/utils/common.py:626` |
| 单卡总显存 | 85,019,590,656 B = 79.18 GiB (81559 MiB) | A 日志 1569/1571/1573/1575 行 OOM 行 `total: 85019590656`; `nvidia-smi` (当前物理 4-7 空闲 81081 MiB) |
| 载入前基线 | 77.36 (TP1/TP2) / 77.41 (TP0) / 77.60 (TP3) GiB, 两轮逐位相同 | A 183/186/189/192 行; B 183/186/190/193 行 |
| `lk_moe` 设备侧字节口径 | `lk_moe.get_device_mem_usage()` = 池 slab/行 + prefill arena + Triton staging ring, 单位 GiB | `srt/mem_cache/kv_cache_configurator.py:218-236`; .so 字符串 `Persistent device bytes owned by lk_moe: pool rows (or the one slab) + prefill arena + the Triton staging ring` (`env/lib/python3.12/site-packages/lk_moe/_lk_moe_C_avx512_amx.so`) |

## 1. 三个变量

### 1.1 变量③ `attn_dp_size`

| 项 | 值 | 依据 |
| --- | --- | --- |
| 字段默认值 | `attn_dp_size = 1` | `srt/arg_groups/fields/parallel.py:81-91` (`A[int, Arg(...)] = 1`) |
| A 解析结果 | `'attn_dp_size': 1, 'dp_size': 1, 'enable_dp_attention': False` | A 日志 76 行 `server_args={...}` |
| B 解析结果 | `'attn_dp_size': 1, 'dp_size': 1, 'enable_dp_attention': False` | B 日志 76 行 `server_args={...}` |
| 唯一差别字段 | A `'ep_size': 4` / B `'ep_size': 1`; 其余 `tp_size=4`, `attn_cp_size=1`, `moe_dp_size=1`, `enable_shared_experts_attn_tp=False`, `shared_experts_tp_size=None`, `moe_a2a_backend='none'` 两轮相同 | A/B 日志 76 行 |
| `attn_dp_size>1` 时的复制语义 (本两轮均未触发) | 注意力按 `attn_tp_size = tp_size // attn_dp_size` 分片, `attn_dp_size>1` 才建 DP 注意力组 | `srt/arg_groups/fields/parallel.py:305-320` (`attn_tp_size`/`attn_dp_enabled` Derived 字段); `srt/distributed/parallel_state.py:2628,2736-2796` |
| 使用点 (与权重复制相关的判定) | `srt/models/deepseek_v4.py:4191,4257,4364,4414,4517`; `srt/models/deepseek_v4_mhc.py:593,671` 均为 `attn_dp_size ==/> 1` 分支, 本两轮取同一侧 | 同上 |

### 1.2 变量② `LK_POOL_SLOTS`

| 项 | 值 | 依据 |
| --- | --- | --- |
| 配置 | A `LK_POOL_SLOTS=48` / B `192`; `LK_POOL_WINDOW=32`, `LK_POOL_INFLIGHT=16`, `LK_POOL_STAGE=1` 两轮相同 | `configs/dsv41_tp4_h100.effective.20261022_211211.conf`, `..._223401.conf` |
| slab 上报 | 两轮均 `[pool] slab: 16.81 GiB (20 layers x 860.62 MiB)` ×4 rank | A 566/569/572/575 行, B 574/576/581/586 行 |
| 槽 ≤ 本地专家上限 | A 48 ≤ 96, B 192 ≤ 384 | A/B 日志 `Initialized lk_moe with {96,384} experts for layer N` |
| slab 字节恒等式 | `48 × 18,800,640 = 192 × 4,700,160 = 902,430,720 B/层`; ×20 层 = 18,048,614,400 B = 16.809 GiB | 见 §3.1 每专家字节 |

### 1.3 变量① `ep_size`

| 项 | 值 | 依据 |
| --- | --- | --- |
| 局部专家数 | A `num_local_experts=96` (=384/4), B `=384` | `srt/layers/moe/fused_moe_triton/layer.py:443-446` (`_num_local_routed = _num_global_routed // storage_ep_size`); 日志 `Initialized lk_moe with 96/384 experts for layer 0 [GPU]` ×4 rank (A 269-273 行, B 273-281 行) |
| MoE 张量并行宽度 | A `moe_tp_size=1` → 每专家全宽 2304; B `moe_tp_size=4` → 每专家 576 | `layer.py:452-454` (`intermediate_size_per_partition = intermediate_size // moe_tp_size`) |
| 补零触发 | 仅 B: `Padding MXFP4 experts from intermediate=576 to 640 ...` ×4 rank; 且仅驻留层 | B 196-199 行; `srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py:74-88` (`_pad_intermediate = intermediate % 128 != 0`), `:142-161` (非驻留层提前 return, 再 `_pad_intermediate_size`) |
| 补零实现尺寸 | `w13 → [E,2·640,K/2]`, `w2 → [E,H,320]`, `w13_scale → [E,1280,160]`, `w2_scale → [E,5120,20]` | `srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py:54-99` |
| 池层是否补零 | 否 (池层权重建在 CPU, 且非驻留层跳过该处理) | `srt/layers/quantization/fp8.py:1687-1689` (`device = "cpu"` for non-resident FusedMoE), `mxfp4_flashinfer_cutlass_moe.py:144-145` |
| 池层权重释放 | 载入后删除 `w13_weight/w2_weight/*_scale_inv` 等 12 个参数 | `layer.py:2045-2056` (`clean_weights_after_loading`, 非驻留层), 调用点 `srt/model_loader/loader.py:1091-1099` |

## 2. 主对账表 (单位 GiB; 1 GiB = 2^30 B)

| 组件 | 计算依据 (file:line / 日志行) | A (ep4+48) | B (ep1+192) | B−A |
| --- | --- | --- | --- | --- |
| 路由专家权重, 驻留主干层 0-19 (20 层) | 每专家 18,800,640 B (全宽) = 4,700,160 B (TP4 切分) = 5,222,400 B (补零 640); `layer.py:452-454`; 96 vs 384 专家 | 33.618 | 37.354 | **+3.735** |
| ├ 其中 128 对齐补零 | `384 × (5,222,400 − 4,700,160) × 20 层` = 4,010,803,200 B | 0 | 3.735 | +3.735 |
| 池 slab, 主干非驻留层 20-39 (20 层) | A `48×20×18,800,640`; B `192×20×4,700,160`; 均 = 18,048,614,400 B; 日志 566/569/572/575 (A), 574/576/581/586 (B) | 16.809 | 16.809 | 0 |
| 池层专家权重 (GPU 侧) | `fp8.py:1687-1689` 建在 CPU; `layer.py:2045-2056` 载入后删除 | 0 | 0 | 0 |
| 池解码缓冲 `pool_hbuf/gout/hpart/gout_part/slot_tab/ids/weights` | `layer.py:2454-2493`; `max_running_requests=16`, `top_k=6`, `H=5120`, `I=2304`(A)/`576`(B), `LK_VERIFY_RESIDENT=0` → `rows=96`; 本行取 `NGK=4` (NGK 由 kernel_caps 决定, 日志未打印; 全范围见 §4.3) | 0.010 (0.005–0.030) | 0.004 (0.002–0.009) | −0.006 |
| 池 H2D staging ring (`ring=8`) | 仅 B 有行 `[pool] H2D staging ON: ring=8 slab=4.70MB` ×4 rank (B 577/580/583/587 行); `8 × 4,700,160 B = 37.6 MB` | 0 | 0.035 | +0.035 |
| 池 prefill arena + Triton staging ring | `LK_GPU_PREFILL_MIN_TOKENS=0` 两轮; 对照: `_20261022_142033.log` (=1024) 有 `[pool] staging ring: 1.68 GiB (96 rows, 3 sets, window 1)` → arena 18.49 GiB, `mem usage=60.20 GB` | 0 | 0 | 0 |
| MTP/dspark 专家 (3 层 × 128 专家) | `speculative_algorithm=None` 两轮; 两轮 lk_moe 注册数 = 40 层 ×4 rank (A/B 均 `[GPU]`80 + `[CPU]`80 行, 最大层号 39) | 0 | 0 | 0 |
| 非专家权重 + CUDA context/内核模块 + 预热缓冲 (实测锚点, 不可再分) | 48/48 分片载入完成、逐层 MoE 转换之前: `77.41 − 37.0345 = 40.3755`; 其中驻留专家 = 33.618 → 本项 = 6.757; 两轮逐位相同 (A 233-260 行, B 231-262 行) | 6.757 | 6.757 | 0 |
| ├ 参照: 非专家权重的 checkpoint 折算 | `/attn_tp=4` 折算 **2.64–4.41 GiB** (attn 1.179 + indexer/compressor 0.019 + engram 0.073 + hc 0.037~0.146 + gate 0.147 + shared 0.330~1.318 + embed/head 0.617 + vision 0.226~0.904 + norm 0.01); 括号内三个范围项 (hc/shared/vision) 的切分方式未从日志确证, 两轮相同 | ~3.5 | ~3.5 | 0 |
| 小计 (专家 + slab + 池缓冲 + 锚点) | `33.618+16.809+0.010+6.757` / `37.354+16.809+0.004+0.035+6.757` | 57.194 | 60.959 | +3.765 |
| 实测总计 (`Load weight end`) | A 654/657/660/664 行 = 58.29/58.52/58.52/58.48 (均值 58.4525); B 665/670/673/676 行 = 69.79/70.03/70.03/69.98 (均值 69.9575) | 58.4525 | 69.9575 | **+11.505** |
| **仍未对上的余量** | 实测总计 − 小计 (H2D staging 若按层分配, 则 B 项再加 0.67) | **+1.26** | **+9.00 (9.00–9.67)** | **+7.74 (7.07–7.74)** |

## 3. 权重侧逐组件 (safetensors header 实测)

### 3.1 每专家字节 (以 `layers.0.ffn.experts.0.*` 为例)

| 张量 | dtype / shape | 字节 | A 每卡 (EP4, 全宽) | B 每卡 (TP4 切分) | B 补零后 |
| --- | --- | --- | --- | --- | --- |
| `w1.weight`/`w3.weight` (gate/up, 2 个) | I8 `[2304,2560]` | 5,898,240 ×2 | 5,898,240 ×2 | 1,474,560 ×2 | 1,638,400 ×2 |
| `w2.weight` (down) | I8 `[5120,1152]` | 5,898,240 | 5,898,240 | 1,474,560 | 1,638,400 |
| `w1/w3.scale` (ue8m0) | F8_E8M0 `[2304,160]` | 368,640 ×2 | 368,640 ×2 | 92,160 ×2 | 102,400 ×2 |
| `w2.scale` | F8_E8M0 `[5120,72]` | 368,640 | 368,640 | 92,160 | 102,400 |
| 合计/专家 | | **18,800,640 B = 17.930 MiB** | 同上 | **4,700,160 B = 4.4825 MiB** | **5,222,400 B = 4.9805 MiB** |

补零后 shape: `w13 [384,1280,2560]` = 1,258,291,200 B; `w2 [384,5120,320]` = 629,145,600 B; `w13_scale [384,1280,160]` = 78,643,200 B; `w2_scale [384,5120,20]` = 39,321,600 B (`mxfp4_flashinfer_trtllm_moe.py:54-99`)。

### 3.2 每层/每卡

| 项 | A | B (未补 / 补零) | 依据 |
| --- | --- | --- | --- |
| 驻留层专家/层/卡 | 1,804,861,440 B = 1.681 GiB | 1,804,861,440 B / 2,005,401,600 B = 1.681 / 1.868 GiB | `96×18,800,640` = `384×4,700,160` = `384×5,222,400`×(576/640) |
| 驻留 20 层合计/卡 | 36,097,228,800 B = 33.618 GiB | 36,097,228,800 B / 40,108,032,000 B = 33.618 / 37.354 GiB | ×20 |
| 池 slab/层/卡 | 902,430,720 B = 860.62 MiB | 902,430,720 B = 860.62 MiB | 日志 slab 行 |

### 3.3 其它组件 (checkpoint 全 rank 合计 → 每卡估算)

| 组件 | checkpoint 合计 | 每卡估算 | 切分依据 |
| --- | --- | --- | --- |
| attention 主体 (`wq_a/wq_b/wkv/wo_a/wo_b`) | 4.716 GiB | 1.179 GiB (`/4`) | MLA 按 attn_tp_size 分片 |
| indexer (`2-36` 层 8 个) + compressor (`2-20` 层 4 个) | 0.076 GiB | 0.019 GiB (`/4`) | 同上 |
| engram (`engram.wkv` 层 1/14 + q/k) | 0.293 GiB | 0.073 GiB (`/4`) | 同上 |
| hc_attn_fn/hc_ffn_fn (F32 `[24,20480]` ×40 层 ×2) | 0.146 GiB | 0.146 或 0.037 GiB | 复制/分片未定 (两轮相同) |
| `ffn.gate.weight/bias/bias_vl` (router) | 0.147 GiB | 0.147 GiB (复制, 不切分) | `[384,5120]` BF16 |
| shared experts (fp8 `[2304,5120]`×3 ×40 层) | 1.318 GiB | 0.330 (若按 attn_tp `/4`) / 1.318 (若复制) | `enable_shared_experts_attn_tp=False` |
| `embed.weight` + `head.weight` (BF16 `[129280,5120]`) | 2.466 GiB | 0.617 GiB (`/4`) | vocab 并行 |
| vision tower + aligner | 0.904 GiB | 0.904 或 0.226 GiB | 复制/分片未定 (两轮相同) |
| 各 norm/sink/base/scale | ~0.02 GiB | ~0.02 GiB | 复制 |
| `engram.embed.*` (196.6 + 6.1 GiB) | 202.7 GiB | 0 (host 表) | 日志 `engram host table layer 1/14: layout=shared, 24170 MiB resident` ×4 rank |
| MTP 命名空间 (`mtp.0/1/2`, 2401 张量, 每层 128 专家) | 7.388 GiB | 0 (未实例化) | §2 行 8 |

## 4. 非权重侧: 随「本地专家数 / 池槽数 / moe_tp」缩放的 GPU 分配清单

### 4.1 随本地专家数 (96 vs 384)

| 分配 | 尺寸式 | A | B | 依据 |
| --- | --- | --- | --- | --- |
| 专家权重/scale 参数缓冲 | `num_local_experts × ...` | 96 | 384 | `layer.py:629-641` 传 `num_experts=self.num_local_experts`; `fp8.py:1734-1754,1830-1857` (I8 紧凑 fp4 + ue8m0) |
| 每专家 SwiGLU 标量 (若走 mxfp4.py 分支) | `fp32 [E] ×3` | 1.2 KB | 4.6 KB | `mxfp4.py:1278-1289` |
| `ffn.gate.weight/bias` (router) | `[384,5120]` BF16, 不随 EP 变 | 3.93 MB | 3.93 MB | checkpoint header |
| `moe_align_block_size`/sorted-token 工作区 | `f(num_experts×block)` (KB 级) | — | — | `fused_moe_triton` 路径, 未持久 |
| FlashInfer CUTLASS MoE workspace | 默认**每次调用**分配并释放, 未传 `workspace_buffer` | 0 | 0 | `flashinfer/fused_moe/core.py:1496-1520`, `:1222-1280`; `srt/` 内 `workspace_buffer=` 的命中均属 VL 模型 attention 路径, MoE/CUTLASS 路径无传参 |
| Humming W4A8 载荷改写临时量 | 未打补丁时 = `E·rows·K·8 B` (A 实测 OOM 申请 18,119,393,280 B) | 0 (已打补丁) | 0 (已打补丁) | `logs/OOM-evidence-20261022_141442-ep4.md`; `flashinfer/fused_moe/prepare.py:203-251` (`torch.int32` + 按专家分块, `chunk_experts = max(1, 8Mi/每专家元素数)`) |

### 4.2 随池槽数 (48 vs 192)

| 分配 | 尺寸式 | A | B | 依据 |
| --- | --- | --- | --- | --- |
| slab (权重+scale 合并) | `slots × 成员层数 × 每槽字节` | 16.809 GiB | 16.809 GiB | 日志 slab 行 |
| H2D staging slab | `ring(8) × 每槽字节 4,700,160 B` | 无此行 | 37.6 MB | B 577/580/583/587 行 (`slab=4.70MB` = 每槽) |
| 槽表/账本 (`pool_slot_tab` 等) | `rows(96) × 4 B` 级 | ~1 KB | ~1 KB | `layer.py:2476-2480` |

### 4.3 随 `moe_tp_size` / `intermediate_per_partition` (2304 vs 576)

| 分配 | 尺寸式 | A (I=2304) | B (I=576) | 依据 |
| --- | --- | --- | --- | --- |
| `pool_hbuf` | `rows(96) × 2I × 4 B` | 1,769,472 B | 442,368 B | `layer.py:2480` |
| `pool_gout` | `16 × H × 4 B` | 327,680 B | 327,680 B | `layer.py:2481-2482` |
| `pool_gout_part` | `rows × H × 4 B` | 1,966,080 B | 1,966,080 B | `layer.py:2493` |
| `pool_hpart` | `NGK × rows × 2I × 4 B` (NGK∈{1..16}) | 1.77–28.31 MB | 0.44–7.08 MB | `layer.py:2492`; `NGK` 由 kernel_caps 决定, 日志未打印 |
| `pool_slot_tab / gathered_ids / gathered_weights` | `rows × {4,4,4} B` | 1,152 B | 1,152 B | `layer.py:2483-2485` |
| `FusedMoE.output_gpu` | `max_num_seqs(16) × H × 4 B` | 327,680 B | 327,680 B | `layer.py:2405-2410` (类级一次分配) |
| 4× 对称性 | `96×2304 = 384×576 = 221,184` 列 | 相同 | 相同 | 见 §3.1 |

### 4.4 lk_moe 自身在日志中报告过的 GPU 分配

| 上报项 | A | B | 依据 |
| --- | --- | --- | --- |
| `lk_moe static device arena` | 16.81 GiB | 16.81 GiB | A 673-676 行; B 685-688 行; `kv_cache_configurator.py:218-236` |
| `[pool] slab` | 16.81 GiB (20×860.62 MiB) | 16.81 GiB (20×860.62 MiB) | A 566/569/572/575; B 574/576/581/586 |
| `[pump] ON` | `slots=48 window=32 inflight=16 log=100` | `slots=192 window=32 inflight=16 log=100` | A 565/568/571/574; B 573/575/579/585 |
| `[pool] H2D staging ON` | 无行 | `ring=8 slab=4.70MB` | B 577/580/583/587 |
| `[pool] staging ring` | 无行 (prefill tier 关) | 无行 | 对照 `_20261022_142033.log:333` |
| `[pool] wave width` / `block plan` / `slab skipped` / `slab cudaMalloc failed` / `stride mismatch` / `L%d ledger unmapped` | 无行 | 无行 | 两日志全文 grep 无命中 (指针表见 .so 字符串) |

## 5. 两次日志逐行对齐 (含 GB/MiB/slab/arena/workspace/bytes/mem 的行)

| 类别 | A | B | 依据 |
| --- | --- | --- | --- |
| `Load weight begin. avail mem` | 77.60(TP3)/77.41(TP0)/77.36(TP1,TP2) | 同上逐位相同 | A 183/186/189/192; B 183/186/190/193 |
| `Memory baseline taken. mem usage` | 0.00 GiB ×4 | 0.00 GiB ×4 | A 167-170; B 167-170 |
| 预热期 `Required memory for warmup / Available memory` | 7 组, 末组 0.15431594848632812/37.03448486328125 | 7 组, 末组逐位相同 | A 233/237/241/248/252/256/260; B 231/235/239/246/250/254/262 |
| `Load weight end` | 58.29/58.48/58.52/58.52, avail 19.31/18.93/18.84/18.84 | 69.79/69.98/70.03/70.03, avail 7.80/7.43/7.34/7.34 | A 654/657/660/664; B 665/670/673/676 |
| `[pool] slab` | 16.81 GiB (20×860.62 MiB) ×4 | 16.81 GiB (20×860.62 MiB) ×4 | 见 §4.4 |
| `[pool] H2D staging ON: ring=8 slab=4.70MB` | 0 行 | 4 行 | 见 §4.4 |
| `Padding MXFP4 experts from intermediate=576 to 640` | 0 行 | 4 行 | B 196-199 |
| `flashinfer_mxfp4 MoE padded the local intermediate size from 576 to 640` | 0 行 | 4 行 | B 274/277/279/281 |
| `Preparing DSv4 MXFP4 experts for FlashInfer SM90 CUTLASS W4A8` | 20 行 | 20 行 | A 270-…; B 275-… |
| `Initialized lk_moe with N experts ... [GPU]/[CPU]` | 96 experts, 80+80 行 | 384 experts, 80+80 行 | 见 §1.3 |
| `[SymmDeviceMemory] Rank: r, Group size: 4, device_idx: r, Signal pad offset: 0` | 24 行 | 0 行 | A 全文; B 无 |
| `FlashInfer AllReduce Fusion enabled and workspace initialized: backend=trtllm, ... max_token_num=2048, hidden_dim=5120` | 8 行 | 0 行 | A 735-738/751-754 |
| `DSV4 memory calculation ... available_bytes` | 9.95 GiB ×4 | 无 (KV 定容前抛错) | A 680-687; B 尾部 `ValueError: ... minimum viable = 1 - available/pre = 0.8988` (B 714/741/768/795) |
| `lk_moe static device arena` | 16.81 GiB | 16.81 GiB | 见 §4.4 |
| `Memory pool end. avail mem` | 12.21/12.30/12.21/12.68 | 无 | A 705-708 |
| `Capture target decode CUDA graph begin ... avail mem` | 11.39/11.48/11.86/11.39 | 无 | A 775-778 |
| OOM/CUDACachingAllocator | `trying to allocate 9,799,991,296 bytes (free: 7,240,744,960 / 7,339,311,104 / 7,744,061,440)`, 4 rank | 无 | A 1569/1571/1573/1575 |
| 终态 | `Segmentation fault` → `Rank 0 scheduler died ... exit code: -11` | `ValueError: Loaded weights leave no GPU memory for the KV cache` | A 尾部; B 尾部 |

## 6. 过程数据 (换算)

| 项 | A | B | 依据 |
| --- | --- | --- | --- |
| 48/48 分片载入完成后 (预热期测量点) 已用 (TP0) | 77.41 − 37.0345 = **40.3755 GiB** | 77.41 − 37.0345 = **40.3755 GiB** | §5 行 4/5 |
| 预热末 → `Load weight end` 增量 (TP0) | 58.48 − 40.3755 = **+18.10 GiB** | 69.98 − 40.3755 = **+29.60 GiB** | 同上 |
| 该增量中的 slab | +16.809 GiB | +16.809 GiB | 日志 slab 行 |
| 该增量中的专家补零 | 0 | +3.735 GiB | §2 |
| 该增量其余部分 | +1.29 GiB | +9.06 GiB | 差值 |
| `Load weight end` 两轮差 | — | — | **+11.505 GiB** |
| 其中已定位 (补零) | — | — | +3.735 GiB |
| 未定位 | — | — | +7.74 GiB (若 H2D staging 按层分配则 7.07 GiB) |
| 参考: 40 层 × 191.25 MiB 补零量 | — | — | 7.471 GiB |
| 参考: 23 层 (20 驻留 + 3 MTP) × 191.25 MiB | — | — | 4.296 GiB |

## 7. 复现性检查

| 运行 | 配置 | `Load weight end` 值 (TP0/TP1/TP2/TP3) |
| --- | --- | --- |
| `serve_dsv41_tp4_h100_20261022_180710.log:649-662` | ep4 + 48 | 58.48 / 58.52 / 58.52 / 58.29 |
| `serve_dsv41_tp4_h100_20261022_195241.log:655-664` | ep4 + 48 | 58.48 / 58.52 / 58.52 / 58.29 |
| `serve_dsv41_tp4_h100_20261022_211211.log:654-664` | ep4 + 48 | 58.48 / 58.52 / 58.52 / 58.29 |
| `serve_dsv41_tp4_h100_20261022_142033.log:645-654` | ep4 + 48 + `LK_GPU_PREFILL_MIN_TOKENS=1024` | 60.16 / 60.20 / 60.20 / 59.97 (arena 18.49 GiB = slab 16.81 + staging ring 1.68) |
| `serve_dsv41_tp4_h100_20261022_223401.log:665-676` | ep1 + 192 | 69.98 / 70.03 / 70.03 / 69.79 |

（`configs/dsv41_tp4_h100.effective.20261022_215036/215055.conf` 为「无 ep + 48 槽」组合, `215440/215501.conf` 为「无 ep + 192 槽」, 无对应日志文件; `logs/` 下无 `serve_dsv41_tp4_h100_2026150*.log`。）

---

## 8. 附录:转换后显式释放的门控补丁 `SGLANG_DSV41_POST_CONVERT_RELEASE`

### 8.1 落点

| 项 | 值 |
| --- | --- |
| 载体 | `env/lib/python3.12/site-packages/sglang/srt/mem_cache/kv_cache_configurator.py` |
| 调用点 | `_profile_available_bytes()` 内,**紧跟上游的 `gc.collect()` 之后、`get_available_gpu_memory(...)` 之前**(`orig6` 行 `:2709` 之后),即"转换后、定容前"的最后时机 |
| 新方法 | `_dsv41_post_convert_release()`,插入在 `_profile_available_bytes` 定义之前 |
| 门 | `SGLANG_DSV41_POST_CONVERT_RELEASE`,默认 `"0"` |
| 备份 | `.orig6`;apply/rollback 脚本同名规则;`--dry-run` 保护、幂等、失败自动还原 |

### 8.2 上游对同一现象的自述(本地支撑)

`kv_cache_configurator.py:2701-2708` 的注释原文:

> "Weight-loading temporaries can still be referenced at this point, and `empty_cache()` (which `get_available_gpu_memory` already calls) **cannot reclaim referenced blocks**. Without collecting first, the KV budget is measured against an understated free-memory figure and the pool can be sized orders of magnitude too small while GPU memory sits idle."

⇒ 上游**已知**此处可能有存活引用,但其处置只有 `gc.collect()`(`:2709`)——`gc.collect()` 对**活引用**无效。这正是本补丁要补的位。

### 8.3 释放了什么 / 为什么安全

| 释放对象 | 依据 |
| --- | --- |
| **驻留 `FusedMoE` 层上的 8 个转换遗留属性**:`w13_weight_packed` / `w2_weight_packed` / `w13_weight_scale` / `w2_weight_scale` / `w13_weight_global_scale` / `w2_weight_global_scale` / `w13_weight_scale_2` / `w2_weight_scale_2` | 它们正是 `FusedMoE.clean_weights_after_loading()`(`fused_moe_triton/layer.py:2048-2053`)的删除清单成员,但该函数**对驻留层提前返回**(`:2046-2047`)⇒ 驻留层上若存在这些名字即为死重 |
| `gc.collect()` + `torch.cuda.empty_cache()`(前后各一次) | 释放一切已无引用的缓存块 |

**明确不释放(必须保留)**:`w13_weight` / `w2_weight` / `w13_weight_scale_inv` / `w2_weight_scale_inv` / `w13_humming_residual_scale` / `w2_humming_residual_scale` / `humming_fc2_act_scale` —— 这 7 个是 `Mxfp4FlashinferCutlassMoEMethod.apply()`(`mxfp4_flashinfer_cutlass_moe.py:266-276`)在运行期实际读取的载荷,补丁以白名单显式排除。**不改动任何权重内容/布局/参数指向**,只求删除对象属性。

### 8.4 预计回收量

| 情形 | 预计 |
| --- | --- |
| 若余量正是这些遗留属性(每层一份 128 对齐后的旧布局) | 约 **3.7 GiB**(= 20 层 × 191.25 MiB,与 §2 已定位的补零增量同量级) |
| 若余量是别的持有者 | **0**,并由诊断输出指名(见 8.5) |

### 8.5 为什么这次"释放"同时是"定位"

补丁把"释放多少"变成**可测量**而非假设:

1. 每个阶段用 `torch.cuda.mem_get_info()` 前后差打一行 `reclaimed=`,三阶段: `gc+empty_cache` → `drop-convert-leftovers` → `gc+empty_cache-2`;
2. **无条件**追加一条 `warning`,**枚举存活的最大 8 组 CUDA 张量**(按 `(dtype, shape)` 聚合字节数),并对每组取一个代表张量调 `gc.get_referrers()` 打印**引用者类型名** ⇒ 直接指出持有者是谁。(早先版本用 `reclaimed < 0.05 GiB` 作为触发门槛,该门槛**无意义**并被废弃:一轮实测三阶段回收 `0.50 GiB`(普通 GC)即越过门槛,而真正要追的 ≈7 GiB 持有者因此未被打印。字节门槛无法区分"普通 GC 回收"与"目标持有者",故改为无条件枚举。)

⇒ 一次运行即可回答"谁还在引用",无需 profiler。

### 8.6 门关闭时的等价性

门≠1 时新增代码只有一条 `__import__("os").environ.get("SGLANG_DSV41_POST_CONVERT_RELEASE", "0") == "1"` 判断(假),`_dsv41_post_convert_release()` **不被调用**;无任何分配、无 `gc.collect()`、无 `empty_cache()`、无属性删除、无日志 ⇒ **逐字节等价、零副作用**。

### 8.7 量具全量化(本轮修订)

| # | 修订 | 落点 |
| --- | --- | --- |
| 1 | **全量枚举**:列出**所有** ≥ **64 MiB** 的存活 CUDA 张量分组(按 `(dtype, shape)` 聚合字节数、GiB 降序、**不设条数上限**);超过 200 组时打印前 200 组并追加一行 `...还有 N 组, 合计 X GiB`;最后一行给总数与总计字节 | `kv_cache_configurator.py` 的 `_diagnose()`(每行带 `ref_by` 引用者类型名) |
| 2 | **非 torch 占用**:同一时刻打印 `torch_allocated` / `torch_reserved` / `mem_get_info` 空闲 / `device_used` 四个原始值,并算出 **`non_torch = device_used − torch_reserved`** | `kv_cache_configurator.py` 新增 `_nontorch()` |
| 3 | **按层空闲轨迹**:每次转换结束调用(每 4 层一行,rank0),打印 `free=X GiB`;**不加额外的 `empty_cache`**,以免抹掉信息;另在"转换全部结束、定容前"打印一行 `free=X GiB (定容前)` | 新增第二文件: `mxfp4_flashinfer_cutlass_moe.py` 的 `process_weights_after_loading()` 末尾(计数用模块级 `_dsv41_convert_trace_n`) |

**为什么(2)是本轮重点**:lk_moe 的 `.so`、cuBLAS/CUTLASS workspace、NCCL 缓冲由 `cudaMalloc` 直接申请,**任何张量遍历都看不到**;而 ep1(本地专家 384)相对 ep4(96)会让这类缓冲可能 ×4。`non_torch` 这一行把"张量看不到的那部分"显式量化出来。

**为什么取消条数上限与字节门槛**:一轮实测三阶段只回收 0.50 GiB(普通 GC),旧门槛 `< 0.05 GiB` 因此被判为"有回收"而**跳过枚举**;且旧设计只列 top-8,一轮根本看不到 ≈7 GiB 的持有者。字节门槛无法区分"普通 GC"与"目标持有者",条数上限则直接遮蔽目标 ⇒ 二者同时取消。

**门与等价性**:仍为 `SGLANG_DSV41_POST_CONVERT_RELEASE`,默认 `"0"`。关闭时两处新增代码各只有一条 env 判断(假)⇒ 不枚举、不统计、不打印、不分配、不改数值行为。

**预计一次运行能看到**:① 3 行 `stage=… reclaimed=…`;② 1 行 `free=… (定容前)`;③ 1 行 `non_torch=…`;④ `live#000…` 起的**完整**张量分组清单(≥64 MiB)+ 总计行;⑤ 转换期每 4 层 1 行 `convert-trace n=… free=…`。
