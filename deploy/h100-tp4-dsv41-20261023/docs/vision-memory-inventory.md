# DeepSeek-V4.1-Flash-1017 视觉部分显存与并发核查

核查对象与只读方式
- 权重：`/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017`（48 个 `.safetensors` 分片）
- 运行时代码：`/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang`（sglang 0.5.15.dev105，lsglang 1.6.1）
- 生效配置：`/mnt/SSD01/models/.llm-env/Lsglang/configs/dsv41_tp4_h100.effective.20261023_013857.conf:54`
- 启动日志：`/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261023_013857.log`
- 权重读取方式：仅读每个分片前 8 字节长度 + header JSON（`struct` + `json`），未读取任何张量数据区；48 个分片 header 合计读取 10,684,928 B（≈ 10.19 MiB）
- 未启动服务、未调用除 `nvidia-smi` 查询以外的 GPU 工具、未修改任何被核查文件

---

## 1. 视觉塔 / projector 权重字节（取自 safetensors header）

### 1.1 张量定位

对 48 个分片 header 全量扫描（正则匹配 `aligner|vision|projector|merger|adapter|image|visual|mm_|patch_embed|deepstack|resampler|connector`）：

| 分片 | 张量总数 | 命中 mm 关键字的张量数 | header 字节 |
| --- | --- | --- | --- |
| `model-00001-of-00048.safetensors` | 263 | 263 | 27,376 |
| `model-00002-of-00048.safetensors` | 4 | 3 | 344 |
| 其余 46 个分片 | — | 0 | — |

- `model-00001-of-00048.safetensors` 的 263 个张量 = `vision.*` 259 个 + `aligner.*` 4 个，**全部**为视觉相关，文件内无任何文本塔张量
- `model-00002-of-00048.safetensors` 的 4 个张量 = `embed.weight [129280,5120]` + `image_end [5120]` + `image_newline [5120]` + `image_start [5120]`
- `vision.patch_embed.proj.weight` 形状 `[1024, 588]`，588 = 3×14²（`deepseek_v41_vit.py:41`）

### 1.2 主表：视觉相关项（未按卡复制的单份字节）

| 项 | 张量数 | dtype | 字节 | GiB | 是否按卡复制 | 依据 file:line / header |
| --- | --- | --- | --- | --- | --- | --- |
| `vision.patch_embed.proj.weight` `[1024,588]` | 1 | BF16 | 1,204,224 | 0.001123 | 是（每卡一份） | shard 00001 header `data_offsets [969302016, 970506240]` |
| `vision.patch_embed.proj.bias` `[1024]` | 1 | BF16 | 2,048 | 0.000002 | 是 | shard 00001 header `data_offsets [969299968, 969302016]` |
| **`vision.patch_embed.proj` 小计** | 2 | BF16 | **1,206,272** | **0.001123** | 是 | 同上 |
| `vision.blocks.N.*` 单层 | 8 | BF16 | 25,702,400 | 0.023937 | 是 | shard 00001 header，32 层逐层字节完全相同 |
| **`vision.blocks` 32 层小计** | 256 | BF16 | **822,476,800** | **0.765991** | 是 | 25,702,400 × 32 |
| `vision.norm.weight` `[1024]` | 1 | BF16 | 2,048 | 0.000002 | 是 | shard 00001 header `data_offsets [969297920, 969299968]` |
| **`vision.*` 合计** | **259** | BF16 | **823,685,120** | **0.767117** | **是** | 上三项之和 |
| `aligner.w1.weight` `[5120,9216]` | 1 | BF16 | 94,371,840 | 0.087891 | 是 | shard 00001 header `data_offsets [10240, 94382080]` |
| `aligner.w1.bias` `[5120]` | 1 | BF16 | 10,240 | 0.000010 | 是 | shard 00001 header `data_offsets [0, 10240]` |
| `aligner.w2.weight` `[5120,5120]` | 1 | BF16 | 52,428,800 | 0.048828 | 是 | shard 00001 header `data_offsets [94392320, 146821120]` |
| `aligner.w2.bias` `[5120]` | 1 | BF16 | 10,240 | 0.000010 | 是 | shard 00001 header `data_offsets [94382080, 94392320]` |
| **`aligner.*` 合计（= mm projector）** | **4** | BF16 | **146,821,120** | **0.136738** | **是** | 上四项之和 |
| `image_start` `[5120]` | 1 | BF16 | 10,240 | 0.000010 | 是 | shard 00002 header |
| `image_newline` `[5120]` | 1 | BF16 | 10,240 | 0.000010 | 是 | shard 00002 header |
| `image_end` `[5120]` | 1 | BF16 | 10,240 | 0.000010 | 是 | shard 00002 header |
| **`image_*` 合计** | 3 | BF16 | **30,720** | **0.000029** | 是 | 上三项之和 |
| **视觉相关总计** | **266** | BF16 | **970,536,960** | **0.903883** | **是** | 823,685,120 + 146,821,120 + 30,720 |

### 1.3 单层 `vision.blocks.N.*` 细分（32 层形状一致，取自 shard 00001 header）

| 子模块 | 形状 | 字节 | 每层 GiB | 32 层合计字节 |
| --- | --- | --- | --- | --- |
| `attn.wqkv.weight` | `[3072,1024]` | 6,291,456 | 0.005859 | 201,326,592 |
| `attn.wqkv.bias` | `[3072]` | 6,144 | 0.000006 | 196,608 |
| `attn.wo.weight` | `[1024,1024]` | 2,097,152 | 0.001953 | 67,108,864 |
| `attn.wo.bias` | `[1024]` | 2,048 | 0.000002 | 65,536 |
| `mlp.w1.weight` | `[5632,1024]` | 11,534,336 | 0.010742 | 369,098,752 |
| `mlp.w2.weight` | `[1024,2816]` | 5,767,168 | 0.005371 | 184,549,376 |
| `norm1.weight` | `[1024]` | 2,048 | 0.000002 | 65,536 |
| `norm2.weight` | `[1024]` | 2,048 | 0.000002 | 65,536 |
| **每层合计** | — | **25,702,400** | **0.023937** | **822,476,800** |

`mlp.w1` 输出 5632 = 2 × 2816（SwiGLU gate+up 打包，`deepseek_v41_vit.py:85-90`）；`attn.wqkv` 输出 3072 = 3 × 1024（`deepseek_v41_vit.py:53-62`，16 heads × 64 = 1024）。

### 1.4 mm projector / merger / adapter 命名核查

| 候选命名 | 是否存在于 checkpoint | 字节 | 依据 |
| --- | --- | --- | --- |
| `multi_modal_projector` | 否 | — | 48 分片 header 全量正则匹配无此键 |
| `vision.merger` / `vision.proj` / `vision.deepstack` / `vision.resampler` | 否 | — | 同上 |
| `aligner.w1` / `aligner.w2` | **是** | 146,821,120 | shard 00001 header；实现见 `deepseek_v41_vit.py:138-151` |
| `image_start` / `image_newline` / `image_end` | **是** | 30,720 | shard 00002 header；`deepseek_v4.py:4808-4810` |

- `aligner` 即 hidden 1024 → 文本 5120 的投影：`in_dim = vision_dim × downsample_ratio² = 1024 × 3² = 9216`（`deepseek_v41_vit.py:141-142`），`w1: 9216→5120`（线性 + bias），`w2: 5120→5120`（线性 + bias），中间 `gelu`（`deepseek_v41_vit.py:151`）
- 9216 = 1024 × 3 × 3 对应 3×3 空间聚合；输出 5120 = `text_config.hidden_size`（`config.json`）

### 1.5 GPU 侧 dtype 修正（非 header 字节）

| 项 | 张量数 | checkpoint 字节 (BF16) | GPU 实际字节 (FP32) | 差值 |
| --- | --- | --- | --- | --- |
| vision 的 RMSNorm 权重（`vision.norm.weight` + 32×`norm1/norm2`） | 65 | 133,120 | 266,240 | +133,120 B = 0.000124 GiB |

依据：
- `deepseek_v41_vit.py:19`：`RMSNorm(dim, eps=1e-6, weight_dtype=torch.float32, force_native=True)`
- `sglang/srt/layers/layernorm.py:474`：`self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))`
- 其余 `nn.Linear`（patch_embed / MLP / Aligner）均未传 `quant_config`（`deepseek_v41_vit.py:41, 85-86, 143-144`，调用点 `deepseek_v4.py:4805-4807` 未传 quant），创建于 `set_default_torch_dtype(model_config.dtype)`（bf16）上下文内（`sglang/srt/model_loader/loader.py:840`）

---

## 2. 是否按 TP 分片

**判定：按卡复制（replicated），每卡各持一整份。**

每卡字节（4 卡 TP=4）：
- `vision.*` 每卡 **0.767117 GiB**
- `aligner.*` 每卡 **0.136738 GiB**
- `image_start/newline/end` 每卡 **0.000029 GiB**
- 合计每卡 **0.903883 GiB**（10 进制 970,536,960 B）
- 4 卡合计 **3.615532 GiB**；相对单副本多出 **2.711649 GiB**

依据：

| 环节 | file:line | 内容 |
| --- | --- | --- |
| 模块构造无 rank 门控 | `sglang/srt/models/deepseek_v4.py:4786-4810` | 条件仅为 `model_type == "deepseek_v41" and vision_n_layers > 0 and not language_model_only`；`self.vision = ViT(args)`、`self.aligner = Aligner(args)` 在该分支内对每个 rank 都执行 |
| 权重装载无 rank 跳过 | `sglang/srt/models/deepseek_v4.py:5966-5980` | `skip_group = "vision"` 仅在 `self.vision is None` 时成立；vision 已构造则 `vision.*` / `aligner.*` / `image_*` 走正常装载 |
| 参数名映射保留原前缀 | `sglang/srt/models/deepseek_v4.py:5232-5237` | `vision.*` → `attn.qkv_proj` / `attn.proj`；`aligner.*`、`image_*` 原样 |
| 视觉注意力声明数据并行 | `sglang/srt/models/deepseek_v41_vit.py:53-62` | `Attention(VisionAttention)` 传 `use_qkv_parallel=True, use_data_parallel=True` |
| 数据并行 ⇒ tp_size=1 | `sglang/srt/layers/attention/vision.py:1123-1124` | `self.tp_size = 1 if use_data_parallel else attn_tp_size`；`self.tp_rank = 0 if use_data_parallel else attn_tp_rank` |
| QKV 线性层不切分 | `sglang/srt/layers/attention/vision.py:1196-1215` | `QKVParallelLinear(..., parallel_group="replicated" if use_data_parallel else "attn_tp")` |
| 输出投影不切分 | `sglang/srt/layers/attention/vision.py:1227-1235` | `RowParallelLinear(..., parallel_group="replicated" if use_data_parallel else "attn_tp")` |
| `"replicated"` 语义 | `sglang/srt/layers/linear.py:121-122` | `if parallel_group == "replicated": return None`（无 group） |
| `None` → (rank 0, size 1) | `sglang/srt/distributed/utils.py:63-65` | `return (0, 1) if group is None else (group.rank_in_group, group.world_size)` |
| 归一化/MLP/Aligner | `sglang/srt/models/deepseek_v41_vit.py:96-99, 121, 143-144` | 原生 `RMSNorm` / `nn.Linear`，无并行包装 |
| 运行日志佐证 | `serve_dsv41_tp4_h100_20261023_013857.log:186-197` | TP0、TP1、TP2、TP3 各自打印 `Multimodal attention backend not set. Use fa3.` / `Using fa3 as multimodal attention backend.` ⇒ 4 个 rank 都构造了视觉注意力模块 |
| 运行日志佐证 | 同上 `:232,246,261,269` | 4 个 rank 各自 `MoE vision-gate prewarm: 1 router spec(s) warmed` |

反向核对（假设值，未发生，仅列字节）：若 `vision.*`/`aligner.*` 按 TP4 列/行切分，则
- `vision.*` 每卡 206,219,264 B = 0.192057 GiB（复制值 ÷ 3.994）
- `aligner.*` 每卡 36,720,640 B = 0.034199 GiB（复制值 ÷ 3.998）
- 合计每卡 0.226255 GiB，4 卡 0.905022 GiB

---

## 3. 图像并发上限（未设 mm 参数时的默认值）

### 3.1 先否证一个开关

- `--mm-max-concurrent-calls`：对 `sglang/srt/**/*.py` 全量检索，**该字符串不存在**；亦不存在 `SGLANG_MM_MAX_*` / `SGLANG_MM_CONCURRENT*` 等环境变量（`sglang/srt/environ.py` 全文检索无匹配）
- 因此该版本没有图像级并发闸门这一层；图像并发由请求级并发 + 每请求上限 + 预处理线程 + prefill token 预算共同决定

### 3.2 mm 相关字段默认值 / 本次生效值

| 字段 | 代码默认值 | 本次生效值 | 位置 |
| --- | --- | --- | --- |
| `limit_mm_data_per_request` | `None`（= 不限） | `None` | `sglang/srt/arg_groups/fields/mm.py:111-117`；日志 `:78` |
| `mm_max_concurrent_calls` | 不存在 | 不存在 | 全量检索无匹配 |
| `max_running_requests` | `None`（自动） | `16` | `arg_groups/fields/schedule.py:31-34`；日志 `:78` |
| `max_queued_requests` | `None` | `None` | `arg_groups/fields/schedule.py:35-38`；日志 `:78` |
| `max_prefill_tokens` | `16384` | `16384` | `arg_groups/fields/schedule.py:67-77`；日志 `:78` |
| `chunked_prefill_size` | `None` | `8192` | `arg_groups/fields/schedule.py:52-55`；日志 `:78` |
| `enable_prefix_mm_cache` | `False` | `False` | `arg_groups/fields/mm.py:56-58`；日志 `:78` |
| `enable_mm_global_cache` | `False` | `False` | `arg_groups/fields/mm.py:118-121`；日志 `:78` |
| `mm_enable_dp_encoder` | `False` | `False` | `arg_groups/fields/mm.py:59-62`；日志 `:78` |
| `mm_process_config` | `None` | `{}`（空） | `arg_groups/fields/mm.py:63-69`；日志 `:78` |
| `mm_processor_worker_num` | `0` → 自动 | `2`（日志明示 auto） | `arg_groups/fields/mm.py:70-75`；解析 `multimodal/processors/base_processor.py:756-776`；日志 `:92` |
| `mm_io_worker_num` | `0` → 自动 | auto 值 `4` | `arg_groups/fields/mm.py:76-81`；`base_processor.py:237,329-341`；日志 `:78` |
| `tokenizer_worker_num` | `1` | `1` | `arg_groups/fields/serving.py:50-53`；日志 `:78` |
| `mm_preprocess_cache_size_mb` | `None`（模型相关默认） | `None` | `arg_groups/fields/mm.py:93-104`（该缓存为 CPU，不预留显存） |
| `image_processor_backend` | `"auto"` | `"auto"` → 实际走 PIL/CPU | `arg_groups/fields/mm.py:122-126`；`processors/deepseek_v41.py:42-63`；日志 `:91,155` |
| CPU 预处理线程池 | `SGLANG_CPU_WORKERS` 未设 → `os.cpu_count()` | `os.cpu_count()` = 192 | `base_processor.py:514-520`；`nproc` = 192 |
| 单图预处理调度 | 每请求内 `asyncio.to_thread`（事件循环默认线程池） | 同左 | `processors/deepseek_v41.py:98,104` |

补充事实：`mm_processor_executor`（上面那 2 个隔离线程）由 `process_and_combine_mm_data_async` 使用（`base_processor.py:2039-2054`）；而 V4.1 处理器完整覆写了 `process_mm_data_async`（`processors/deepseek_v41.py:76-127`），调用点是 `sglang/srt/managers/tokenizer_manager.py:1216,1231`，不经过该 executor。其单图工作走 `asyncio.to_thread`。

### 3.3 图像 token 上限如何参与调度

| 量 | 值 | 依据 |
| --- | --- | --- |
| `vision_max_n_token`（= config 的 `max_image_tokens`） | `1024` | `config.json` → `sglang/srt/configs/deepseek_v41.py:13`（映射 `max_image_tokens`→`vision_max_n_token`）→ `sglang/srt/configs/deepseek_v4.py:128` |
| `vision_min_pixels`（= config 的 `min_pixels`） | `295936` | `sglang/srt/configs/deepseek_v4.py:129` |
| `vision_max_wh_ratio`（= config 的 `max_wh_ratio`） | `None`（不触发裁切） | `sglang/srt/configs/deepseek_v4.py:130`；判空分支 `multimodal/deepseek_v41_image_processing.py:73-77,107-110` |
| 每图 LLM token 数公式 | `n_llm_h × (n_llm_w + 1) + 2` | `multimodal/deepseek_v41_image_processing.py:25-26` |
| 上限施加点 | `safe_resize(...)` 内 `assert num_image_tokens ≤ max_n_token` | `multimodal/deepseek_v41_image_processing.py:54-67, 84-92` |
| 下限施加点 | `width*height < min_pixels` 时按 `(min_pixels/(w*h))**0.5` 放大 | `multimodal/deepseek_v41_image_processing.py:78-81` |
| 图像 token 是否进 prefill 预算 | **是**：处理器把展开后的占位 token 直接写入 `input_ids` | `multimodal/processors/deepseek_v41.py:112-114`（`count = len(image_token_types(lh, lw))`，`tokens.extend([image_token_id]*count)`） |
| 该 token 数进入调度 | `items_offset` 参与 `_count_mm_tokens_in_extend` | `managers/mm_schedule.py:657-676, 737-741` |

由上述常量纯算式得到（`p=14`、`r=3`、`min_pixels=295936`）：

| 量 | 值 |
| --- | --- |
| 544×544 图（= min_pixels 正方形） | best 546×546 → ViT patch 网格 39×39 = 1521 patch → LLM 网格 13×13 → **184 token** |
| 采样到的每图最小 token 数 | 176（928×200 → 6×28 LLM 网格） |
| 每图 token 上限 | 1024 |
| 1536×1536 图 | LLM 网格 31×31 → **994 token**；ViT patch 网格 93×93 = 8649 patch |
| 一个 8192 token 的 prefill chunk 可容纳的最小图数 | 8192 ÷ 184 = **44**（按 176 则为 46） |
| 一个 16384 token prefill batch 可容纳的最小图数 | 16384 ÷ 184 = **89** |
| 1M 上下文可容纳的最小图数 | 1048576 ÷ 184 = **5698** |

### 3.4 一轮（= 一次 prefill forward）内图像处理路径

| 环节 | 行为 | 位置 |
| --- | --- | --- |
| 请求级并发 | ≤ `max_running_requests` = 16（`max_queued_requests` 不限） | 配置见 §3.2 |
| 每请求图片数 | 默认不限；上界由 prompt token 预算给出 | `arg_groups/fields/mm.py:111-117`，`tokenizer_manager.py:1485-1498`（仅在字段非空时校验） |
| 单请求内预处理 | 对图片逐张串行 for 循环 | `multimodal/processors/deepseek_v41.py:89-122` |
| GPU 侧 ViT 调用次数 | **每模态每轮 1 次**：把该轮所有请求的全部 cache-miss 图合成一次 `data_embedding_func` 调用 | 注释 `managers/mm_schedule.py:551-554`；收集 `:610-615`；单次调用 `:403-410` |
| 单次调用的批量 | `get_image_feature(items)` 内部对 item 逐个串行执行 `vision→aligner` | `models/deepseek_v4.py:4916-4942` |
| 一轮内 GPU 上的图像张量峰值 | 未实测（本次只读，未起服务） | 见「未确认项」 |

---

## 4. 显存中的复制 / 缓存路径

### 4.1 传输路径

| 项 | 本次生效值 | 依据 |
| --- | --- | --- |
| `mm_feature_transport` | **`cpu`** | `arg_groups/fields/mm.py:138-152`（默认 `None`）→ `arg_groups/serving_hook.py:793-850`（`nnodes == 1` ⇒ `cpu`，`:818-819`）→ 日志 `:78` 打印 `'mm_feature_transport': 'cpu'` |
| `SGLANG_USE_CUDA_IPC_TRANSPORT` | 未设（conf 中无该导出） | `configs/dsv41_tp4_h100.effective.20261023_013857.conf` 环境变量段 |
| CUDA IPC 特征池是否创建 | 否 | `multimodal/processors/base_processor.py:439-463`（仅 `use_cuda_ipc` 时创建 `MmItemMemoryPool`） |
| `keep_mm_features_on_device` | `False` | `multimodal/processors/base_processor.py:466-468`（`transport in ("cuda_ipc","cuda_vmm")` 才为 True） |
| 处理器内特征是否上 GPU | 否（走 CPU） | `multimodal/processors/deepseek_v41.py:107-108` |
| 模型侧特征上 GPU 点 | `get_image_feature` 内 `torch.as_tensor(item.feature, device=device)` + `patches.to(dtype)` | `models/deepseek_v4.py:4924-4930` |

### 4.2 `mm_runtime_reservation_gb(...)` 的实现与本次取值

实现（`sglang/srt/mem_cache/kv_cache_configurator.py:198-215`）：

```
if not is_multimodal: return 0.0
reserved_mb = SGLANG_VLM_CACHE_SIZE_MB.get()
if mm_feature_transport in ("cuda_ipc", "cuda_vmm"):
    reserved_mb += SGLANG_MM_FEATURE_CACHE_MB.get()
… log "Reserving %.2f GB of the KV budget for post-sizing multimodal allocations …" (reserved_mb / 1024)
return reserved_mb / 1024
```

| 输入 | 默认值 | 本次值 | 位置 |
| --- | --- | --- | --- |
| `SGLANG_VLM_CACHE_SIZE_MB` | `EnvInt(100)` | `100`（未设） | `sglang/srt/environ.py:1515` |
| `SGLANG_MM_FEATURE_CACHE_MB` | `EnvInt(1 * 1024)` = 1024 | 1024，但**未参与相加** | `sglang/srt/environ.py:1539` |
| `is_multimodal` | — | True | 模型有 `vision_config` |
| `mm_feature_transport` | `None` → `cpu` | `cpu` | 见 §4.1 |

计算：`reserved_mb = 100 + 0 = 100` → 返回值 `100 / 1024 = 0.09765625`（以 GB 计）→ 日志格式 `%.2f` 输出 **`0.10 GB`**。

扣减点：
- `sglang/srt/model_executor/model_runner_components/kv_pool_runtime.py:121-124`（取回该值）、`:142-147`（`budget_bytes = int(max(0.0, free_gb - headroom_gb - mm_reservation_gb) * (1 << 30)) + …`）
- `sglang/srt/mem_cache/kv_cache_configurator.py:2916-2919`（`rest_memory = available_gpu_memory - slack_gb - mm_reservation_gb - _lk_moe_device_reservation_gb()`）
- 实际扣除字节数 = `0.09765625 × 2³⁰` = **104,857,600 B = 100 MiB**
- 扣减按**每个 rank 各扣一次**：日志 `:711,713,714,717` 在 TP0/TP1/TP2/TP3 各打印一行 `Reserving 0.10 GB …`

随什么变化：

| 条件 | 返回值 (GB) | 日志文本 | 每卡扣除字节 |
| --- | --- | --- | --- |
| `is_multimodal = False` | 0 | 无该行 | 0 |
| 本次：transport = `cpu`（`mm.py:139` 的取值集合只有 `cpu`/`cuda_ipc`/`cuda_vmm`），`SGLANG_VLM_CACHE_SIZE_MB = 100` | 0.09765625 | `Reserving 0.10 GB …` | 104,857,600 |
| transport ∈ {`cuda_ipc`, `cuda_vmm`}（默认 `SGLANG_MM_FEATURE_CACHE_MB = 1024`） | 1.09765625 | `Reserving 1.10 GB …` | 1,178,599,424 |
| `SGLANG_VLM_CACHE_SIZE_MB = 0` | 0（`cpu` 下） | 无该行 | 0 |

### 4.3 图像 embedding 缓存（GPU 侧）

| 项 | 值 / 说明 | 依据 |
| --- | --- | --- |
| 缓存实现 | `MultiModalStaticCache`（服务级、`OrderedDict` + LRU） | `sglang/srt/mem_cache/multimodal_cache.py:76-156` |
| 容量参数 | `SGLANG_VLM_CACHE_SIZE_MB × 1024 × 1024` = 100 × 1MiB = **104,857,600 B = 0.097656 GiB** | `mem_cache/kv_cache_builder.py:443-444` |
| 实例化位置 | `build_kv_cache(...)` 内，由 `scheduler.py:577` 调用 ⇒ **每个 rank 一份** | `sglang/srt/managers/scheduler.py:577`；`mem_cache/kv_cache_builder.py:443-444`；`managers/mm_schedule.py:21,25-27` |
| 缓存对象 | `EmbeddingResult(embedding=<tensor>)`，即 ViT+aligner 输出（hidden 5120, bf16 = 10,240 B/token） | `managers/mm_schedule.py:302-312`（full 路径）、`:403-435`（per-image 路径） |
| 张量设备 | CUDA：`get_image_feature` 内 `span = torch.empty((len(types), hidden_size), device=device, …)` | `models/deepseek_v4.py:4934-4941` |
| 写入时行为 | 若缓存项是整批 tensor 的切片，会 `clone()` 以免钉住整批 buffer | `mem_cache/multimodal_cache.py:112-126` |
| 容量换算 | 100 MiB ÷ 10,240 B/token = **10,240 token** ⇒ ≈ 10 张 @1024 token/图，或 ≈ 55 张 @184 token/图 | 由前两行推得 |
| 驱逐 | 超限时按 LRU `popitem(last=False)` | `mem_cache/multimodal_cache.py:115-119` |

### 4.4 其他缓存 / 关闭项

| 项 | 本次状态 | 依据 |
| --- | --- | --- |
| prefix mm cache（`enable_prefix_mm_cache`） | `False` | `arg_groups/fields/mm.py:56-58`；日志 `:78` |
| global mm embedding cache（`--enable-mm-global-cache`，mooncake） | `False` | `arg_groups/fields/mm.py:118-121, 127-134`；日志 `:78`（`'mm_global_cache_backend': 'mooncake'`） |
| 预处理 artifact 缓存（`mm_preprocess_cache_size_mb`） | `None`；文档明示 `does not reserve GPU memory`（CPU） | `arg_groups/fields/mm.py:93-104` |
| CUDA IPC / VMM 特征池 | 未启用（transport = cpu） | `base_processor.py:439-463`；`arg_groups/serving_hook.py:870-912` |
| vision 是否进 CUDA graph | 日志中 prefill graph 为 `'backend': 'disabled'`；decode graph 为 `'backend': 'full'`，`bs: [1,2,4,8,12,16]`；视觉塔只在非 decode / 非 target_verify 且本 batch 含 image 的 forward 中执行 | 日志 `:78`；`models/deepseek_v4.py:5019-5028` |
| `VisionAttention.workspace_buffer` | 未传 ⇒ `None`（`deepseek_v41_vit.py:53-62` 未传该参数） | `layers/attention/vision.py:1110, 1237` |

---

## 5. 与「KV 定容缺口 ≈0.79 GiB/卡」的并列数字

| 项 | 每卡 GiB | 4 卡合计 GiB | 是否随 4 卡复制而放大 | 依据 |
| --- | --- | --- | --- | --- |
| `vision.*` 权重 | 0.767117 | 3.068468 | **是（×4）** | shard 00001 header；`deepseek_v4.py:4786-4810` |
| `aligner.*`（mm projector） | 0.136738 | 0.546952 | **是（×4）** | shard 00001 header；同上 |
| `image_start`/`image_newline`/`image_end` | 0.000029 | 0.000114 | **是（×4）** | shard 00002 header；同上 |
| vision RMSNorm 由 BF16 装为 FP32 的增量 | 0.000124 | 0.000496 | **是（×4）** | `deepseek_v41_vit.py:19`；`layers/layernorm.py:474` |
| **视觉相关小计（权重侧）** | **0.904008** | **3.616031** | **是（×4）** | 上四行之和 |
| mm embedding 缓存上限（KV 预算预留） | 0.097656 | 0.390625 | **是（每 rank 一份）** | `environ.py:1515`；`kv_cache_builder.py:443-444`；`kv_cache_configurator.py:198-215` |
| CUDA IPC/VMM 特征池预留 | 0.000000 | 0.000000 | —（本次未启用） | `arg_groups/serving_hook.py:818-819`；日志 `:78` |
| **视觉相关合计（权重侧小计 + mm 缓存预留）** | **1.001664** | **4.006656** | **是** | 0.904008 + 0.097656 |
| KV 定容缺口（外部给定，本次引用） | ≈0.79 | ≈3.16 | — | 见下方佐证 |
| 每卡 GPU 总显存（查询值） | 79.647461（81,559 MiB） | — | — | `nvidia-smi --query-gpu=memory.total`，8 张 H100 均为 81559 MiB |

缺口佐证（`serve_dsv41_tp4_h100_20261023_013857.log:744`）：
```
ValueError: Loaded weights leave no GPU memory for the KV cache under
--mem-fraction-static=0.88. Raise --mem-fraction-static above 0.890
(minimum viable = 1 - available/pre = 0.8894).
```
- 触发公式：`slack_gb = pre_model_load_memory × (1 − mem_fraction_static)`，`pre_model_load_memory` = 装载权重前可用显存（`kv_cache_configurator.py:2886-2891`；`distributed/bootstrap.py:187-207`）
- 相对量：`0.8894 − 0.88 = 0.0094`（相对 `pre_model_load_memory` 的比例）
- `pre_model_load_memory` 未打印在该日志中；若按 GPU 总显存 81559 MiB 计，`0.0094 × 79.647461 GiB = 0.748686 GiB/卡`
- 该次尝试中，`Reserving 0.10 GB` 四行（`:711,713,714,717`）出现在该异常之前

---

## 未确认项

| 项 | 未确认内容 | 原因 / 边界 |
| --- | --- | --- |
| ViT 前向激活峰值 | 一轮内 GPU 上 patch/attention 中间张量的实测峰值（含 fa3 workspace） | 本次为只读核查，未启动服务、未跑前向；代码路径上 `VisionAttention.workspace_buffer` 为 `None`（`vision.py:1110,1237`） |
| 每卡实测常驻差值 | 无服务运行快照，无法给出「加载视觉权重前后」的 `nvidia-smi` 差值 | 未占用 GPU |
| `pre_model_load_memory` 精确值 | 日志未打印该值，故 §5 的 `0.0094 × pre` 只能按 GPU 总显存换算 | 日志 `:744` 仅给出比例 0.8894 |
| 「≈0.79 GiB/卡」的原始出处 | 未在本日志中找到直接等于 0.79 的打印行，该值按外部给定引用 | 本次只读范围内可得的最近似换算为 0.748686 GiB/卡 |
| `vision.*` 是否被任何量化路径改写 | 代码上 `ViT`/`Aligner` 未接收 `quant_config`（`deepseek_v4.py:4805-4807`），故按 BF16 计；未做权重 dump 逐张核对 | 未导出模型参数 |
| 1536×1536 以上极端长宽比图的 token 数 | §3.3 的 176/184 为纯算式采样结果，未覆盖全部像素组合 | 采样步长 4 px，范围 200–4096 |
| CPU 预处理实际并发度 | `SGLANG_CPU_WORKERS` 未在 conf 中设置 ⇒ 取 `os.cpu_count()` = 192（`nproc` 查询值）；实际线程创建数未在日志中确认 | `base_processor.py:514-520` |
| `mm_processor_worker_num` 的 2 个线程是否被 V4.1 路径使用 | 代码路径显示 V4.1 覆写的 `process_mm_data_async` 不经过 `mm_processor_executor`；该 executor 是否在其他调用点被触达未逐一验证 | `processors/deepseek_v41.py:76-127`；`base_processor.py:2039-2054` |
| 视觉权重在各 rank 的装载字节是否完全一致 | 仅凭代码路径 + 「4 个 rank 各自构造视觉注意力」的日志推断；未逐 rank dump 参数 | 未启动服务 |
| 其他 DeepSeek 目录/其他 conf 的视觉配置差异 | 本次只核查 `DeepSeek-V4.1-Flash-1017` 与 `dsv41_tp4_h100.conf` 定稿版 | 核查范围限定 |

---

## 证据文件

- 求和与形状原始数据：`/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/vision_mem_sums.json`
- 权重 header 来源：`/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017/model-00001-of-00048.safetensors`（header 27,376 B）、`/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017/model-00002-of-00048.safetensors`（header 344 B）
