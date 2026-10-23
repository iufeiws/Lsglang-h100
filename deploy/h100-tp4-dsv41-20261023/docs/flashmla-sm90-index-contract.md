# FlashMLA SM90 稀疏 decode 内核索引契约(源码定稿)

> 目的:把 SM90 上 `torch.ops.sgl_kernel.sparse_decode_fwd` 的**索引/元数据契约**从源码钉死,供
> "合法索引 + 非法元数据 ⇒ 越界" 这一类失败模式做归因。
>
> **本文全部结论来自静态源码阅读**,不涉及 GPU 运行、不启动服务、不跑测试。

## 0. 元信息 / 可复核性

| 项 | 值 |
| --- | --- |
| 仓库 | `sgl-project/FlashMLA`(SSH) |
| 固定 pin | `3620b63fc4ebe33950c50e1b7701b6607f44150e` — *"misc: set v41 format as defaultx (#15)"* |
| 本地路径 | `/mnt/SSD01/models/.llm-env/FlashMLA-3620b63` |
| 子模块 | 仅 `csrc/cutlass`(头文件依赖)。**本次未拉取**:所需源码全部在主仓 `csrc/` 内,已用 `git ls-tree -r HEAD csrc/kernels/sm90/decode/sparse/` 逐文件确认 |
| 被分析的调用点 | `/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/srt/layers/attention/deepseek_v4_backend.py:4267-4282` |
| 被分析的算子实现 | `/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sgl_kernel/flashmla_ops.abi3.so`(wheel 二进制,与上表 pin 的 API 面一致,见 §5.4) |
| 图 | `/mnt/SSD01/models/.llm-env/Lsglang/docs/figs/flashmla_sm90_index_domain.png`(由 `/mnt/SSD01/models/.llm-env/Lsglang/scripts/plot_flashmla_sm90_index_domain.py` 纯算术生成) |

复核命令(不产生任何 GPU 负载):

```bash
cd /mnt/SSD01/models/.llm-env/FlashMLA-3620b63
git rev-parse HEAD                 # 3620b63fc4ebe33950c50e1b7701b6607f44150e
sed -n '516,545p' csrc/kernels/sm90/decode/sparse/splitkv_mla.cuh
sed -n '224,366p' csrc/api/sparse_decode.cpp
```

---

## 1. 一次 SM90 调用的内核构成

`csrc/api/sparse_decode.cpp` 里,`enable_split_kv` 在 SM90 上**恒为真**:

```
:276   bool enable_split_kv = arch.is_sm90a() || !(topk + extra_topk <= 640);
```

`Arch::is_sm90a()` 即 `major==9 && minor==0`(`csrc/api/common.h:44-46`),H100 命中 ⇒ SM90 **永远走 split-KV 分支**。
因此一次 `sparse_decode_fwd` 在 SM90 上会启动/访问以下四类对象:

| # | 对象 | 启动位置 | 是否消费 `token_index` |
| --- | --- | --- | --- |
| 1 | `get_mla_metadata_kernel`(`<<<1,32,smem>>>`) | `sparse_decode.cpp:481-495` → `get_decoding_sched_meta.cu:208-233` | 否(只读 `topk_length`) |
| 2 | SM90 稀疏主循环 `flash_fwd_splitkv_mla_fp8_sparse_kernel` | `splitkv_mla.cuh:684`,由 `:779` cluster launch 启动 | **是(唯一)** |
| 3 | `run_flash_mla_combine_kernel` | `sparse_decode.cpp:546` | 否(只读 `num_splits`) |
| 4 | 中间缓冲 `tile_scheduler_metadata`/`num_splits`/`lse_accum`/`o_accum` | `sparse_decode.cpp:476-521` | — |

`gridDim = (NUM_M_BLOCKS, s_q, num_sm_parts)`(`:773`),`NUM_M_BLOCKS = NUM_HEADS/64`(`config.h:20`)。

---

## 2. 表 1:`token_index` 取值 → 内核行为 → 安全性

内核里 `token_index` 只有 5 处使用,核心是这两行(`splitkv_mla.cuh:538-539`):

```cpp
int block_index      = token_index == -1 ? 0 : (int)((uint32_t)token_index/(uint32_t)page_block_size);
int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;   // -1(=UINT_MAX) 的余数必然 < page_block_size
```

以本部署实际口径(page_block_size=256,V4:576 B 数据 + 8 B scale/token,页 stride=256×584=149504 B)为例:

| `token_index` 取值 | 内核行为(page / row / 字节偏移) | 安全/不安全 | 依据(pin `3620b63` + 文件:行) |
| --- | --- | --- | --- |
| `0 ≤ t < num_blocks*page_block_size` | `page=t/256`,`row=t%256`,偏移 `page*149504 + row*576` | **安全**(设计内) | `splitkv_mla.cuh:538-539, 554-555`;`README.md:168-173` |
| `-1`(哨兵) | `page=0`,`row=255`;随后 fp8 数据、RoPE 数据、scale **全部清零**;softmax 由 `is_kv_valid` 置 `-inf` 屏蔽 | **安全**:仍是**页 0 内的合法地址**,读取被丢弃 | 地址: `:538-539`;清零: `:577-581`(scale)、`:594-595`(nope)、`:617-618`(rope);屏蔽: `:630-644` → `:54-56` |
| `-2`、`-123456`、`INT_MIN` 等**其它负数** | 走"非 -1"分支,`(uint32_t)t` 变成 ~4.29e9;`page = (uint32_t)t/256` ≈ 1.6e7~8.4e6,**无任何上界校验** | **不安全**:偏移 1.25e12 ~ 2.51e12 B,远超任何显存分配 ⇒ IMA / SIGSEGV | `:538-539`(唯一处理的值是 -1,`token_index == -1` 三元表达式);`README.md:173` 也只规定 -1 |
| `t ≥ num_blocks*page_block_size`(正数越界) | `page` 超出页数,地址线性外推;**无上界校验** | **不安全**:偏移随 `t` 线性超出缓冲区 | `:538-539`;`params.num_blocks` 在 `devfunc` 中**从未参与地址计算**(仅出现在 `params.h` 与 API 层) |
| `topk` 位置 ≥ `topk_length` 的条目(**仅 V4**) | 被**强制改写为 -1**(位置守卫),即等价于上一行"安全"分支 | **安全**,但守卫只在 `MODEL_TYPE==V4` 时编译进来 | `:531-536`(`if constexpr (MODEL_TYPE == ModelType::V4)`);V32/V32_NO_ROPE 无此守卫,`is_index_valid` 仅判 `index != -1`(`:632-638`) |

补充事实:

* 数据读取用 `ld.global.nc...v4.s32`(`components/dequant.h:52-80`),**无谓词、无边界填充**(不是 TMA),因此"远越界地址"是硬 IMA,不会被静默补零。
  (对比:`README.md:192` 给 **prefill** 内核写的规则是"`-1` 或 `>= s_kv` 都算无效";SM90 稀疏 **decode** 内核**没有**实现 `>= s_kv` 这一条,只实现了 `-1`。)
* 页内偏移的**上界**由 `% page_block_size` 自动成立(对任意 `page_block_size`,余数 `< page_block_size`),所以 -1 的"页 0 + 任意 row"始终落在页 0 内 —— 前提是池子里至少有一页。
* `is_kv_valid` 写入用的位置映射是 `lane_idx*2 + off`(`:639-643`),而 K 载入用的位置映射是 `warp_idx*8 + lane_idx%8 + round*32`(`:463, 511`)。两者是 0..63 的不同置换,**均在界内**(不产生非法访存),是否会错配掩码属于数值问题,见 §11 "未证实事项"。
* 上游自测只在 decode 侧注入 `-1`:`tests/lib.py:291-303`(`abs_indices` 的无效候选在 decode 生成里只有 `[-1]`;`>= topk_length` 的位置被显式写成 `-1`);注入 `INT_MIN/INT_MAX/114514/...` 的那组候选(`tests/lib.py:149`)只服务 **prefill** 路径。

![SM90 稀疏 decode:不同 token_index 取值产生的 KV 字节偏移](/mnt/SSD01/models/.llm-env/Lsglang/docs/figs/flashmla_sm90_index_domain.png)

---

## 3. 表 2:`page_block_size` 口径

| 问题 | 源码事实 | 依据 |
| --- | --- | --- |
| 从哪里来 | `page_block_size = kv.size(1)` —— **取自传入张量的形状**,不是独立参数 | `api/sparse_decode.cpp:257, 441` |
| 允许的范围 | 内核只用它做 `uint32_t` **除/模**(页号、页内偏移)和**页内 scale 区偏移** `page_block_size*(HEAD_DIM_NOPE+HEAD_DIM_ROPE*2)`;设备侧/主机侧**没有任何**"2 的幂""≤ 某值""等于 64/128/256"的断言 | 除法/模: `splitkv_mla.cuh:538-539`;scale 区: `:555`;主机断言清单: `:688-706`(只断言 `h_kv==1`、`topk%64==0`、`d_qk/d_v/h_q`、`stride_kv_row==BYTES_PER_TOKEN`) |
| API 侧是否限制 | 只做自洽检查 `KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, bytes_per_token)`,即"你声明的页大小 = 你给的形状";**不限制具体数值** | `api/sparse_decode.cpp:354` |
| 上游自测实际跑过的值 | `64`、`61`、`576`、`256`(其中 61/576 明确非 2 的幂、且 576>s_k) | `tests/test_flash_mla_sparse_decoding.py:125, 142, 155-168` |
| vLLM 用 64 vs Lsglang 用 256 | 二者都是**该内核允许的取值**;差异不在内核,而在**生产者与内核必须用同一个页大小**(索引编码 `page*page_size+offset` 的 `page_size` 必须等于 `kv.size(1)`) | 编码定义: `README.md:172`;生产者侧见 §9 |
| 只读契约的隐含要求 | 内核按 `page*bytes_per_token` 线性步进(token 行 stride=584)并假设"页 = [page_size×576 数据][page_size×8 scale]",要求 `kv.stride(0) ≥ page_block_size*bytes_per_token`;**但 API 只校验了 `stride(1)==bytes_per_token`,没有校验 `stride(0)`** | 数据行: `:554`;scale 行: `:555`;API 校验: `api/sparse_decode.cpp:356` |

---

## 4. 表 3:`topk_length` 语义

读取方式是**每 batch 一个值**:`params.topk_length + batch_idx`(`splitkv_mla.cuh:170, 172`),形状必须是 `[b]`(`api/sparse_decode.cpp:361`)。

| 取值 | 内核行为 | 安全/不安全 | 依据 |
| --- | --- | --- | --- |
| `None`(不传) | `topk_length = params.topk`(indices 宽度),全部位置都"有效",没有位置守卫 | 安全**仅当**每个条目都是合法 loc 或 -1 | `:170`;`is_index_valid` 分支: `:632-638` |
| `0` | `orig_topk_padded = max(ku::ceil(0,64),64) = 64` ⇒ 仍处理 1 个 64 块的迭代;V4 位置守卫 `pos >= 0` 恒真 ⇒ 该块 64 个条目**全部改写为 -1** ⇒ 页 0 + 全零 + 全屏蔽;`L==0` 时输出缩放取 0(不产生 NaN) | **安全** | 上界: `:171`;`ku::ceil(a,b)=(a+b-1)/b*b`: `kerutils/include/kerutils/host/host.h:83-86`;守卫: `:533-536`;`L==0`: `:306-308, 323, 340`;调度侧同样把 0 抬成 1: `get_decoding_sched_meta.cu:20-22, 56` |
| `0 < L ≤ topk`(正常) | 处理 `ceil(L/64)` 个块,块内位置 `≥ L` 的条目被守卫改写成 -1 | **安全**(推荐形态) | `:171, 533-536` |
| `L < 0` | `ku::ceil(负,64) = ((a+63)/64)*64 ≤ 0` ⇒ `max(...,64)=64`,即退化成"1 个块 + 全部守卫为无效" | **安全**(但语义等于空) | `:171` + `host.h:83-86` + `:533-536` |
| `L > topk`(**超过 indices 宽度**) | `num_orig_kv_blocks = ceil(L,64)/64 > topk/64`;原块循环上界是 `min(num_orig_kv_blocks, end_block_idx)`,而 `end_block_idx` 也由同一个 `L` 推出 ⇒ **会预取/读取 `gIndices` 数组尾之后的内存**;取回的是垃圾整数,随后**无校验地**进入地址计算 | **不安全**:两段越界——先是 indices 数组越读(≤63×4 B/行),再是任意 KV 地址(见 §2) | `:171, 176, 180, 520, 526-528, 658`;API **没有** `topk_length ≤ indices.size(-1)` 的交叉检查(`api/sparse_decode.cpp:360-364` 只有形状) |
| `extra_topk_length` 同规则 | `total_topk_padded = orig_topk_padded + ceil(extra_topk_length,64)`;`extra_topk_length > extra_topk` 时同样越读 `extra_indices` | 同 `L > topk` | `:172-173, 499, 523, 528, 662-668` |

> 结论(表 3):`topk_length` 的**值域本身**(0/负数/正常)都被内核兜住了;真正危险的是**它与 `indices.shape[-1]` 的关系**——这个关系**没有任何一层检查**(Python 层 `flash_mla.py` 不查,`sparse_decode.cpp` 不查,内核不查)。

---

## 5. 格式推断(`584 / 528 / 288`)

### 5.1 三种"view 最后一维"对应什么布局

| 字节/token | 枚举 | 物理布局(由 `KVCacheFormat` 定义) | 依据 |
| --- | --- | --- | --- |
| **584** | `V4` | 每 token 448 B fp8 NoPE + 128 B bf16 RoPE = 576 B 数据;**scale 独立成区**:页内 `page_size*576 + rel*8`,8×ue8m0 | `kv_cache_format.h:12-38`(`QUANT_TILE=64`,`NUM_SCALES=8`,`SCALE_BYTES=1`);`splitkv_mla.cuh:554-556` |
| **528** | `V41` **或** `V32_NO_ROPE`(**碰撞**) | `V41`:512 B fp8 全量化(含 RoPE),scale 独立成区 `page_size*512 + rel*16`,16×ue8m0,量化粒度 32。`V32_NO_ROPE`:512 B fp8 + **行内** 16 B(4×fp32)scale,无 RoPE 区 | `kv_cache_format.h:28-31`;`V4.1` 侧见 `:17-21`;`V32_NO_ROPE` 侧见 `config.h:24-32` + `splitkv_mla.cuh:704-705` |
| **288** | `V41_FP4` | 512 维 fp4 两两打包 = 256 B + 32 B e4m3 scale;**只允许出现在 `extra_kv`** | `api/sparse_decode.cpp:352`(`TORCH_CHECK(model_type != V41_FP4, "The fp4 KV cache is only supported as extra_kv")`) |

### 5.2 判错时会发生什么

| 场景 | 结果 | 依据 |
| --- | --- | --- |
| 未传 `kv_format` | `model_type` **只由** `kv.size(3)` 决定(`detect_kv_cache_format_for_headdim_512`);随后形状检查拿的正是该格式自己的常量 ⇒ **形状检查恒真**,格式与字节数不可能"矛盾通过" | `api/sparse_decode.cpp:343, 354`;`api/common.h:175-187` |
| 显式传了 `kv_format` 但与字节数不符 | `TORCH_CHECK(model_type == *kv_format_hint)` 直接报错。**不会**带着错步长继续跑 | `api/sparse_decode.cpp:338-351` |
| 格式判成另一种"同字节数"格式(528 的两种) | **不是地址/步长错误,而是数值错误**:两种布局的行 stride 与 scale 位置不同(576/512/行内),但在 `page_size*528` 字节的页内互相都放得下,读取不会越出该页(见 §5.3) | `config.h:26-32` 对 `V32_NO_ROPE`(NOPE=512,ROPE=0,scale 行内)与 `V41`(scale 分区) |
| 528 被当成 `V41`(**本 pin 的默认行为**) | SM90 侧 **无 V41 实现** ⇒ 报"不支持全部所需特性"并抛异常(干净失败,不是段错误) | `api/sparse_decode.cpp:54-67`(SM90 实现的特性表里没有 `V41_KVCACHE_FORMAT`/`V41_FP4_KVCACHE_FORMAT`);`api/common.h:245-277`(`fprintf` + `TORCH_CHECK(false)`) |
| 288 当主 cache | 直接报错 | `api/sparse_decode.cpp:352` |

### 5.3 为什么"格式判错"本身不构成越界

内核每个格式的单页读取上界都恰好等于 `page_size * BYTES_PER_TOKEN = stride(0)`:

| 格式 | 单页最大触及字节 | 计算 |
| --- | --- | --- |
| V4 | `(page-1)*576 + 576` 数据、`page*576 + (page-1)*8 + 8` = `page*584` scale | `splitkv_mla.cuh:554-557` |
| V32_NO_ROPE | `(page-1)*528 + 512 + 16` = `page*528` | `:554-557` + `:543-551` |
| V32 | `(page-1)*640 + 656 ≤ page*656` | `:554`, `:604` |

即:**只要索引合法 + `stride(0)` 不小于 `page_size*bytes_per_token`,格式不会额外制造越界**;格式错 ⇒ 数值错。真正的越界入口是索引值(§2)与 `topk_length`/indices 宽度失配(§4)。

### 5.4 部署侧对照

* Lsglang 侧 `select_dsv4_kv_layout()` 明确写了"V4.1 布局只在 SM100/SM103 FlashMLA 或 gfx950 AITER 上存在";非 SM100 且 `mode="auto"` ⇒ **返回 `KVLayout.V4`** ⇒ `bytes_per_token = 584`。
  依据:`/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:186-221`(第 218-219 行)。
  本部署 `dsv4_attn_backend = flashmla`(启动日志 `server_args`),故本部署的主 cache(last dim)**是 584**,不落入 528 碰撞。
* 安装的 wheel 里 SM90 稀疏内核的实例化集合(**从 `.so` 符号读出**)为 `{ModelType 0, 1, 4} × {64, 128}` = `{V32, V4, V32_NO_ROPE} × {h64, h128}` —— 与 pin 的 `csrc/kernels/sm90/decode/sparse/instantiations/` 六个文件**一一对应,没有 V41**:

```bash
strings -a /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sgl_kernel/flashmla_ops.abi3.so \
  | grep -o 'run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType[0-9]*ELi[0-9]*E' | sort -u
# ModelType0/1/4 × Li64/Li128   (枚举序: V32=0, V4=1, V41=2, V41_FP4=3, V32_NO_ROPE=4)
```
  wheel 同时含 `V41_KVCACHE_FORMAT` / `kv_format` 等符号 ⇒ API 面与本 pin 一致(能否精确等于本 pin 的**检测顺序**,见 §11)。

---

## 6. 同一块 528 B/token 缓存的两种解析(sched_meta vs tensor)

`sgl_kernel/flash_mla.py:126` 的说明与 C++ 侧实现一致,但两条路径的**结果不同**:

| 路径 | 触发条件 | `kv_format` 传参 | 528 字节解析成 | SM90 上后果 |
| --- | --- | --- | --- | --- |
| **sched-meta 对象** | `isinstance(tile_scheduler_metadata, FlashMLASchedMeta)` → `flash_mla.py:137-156` → `_flash_mla_with_kvcache_sched_meta` → `:302-317` | **不传**(`None`) | `V41`(本 pin 起,528 默认判为 V4.1) | **无 V41 内核 ⇒ 干净报错** |
| **tensor 元数据** | 走 `flash_mla.py:172-196` | 若 `q.shape[-1]==512 and k_cache.shape[-1]==528` 则由 Python **显式改写**为 `"V32_NO_ROPE"` | `V32_NO_ROPE` | 走 SM90 的 `ModelType::V32_NO_ROPE` 内核 |

* 两条路径**最终调用的是同一个算子**(同一个 `sparse_decode_fwd`)与**同一套形状/dtype 校验**;区别只在 `model_type` 这个模板参数的取值。
* 因此对同一块 528 B/token 缓存:两条路径的**地址公式不同**(`V41`:scale 区在页内 `page*512+rel*16`;`V32_NO_ROPE`:scale 行内 `rel*528+512`),但**都以 `page_size*528` 为页内上界**(§5.3)⇒ 不是"合法索引 + 错步长 = 越界"的来源;它是**同一字节流的两种不同解释**(数值不同),且在 SM90 上 `V41` 这条路根本跑不到内核里(先被特性检查拦下)。
* 本部署调用点 `deepseek_v4_backend.py:4267-4282` 传的是 `FlashMLASchedMeta` **对象**(由 `_create_flashmla_metadata()` = `flash_mla.get_mla_metadata()[0]` 生成,`:222-226`),即走**上表第一条**路径;但本部署缓存是 **584(V4)**,判定为 `V4` 且 SM90 有该内核 ⇒ 两条路径在本部署**都不会**因格式产生分歧。

---

## 7. `extra_k_cache` 与主 cache 的读地址组合

| 事实 | 依据 |
| --- | --- |
| 主/额外两块 cache 的基址、页大小、行 stride、索引数组、长度**各自独立**,地址计算中**没有任何跨 buffer 相加** | 分支赋值: `splitkv_mla.cuh:488-504`(`k_ptr`/`k_block_stride`/`k_row_stride`/`page_block_size` 全部按 `IS_EXTRA_BLOCK` 二选一);索引基址: `:469`(`gExtraIndices`)与 `:468`(`gIndices`) |
| 一处共享:预取寄存器 `nxt_token_indexs` 在两个区之间被复用,但"第一个 extra 块"强制从 `gExtraIndices` 直接装载,跨区不会串用旧值 | `:471-476`(初始预取)、`:516-529`(`IS_FIRST_EXTRA_BLOCK` 分支)、`:662-668`(进入 extra 区的三种调用) |
| SM90 的 V4 实现要求**两块 cache 的行 stride 都是 584**:`stride_extra_kv_row == BYTES_PER_TOKEN` | `:695-700`(`KU_ASSERT`,该断言在 `-DNDEBUG` 下同样生效:`kerutils/include/kerutils/host/host.h:59-73`) |
| 合法格式对只有"相同"或 `(V41, V41_FP4)`;额外 cache 允许的 fp4(288)只与主 V41 搭配 | `kv_cache_format.h:40-43`;`api/sparse_decode.cpp:353` |
| 因此 288 只出现在 V4.1(fp4 extra)体系;本部署 SM90 + V4 主 cache 时,Lsglang 侧 `resolve_compressed_kv_layout()` 对所有压缩比都返回 `V4`(584),与内核断言一致 | `deepseek_v4_memory_pool.py:160-183` |
| 越出各自 buffer 的唯一入口仍是**索引值**:`extra` 索引 ≥ `extra_num_blocks*extra_page_block_size`(只对页 0 例外,即 -1) | `:538-539` 同一段代码 + `:500-503` 的独立基址 |

---

## 8. 表 4:生产者只出现 `{合法 loc, -1}` 是否足够?

以 Lsglang 的实际生产者为例(初值全 `-1`、长度 `clamp(..., max=index_topk)`):

```
deepseek_v4_backend.py:368   sparse_topk_lengths = torch.clamp(topk_lengths_clamp1, max=topk)
deepseek_v4_backend.py:369-376  page_indices = _pad_last_dim(torch.full((rows, topk), -1, int32))
deepseek_v4_backend.py:214-220  _pad_last_dim(...) 用 F.pad(..., value=-1)  ← 列填充值也是 -1
dsv41_request_window.py:60-118  indices = torch.where(valid, ..., -1);  lengths = valid.sum(-1)
```

**是的——在以下 5 个条件同时成立时,`{合法 loc, -1}` + `topk_length`(含 0)不产生越界**:

| # | 条件 | 不满足会怎样 | 依据 |
| --- | --- | --- | --- |
| C1 | 无效位**恰好等于 -1**(不能是 -2/`INT_MIN`/其它负数) | ~2.5e12 B 偏移 ⇒ IMA | §2 `splitkv_mla.cuh:538-539` |
| C2 | 每个合法 loc 都 `<` **它所服务的那个 buffer** 的 `页数 × 页大小`(主/extra 是两个独立空间) | 线性外推越出该 buffer | §2、§7 |
| C3 | `topk_length ≤ indices.shape[-1]`(extra 同理) | 先越读 indices 数组,再把垃圾整数当地址 | §4 |
| C4 | `indices.shape[-1] % 64 == 0` 且 `> 0` | 主机断言 `topk % 64 == 0` 抛错(干净失败) | `splitkv_mla.cuh:691` |
| C5 | 池子里至少存在页 0(总页数 ≥ 1) | -1 的"页 0 行 255"会落到分配之外 | `:538-539` 的页 0 重映射 |

`topk_length == 0` 本身**不危险**(内核把块数兜成 1,并把该块全部改写为 -1,见 §4);本部署的 `clamp(max=index_topk)`(=512,且 512%64==0)恰好同时满足 C3 与 C4。

**注意:C1~C5 中没有任何一条由 `sparse_decode_fwd` 校验**。C++ 层只做"形状/dtype/连续性/自洽"检查(`api/sparse_decode.cpp:248-364`),从不读取 `indices` 的内容:

```
:312   KU_CHECK_DTYPE(indices, torch::kInt32)
:323   KU_CHECK_LAST_DIM_CONTIGUOUS(indices)
:360   KU_CHECK_SHAPE(indices, b, s_q, topk)
```

---

## 9. 本部署(Lsglang DSV4.1 / SM90)侧的"不越界"条件清单

| # | 必须成立的条件 | 本部署是否成立 / 证据 |
| --- | --- | --- |
| L1 | KV/extra 的行 stride = 该格式的 `BYTES_PER_TOKEN`(V4 ⇒ 584) | 成立:主 cache 布局 `V4` ⇒ `get_swa_key_bytes_per_token()=584`(`deepseek_v4_memory_pool.py:218-219, 1962-1968`),视图由 `deepseek_v4_backend.py:4035-4056` 按 584 构造 |
| L2 | `kv.size(1)`(页大小)与**索引生产者**使用的页大小一致 | 成立:内核用 `kv.size(1)`,生产用 `request_window.page_size`/`get_swa_key_page_size()` 同一来源(`dsv41_request_window.py:151-155, 234-241`) |
| L3 | `stride(0) ≥ 页大小 × bytes_per_token` | 成立:切片 `[:, :page_size*k]` + `.view()` 只把 `stride(1)` 钉到 584,**`stride(0)` 保留底层 buffer 的页步长** = 池的 `bytes_per_page_padded = page_bytes(256) = ceil(256*584/576)*576 = 149760 ≥ 149504`(`deepseek_v4_memory_pool.py:297-306`;`deepseek_v4_backend.py:4035-4042`)⇒ 页步长不小于一页的 data+scale 总量;**API 不校验此项**,靠 Lsglang 侧构造保证 |
| L4 | 无效位只用 -1 | 成立:`_pad_last_dim` 填充 -1(`:214-220`)、`window_layout` 的 `torch.where(..., -1)`(`dsv41_request_window.py:60-118`)、低压缩比索引初值 -1(`:369-376`);**注意 E 补丁(哨兵门)把 pad 行改成 `page=-1, length=0`,两者都在内核安全分支内**(`SGLANG_DSV41_SWA_PAD_SENTINEL=1`) |
| L5 | 合法 loc 不越出各自 buffer | 成立:`window_layout` 的 slot 上界 `history_rows + n ≤ workspace.size`(`dsv41_request_window.py:172-198` 会按 `layout.size` 扩张 workspace,不足则报错) |
| L6 | `topk_length ≤ indices.shape[-1]` | 需部署侧确认:`sparse_topk_lengths = clamp(..., max=index_topk)`,`index_topk=512`(config.json)、`swa_page_indices` 宽度亦为 512(`:4120-4122` 断言 `%64==0`);c128 路径是 `+ SWA_WINDOW` 后的另一个长度(`:619`),**属于需要单独盯的口径** |
| L7 | 行 stride 单位 = 1 字节(缓冲区 dtype 必须是 fp8/int8/uint8) | 成立:`kv_cache_dtype=fp8_e4m3`;API 亦检查(`api/sparse_decode.cpp:308`) |
| L8 | `h_q ∈ {64,128}`、`d_qk ∈ {512,576}`、`d_v=512`、`h_kv=1`、`topk % 64 == 0` | 成立:TP4 × 64 heads;API/内核断言覆盖(`api/sparse_decode.cpp:282-284, 373-386`;`splitkv_mla.cuh:691-694`) |

---

## 10. 首次调用的动态路径,以及与 CUDA graph capture 相关的操作

SM90 上 `tile_scheduler_metadata` 的**第一次**调用一定是 `None`:

* Lsglang 只在 `_fast_flashmla_sched_shape()` 为真时预算调度元数据,而该函数要求 `is_blackwell` ⇒ **SM90 恒 False** ⇒ 预计算被跳过。
  依据:`deepseek_v4_backend.py:241-242, 258-262`;`_maybe_precompute_flashmla_sched_meta` 的早退:`:257-262`。
* 于是 C++ 侧走"现场分配 + 现场算"分支(`api/sparse_decode.cpp:473-496`);返回值随后被 Python 缓存:`flash_mla.py:341-342`(`sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata`)。

该分支的动作清单(逐条对应源码):

| 动作 | 源码 | 类别 |
| --- | --- | --- |
| `tile_scheduler_metadata = torch::empty({num_sm_parts, 8}, int32)` | `sparse_decode.cpp:476` | **torch 分配器**(capture 期走 graph 私有池) |
| `num_splits = torch::empty({b+1}, int32)` | `:477` | 同上 |
| `cudaGetDevice(&dev)` | `get_decoding_sched_meta.cu:212` | 主机侧查询 |
| `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin, ...)` | `:213-214` | 主机侧查询 |
| `cudaFuncSetAttribute(get_mla_metadata_kernel, MaxDynamicSharedMemorySize, smem)` | `:217-220` | 主机侧函数属性设置(**每次调用都会执行**) |
| `get_mla_metadata_kernel<<<1,32,smem,params.stream>>>` | `:221` | **在当前流上启动内核** ⇒ capture 期会被捕获成图节点 |
| `printf(...)` + low-smem 回退内核 | `:223-230` | 仅当 `b*5+1` 个 int 超过每块 opt-in smem 时进入(设备端 `printf`) |
| `CHECK_CUDA_KERNEL_LAUNCH()` = `cudaGetLastError()` | `:232` | 主机侧 |

* 这段路径里**没有** `cudaMalloc`/`cudaFree`/`cudaStreamSynchronize`/`cudaMemcpy`/主机 `malloc`(全文件检索 `get_decoding_sched_meta.cu` 与 `run_get_decoding_sched_meta_kernel` 可复核)。
* 因此"`tile_scheduler_metadata=None` ⇒ 首调用才分配"这一点在**capture 之前**就已经被 eager 预热调用消耗掉;**capture 期通常传入的是缓存下来的 buffer**(`flash_mla.py:341-342` 已写回),不会重复分配。
* 由此产生的**真实副作用是"调度元数据只算一次"**:`_maybe_precompute_flashmla_sched_meta` 与 `flash_mla.py` 都只在 `tile_scheduler_metadata is None` 时计算,而 Python 会把它缓存下来;后续调用即便 `topk_length` 变了也**不会重算**(`flash_mla.py:286-297` 的复用断言只比较 b/s_q/h_q/页大小/头数/`topk`/`extra` 宽度,不比较 `topk_length` 数值)。
  内核侧仍会按**当前** `topk_length` 重算 `orig_topk_padded`(`splitkv_mla.cuh:170-177`),但**最后一个请求**的 `end_block_idx` 取自缓存的 `sched_meta`(`:180-181`)——当缓存的 `end_block_idx` 大于当前算出的 `num_orig_kv_blocks` 时,内核会进入 extra 区(`:662-668`),而此时若 `params.extra_indices == nullptr`(如 compress_ratio=0 的层未提供 extra cache),`gExtraIndices` 是空指针运算(`:469`)⇒ **该分支下会发生非法访存**。此条为源码可见的机制性风险,触发条件是"同一 `FlashMLASchedMeta` 对象被不同 `topk_length` 复用";**该条件的完整可证伪形式、以及它在 capture 期是否可达,见 §13.3**(结论:capture 期 `length ∈ {0,1}` ⇒ 需要"先 >64 后 ≤64"复用同一实例,图内不可达)。

---

## 11. 未证实事项(明确区分于上文的源码事实)

1. **安装 wheel 与本 pin 的精确一致性**:wheel 内可见 `kv_format`、`V41_KVCACHE_FORMAT`、`V32_NO_ROPE_KVCACHE_FORMAT` 等符号,且 SM90 稀疏实例化为 `{V32,V4,V32_NO_ROPE}×{64,128}`,与本 pin 一致;但 `detect_kv_cache_format_for_headdim_512` 里"528 优先判成 `V4.1` 还是 `V32_NO_ROPE`"**只是常量顺序,无法从二进制字符串判定**。本部署缓存为 584,两种顺序都判 `V4`,故不影响本部署结论。
2. **掩码位置映射**:`is_kv_valid` 的写入位置(`:639-643`)与 K 载入的位置(`:463, 511`)是 0..63 的两个不同置换;**均为界内,不构成非法访存**;是否造成"掩错 token"的数值问题需要数值对拍(本次不跑)。
3. **§10 末尾的"缓存调度元数据 + 变长度"组合**是否在真实捕获/回放序列里出现,需要运行期观测(本次不跑)。
4. 本文未使用 compute-sanitizer / core dump —— 按指令"纯静态、零 GPU"。

---

## 12. 复核用命令汇总(全部只读)

```bash
R=/mnt/SSD01/models/.llm-env/FlashMLA-3620b63
# 1) token_index 的全部使用点
grep -n "token_index" $R/csrc/kernels/sm90/decode/sparse/splitkv_mla.cuh
# 2) API 层的全部校验(无索引值检查)
sed -n '246,365p' $R/csrc/api/sparse_decode.cpp
# 3) 格式常量
sed -n '1,50p' $R/csrc/kernels/kv_cache_format.h
# 4) 首调用动态路径
sed -n '208,233p' $R/csrc/kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
# 5) 安装 wheel 的 SM90 实例化集合
strings -a /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sgl_kernel/flashmla_ops.abi3.so \
  | grep -o 'run_flash_splitkv_mla_fp8_sparse_kernelIL9ModelType[0-9]*ELi[0-9]*E' | sort -u
# 6) Lsglang 侧口径
grep -n "select_dsv4_kv_layout" -A 36 /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
# 7) 图(纯算术)
python3 /mnt/SSD01/models/.llm-env/Lsglang/scripts/plot_flashmla_sm90_index_domain.py
```

---

# 13. 追加:四条假设的判定 + 两条可静态定论

前提(由索引侧另一条线给出,本轮直接采用):capture 期 `indices` = **一个小非负整数 + 其余全 `-1`**(活跃分支 `window_layout`,每行恰 1 个 valid),或全 `-1`;`lengths ∈ {0,1}`;`extra_indices` 全 `-1`(初值未写入)。

## 13.1 表 5:`page_block_size` 的来源 / 值域 / 是否必须与生产者一致

| 问题 | 结论 | 依据 |
| --- | --- | --- |
| **从哪来** | **取自张量形状 `kv.size(1)`(第 1 维)**,既不是独立参数也不是常量;`splitkv_mla.h:1-11` 只有函数声明,不含任何页大小约束 | `api/sparse_decode.cpp:257` → `:441`(写入 params)→ `splitkv_mla.cuh:494`(主 cache)/ `:500`(extra cache) |
| 合法取值域 | 任意正整数;页号用 `uint32_t` **除法**、页内偏移用**取模**(非移位),故**不要求 2 的幂**,也无上界断言;唯一相关硬约束是 `topk % 64 == 0` | `splitkv_mla.cuh:538-539`;`:691`;上游自测跑过 61 / 64 / 256 / 576:`tests/test_flash_mla_sparse_decoding.py:125, 142, 155-168` |
| 形状自洽 | API 只检查"你声明的页大小 = 张量的第 1 维" | `api/sparse_decode.cpp:354` |
| **是否必须与生产者构造 `indices` 时的页大小一致** | **必须一致**。索引编码定义为 `page_index*page_block_size + offset_within_page`,消费端用 `t/P_c`、`t%P_c` 反解,`P` 不一致即解出另一格 | 编码:`README.md:172`;反解:`splitkv_mla.cuh:538-539` |
| 不一致的后果 | **方向决定后果**:若**生产者页大小 `P_p` > 内核页大小 `P_c`**,页号被放大约 `P_p/P_c` 倍 ⇒ `page*stride0` 可越出缓冲区 ⇒ **会越界**;若 `P_p < P_c`,页号变小 ⇒ **只错页、不越界**(地址仍在缓冲区内,但读到别的 token) | 同上;地址式:`:538-539, 554-555` |
| 本部署为什么必然一致 | **单变量来源**:`swa_page_size` 同时决定 allocator 物理页、窗口 workspace、`RequestWindow.page_size` 与内核视图的第 1 维 ⇒ 生产者与消费者同源 | allocator:`deepseek_v4_memory_pool.py:1124-1132`;`RequestWindow`:同文件 `:1141-1145`;内核视图:`deepseek_v4_backend.py:4035-4042`(`get_swa_key_page_size()`);生产者:`dsv41_request_window.py:151-155` |
| 与 vLLM 的 64 / Lsglang 的 256 的关系 | 二者都是**各自部署内部的约定值**,内核侧对具体数值无要求;差异本身不构成越界,只有"部署内部两端不一致"才构成 | 表 2 / 本表 |

## 13.2 C2(`loc < 各自 buffer 的 页数×页大小`)能否静态判定

**能,且判定为成立**(在主 cache 与 extra cache 上分别成立):

| 对象 | 静态判定 | 算式与数值 |
| --- | --- | --- |
| 主 cache(`swa_k_cache`) | **成立** | 生产者 loc 上界 = `layout.size = history_rows + rows = groups*window + n`(`dsv41_request_window.py:88-118`);buffer = `RequestWindow.workspace`(:216-243),行数 = `ceil(workspace_rows/page_size)*page_size`(:172-198),且 `activate()` 在尺寸不足时**直接抛 RuntimeError**(:194-198)⇒ `max(loc)+1 ≤ 行数 = dim0*page_size` 由**构造**保证 ⇒ C2 成立。代入 capture:b=16、window=128、page_size=256 ⇒ `layout.size = 129×16 = 2064 ≤ dim0×256` ⇒ `dim0 ≥ 9`(实际 workspace_rows = `(num_req_slots+1)×128 + max(chunked_prefill, …)`,远大于 2064) |
| extra cache | **成立(依前提)** | 前提"capture 期 `extra_indices` 全 -1" ⇒ 唯一无效值走页 0 分支(`splitkv_mla.cuh:538-539`)⇒ 无条件在界内(池 ≥1 页) |

* **需要运行期才知道的量只有一个**:`dim0`(= workspace 行数 / page_size)的具体数值;但上述**不等式的成立并不依赖**它,依赖的是 `_ensure_workspace` 的向上取整 + `activate()` 的 RuntimeError 兜底。
* 因此本部署残余风险**不在 C2**;按 §8 的 C1~C5,仍需盯的是 **C3**(`topk_length ≤ indices.shape[-1]`):c1/c2 的 extra 由 `clamp(..., max=index_topk=512)` 得到、宽度同为 512 ⇒ 成立;c128 路径(`+SWA_WINDOW`,`:619`)不在本 config 的 `present_ratios` 内(`has_c128=False`)。

## 13.3 §10 那条"调度元数据缓存"风险的**可证伪形式**

要走到 `splitkv_mla.cuh:469` 的空指针指针运算(`int* gExtraIndices = params.extra_indices + …`),以下四条必须同时成立:

1. `extra_indices is None` ⇔ 该层 `compress_ratio == 0`(`deepseek_v4_backend.py:4028-4032`)⇒ **本部署确实可能**:config 的 `compress_ratios` 含 5 个 0(layer 0,1,40,41,42);
2. 进入 extra 分支(`:662`,`args.num_orig_kv_blocks < args.end_block_idx`)。对分区内**非末位**请求,`end_block_idx` 由**当前** `topk_length` 重算(`:180`),且 extra 为 None ⇒ `extra_topk_length = 0` ⇒ 两者恒相等 ⇒ 不成立;**只有分区末位请求**使用缓存的 `sched_meta.end_block_idx`(`:181`);
3. 该缓存值来自同一 `FlashMLASchedMeta` 实例**更早一次**调用,且那次 `ceil(topk_length,64)+ceil(extra_topk_length,64) ≥ 128`,而本次 `max(ceil(topk_length,64),64) = 64`(即**先 >64 后 ≤64**);
4. 该实例在两次调用之间**存活**(缓存点:`flash_mla.py:341-342`;创建点:`deepseek_v4_backend.py:902-935`;构造点:`DSV4AttnMetadata(...)` @ `:4913`,工厂 `make_core_attn_metadata()` @ `:4767`)。

> **可证伪命题**:若"同一个 `FlashMLASchedMeta` 实例,在 ratio-0 层上被两次不同 forward 复用,且两次的 `ceil(topk_length,64)` 分别 ≥128 与 =64",则可到;否则不可到。
> 若实例只在一个 forward 内跨层复用(各层 `topk_length` 相同)或每次 forward 新建,则条件 3 不可能成立 ⇒ **不可达**。
> 与 capture 前提的关系:capture 期 `length ∈ {0,1}` ⇒ `ceil(·,64) = 64` 恒成立 ⇒ 图内调用**不可能**成为"先 >64"的那一次;必须是"图外 eager 调用与图内调用共用同一实例"。

* decode 侧核心元数据的具体构造点在本轮**未定位**(`init_forward_metadata_decode:1569-1586` 只返回 `DSV4RawDecodeMetadata`;`make_core_attn_metadata()` 的可见调用点 `:1638/:2349/:2406/:2513` 分别落在 prefill / target_verify / draft_extend 路径)⇒ 实例是否跨 forward 存活列为未证实事项(§11)。

## 13.4 四条假设的判定(接上一轮的四问)

| # | 假设 | 静态结论 | 依据 |
| --- | --- | --- | --- |
| 1 | 缓存格式/步长判错 ⇒ "合法索引 + 错步长 = 越界" | **本部署不成立**:格式由 last dim 反推(:343)且随后 shape check 用的是该格式自身常量(:354)、行 stride 另有断言(:356 与 `splitkv_mla.cuh:697`),格式判错只改**数值**;SM90 无 V41/V41_FP4 内核 ⇒ 528/288 是**干净报错** | §5、§6 |
| 2 | extra(288) 与主 cache 的读地址组合越出各自 buffer | **不成立**:两套基址/页大小/行 stride/索引数组完全独立(:488-504);288 只属 V4.1/SM100;本部署 extra 也是 V4=584(:695-700 断言) | §7 |
| 3 | 首调用(`None`)动态路径与 capture 冲突 | **源码中无 `cudaMalloc`/`cudaStreamSynchronize`/主机 malloc**;仅 `torch::empty` 分配 + `cudaGetDevice` + `cudaDeviceGetAttribute` + `cudaFuncSetAttribute` + 当前流 launch(`get_decoding_sched_meta.cu:208-233`) | §10 |
| 4 | 内核自身对 `topk_length ∈ {0,1}` + `indices` 含 `-1` 的限制 | **无越界限制**:长度 0/1 都被兜住(块数抬到 ≥64、V4 位置守卫把超出部分改写为 -1、`L==0` 时缩放取 0);`-1` 是唯一被特判并重映射到页 0 的值,读取被清零+屏蔽 | §2、§4、§8 |
| 残余 | 用 §13.3 的可证伪形式描述 | 需"先 >64 后 ≤64 复用同一实例",capture 图内不可达 | §13.3 |

---

# 14. 追加:wheel 二进制反汇编核对(`flashmla_ops.abi3.so` vs pin `3620b63`)

目的:把 §11 第 1 条("wheel 与 pin 是否精确一致")从"无法从字符串判定"升级为**反汇编级核对**。
对象:`/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sgl_kernel/flashmla_ops.abi3.so`(24,746,848 B)
工具:`/usr/local/cuda/bin/cuobjdump`(compute-sanitizer 不需要;全程零 GPU)
原始输出(可复核):
`/mnt/SSD01/models/.llm-env/Lsglang/logs/bincheck/01_list_elf.txt`、
`/mnt/SSD01/models/.llm-env/Lsglang/logs/bincheck/02_symbols_sm90a.txt`、
`/mnt/SSD01/models/.llm-env/Lsglang/logs/bincheck/03_sass_{13,16,19,25}.txt`、
`/mnt/SSD01/models/.llm-env/Lsglang/logs/bincheck/xelf/`(171 个已抽取 cubin)

## 14.1 cubin 清单 + 承载函数的判定方法

```bash
cuobjdump --list-elf  $SO                       # 171 行
cuobjdump -xelf all   $SO                       # 抽取到 CWD
for f in *.sm_90a.cubin; do cuobjdump -symbols "$f" | grep -q flash_fwd_splitkv_mla_fp8_sparse_kernel && echo "$f"; done
```

| arch | 数量 | 说明 |
| --- | --- | --- |
| `sm_90a` | 57 | Hopper;`.headerflags` 含 `EF_CUDA_ACCELERATORS`,`.target sm_90a` |
| `sm_100` | 57 | Blackwell |
| `sm_103a` | 57 | Blackwell |
| `sm_90`(非 a) | **0** | **不存在** ⇒ 二进制侧不存在"按非-a 目标编译"的情形 |

**判定"哪些 cubin 属于 `sparse_decode_fwd`"的依据**:按 mangled 命名空间 `sm90::decode::sparse` 匹配
`_ZN4sm906decode6sparse39flash_fwd_splitkv_mla_fp8_sparse_kernelI…`(与 §5.4 里出现的 `sm100…` 符号区分开)。
命中 **恰 6 个** sm_90a cubin,每个 cubin 只含 **1 个 function**(`grep -c "Function :"` = 1):

| cubin | 字节 | 模板实参 | 对应源码实例化文件 |
| --- | --- | --- | --- |
| `flashmla_ops.abi3.13.sm_90a.cubin` | 117,712 | `ModelType1` = **V4**, `<64>` | `instantiations/v4_persistent_h64.cu` |
| `flashmla_ops.abi3.16.sm_90a.cubin` | 98,000 | `ModelType1` = **V4**, `<128>` | `instantiations/v4_persistent_h128.cu` |
| `flashmla_ops.abi3.19.sm_90a.cubin` | 80,848 | `ModelType0` = V32, `<64>` | `v32_persistent_h64.cu` |
| `flashmla_ops.abi3.22.sm_90a.cubin` | 75,472 | `ModelType0` = V32, `<128>` | `v32_persistent_h128.cu` |
| `flashmla_ops.abi3.25.sm_90a.cubin` | 80,208 | `ModelType4` = V32_NO_ROPE, `<64>` | `v32_no_rope_persistent_h64.cu` |
| `flashmla_ops.abi3.28.sm_90a.cubin` | 74,832 | `ModelType4` = V32_NO_ROPE, `<128>` | `v32_no_rope_persistent_h128.cu` |

> 注意:`sm100::decode::sparse::head64` 等符号也出现在若干 **sm_90a** cubin 里(同一 TU 对三档 arch 都编译),
> 但 SM90 的派发走的是 `Decode_Sm90_Impl` → `sm90::decode::sparse`(§1),故与本次核对无关。

## 14.2 地址算式核对(以本部署实例 `abi3.13` = V4/h64 为例)

```bash
cuobjdump -sass -arch sm_90a flashmla_ops.abi3.13.sm_90a.cubin > 03_sass_13.txt   # 12,377 行
```

**参数空间基址 = 0x210**(由 3 个独立锚点推出,而非假设):

| 参数偏移 | 结构体字段(偏移) | 二进制中的用法(与源码对应) |
| --- | --- | --- |
| `c[0x0][0x258]` | `indices`(+0x48) | `LDG.E.CONSTANT ...,+0x0` 与 `...,+0x80`;++0x80 = 32 个 int32 = `TOPK_BLOCK_SIZE/2`,正是源码两轮预取 `round*32 + …`(`:474-476`/`:519-520`) |
| `c[0x0][0x260]` | `topk_length`(+0x50) | `UIMAD.WIDE …,UR8,0x4,…`(即 `+batch_idx*4`)→ `R2UR UR13` → 与位置比较(`:170`、`:533`) |
| `c[0x0][0x2b4]` | `stride_kv_block`(+0xa4) | 与 `block_index` 相乘,构成页步长项(`:495`) |
| `c[0x0][0x22c]` / `0x234` / `0x238` | `sm_scale_div_log2`(+0x1c) / **`page_block_size`(+0x24)** / `topk`(+0x28) | 见下 |
| `0x250/0x258/0x260/0x268/0x270/0x290/0x298/0x2a0` | `kv/indices/topk_length/attn_sink/lse/extra_kv/extra_indices/extra_topk_length` | **全部命中指针字段**(64 位装载)✓ |

### (a) 是否先判 `t == -1` 并走页 0 分支 —— **是**

```
/*ac20*/  ISETP.GE.AND P0, PT, R18, UR13, PT ;     // 位置 >= topk_length   (源码 :533)
/*ac40*/  SEL R31, R12, 0xffffffff, !P0 ;           // 超出长度 ⇒ token_index = -1(:534)
/*ac50*/  ISETP.NE.AND P0, PT, R31, -0x1, PT ;      // token_index == -1 ?   (:538)
/*ac90*/  IMAD.MOV.U32 R22, RZ, RZ, RZ ;            // block_index 先置 0
/*ae10*/  @P0 IMAD.MOV.U32 R22, RZ, RZ, R18 ;       // 仅当 t != -1 才覆盖为商 ⇒ 页 0 分支
/*aeb0*/  ...
/*aef0*/  IMAD.WIDE R20, R14, 0x240, R22 ;          // page_block_size*576 + 页基址(scale 区)
```
余数一侧**不参与该谓词**(`SEL R29, R38, R29, !P3` 只处理除零保护),故 `-1` 时 `row = UINT_MAX % page_block_size = page_block_size-1`
—— 与源码注释"UINT_MAX%page_block_size < page_block_size,所以不会有非法访问"完全一致。

### (b) `(uint32)t / page_block_size` —— 运行时整数除法(标准软件序列)

```
/*ab50*/  LDC R14, c[0x0][0x234] ;                  // R14 = page_block_size (运行时值,非立即数)
/*abe0*/  I2F.U32.RP R22, R14 ;
/*ac60*/  MUFU.RCP R22, R22 ;
/*ac80*/  VIADD R20, R22, 0xffffffe ;
/*aca0*/  F2I.FTZ.U32.TRUNC.NTZ R21, R20 ;
/*ace0*/  IMAD.HI.U32 R30, R21, R23, R20 ;          // 商/余数迭代
/*ad30*/  @P0 IMAD R21, R14, R21, R31 ;
…
/*add0*/  @P3 VIADD R18, R18, 0x1 ;                 // 两级 ±1 校正
```
* 除数来自**参数空间**(`LDC …c[0x0][0x234]`),因此**没有被特化成立即数**,也没有编译成移位 —— 与 `:538-539` 的 `(uint32)…/(uint32)page_block_size` 语义一致。
* 该序列同时给出**商(block_index)**与**余数(rel_idx_in_block)**:源码里 `/` 与 `%` 共用同一次除法,二进制亦如此。

### (c) `(uint32)t % page_block_size` —— 同序列的余数支路,且含**除零保护**

```
/*ac70*/  ISETP.NE.U32.OR P4, PT, R14, RZ, !P0 ;    // (page_block_size != 0) || (t == -1)
/*ae00*/  ISETP.NE.U32.AND P3, PT, R14, RZ, PT ;
/*aea0*/  SEL R29, R38, R29, !P3 ;                  // 除数为 0 时取哨兵,而非未定义行为
```

### (d) 地址里**是否出现 `num_blocks`** —— **否(0 次)**

对**全部**参数偏移做直方图后:

| 偏移 | 字段 | 出现次数 |
| --- | --- | --- |
| `c[0x0][0x230]` | **`num_blocks`**(+0x20) | **0** |
| `c[0x0][0x280]` | **`extra_num_blocks`**(+0x70) | **0** |
| `c[0x0][0x234]` | `page_block_size`(+0x24) | 1 |
| `c[0x0][0x238]` | `topk`(+0x28) | 3 |

⇒ 二进制层面同样**不存在**对索引的上界校验;两个 cache 的页数都不参与寻址。这与源码"`num_blocks` 从不参与地址计算"一致。

### (e) 两项地址与源码 `:554-555` 对齐

```
/*ae90*/  IMAD.WIDE.U32 R22, R22, UR12, RZ ;        // block_index * stride_kv_block   (UR12 = c[0x0][0x2b4])
/*b070*/  IMAD.WIDE R22, R29, 0x240, R22 ;          // + rel_idx_in_block * 576        (0x240 = 448 + 64*2)
/*aef0*/  IMAD.WIDE R20, R14, 0x240, R22 ;          // scale 区: page_block_size * 576
/*b000*/  IMAD R20, R18, 0x8, R21 ;                 //           + rel_idx_in_block * 8  (NUM_SCALES=8)
/*b030*/  ISETP.NE.AND P2, PT, R31, -0x1, PT ;      // -1 ⇒ scale/数据清零分支(源码 :577/:594/:617)
```

## 14.3 指令集核对(`sm_90a` 专属指令)与 arch 门控

`abi3.13`(V4/h64,该 cubin **只有 1 个函数**,故"出现在哪个函数"= 该稀疏 decode 内核):

| 指令(Hopper SASS 助记符) | 计数 | 含义 | 对应源码 |
| --- | --- | --- | --- |
| `HGMMA.64…` | 40 | warpgroup MMA(wgmma;Hopper 为 `HGMMA`) | `cute::gemm`(`components/helpers.h:22+`) |
| `UTMALDG` | 16 | TMA 装载(`cp.async.bulk.tensor`) | Q 的 `launch_tma_copy` / `SM90_TMA_LOAD` |
| `UTMASTG` | 2 | TMA 存储 | `SM90_TMA_STORE_5D`(`config.h:237`) |
| `UBLKCP` | 16 | `cp.async.bulk`(非 tensor) | `SM90_BULK_COPY_S2G`(`config.h:265`) |
| `SETMAXREG.*` | 3 | 2×`DEALLOC.CTAPOOL` + 1×`TRY`(CTA-pool 寄存器预算) | `warpgroup_reg_alloc<192>`/`dealloc<160>`(`splitkv_mla.cuh:187/367/456`) |
| `SETMAXNREG` | 0 | 该助记符未出现(寄存器预算类提示,不影响访存) | 同上 |

**文字面检索补充**:`WGMMA`=0、`CP.ASYNC.BULK`=0、`LDGSTS`=0 —— 因为 Hopper SASS 里 wgmma/bulk 分别写作 `HGMMA`/`UTMALDG`/`UBLKCP`,不是 PTX 助记符。

**源码是否用 `sm_90a` 宏门控关键代码**:`grep -rn "__CUDA_ARCH_FEAT_SM90_ALL\|SM90_ALL"` 覆盖 `csrc/` 全部 `.cu/.cuh/.h/.hpp` ⇒ **0 命中**。
SM90 稀疏内核体唯一的 arch 门是:

```cpp
// csrc/kernels/sm90/decode/sparse/splitkv_mla.cuh:89
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
…
#else
    if (cute::thread0()) CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
#endif
```

`__CUDA_ARCH__ == 900` 对 `-arch=sm_90` 与 `-arch=sm_90a` **同时成立**(`a` 只影响可用的指令集,不改变 `__CUDA_ARCH__`),
且没有 `SM90_ALL` 之类的二次门控 ⇒ **不存在"用非-a 目标编译时静默少编译一段主循环/地址代码"的机制**;
反过来,若真按非-a 编译,`HGMMA`/TMA 内联汇编会**编译失败**而不是静默裁剪。
结合 14.1(wheel 只提供 `sm_90a` cubin,且 H100=9.0a 可加载),该假设在本产物上不成立。

## 14.4 实例化集合比对

| 侧 | 集合 | 结论 |
| --- | --- | --- |
| wheel(sm_90a cubin 符号) | {V32, V4, V32_NO_ROPE} × {h64, h128} = **6** | — |
| pin 源码 `csrc/kernels/sm90/decode/sparse/instantiations/` | 正好 **6** 个 `.cu`(v4_h64/h128、v32_h64/h128、v32_no_rope_h64/h128) | **一一对应,无缺失** |
| pin 源码 `Decode_Sm90_Impl` 特性表(`api/sparse_decode.cpp:54-67`) | 无 `V41_KVCACHE_FORMAT`/`V41_FP4_KVCACHE_FORMAT` | 与"SM90 无 V41 内核"一致,**无多余裁剪** |

Python 侧同一 wheel 的 `flash_mla.py` 两条路径已在 §6 核对(sched-meta 对象 ⇒ `kv_format=None` ⇒ 528 判 V4.1;tensor ⇒ 显式 `V32_NO_ROPE`)。

## 14.5 一句话结论

**二进制 `sparse_decode_fwd` 与 pin 源码在地址算式与关键指令上一致** —— 运行时 `page_block_size` 经 `LDC c[0x0][0x234]` 取出后走标准 u32 除/模序列(商=页号、余=页内行)、`-1` 走"预置 0 + 条件覆盖"的页 0 分支、`576`/`8` 两项布局常量与 `stride_kv_block` 项齐备、位置守卫在、且 `num_blocks`/`extra_num_blocks` **在二进制中完全不出现**;`HGMMA`/`UTMALDG`/`UTMASTG`/`UBLKCP` 说明 sm_90a 主循环完整编译进来,源码也没有 `SM90_ALL` 之类的二次门控。因此**它不能解释"合法索引下的非法访存"**;在索引合法的前提下,该算子能越界的入口只剩 (i) `page_block_size == 0`(SASS 有除零保护,但会产生哨兵余数;本部署视图下不可达)与 (ii) 调用方给出的 `stride(0) < 页大小 × bytes_per_token`(API 不校验,§3 表 2)。
