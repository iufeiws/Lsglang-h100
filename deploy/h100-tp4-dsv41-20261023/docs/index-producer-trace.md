# 索引生产者追溯: `full_to_swa` / `req_to_token` / SWA page indices

| 项 | 值 |
| --- | --- |
| 目标 | 静态核对 `BuildCausalSwaPageIndices` 的索引生产者,判定能否产生非法编码 |
| 核查对象 | `/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages`(即实际被 import 的代码) |
| 方式 | **纯静态阅读**。未占 GPU、未启动服务、未跑任何复现、未改 site-packages |
| 约束遵守 | 未读取/未引用 `/mnt/SSD01/models/.llm-env/deprecated/` 或任何 `*_DEPRECATED*` 路径 |
| 备注 | 目录内存在 `.orig4` 等**备份副本**(先前补丁链),本报告一律引用**无后缀**的实际文件 |

---

## 0. 结论摘要(仅陈述阅读所得)

| 问题 | 阅读所得 |
| --- | --- |
| `full_loc == -1` 且 `valid == True` 是否可能由 `other=-1` 产生 | **不能**。`:592` 的 `mask=valid` ⇒ `other=-1` 只在 `valid=False` 的 lane 生效 |
| 第二次 load(`:595`)是否有地址上界约束 | **没有**。mask 只用 `valid`,不校验 `full_loc` 的值域 |
| `full_to_swa` 是否为未初始化残留 | **否**。两处构造(`swa.py:103`、`:750`)都是 `torch.zeros` / `torch.arange`,**没有 `torch.empty`** |
| `req_to_token` 是否为未初始化残留 | **否**。`memory_pool.py:404-406` 是 **`torch.zeros`**(int32),不是 `torch.empty` |
| 是否存在"读未写行 ⇒ 超大正整数"的路径 | **未找到**。表初值恒为 `0`(合法编码),非垃圾 |
| `full_to_swa[-1]` 是否属越界 | **不属**。表**末尾专门放了一个 `-1` 哨兵元素**供其索引,注释与 torch 参考实现均显式依赖该语义 |
| SM90 内核对 `-1`/负数/topk_length 的专门语义 | **本报告未取得证据,不作断言**(见 §6) |

---

## 1. `BuildCausalSwaPageIndices` / Triton kernel 全文

### 1.1 分发壳(`dsv4_attn_metadata_kernels.py:388-437`)

```python
388  class BuildCausalSwaPageIndices:
390      def execute(cls, *args, **kwargs):
391          if _inputs_on_cuda(*args, **kwargs):
392              return cls.triton(*args, **kwargs)
393          return cls.torch(*args, **kwargs)
```

`torch`/`triton` 两个类方法签名一致,参数为 `req_to_token`、`full_to_swa_mapping`、`req_pool_indices_repeated`、`seq_lens_casual`、`swa_window`、`page_index_aligned_size`、可选的 `swa_replay_start`。

### 1.2 Triton kernel 原文(`:563-596`,逐行)

```python
563  @triton.jit
564  def _causal_swa_page_indices_kernel(
565      req_to_token_ptr, full_to_swa_ptr, req_pool_ptr, seq_lens_ptr,
569      swa_replay_start_ptr, out_ptr, rt_stride, swa_window, padded_width,
574      HAS_SWA_REPLAY_START: tl.constexpr, BLOCK_K: tl.constexpr,
576  ):
577      row = tl.program_id(0)
578      pos = tl.load(seq_lens_ptr + row).to(tl.int64) - 1
579      rp = tl.load(req_pool_ptr + row).to(tl.int64)
580      base = req_to_token_ptr + rp * rt_stride
581      out_base = out_ptr + row.to(tl.int64) * padded_width
582      if HAS_SWA_REPLAY_START:
583          floor = tl.load(swa_replay_start_ptr + row).to(tl.int64)
584      else:
585          floor = tl.zeros((), dtype=tl.int64)
586
587      for k0 in range(0, padded_width, BLOCK_K):
588          k = k0 + tl.arange(0, BLOCK_K)
589          kmask = k < padded_width
590          off = pos - k.to(tl.int64)
591          valid = (k < swa_window) & (off >= floor) & kmask
592          full_loc = tl.load(base + tl.where(valid, off, 0), mask=valid, other=-1).to(
593              tl.int64
594          )
595          swa = tl.load(full_to_swa_ptr + full_loc, mask=valid, other=-1).to(tl.int32)
596          tl.store(out_base + k, tl.where(valid, swa, -1), mask=kmask)
```

`off` / `valid` / `base` 的定义:

| 符号 | 定义处 | 内容 |
| --- | --- | --- |
| `base` | `:580` | `req_to_token_ptr + rp * rt_stride`,其中 `rp` 为 `:579` 读入的 `req_pool[row]` |
| `off` | `:590` | `pos - k`,`pos = seq_lens[row]-1`(`:578`)⇒ 该行内的**位置偏移**(非内存偏移) |
| `valid` | `:591` | `(k < swa_window) & (off >= floor) & kmask`;`floor` 来自 `:582-585`(有 replay 起点则用之,否则 0) |

### 1.3 逐条回答第 1 问

| 问 | 读到的行为 |
| --- | --- |
| `full_loc == -1` 且 `valid == True` 是否能发生 | 由 `other` **不能**:`:592` 的 `mask=valid`,故 `other=-1` 只填 `valid=False` 的 lane。**唯一可能**是 `req_to_token[rp, off]` 这一格**本身就存着 `-1`** |
| 第二次 load(`:595`)的地址上界约束 | **无**。`:595` 的 `mask` 只用 `valid`,不检查 `full_loc` 的数值范围;上界只能来自 `full_loc` 的**来源天然受限**(见 §2/§3) |
| `other=-1` 生效情形 | ① `:592` 当 `valid=False`(该 lane 的 `full_loc` 得 `-1`,不产生实际读取);② `:595` 当 `valid=False`(该 lane 的 `swa` 得 `-1`) |
| `:596` 的写出值 | `tl.where(valid, swa, -1)`,mask 为 `kmask` ⇒ 非 `valid` 的列写 `-1`,越界列不写 |

### 1.4 与 torch 参考实现的对照(**关键**)

`build_causal_swa_page_indices`(`:524-560`)是 `BuildCausalSwaPageIndices.torch` 的实现:

```python
539      offsets = pos_causal.unsqueeze(1) - torch.arange(
540          swa_window, dtype=torch.int32, device=device
541      ).unsqueeze(0)
542      if swa_replay_start is None:
543          invalid_offset_mask = offsets < 0
544      else:
545          invalid_offset_mask = offsets < swa_replay_start.to(offsets.dtype).unsqueeze(1)
546      offsets.masked_fill_(invalid_offset_mask, 0)
547      raw_indices = req_to_token[req_pool_indices_repeated[:, None], offsets]
548      assert raw_indices.shape == (num_qo_tokens, swa_window)
549      raw_indices.masked_fill_(invalid_offset_mask, -1)
550      swa_indices = full_to_swa_mapping[raw_indices]
551      swa_indices = swa_indices.to(torch.int32)
```

⇒ torch 路径在 `:549` 把无效格显式写成 `-1`,并在 `:550` **以 `-1` 为索引访问 `full_to_swa_mapping`**。这解释了 §2 中"表尾多一个 `-1` 元素"的存在理由。

---

## 2. `full_to_swa_mapping` 的完整生命周期

### 2.1 分配与初始化(两处,均非 `torch.empty`)

| # | 位置 | 原文 | 尺寸 / 初值 |
| --- | --- | --- | --- |
| A | `mem_cache/allocator/swa.py:101-112` | `# Trailing -1: a last_loc of -1 (no prefix) indexes it, so alloc_extend and alloc_decode see -1 on the SWA side as well.`<br>`self.full_to_swa_index_mapping = torch.cat([torch.zeros(size + self.page_size, dtype=torch.int64, device=device), torch.tensor([-1], dtype=torch.int64, device=device)])` | 长度 `size + page_size + 1`;**前 `size+page_size` 行恒为 `0`**;**末 1 行为 `-1`** |
| B | `mem_cache/allocator/swa.py:750-755` | `self.full_to_swa_index_mapping = torch.cat([torch.arange(size_swa + page_size, dtype=torch.int64, device=device), torch.tensor([-1], dtype=torch.int64, device=device)])` | 长度 `size_swa + page_size + 1`;前段为**恒等映射 `0..size_swa+page_size-1`**;末 1 行为 `-1` |

**⇒ 两处都使用 `torch.zeros` / `torch.arange`,目录内不存在 `full_to_swa*` 由 `torch.empty` 构造的写法**(对 `torch.empty` 的定向检索无命中)。

### 2.2 注册(谁把它交给 KV pool)

| 位置 | 内容 |
| --- | --- |
| `mem_cache/deepseek_v4_memory_pool.py:1111` | `self.full_to_swa_index_mapping = None`(初值) |
| `mem_cache/deepseek_v4_memory_pool.py:1270-1271` | `def register_mapping(self, full_to_swa_index_mapping): self.full_to_swa_index_mapping = full_to_swa_index_mapping` |
| `mem_cache/unified_memory_pool.py:1828, 1857` | 同上模式 |
| `mem_cache/base_swa_memory_pool.py:27` | `def register_mapping(...)` |
| `mem_cache/allocator/swa.py:763` | `self._kvcache.register_mapping(self.full_to_swa_index_mapping)` |

### 2.3 消费点

| 位置 | 内容 |
| --- | --- |
| `kernels/ops/attention/dsv4_attn_metadata_kernels.py:550` | torch 参考:`full_to_swa_mapping[raw_indices]`(`raw_indices` 可为 `-1`) |
| `layers/attention/dsv4/compressor.py:294` | `full_to_swa_index_mapping=token_to_kv_pool.full_to_swa_index_mapping` |
| `kv_canary/endpoint.py:70-71` | `if self.full_to_swa_index_mapping is not None: out_cache_loc_for_canary = self.full_to_swa_index_mapping[out_cache_loc]` |
| `speculative/dspark_components/dspark_verify.py:866`、`dspark_kv_inject.py:149` | 作为参数传入 |
| `swa.py:136-140` | **ring 模式会改写 `self._size_swa = req_to_token_pool.available_size() * self._swa_ring_cost`**(注:改的是尺寸变量,不是表内容) |

### 2.4 "是否存在未被写过的行会被读"

| 观察 | 依据 |
| --- | --- |
| 表行初值 | A 路径 ⇒ 全 `0`;B 路径 ⇒ 恒等值。**均非未初始化垃圾** |
| 因此读未写行的效果 | 得到 `0` 或恒等值 ⇒ 都是**合法编码**,不会产生"超大正整数" |
| **`-1` 行的语义** | 表尾那 1 行是**为 `full_loc == -1` 专门预留**的(注释 `:101-102` 与 torch 参考 `:549-550` 双向印证)⇒ `full_to_swa_ptr + (-1)` 落在**表内** |
| 未见到的写法 | 未找到任何 `torch.empty` 生成该表的代码路径 |

---

## 3. `req_to_token` 的生产者与初始化时点

### 3.1 分配与初始化

`mem_cache/memory_pool.py:386-409`:

```python
397      self.size = size
398      # +1 padding row at index 0: cuda-graph padded batches default
399      # req_pool_indices to 0, so dummy reads/writes land here harmlessly.
400      self._alloc_size = size + 1
401      self.max_context_len = max_context_len
402      self.device = device
403      with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
404          self.req_to_token = torch.zeros(
405              (self._alloc_size, max_context_len), dtype=torch.int32, device=device
406          )
407      self.free_slots = list(range(1, self._alloc_size))
408      self.req_generation = torch.zeros(self._alloc_size, dtype=torch.int64)
```

| 观察 | 依据 |
| --- | --- |
| 初始化方法 | **`torch.zeros`**,dtype `torch.int32` —— **不是 `torch.empty`** |
| 行 0 的用途 | 注释原文:`"+1 padding row at index 0: **cuda-graph padded batches default `req_pool_indices` to 0, so dummy reads/writes land here harmlessly**"` |
| 分配域 | `:407 free_slots = list(range(1, self._alloc_size))` ⇒ 第 0 行**不参与正常分配**,专供 cudagraph padding |
| 初值 | 全 `0` ⇒ `full_loc = 0`(合法,表内) |

### 3.2 写者(全目录检索 `req_to_token_pool.write`)

```
sglang/srt/mem_cache/allocation.py:97, 101, 663
sglang/srt/mem_cache/radix_cache.py:531
sglang/srt/mem_cache/unified_radix_cache.py:1256
sglang/srt/mem_cache/storage/lmcache/lmcache_unified_radix_cache.py:833
sglang/srt/disaggregation/decode.py:2210, 2314
sglang/srt/disaggregation/decode_hicache_mixin.py:312
sglang/srt/hardware_backend/npu/dsv4/dsv4_common_hooks.py:184, 219   (NPU, 本机不用)
```

`ReqToTokenPool.write` 本体定义在 `mem_cache/memory_pool.py:411-412`:`def write(self, indices, values): self.req_to_token[indices] = values`。

### 3.3 cudagraph 虚拟请求的元数据构造

| 位置 | 内容 |
| --- | --- |
| `layers/attention/deepseek_v4_backend.py:2398` | `def make_forward_metadata_from_raw_decode(`(定义) |
| `layers/attention/deepseek_v4_backend.py:2571` | `self.forward_metadata = self.make_forward_metadata_from_raw_decode(`(使用) |
| `layers/attention/deepseek_v4_backend_hip_radix.py:1565, 1664` | ROCm 侧同一函数的定义/使用(本机不用) |

**未取得的结论**:该函数内部对 `req_pool_indices` / `seq_lens` 的具体填法未在本轮读完(`:2398` 起的长函数体),故"capture 期 slot 行的值"这一点**不在此报告中断言**;已确认的相邻事实是 §3.1 的"行 0 全零 + 专供 padding"。

---

## 4. 最终传给 `flash_mla_with_kvcache` 的 `indices` / `extra_indices_in_kvcache` 的可能值域

由 §2、§3 阅读所得的**值域来源链**:

| 环节 | 可能取值 | 依据 |
| --- | --- | --- |
| `req_to_token[rp, off]` | `0`(未写行/保留行 0) 或 合法页 id(写过) | `memory_pool.py:404-406`(全零初值);写者见 §3.2 |
| `full_loc`(= 上者转 int64) | 同上一行;**未见表内可存负值的写入路径**;`-1` 仅由 `:592` 的 `other` 在 `valid=False` 时给出 | `:592` |
| `full_to_swa_mapping[full_loc]` | `0`(A 路径未写行) / 恒等值(B 路径) / `-1`(表尾哨兵) | `swa.py:103-112`, `:750-755` |
| `swa_page_indices` 写出 | `where(valid, swa, -1)`;越界列不写 | `:596` |
| 传给 `flash_mla_with_kvcache` 前的对齐 | `match_num_queries`:`swa_page_indices` 补 `0`、`swa_topk_lengths` 补 `1`、`extra_indices` 补 `-1` | `layers/attention/deepseek_v4_backend.py:4060-4070` |
| `extra_indices_in_kvcache` | 由 `extra_*` 参数传入,值域同上链路 | 同上一行 |

---

## 5. 表格:非法编码类型 → 产生处 → capture 期是否可能发生

| # | 非法编码类型 | 由哪段代码产生 | 在 CUDA graph capture 期是否可能发生 |
| --- | --- | --- | --- |
| 1 | `full_loc == -1`(由 `other` 造成) | `:592` 的 `other=-1` | **否** —— 只发生在 `valid=False` 的 lane,而这些 lane 在 `:595`/`:596` 也不产生有效写出 |
| 2 | `full_loc == -1`(由表内容造成) | `req_to_token` 中确有 `-1` 的行(写入方见 §3.2) | **语义上被支持**:torch 参考 `:549-550` 显式以 `-1` 索引,表尾 `:101-102`/`:750-755` 专置哨兵行 ⇒ 落在表内 |
| 3 | 超大正整数页号 | 需要 `req_to_token` 被写入超出表长的值 | **未找到该路径**;表长按**全池** `size + page_size` 分配(`:103-112`),而页号来自同池分配器 |
| 4 | 未初始化垃圾(任意大值) | 需要该表由 `torch.empty` 构造 | **否** —— 两张表均 `torch.zeros`/`torch.arange`(`memory_pool.py:404-406`;`swa.py:103`, `:750`) |
| 5 | `swa_page_indices` 中的 `-1`(无效位) | `:596` 的 `where(valid, swa, -1)`;`match_num_queries` 的 padding | **是**(正常语义,非 bug) |
| 6 | `topk_lengths` 的 padding `1` | `deepseek_v4_backend.py:4068` `value=1` | **是**;其为"声称有 1 个有效索引",是否合法取决于内核语义(§6) |

---

## 6. 两条边界:本报告**未能**提供证据

用户提示中提到的以下两点,本轮**未取得**源码级证据,**不作断言**,仅登记为待核实项:

| # | 待核实项 | 本轮状态 |
| --- | --- | --- |
| 1 | "SM90 内核对 `-1` 有专门处理,对其它负数会按无符号回绕" | **未核实**。`sgl_kernel/flash_mla.py` 与 `flashinfer/data/csrc/` 内**未定位到** SM90/Hopper 侧 `sparse_decode_fwd` 的实现源码(目录内只有 `sparse_mla_sm120/*` 与 `cake_dsv4/*` 含 `topk_length` 等符号) |
| 2 | "`topk_length` 的 mask 语义" | **未核实**。同因:`:302-315` 只给出调用点与返回值元数(`out, lse, new_tile_scheduler_metadata, new_num_splits`),未见其 C++ 侧对该参数的判读 |

已确认的相邻事实(供下一步使用):`sgl_kernel/flashmla_ops.abi3.so` 内嵌 cubin 为 **57 × `sm_90a` + 57 × `sm_100` + 57 × `sm_103a`**(`cuobjdump --list-elf`;**普通 `sm_90` 为 0 条**),该文件内 `sparse_decode_fwd` 字符串命中 2 处;`sgl_kernel/flash_ops.abi3.so` 的 cubin 为 `sm_80`/`sm_86`/`sm_90`(各 372 条),该文件内 `sparse_decode_fwd` 命中 0 处。

> **⚠️ 口径更正(作者自身错误,已作废)。** 本行早期版本(以及据此发出的报文)写作 `sm_90` / `sm_100` / `sm_103` 并据此把"编译目标不是 `sm_90a`"当作强疑点。**该结论错误。**
> 根因:当时取样用了**有损正则** `grep -oE "sm_[0-9]+"`,它会吃掉架构后缀字母(`sm_90a` → `sm_90`)。
> 用保留后缀的 `grep -oE "sm_[0-9]+[af]?"` 复核得:`sm_90a` **57** / `sm_100` **57** / `sm_103a` **57**,普通 `sm_90` **0** —— 与 binary-vs-source 核对结论一致。
> ⇒ **"SM90 目标不对"这条疑点不成立**,不应再据此推断。

---

## 7. 追加:capture 虚拟请求的元数据来源(第 1 条线索闭环)

### 7.1 **关键发现:用户分析的 kernel 在**本配置**下不在活跃路径上**

`make_core_attn_metadata` 对 `swa_page_indices` 有**三条互斥分支**,取哪条取决于 `self.token_to_kv_pool.request_window is not None`(`deepseek_v4_backend.py:4837`):

| 分支 | 条件 | `swa_page_indices` 生产者 | 位置 |
| --- | --- | --- | --- |
| **① request-window(本配置)** | `token_to_kv_pool.request_window is not None` | `window_layout(...)`(`mem_cache/dsv41_request_window.py:60-120`)→ `swa_page_indices = _pad_last_dim(request_layout.indices)`、`swa_topk_lengths = request_layout.lengths` | `deepseek_v4_backend.py:4837-4865` |
| ② DSpark draft | `dspark_block_size is not None` | `get_dspark_swa_page_indices(...)`(`:4880-4885`)或复用 `dspark_swa_buffers`(`:4887`) | `:4866-4892` |
| ③ **用户分析的 kernel** | 前两者皆不成立 | **`BuildCausalSwaPageIndices.execute(...)`** | `:4893-4909` |

`request_window` 的设置条件(`mem_cache/deepseek_v4_memory_pool.py:1110-1152`):

```python
1114     encoder_replay = get_exec().features.enable_encoder_swa_bounded_replay
1118     self.needs_paged_swa_allocator = not encoder_replay or (...)
1121     if encoder_replay and not is_draft_worker:
1122         from sglang.srt.mem_cache.dsv41_request_window import RequestWindow
1141         self.request_window = RequestWindow(...)
```

⇒ 即 **`--enable-decoder-swa-bounded-replay` 打开时**(`encoder_replay=True`),走 **分支 ①**,`BuildCausalSwaPageIndices` **根本不被调用**;关闭该 flag 时 `request_window` 保持 `None`(`:1110`),才落到 **分支 ③**。

**本配置(conf 显式开了 `--enable-decoder-swa-bounded-replay`)⇒ 活跃分支是 ①。**

### 7.2 分支 ① 的值域(`dsv41_request_window.py:60-92` 原文关键行)

```python
 81      write_loc = (history_rows + offset).to(torch.int32)
 82      lookback = torch.arange(window, device=device)
 83      seen_pos = pos[:, None] - lookback
 84      old = seen_pos < first_pos[:, None]
 85      old_loc = group[:, None] * window + (seen_pos - (first_pos[:, None] - window))
 86      new_loc = history_rows + group_first[:, None] + seen_pos - first_pos[:, None]
 87      valid = seen_pos >= 0
 88      if floor is not None:
 89          floor = floor.to(torch.int64)
 90          valid &= seen_pos >= floor[:, None]
 91      indices = torch.where(valid, torch.where(old, old_loc, new_loc), -1).to(torch.int32)
 92      lengths = valid.sum(-1).to(torch.int32)
```

| 输出 | 可能取值 | 依据 |
| --- | --- | --- |
| `indices` | `old_loc`/`new_loc`(由 `group`/`pos`/`window`/`history_rows` 等**计算**得到,非内存读) 或 **`-1`** | `:91` |
| `lengths` | `valid.sum(-1)` ⇒ **`0 .. window`(含 0)** | `:92` |
| `write_loc` | `history_rows + offset` | `:81` |

### 7.3 `extra_indices_in_kvcache` / `extra_k_cache` 的生产者与值域

| 符号 | 产生处 | 内容 |
| --- | --- | --- |
| `extra_k_cache` | `deepseek_v4_backend.py:4029` `token_to_kv_pool.get_extra_key_buffer(layer_id)`;随后 `:4044-4056` 只做 **切片 + `.view`**(`extra_k_cache[:, :extra_page_size*extra_total_dim].view(...)`) | **KV 池内固定 buffer 的一个视图**,不重新分配 |
| `extra_indices` | `:4030` `core_attn_metadata.sparse_page_indices(compress_ratio)`;`:4090` `match_num_queries(extra_indices, value=-1)`(padding 用 **-1**) | 见下 |
| `extra_topk_lengths` | `:4031-4033` `core_attn_metadata.sparse_topk_lengths(compress_ratio)`;`:4090` 后 `match_num_queries(..., value=1)` | 见下 |
| `c4_sparse_page_indices` 初值 | `:887-892` `torch.full((rows, self.index_topk), -1, dtype=torch.int32, device=...)`,再 `:893 _pad_last_dim(...)` | **初值全 `-1`** |
| `c4_sparse_topk_lengths` | `:884-886` `torch.clamp(self.c4_topk_lengths_clamp1, max=self.index_topk)` | 由 `c4_topk_lengths_clamp1` 决定,上界 `index_topk` |
| **唯一的 `torch.empty`** | `:895-897` `if is_prefill: self.c4_sparse_raw_indices = torch.empty_like(self.c4_sparse_page_indices)` | **仅在 `is_prefill=True` 时分配**;decode/capture 路径(`is_prefill=False`)不分配 |

### 7.4 二元问题的回答

> **在 capture 虚拟请求下,传给 `sparse_decode_fwd` 的 `indices` 值域是否只可能是 {合法 slot, 0(padding 行映射), -1}?**

**是。** 依据:

| 路径 | 依据 |
| --- | --- |
| 活跃分支 ①(本配置) | `dsv41_request_window.py:91` 的 `torch.where(valid, …, -1)` ⇒ 只产出**计算得到的 loc** 或 `-1`;`:92` 的 `lengths ∈ [0, window]`(含 0) |
| 备用分支 ③(关掉 flag 时) | `dsv4_attn_metadata_kernels.py:596` `tl.where(valid, swa, -1)`;`:595` 的 `swa` 取自 `full_to_swa` 表,而该表初值恒为 `0`/恒等值/末尾 `-1`(§2)⇒ 值域 = {合法 loc, 0, -1} |
| `extra_indices_in_kvcache` | `:4090` `match_num_queries(..., value=-1)` + `:887-892` 的 `torch.full(..., -1)` 初值 ⇒ {-1, 被 indexer 填入的合法 slot} |

**未找到任何可产生"超大正整数"或未初始化垃圾的写入路径**;唯一 `torch.empty`(`:895-897`)受 `is_prefill` 门控,decode/capture 期不参与。

### 7.5 `swa_k_cache` / `extra_k_cache` 的指针来源(第 4 问)

| 项 | 原文 | 结论 |
| --- | --- | --- |
| `swa_k_cache` | `:4025` `swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)`;`:4036` `assert swa_k_cache.ndim == 2`;`:4040-4042` 切片 + `.view(shape[0], swa_kv_page_size, 1, k_cache_total_dim)` | **KV 池内 buffer 的视图**(不是新分配) |
| `extra_k_cache` | `:4029` `token_to_kv_pool.get_extra_key_buffer(layer_id)`;`:4049-4056` 切片 + `.view(...)` | 同上 |
| `:4038` 注释原文 | `# The kernel detects each cache's format from the last dim of this view: 584 (V4), 528 (V4.1 fp8) or 288 (V4.1 fp4, extra cache only).` | 视图的最后一维即"格式判定依据" |
| **capture 期是否会重新分配** | KV 池在构造期一次性分配;本模型 `post_capture_kv_sizing_planned()` 为 **False**(MLA 分支,见 `SEGV-145831-analysis.md`),即**不做捕获后 resize** | 烘进图的指针在 capture→replay 之间**保持不变** |

---

## 8. 追加:`DSV4RawDecodeMetadata` 三个字段在图表捕获期的实际取值(闭环)

### 8.1 上溯路径

| 步 | 位置 | 原文要点 |
| --- | --- | --- |
| 1 | `decode_cuda_graph_runner.py:1274` | `attn_backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)` —— 捕获期入口 |
| 2 | `deepseek_v4_backend.py:2671-2679` | `def init_forward_metadata_out_graph(self, forward_batch, in_capture=False)`;`bucket = _GraphBucket.of(forward_batch.forward_mode)`;`bs = forward_batch.batch_size`;**`req_pool_indices = forward_batch.req_pool_indices`**;**`seq_lens = forward_batch.seq_lens`** |
| 3 | `deepseek_v4_backend.py:2682-2696` | `if in_capture:` … **`out_cache_loc = torch.zeros_like(seq_lens)`**;注释原文:`# Captured graph does no real cache writes, so synthesize a dummy out_cache_loc per bucket (replay supplies the real value).` |
| 4 | `deepseek_v4_backend.py:2734-2754` | 仅 `DECODE_OR_IDLE` 桶:`assert out_cache_loc is not None`;`out_cache_loc_padded = F.pad(out_cache_loc, (0, bs - len(out_cache_loc)), mode="constant", **value=0**)`;再交 `init_forward_metadata_decode(...)` |
| 5 | `deepseek_v4_backend.py:1569-1586` | `init_forward_metadata_decode` 是**纯转交**:`return DSV4RawDecodeMetadata(req_pool_indices=…, seq_lens=…, out_cache_loc=…)`,**不修改任何值** |
| 6 | `deepseek_v4_backend.py:2570-2573` | `init_forward_metadata_in_graph`:将 `DSV4RawDecodeMetadata` 经 `make_forward_metadata_from_raw_decode` 升级为 full metadata(注释:`Upgrade Raw->Full so compress + core_attn + indexer materialization is recorded inside the cuda graph`) |

### 8.2 捕获期这三个字段的**确定值**

| 字段 | 捕获期取值 | 依据 |
| --- | --- | --- |
| `req_pool_indices` | **恒 `0`** | ① `cuda_graph_buffer_registry.py:99-102` 的 `ZERO` 策略原文:`ZERO — Reset the padded region to 0 (e.g. out_cache_loc / req_pool_indices — **padded rows must point at slot 0 so dummy attention reads land harmlessly**).`;② `cuda_graph_buffer_registry.py:557-558` 明确列出 `req_pool_indices / out_cache_loc / mamba_track_* -> ZERO`;③ `input_buffers.py:44-56` 的 `INDEX_SEMANTIC_BUFFERS` 含 `req_pool_indices`/`out_cache_loc`/`positions`,注释称 *"build_decode_registry asserts its ZERO-policy slots against this set"*;④ 缓存的第 0 行是**保留行且恒为 0**(`memory_pool.py:398-407`) |
| `seq_lens` | **恒 `1`** | ① `deepseek_v4_backend.py:3328-3329` `def get_cuda_graph_seq_len_fill_value(self): return 1`;② `decode_cuda_graph_runner.py:386-387` `self.seq_len_fill_value = self.attn_backend.get_cuda_graph_seq_len_fill_value()`;③ `decode_cuda_graph_runner.py:1155-1156` **`self.buffers.seq_lens.fill_(self.seq_len_fill_value)`**;④ `seq_lens` 走 `FILL_SENTINEL`(`cuda_graph_buffer_registry.py:96-98, 275-280`) |
| `out_cache_loc` | **恒 `0`** | ① `deepseek_v4_backend.py:2689` `torch.zeros_like(seq_lens)`;② `:2742-2747` 的 padding 值亦为 `0`;③ `ZERO` 策略(同上) |

补充事实(用于排除"IDLE 覆盖"):`decode_cuda_graph_runner.py:301` **`self.capture_forward_mode = ForwardMode.DECODE`** ⇒ 捕获走 **DECODE**(非 IDLE),故 `deepseek_v4_backend.py:2705-2719` 那段 IDLE 覆盖(`seq_lens=ones`/`req_pool_indices=zeros`/`out_cache_loc=zeros`)**不作用于本路径**。

### 8.3 由此推出的 `swa_page_indices` / `lengths` / `extra_indices` 具体值

**活跃分支 ①(`window_layout`,`mem_cache/dsv41_request_window.py:60-120`)**,输入 `req=0`、`pos≈0`(`seq_lens=1` ⇒ 位置从 0 起):

| 输出 | 具体值 | 推导依据 |
| --- | --- | --- |
| `seen_pos = pos - lookback` | `0, -1, …, -(window-1)` | `:83` |
| `valid = seen_pos >= 0`(且 `>= floor` 若给) | **每行恰好 1 个 `True`**(即 `lookback=0`);若 `floor > 0` 则 **0 个** | `:87-90` |
| `indices = torch.where(valid, where(old,old_loc,new_loc), -1)` | **1 个由 `group`/`first_pos`/`window`/`history_rows` 算出的小非负整数 + 其余全 `-1`**;`floor>0` 时**全 `-1`** | `:91` |
| `lengths = valid.sum(-1)` | **`1`**(或 `floor>0` 时 `0`) | `:92` |
| `extra_indices` | `-1`(初值 `torch.full(..., -1)`,或 `match_num_queries(..., value=-1)` 的 padding);无写入时保持 `-1` | `:887-892`;`:4090` |

**备用分支 ③(仅当 bounded-replay 关闭)**:`full_loc = req_to_token[0, off]`。第 0 行为保留行且**恒 0**(`memory_pool.py:404-406` 零初始化 + `:407` 不参与分配)⇒ `full_loc = 0` ⇒ `swa = full_to_swa[0]` = **`0`**(A 路径表初值,`swa.py:103-112`)⇒ `indices ∈ {0, -1}`。

### 8.4 可证伪的结论(第 3 问要求的形式)

| 情形 | 由代码决定 | 后果 |
| --- | --- | --- |
| **X(即代码事实)**:`req_pool_indices ≡ 0`、`seq_lens ≡ 1`、`out_cache_loc ≡ 0` | §8.2 四条独立依据 | `indices ∈ {计算得到的小整数 loc, -1}`、`lengths ∈ {0, 1}`(分支①)或 `indices ∈ {0, -1}`(分支③)⇒ **索引必然落在合法范围内**;`req_to_token` 行 0 恒 0,`full_to_swa[0]=0` 在表内,`-1` 命中表尾哨兵 |
| **Y1**:`req_pool_indices` 指向非 0 的**未写过** slot 行 | 需图运行器的 dummy FB 偏离 ZERO 策略 | 该行初值亦为 `0`(全表 `torch.zeros`)⇒ 仍得 `full_loc = 0`;但会读到**别行**的页号 ⇒ 语义错误而非越界 |
| **Y2**:`req_pool_indices` **超出 `_alloc_size`**(即 `rp ≥ size+1`) | 需写入越界 slot 号 | `base = req_to_token_ptr + rp*rt_stride`(`dsv4_attn_metadata_kernels.py:580`)直接越界 ⇒ **这是唯一能产生真正越界的形态**;本轮**未找到**产生该值的路径 |
| **Y3**:`seq_lens = 0` | 需偏离 fill 值 `1` | `pos = -1` ⇒ `seen_pos` 全负 ⇒ `lengths = 0`、`indices` 全 `-1` ⇒ 安全但无语义(不会越界) |
| **Y4**:`seq_lens` > 实际写入长度 | 需 `seq_lens` 与 `out_cache_loc` 不一致 | `pos` 指向未写位置 ⇒ 读到 `0` ⇒ 映射到 page 0 ⇒ **合法但语义错误** |

**结论(可证伪形式)**:若 §8.2 三个字段确为 `0 / 1 / 0`,则索引**必然合法**,用户提出的"捕获期产生无效 SWA/extra KV 索引"**不成立**;反之,若要证成该假设,必须给出**一个使 `req_pool_indices ≥ size+1` 的写入路径**(Y2 形态)—— 本轮遍历未见此路径。其余偏离(Y1/Y3/Y4)只能造成**语义错误**,不产生越界。
