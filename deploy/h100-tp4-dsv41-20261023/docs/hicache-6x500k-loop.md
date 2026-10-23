# HiCache 命中验证：6 条 ~500K prompt 按 1..6,1..6 循环两遍

- 条件：TP4（物理卡 4-7）/ `resident=0-19` / `LK_POOL_SLOTS=192` / `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` /
  `--enable-hierarchical-cache --hicache-ratio 6` / dspark 开 / `--max-total-tokens 4194304`（profile 1,984,000 ⇒ 实际 KV = 1,984,000）/ 端口 18010
- 负载：6 条互不相同的**随机汉字** prompt（346,620 字符/条），两遍顺序均为 1,2,3,4,5,6；每请求 `max_tokens=8`
- 命中判据：服务端日志 `Prefill batch … #cached-token:`（响应体**不含** `prompt_tokens_details.cached_tokens`，实测返回 None）
- 原始 CSV：`/mnt/SSD01/models/.llm-env/Lsglang/logs/hicache-6x500k-20261023_2222.csv`

| # | prompt_tokens | 第1遍耗时(s) | 第2遍耗时(s) | 第2遍 #cached-token | 命中率 |
|---|---|---|---|---|---|
| 1 | 501111 | 248.750 | 3.065 | 499712 | 99.72% |
| 2 | 500395 | 238.673 | 1.771 | 500224 | 99.97% |
| 3 | 501350 | 348.686 | 3.324 | 499712 | 99.67% |
| 4 | 500357 | 533.285 | 1.731 | 500224 | 99.97% |
| 5 | 501003 | 543.804 | 2.934 | 499712 | 99.74% |
| 6 | 501239 | 549.865 | 1.837 | 500992 | 99.95% |
| **合计** | 3005455 | **2463.1** | **14.7** | — | — |

## HiCache 主机侧分配（启动日志原文）

- `Allocating 9.04 GB host memory for V4 paged pool 'swa'`（每 rank）
- `Allocating 6.96 GB host memory for V4 paged pool 'deepseek_v4_c1'`（每 rank）
- `Allocating 10.45 GB host memory for V4 paged pool 'deepseek_v4_c2'`（每 rank）
- `Allocating 0.81 GB host memory for V4 paged pool 'deepseek_v4_c1_indexer'`（每 rank）
- `Allocating 1.21 GB host memory for V4 paged pool 'deepseek_v4_c2_indexer'`（每 rank）
- 每 rank 合计 **28.47 GB**；4 rank 合计 **113.88 GB**
- `Tree cache initialized: source=default impl=UnifiedRadixCache hybrid_swa=True hybrid_ssm=False hicache_attached=True`（未开 HiCache 时为 `hicache_attached=False`）
- 测试期间主机内存：used 239 GiB / available 1036 GiB

## 同口径 decode 对照（16 个 diverge prompt, temp 0.7, max_tokens 400）

| | HiCache 关（实例 202722） | HiCache 开（实例 211417） |
|---|---|---|
| decode p50 | 162.3 tok/s | 161.5 tok/s |
| accept len p50 | 2.59 | 2.60 |

## 冷 prefill（HiCache 开，8K 分块）

| 负载 | 结果 |
|---|---|
| 全新随机 prompt 100,129 token | 109.36 s ⇒ 约 916 tok/s |
| 6×500K 第 1 遍逐条（同一次运行内） | 501,111 tok / 248.75 s（2,014 tok/s）→ 549.87 s（909 tok/s） |
