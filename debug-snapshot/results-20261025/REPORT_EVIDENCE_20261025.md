# 部署取证 — 提交 1cf916ebc0 (GPU expert cache)

生成: 2026-10-25T17:10:29+08:00

## 1. 提交
```
1cf916ebc0f33d63dca25aead3d5a96beaf54a84
feat(lk-moe): GPU expert cache for offloaded layers, misses read over UVA
作者 Claude, 2026-10-24 21:31:59 +0000
基线 7f664cf6a6 (父提交)
远端 origin/debug/dspark-variant-matrix-20261025
```

### 相对 7f664cf6a6 净差异 (8 文件, +517/-21)
```
 debug-snapshot/ANALYSIS-20261025.md                |  36 ++-
 python/sglang/kernels/ops/moe/lk_verify_uva.py     |  41 +++-
 .../srt/layers/moe/fused_moe_triton/layer.py       |  90 +++++++-
 python/sglang/srt/layers/moe/gpu_expert_cache.py   | 257 +++++++++++++++++++++
 .../runner_backend/full_cuda_graph_backend.py      |   8 +
 python/sglang/srt/utils/lk_moe_env.py              |  26 ++-
 test/manual/kernels/test_lk_verify_uva.py          |  46 +++-
 test/manual/layers/test_gpu_expert_cache_policy.py |  34 +++
 8 files changed, 517 insertions(+), 21 deletions(-)
```

## 2. 部署副本 md5 (部署副本 == 仓库版本)
```
FILE                                                           MD5
kernels/ops/moe/lk_verify_uva.py                               de652c16a0311f988d1ab0b826e0cdec
srt/layers/moe/fused_moe_triton/layer.py                       9ae6df7d3c28fe5621de9a4af9a09246
srt/layers/moe/gpu_expert_cache.py                             272c13a06e85e3eb1299b56acb9533a0
srt/model_executor/runner_backend/full_cuda_graph_backend.py   d2cc8403b5ba14c2e35e1ff61cd3dc9a
srt/utils/lk_moe_env.py                                        45d7471ea148c5bfe17257ff79acc4d8
```

## 3. 真卡测试
```
CUDA_VISIBLE_DEVICES=GPU-7561b28f-b049-bbdb-31f8-16efeb06663d,GPU-fb59db94-0e4e-4c4f-8ae5-4d0fdd1e0f0b,GPU-3c5d7267-445c-4679-bce9-503bedd0c920,GPU-3c0f2856-a6e3-b9d5-fd10-a08f55579fff
=== started 2026-10-25T16:36:33+08:00 ===
.............                                                            [100%]
=============================== warnings summary ===============================
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED test/manual/kernels/test_lk_verify_uva.py::test_self_check_rejects_wrong_addresses
PASSED test/manual/kernels/test_lk_verify_uva.py::test_draft_miss_matches_reference[False-0.0-1]
PASSED test/manual/kernels/test_lk_verify_uva.py::test_draft_miss_matches_reference[False-0.0-3]
PASSED test/manual/kernels/test_lk_verify_uva.py::test_draft_miss_matches_reference[False-7.0-1]
PASSED test/manual/kernels/test_lk_verify_uva.py::test_draft_miss_matches_reference[False-7.0-3]
PASSED test/manual/kernels/test_lk_verify_uva.py::test_draft_miss_matches_reference[True-0.0-1]
PASSED test/manual/kernels/test_lk_verify_uva.py::test_draft_miss_matches_reference[True-0.0-3]
PASSED test/manual/kernels/test_lk_verify_uva.py::test_draft_miss_matches_reference[True-7.0-1]
PASSED test/manual/kernels/test_lk_verify_uva.py::test_draft_miss_matches_reference[True-7.0-3]
PASSED test/manual/kernels/test_lk_verify_uva.py::test_cache_mode_computes_every_route_and_counts_them
PASSED test/manual/layers/test_gpu_expert_cache_policy.py::test_empty_slots_take_the_hottest_experts_first
PASSED test/manual/layers/test_gpu_expert_cache_policy.py::test_budget_and_gain_limit_the_swaps
PASSED test/manual/layers/test_gpu_expert_cache_policy.py::test_never_swaps_in_a_resident_or_unrouted_expert
13 passed, 2 warnings in 24.77s
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

real	0m26.626s
user	1m17.976s
sys	0m8.076s
PYTEST_EXIT=0
=== finished 2026-10-25T16:36:59+08:00 ===
```

## 4. 启动与运行期日志
```
health 就绪: 2026-10-25 17:04:26  t+1287s (启动 ts 16:43:20) => 约 21.5 分钟
PID=59756 PGID=59756

--- LK_GPU_CACHE_SLOTS 层行 (20 层 x 4 rank = 80 行; 取 layer 20 与 layer 39) ---
[2026-10-25 16:52:52 TP2 EP1] LK_GPU_CACHE_SLOTS: layer 20: 104 VRAM slots (0.91 GiB), 192 experts pinned on host (1.68 GiB)
[2026-10-25 16:53:05 TP1 EP0] LK_GPU_CACHE_SLOTS: layer 39: 104 VRAM slots (0.91 GiB), 192 experts pinned on host (1.68 GiB)

--- [gpu-expert-cache] 全部行 ---
[2026-10-25 17:04:20 TP0 EP0] [gpu-expert-cache] 20 layers, last 687s: routes=143789 hit=49.9% swaps=76
[2026-10-25 17:04:20 TP2 EP1] [gpu-expert-cache] 20 layers, last 687s: routes=110371 hit=50.8% swaps=73
[2026-10-25 17:04:20 TP1 EP0] [gpu-expert-cache] 20 layers, last 687s: routes=143789 hit=49.9% swaps=76
[2026-10-25 17:04:20 TP3 EP1] [gpu-expert-cache] 20 layers, last 687s: routes=110371 hit=50.8% swaps=73
[2026-10-25 17:04:50 TP0 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=1767 hit=73.1% swaps=278
[2026-10-25 17:04:50 TP3 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=1833 hit=76.9% swaps=288
[2026-10-25 17:04:50 TP1 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=1767 hit=73.1% swaps=278
[2026-10-25 17:04:50 TP2 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=1833 hit=77.1% swaps=288
[2026-10-25 17:05:20 TP0 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:05:20 TP1 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:05:20 TP3 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:05:20 TP2 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:05:50 TP1 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:05:50 TP0 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:05:50 TP3 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:05:50 TP2 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:06:20 TP0 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=4984 hit=83.4% swaps=520
[2026-10-25 17:06:20 TP1 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=4984 hit=83.4% swaps=520
[2026-10-25 17:06:20 TP3 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=5096 hit=81.7% swaps=528
[2026-10-25 17:06:20 TP2 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=5096 hit=81.6% swaps=528
[2026-10-25 17:06:50 TP0 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:06:50 TP1 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:06:50 TP3 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:06:50 TP2 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:07:20 TP0 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:07:20 TP3 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:07:20 TP1 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:07:21 TP2 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:07:50 TP0 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:07:50 TP3 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:07:50 TP1 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:07:51 TP2 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:08:20 TP0 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:08:20 TP3 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:08:20 TP1 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:08:21 TP2 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:08:50 TP0 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:08:50 TP3 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:08:51 TP1 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:08:51 TP2 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:09:21 TP0 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:09:21 TP3 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:09:21 TP1 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:09:21 TP2 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:09:51 TP0 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:09:51 TP3 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:09:51 TP1 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:09:51 TP2 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:10:21 TP0 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:10:21 TP3 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:10:21 TP1 EP0] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0
[2026-10-25 17:10:21 TP2 EP1] [gpu-expert-cache] 20 layers, last 30s: routes=0 hit=0.0% swaps=0

--- [pool] slab: 真实行数 (banner 除外) ---
0
0
```

## 5. 冒烟
```
smoke run 20261025_170600  base=http://127.0.0.1:18010
=== [1] chat completion ===
--- response ---
{"id":"04c8170d97304f64ab15a66ff8115340","object":"chat.completion","created":1792919162,"model":"DeepSeek-V4.1-Flash-1017","choices":[{"index":0,"message":{"role":"assistant","content":"17*23 is 391.","reasoning_content":null},"logprobs":null,"finish_reason":"stop","matched_stop":1}],"usage":{"prompt_tokens":18,"total_tokens":26,"completion_tokens":8,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default","weight_versions":[{"version":"default","start":0,"end":8}]}}
=== [2] tool call (max_tokens=256) ===
--- response ---
{"id":"21a80b6d2cfd4ca4b389f91b81ee8a89","object":"chat.completion","created":1792919165,"model":"DeepSeek-V4.1-Flash-1017","choices":[{"index":0,"message":{"role":"assistant","content":"I'll check the current weather in Shanghai for you.","reasoning_content":null,"tool_calls":[{"id":"call_62b9b14b8cfa4022b7189bf4","index":0,"type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Shanghai\"}"}}]},"logprobs":null,"finish_reason":"tool_calls","matched_stop":null}],"usage":{"prompt_tokens":286,"total_tokens":336,"completion_tokens":50,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default","weight_versions":[{"version":"default","start":0,"end":50}]}}
```

## 6. 启动自检结果 (退出码 1)
见 results/selfcheck_20261025.txt
