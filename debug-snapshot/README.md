# DSV4.1-Flash-1017 / TP4×H100 / EP2 / DSPARK —— 各版本实测记录（2026-10-24 ~ 25）

本文记录：版本、递交、与原版的文件差异、实测数字、每个数字所用的测试代码、原始数据位置。

---

## 1. 环境（除表中注明外，所有行相同）

| 项 | 值 |
|---|---|
| 模型 | `/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017` |
| 硬件 | 4×H100 80GB（物理卡 4–7，`CUDA_VISIBLE_DEVICES` 用 UUID 固定） |
| 并行 | `--tensor-parallel-size 4 --ep-size 2`（EP2），`LK_POOL_SLOTS=104`，`LK_GPU_RESIDENT_MOE_LAYERS=0-19` |
| KV / 缓存 | `--max-total-tokens 4194304`，`--enable-hierarchical-cache --hicache-ratio 6` |
| 其它 | ctx 1M，chunked-prefill 8192，mem-fraction 0.88，max-running-requests 16，`--disable-flashinfer-autotune` |
| 权重精度 | W4A16（`flashinfer_mxfp4_moe_precision='default'`），除表中注明 |
| `reject_sampling` Q 容差补丁 | 关闭（所有行） |
| `flashinfer/fused_moe/prepare.py` 分块补丁 | 关闭，除表中注明 |
| 启动脚本 | `/mnt/SSD01/models/.llm-env/Lsglang/scripts/serve_dsv41_tp4_h100.sh` |
| conf | `/mnt/SSD01/models/.llm-env/Lsglang/configs/dsv41_tp4_h100.conf`（各轮快照见 `results/`） |
| 端点 | `http://127.0.0.1:18010`，key `sk-dsv41-h100-interop-2026` |

“原版” = upstream `main` = `60f6658524`。

---

## 2. 主表 A：版本、递交与分支、开关参数、运行态取了哪些文件

“版本”= 该轮**运行态**的概述（不是提交号）。递交号只写在“递交 / 分支”列。
`LK_VERIFY_GEMM` = 该轮实际生效的 verify GEMM 开关。
“与 `main` 不同的文件”= 该轮实际加载的 `python/sglang` 中与 upstream `main`(`60f6658524`) 内容不同的文件，逐个列出并标注内容取自哪里。

| # | 版本（运行态概述） | 递交 / 分支 | DSPARK | LK_VERIFY_GEMM | 权重精度 | rejection sampling | Q 容差补丁 | PRMT | prepare 分块补丁 | 与 `main` 不同的文件（内容来源） |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 无 DSPARK；无 PRMT；无分块补丁 | `a69d8616ce` — `deploy/h100-tp4-dsv41-nodspark-20261024`（已推送） | 关 | — | W4A16（不传 precision） | 关 | **开** | 无 | 无 | `lk_verify.py`(新增←`9591d76e4f`)、`layer.py`(←`9591d76e4f`)、`model_runner.py`(←`9591d76e4f`)、`decode_cuda_graph_runner.py`(←`9591d76e4f`)、`forward_batch.py`(←`9591d76e4f`)、`deepseek_v4_dspark.py`(←`9591d76e4f`)、`standard.py`(←`30e2bc11f7`)、`reject_sampling.py`(←`6c2b980aa8`) |
| 2 | 无 DSPARK；**W4A8** + prepare 分块补丁 | `292849bba8` — `experiment/prmt-nodspark-20261024`（已推送） | 关 | false | **W4A8**（`--flashinfer-mxfp4-moe-precision fp8`） | 关 | 关 | **有**（`lk_verify.py` 取自 `25da0f80fb`） | **有**（site-packages 内 `flashinfer/fused_moe/prepare.py`，不在仓库） | `lk_verify.py`(新增←`25da0f80fb`，含 PRMT)、`layer.py`(←`d06c0d72`)、`model_runner.py`(←`9591d76e4f`)、`decode_cuda_graph_runner.py`(←`9591d76e4f`)、`forward_batch.py`(←`9591d76e4f`)、`deepseek_v4_dspark.py`(←`9591d76e4f`)、`standard.py`(←`30e2bc11f7`)、`dspark_verify.py`(←`d06c0d72`) |
| 3 | DSPARK；verify GEMM 改为 opt-in 的提交 | `f3911c5e` — `fix/h100-dspark-verify-gemv-default`（已推送） | 开 | false | W4A16 | 关 | **开** | 无 | 无 | `lk_verify.py`(新增←`9591d76e4f`)、`layer.py`(←`f3911c5e`)、`model_runner.py`(←`9591d76e4f`)、`decode_cuda_graph_runner.py`(←`9591d76e4f`)、`forward_batch.py`(←`9591d76e4f`)、`deepseek_v4_dspark.py`(←`9591d76e4f`)、`standard.py`(←`30e2bc11f7`)、`reject_sampling.py`(←`6c2b980aa8`) |
| 4 | DSPARK；GEMV | **无递交**：代码树 = `a69d8616ce`（同 #1，未改代码）；运行态只改 conf `SPEC_ENABLED="0"→"1"`，serve 脚本未加 rejection 参数；生效快照见 §4 的 `…190831.conf` | 开 | true | W4A16 | 关 | 关 | 无 | 无 | 同 #1 的 8 个文件（`reject_sampling.py` 除外，见本行 Q 补丁=关） |
| 5 | DSPARK；GEMV；rejection 开 | **无递交**：代码树 = `a69d8616ce`（同 #1，未改代码）；运行态在 #4 基础上给 serve 脚本加 `--speculative-use-rejection-sampling`；生效快照见 §4 的 `…194238.conf` | 开 | true | W4A16 | **开** | 关 | 无 | 无 | 同 #4 |
| 6 | DSPARK；GEMV；含 accept-ownership 修复 | `d06c0d72` — `fix/dspark-accept-result-ownership`（已推送） | 开 | false | W4A16 | 开 | 关 | 无 | 无 | `lk_verify.py`(新增←`9591d76e4f`)、`layer.py`(←`f3911c5e`)、`model_runner.py`(←`9591d76e4f`)、`decode_cuda_graph_runner.py`(←`9591d76e4f`)、`forward_batch.py`(←`9591d76e4f`)、`deepseek_v4_dspark.py`(←`9591d76e4f`)、`standard.py`(←`30e2bc11f7`)、`dspark_verify.py`(←`d06c0d72`) |
| 7 | 同 #6 | 同 #6 | 开 | false | W4A16 | 开 | 关 | 无 | 无 | 同 #6 |
| 8 | DSPARK；含 full-rows + accept-ownership 修复；**GEMM 开** | `e6a1cef36d` — `fix/lk-target-verify-full-rows`（已推送） | 开 | true | W4A16 | 开 | 关 | 无 | 无 | 同 #6 |
| 9 | 同 #8，但 **GEMM 关** | 同 #8 | 开 | false | W4A16 | 开 | 关 | 无 | 无 | 同 #6 |
| 10 | DSPARK；**首个 DSPARK 改动（`9591d76e4f`）之前**的代码 + EP2 修复 | `d83c3d1ac2` — `experiment/predspark-plus-ep2fix`（已推送） | 开 | 不适用 | W4A16 | 开 | 关 | 无（`lk_verify.py` 不存在） | 无 | `standard.py`(←`30e2bc11f7` 的 8 行守卫)；其余全部 = `main` |

各差异文件的完整路径：`python/sglang/` + 下表中的路径。

| 表中短名 | 完整路径 |
|---|---|
| `lk_verify.py` | `kernels/ops/moe/lk_verify.py` |
| `reject_sampling.py` | `kernels/ops/speculative/reject_sampling.py` |
| `layer.py` | `srt/layers/moe/fused_moe_triton/layer.py` |
| `standard.py` | `srt/layers/moe/token_dispatcher/standard.py` |
| `model_runner.py` | `srt/model_executor/model_runner.py` |
| `decode_cuda_graph_runner.py` | `srt/model_executor/runner/decode_cuda_graph_runner.py` |
| `forward_batch.py` | `srt/model_executor/runner_utils/forward_batch.py` |
| `deepseek_v4_dspark.py` | `srt/models/deepseek_v4_dspark.py` |
| `dspark_verify.py` | `srt/speculative/dspark_components/dspark_verify.py` |

## 3. 主表 B：实测数字

**分数口径统一为「通过数 / 总数」**（复述 = 输出逐字包含 `DeepSeek-V4.1-Flash-1017` 的次数；工具调用 = 返回非空 `tool_calls` 且 `finish_reason == "tool_calls"` 的次数）。
标记：`✅` = 该项全部通过；`❌` = 该项存在失败。括号内为所用的脚本与参数。

| # | 版本（同主表 A） | 贪心复述 通过/总数 | 采样复述 通过/总数 | 工具调用 通过/总数 | decode tok/s | ms/step | accept len | 服务日志 |
|---|---|---|---|---|---|---|---|---|
| 1 | 无 DSPARK；W4A16 | 4/4 ✅ (n=4,mt48,`_compare_nodspark.py`) | 未测 | 1/1 ✅ (n=1,temp0,`_compare_nodspark.py`) | 90.76 | 11.02 | n/a | `20261024_164326` |
| 2 | 无 DSPARK；W4A8+分块 | 未测 | 10/10 ✅ (n=10,mt48,`_recall_temp.py 0.7 10`)；另有 6/6 ✅ (n=6) | 5/5 ✅ (n=5,temp0.7,内联) | 91.56 | 10.92 | n/a | `20261024_214357` |
| 3 | DSPARK；GEMV opt-in；Q 开 | 0/10 ❌ (n=10,mt40,`_recall10.py`)；2/4 ❌ (n=4,mt48,`_compare_nodspark.py`) | 未测 | 0/4 ❌ (n=3,temp0 内联 + n=1,temp0 `_compare_nodspark.py`) | 未测 | 未测 | 3.26 (n=5) | `20261024_180111` |
| 4 | DSPARK；GEMV；Q 关 | 2/10 ❌ (n=10,mt40,`_recall10.py`)；2/4 ❌ (n=4,mt48,`_compare_nodspark.py`) | 未测 | 0/4 ❌ (n=4,temp0,内联) | 未测 | 未测 | 3.43 (n=3) | `20261024_190831` |
| 5 | DSPARK；GEMV；rejection 开；Q 关 | 0/10 ❌ (n=10,mt48,`_recall_temp.py 0.0 10`) | 3/6 ❌ (n=6,mt48,`_recall_temp.py 0.7 6`) | 0/5 ❌ (n=5,temp0.7,内联) | 未测 | 未测 | 3.52 (n=7) | `20261024_194238` |
| 6 | +accept-ownership；GEMM 关；Q 关 | 未测 | 4/6 ❌ (n=6,mt48,`_recall_temp.py 0.7 6`) | 0/5 ❌ (n=5,temp0.7,内联) | 未测 | 未测 | 3.46 (n=5) | `20261024_201212` |
| 7 | 同 #6 | 0/10 ❌ (n=10,mt48,`_recall_temp.py 0.0 10`) | 未测 | 0/4 ❌ (n=4,temp0,内联) | 未测 | 未测 | 3.45 (n=8) | `20261024_210000` |
| 8 | +full-rows；GEMM 开；Q 关 | 8/10 ❌ (n=10,mt48,`_recall_temp.py 0.0 10`)；19/20 ❌ (n=20,mt32,`_recall_det.py`) | 10/10 ✅ (n=10,mt48,`_recall_temp.py 0.7 10`) | 10/10 ✅ (n=5,temp0 + n=5,temp0.7,内联) | 31.73 | 31.52 | 2.90 (n=28) | `20261024_222630` |
| 9 | +full-rows；GEMM 关；Q 关 | 20/20 ✅ (n=20,mt32,`_recall_det.py`) | 7/10 ❌ (`_recall_temp.py 0.7 10`)；18/20 ❌、16/20 ❌ (`_recall_samp20.py` ×2) | 10/10 ✅ (n=5,temp0 + n=5,temp0.7,内联) | 30.85 | 32.41 | 3.55 (n=22) | `20261024_230027` |
| 10 | 首个 DSPARK 改动之前 + EP2 修复 | 19/20 ❌ (n=20,mt32,`_recall_det.py`) | 19/20 ❌ (n=20,mt32,`_recall_samp20.py`) | 10/10 ✅ (n=5,temp0 + n=5,temp0.7,内联) | 31.93 | 31.32 | 3.32 (n=18) | `20261024_233544` |

补充事实：

- #8 贪心 19/20：同一次测试中出现 2 种输出，19 次为 `# 交接说明 — DeepSeek-V4.1-Flash-1017 / TP4(H100) / EP2 现状`，1 次为 `# 交接说明 — DeepSeage-V4.1-Flash-1017 / TP4(H100) / EP2 现状`。#10 贪心 19/20 同样是这 2 种输出（19 + 1 次 `DeepSeage`）。
- #1 未开 DSPARK 时，服务日志中不出现 `accept len` 字段。
- #8 的两次贪心测试来自两个脚本（`_recall_temp.py` n=10 与 `_recall_det.py` n=20），二者结果分别为 8/10 与 19/20。
- Q 容差补丁的运行时开关状态（#1、#3 为开，其余为关）来自当次同步操作记录；服务日志不记录该状态。

## 4. 各轮生效 conf 快照与命令行关键项

每轮启动时 `serve_dsv41_tp4_h100.sh` 会把**当时生效的完整 conf 与最终命令行**存到
`/mnt/SSD01/models/.llm-env/Lsglang/configs/dsv41_tp4_h100.effective.<起始时刻>.conf`；
这些文件已复制到本目录 `results/`。下表是从这些快照的命令行里直接读出的关键项。

| # | 生效快照（`results/` 内） | `--speculative-algorithm DSPARK` | `--speculative-use-rejection-sampling` | `LK_VERIFY_GEMM` 环境变量 | `--flashinfer-mxfp4-moe-precision` | `--api-key` |
|---|---|---|---|---|---|---|
| 1 | `dsv41_tp4_h100.effective.20261024_164326.conf` | 无 | 无 | 未导出 | 不传（=default） | sk-… |
| 2 | `dsv41_tp4_h100.effective.20261024_214357.conf` | 无 | 无 | `false` | `fp8` | sk-… |
| 3 | `dsv41_tp4_h100.effective.20261024_180111.conf` | 有 | 无 | 未导出 | 不传 | sk-… |
| 4 | `dsv41_tp4_h100.effective.20261024_190831.conf` | 有 | 无 | 未导出 | 不传 | sk-… |
| 5 | `dsv41_tp4_h100.effective.20261024_194238.conf` | 有 | 有 | 未导出 | 不传 | sk-… |
| 6 | `dsv41_tp4_h100.effective.20261024_201212.conf` | 有 | 有 | `false` | 不传 | sk-… |
| 7 | `dsv41_tp4_h100.effective.20261024_210000.conf` | 有 | 有 | `false` | 不传 | sk-… |
| 8 | `dsv41_tp4_h100.effective.20261024_222630.conf` | 有 | 有 | `true` | 不传 | sk-… |
| 9 | `dsv41_tp4_h100.effective.20261024_230027.conf` | 有 | 有 | `false` | 不传 | sk-… |
| 10 | `dsv41_tp4_h100.effective.20261024_233544.conf` | 有 | 有 | `false` | 不传 | sk-… |

（`LK_VERIFY_GEMM` 的开关状态取自快照内的 `export LK_VERIFY_GEMM=…` 行；“未导出”表示当轮环境里没有这个变量。）

#1 与 #2 的差异（两份快照对照）：`--speculative-*` 三项均无 → #2 多了
`--flashinfer-mxfp4-moe-precision fp8` 与 `export LK_VERIFY_GEMM=false`。

## 5. 测试脚本（`scripts/`，原样收录）

| 文件 | 用途 | 参数（脚本内硬编码或命令行） |
|---|---|---|
| `_recall_det.py` | 复述 | 材料 = `HANDOFF.md` 全文；`max_tokens=32`、`temperature=0.0`、`N=20`；输出“不同输出种类”计数 |
| `_recall_samp20.py` | 复述 | 同上但 `temperature=0.7`、`N=20` |
| `_recall_temp.py <temp> <N>` | 复述 | `max_tokens=48`；`temp`、`N` 由命令行给 |
| `_recall10.py` | 复述 | `max_tokens=40`、`temperature=0.0`、`N=10` |
| `_compare_nodspark.py` | 复述 ×4 + 工具 ×1 + 自由文本 ×1 | 复述 `max_tokens=48 temp=0`；工具 `max_tokens=256 temp=0`；自由文本 `max_tokens=256 temp=0.7` |
| `bench_ep2_streams.py` | decode / ms-per-step / TTFT / 显存采样 | `--k 1 --max-tokens 1000(或600) --temp 0.7 --api-key …` |
| `parse_serve_log.py` | 从服务日志提取 `accept len` / `gen throughput` / `[pool]` 等 | `--tp 0`（默认） |

各行的复述数字来自哪个脚本：

| 行 | 贪心复述 | 采样复述 |
|---|---|---|
| #1 | `_compare_nodspark.py` (n=4,mt48) | — |
| #2 | — | `_recall_temp.py 0.7 10` |
| #3 | `_recall10.py` (n=10,mt40)；`_compare_nodspark.py` (n=4,mt48) | — |
| #4 | `_recall10.py` (n=10,mt40)；`_compare_nodspark.py` (n=4,mt48) | — |
| #5 | `_recall_temp.py 0.0 10` | `_recall_temp.py 0.7 6` |
| #6 | — | `_recall_temp.py 0.7 6` |
| #7 | `_recall_temp.py 0.0 10` | — |
| #8 | `_recall_det.py` (n=20,mt32)；`_recall_temp.py 0.0 10` | `_recall_temp.py 0.7 10` |
| #9 | `_recall_det.py` (n=20,mt32) | `_recall_temp.py 0.7 10`；`_recall_samp20.py` ×2 |
| #10 | `_recall_det.py` (n=20,mt32) | `_recall_samp20.py` (n=20,mt32) |

---

## 6. 工具调用测试：内联命令

工具调用测试未使用脚本文件。请求体结构在各次之间相同，均为：单轮 user 消息 + 1 个 `get_weather` 工具 + `tool_choice="auto"` + `max_tokens=256`。各次差异在 `n` 与 `temperature`（见主表括号内标注：n=1/3/4/5/10，temp=0/0.7）。

以下是**最后一次实际执行**的命令（2026-10-24 23:59，服务 `20261024_233544`），原样照抄：

```bash
cd /tmp && K="sk-dsv41-h100-interop-2026"; B=http://127.0.0.1:18010
for t in 0 0.7; do
for i in 1 2 3 4 5; do
curl -s -m 300 $B/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $K" \
 -d "{\"model\":\"DeepSeek-V4.1-Flash-1017\",\"messages\":[{\"role\":\"user\",\"content\":\"北京现在天气怎么样？请调用工具查询。\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"查天气\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"tool_choice\":\"auto\",\"max_tokens\":256,\"temperature\":$t}" \
 | /mnt/SSD01/models/.llm-env/Lsglang/env/bin/python -c "
import json,sys;d=json.load(sys.stdin);c=d['choices'][0];m=c['message']
ok = m.get('tool_calls') is not None
print('  temp=$t #$i %s finish=%-11s %s' % ('✅' if ok else '❌', c.get('finish_reason'), json.dumps(m.get('tool_calls'),ensure_ascii=False)[:70] if ok else repr((m.get('content') or '')[:42])))"
done; done
```

判定条件：`tool_calls` 非空且 `finish_reason == "tool_calls"`。

多轮工具结果回填的命令（#8/#9/#10 各执行 1 次）：

```bash
curl -s -m 300 $B/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $K" \
 -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"北京天气如何？"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"北京\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"晴，25°C，湿度40%"}],"max_tokens":200,"temperature":0.7}'
```

---

## 7. 原始数据位置（均在本目录内）

| 路径 | 内容 |
|---|---|
| `logs/serve_dsv41_tp4_h100_20261024_<HHMMSS>.log` | 主表 10 行的服务日志原文 |
| `results/bench_*.json` / `.csv` / `.vram.csv` | `bench_ep2_streams.py` 的逐流结果、摘要、显存采样 |
| `results/dsv41_tp4_h100.effective.*.conf` | 各轮启动时服务端保存的生效参数快照（22 份） |
| `results/dsv41_tp4_h100.conf.current` | 本文写作时的 conf |
| `scripts/` | §5 列出的 7 个脚本原件 |

行 ↔ 日志对应：

| 行 | 服务日志 |
|---|---|
| 1 | `logs/serve_dsv41_tp4_h100_20261024_164326.log` |
| 2 | `logs/serve_dsv41_tp4_h100_20261024_214357.log` |
| 3 | `logs/serve_dsv41_tp4_h100_20261024_180111.log` |
| 4 | `logs/serve_dsv41_tp4_h100_20261024_190831.log` |
| 5 | `logs/serve_dsv41_tp4_h100_20261024_194238.log` |
| 6 | `logs/serve_dsv41_tp4_h100_20261024_201212.log` |
| 7 | `logs/serve_dsv41_tp4_h100_20261024_210000.log` |
| 8 | `logs/serve_dsv41_tp4_h100_20261024_222630.log` |
| 9 | `logs/serve_dsv41_tp4_h100_20261024_230027.log` |
| 10 | `logs/serve_dsv41_tp4_h100_20261024_233544.log` |

---

## 8. 数据来源与已知的缺失

1. 复述测试的参数：`n` = 4/6/10/20，`max_tokens` = 32/40/48，逐行见 §3 与 §5。
2. 工具调用测试为内联命令，`n` = 1/3/4/5/10，`temperature` = 0 或 0.7，逐行见 §3 括号内标注。
3. 模型输出正文只在终端打印，**未逐条落盘**；复述列的 ✅/❌ 依据是当时终端输出。各服务日志中不含模型正文（sglang 只记录 token 计数与池统计）。
4. `accept len` 取自服务端 `Decode batch` 行的滑动统计；各行的采样条数 n 见 §3。
5. `decode` / `ms/step` 来自 `bench_ep2_streams.py --k 1`（单流、16 个互异 prompt 的第 1 个、temp 0.7）；主表中标“未测”的行未做该测量。
6. `LK_VERIFY_GEMM` 在 `d83c3d1ac2`（#10）中不被任何代码读取；该行的 conf 里虽写有该变量，实际不生效。
7. 本目录的文件中，`logs/*.log` 与 `results/*.csv` 被仓库 `.gitignore`（`*.log` 等规则）匹配，是用 `git add -f` 强制加入的。
