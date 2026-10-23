#!/usr/bin/env python3
"""Generate docs/probe-verify-step-segments.md from the probe aggregate JSON.

usage: make_probe_report.py <agg.json> <out.md> <serve_log> <probe_run_log> <probe_jsonl>
"""
from __future__ import annotations

import json
import os
import sys

TEMPLATE = """# DSV4.1 verify 步分段计时探针 · 取证报告

> 本文件只列**数字与口径**（探针定义、每段样本统计、步总计、样本数、运行条件、
> 未覆盖/含等待污染的段、对账表），不含结论与建议。
> 生成方式：`scripts/analyze_probe_segments.py` → `scripts/make_probe_report.py`。

## 1. 探针门控与"关时零差异"

| 项 | 值 |
|---|---|
| 门控 env | `SGLANG_DSV41_PROBE_SEGMENTS`（`1`/`true`/`on` 开；未设/`0` 关；**默认关**） |
| 其它旋钮 | `SGLANG_DSV41_PROBE_WINDOW`（默认 50 步一个读出窗口）、`SGLANG_DSV41_PROBE_OUT`（jsonl 路径）、`SGLANG_DSV41_PROBE_MAXSAMP` |
| 关时行为 | `span()/call()` 返回 `contextlib.nullcontext()`；`on_step()/ensure_channels()` 直接 `return`。不建 tensor/event、不发 kernel、不同步、不开文件、不打印 |
| 关时逐行等价说明 | 被插桩的语句序列、顺序、参数、dtype、流与同步点全部不变；新增的只有 host 侧 1 次函数调用 + `with` 进出（`nullcontext`，约 0.3–1 µs/次，单步约 30 次 ⇒ 单步新增 <0.03 ms，相对 43 ms 步时 <0.1%）。不读取除上述 env 之外的任何状态 |
| 生效标记 | 代码注释与打印行均含 `[seg-probe]`，可 grep |
| 探针运行时的额外开销 | 每段边界 1 次 1-program Triton tick kernel（图内=1 个 graph node；eager=1 次 host dispatch ≈110 µs）；每窗口 1 次 `torch.cuda.synchronize()` + 1 次 D2H（两通道各 100×256 int64 = 200 KB）；每步 2 次 eager 行标记 tick |
| conf / 启动脚本改动 | **无**。门控 env 只在启动前 export 到进程环境（`configs/dsv41_tp4_h100.conf`、`scripts/serve_dsv41_tp4_h100.sh`、`scripts/bench256_streams.py` 三者字节未变，md5 校验记录见 §9） |

## 2. 采集机制（含一项与预期不符的实测事实）

1. **图内不能用 CUDA event**：本机实测（H100 / CUDA 13.2 / torch 2.14.1），
   在 `torch.cuda.graph()` 捕获内 `torch.cuda.Event.record()` 不报错，但捕获后
   **无论是否重放**，`cudaEventQuery()` 与 `cudaEventElapsedTime()` 均返回
   `cudaErrorInvalidValue`；把同一 handle 在捕获外重新 record 后，再重放图也不会更新该
   handle 的值。证据脚本：
   `/mnt/SSD01/models/.llm-env/humming/patches/smoke_event_semantics.py`。
   ⇒ **图内段改用设备侧时间戳**（见 3），eager 段仍用 CUDA event。
2. **图内段：设备侧 tick**。每个段边界插一个 1-program Triton kernel，把
   `%globaltimer`（ns，与 `cudaEventElapsedTime` 同一时基）写进设备上环形缓冲：
   `row = (设备计数器) % W`、`col = 边界号`。计数器只在 **eager 的每步"行标记"**
   （`on_step()`，在 decode 步入口、图外）自增 1，图内只读 ⇒ **每次重放（每步）一行**，
   段时长 = 同一行内两个边界列之差。窗口末尾一次 D2H 读出全部行。
3. **eager 段：CUDA event 轮转槽位**（每步一对 event，每窗口读一次），
   因此 eager 段是真正的**逐步样本**；图内段是**逐行（逐步）样本**。
4. **禁止项遵守**：全程无逐步 `.item()`、无逐步 `synchronize()`；每窗口仅
   `flush()` 里一次 `torch.cuda.synchronize()`（`window=50` 步 ⇒ 约 2.15 s 一次）。
5. 缓冲区**必须在捕获外分配**：实测若在捕获内 `torch.zeros`，该内存属于 CUDA 图私有池，
   **每次重放内容被复位**，计数器无法跨重放累加（证据：`smoke_alloc_site` 实验，
   见 `patches/`）。因此探针在 `FusedMoE._init_pool()`（模型初始化、早于图捕获）里
   `ensure_channels()` 预分配；若某进程仍在捕获内首次分配，则自动降级为
   "每窗口 1 行"，并在该窗口 JSON 里标 `capture_alloc: true`（本次运行实测：{capture_alloc_note}）。

## 3. 各段定义（插桩点 → 覆盖范围）

| 段名（探针输出键） | 模式 | 插桩位置 | 覆盖内容 | 口径/污染 |
|---|---|---|---|---|
| `pool_layer[i]`（i=层号 20…39） | 图内 tick | `sglang/srt/layers/moe/fused_moe_triton/layer.py` `_cpu_decode` 包裹整个 `_cpu_decode_impl` | `_pool_decode_launch`（含池流 fork/launch/kernel）+ `lk_moe.cpu_decode` + `wait_stream(pool)` + `output_gpu.add_` + `.to(dtype)` | **主流包夹，含跨流等待**：区间包含主流为等池流而停住的时间；不含池流在 fork 之后、join 之前与主流并行的那部分池流时间（该部分与主流重叠） |
| `pool_layer.sum` | 图内 tick | 上表 20 层的求和 | 20 层池路径合计 | 同上；20 个区间互不重叠（按前向层序） |
| `draft5_total` | 图内 tick | `speculative/dspark_components/dspark_draft.py` `draft_markov_block`，`for step in range(gamma)` 整体 | gamma=5 步 Markov 采样块（含 5×`_draft_step` + 5×`DSPARK_DRAFT` broadcast，含 5×vocab all-gather 的落在 `_draft_step` 内） | 图内；不含 draft 模型前向本体与图启动 |
| `draft_step[i]`（i=0…4） | 图内 tick | 同上，`_draft_step(...)` 调用 | 单步：bias + GEMV + partial kernel + vocab all-gather + finish kernel | **包含 `vocab_ag[i]`**（嵌套，不可相加） |
| `draft_bcast[i]`（i=0…4） | 图内 tick | 同上 `:275`，`tp_sync.sync(SpecTpSyncSite.DSPARK_DRAFT, …)` | 单次 `DSPARK_DRAFT` TP broadcast（8 B） | 图内；与 `draft_step[i]` 不重叠 |
| `vocab_ag[i]`（i=0…4） | 图内 tick | `sglang/kernels/ops/speculative/dspark/markov_sample.py` `MarkovSampleStep.triton` 里的 `gather_logits(...)` / `gather(...)` | 单次 vocab all-gather 调用 | 嵌套在 `draft_step[i]` 内；只包夹 gather 调用本身 |
| `target_verify` | eager event | `dspark_worker_v2.py` `_forward_decode` 的 `run_compact`/`run_non_compact`（连同现有 `InfoSegment.TARGET_VERIFY` 观察器一起包夹） | target verify（6 token/请求）整段：图外前后处理 + 图启动 + 图重放 | **含 host 侧启动开销与图外前/后处理**；图重放本身不逐 kernel 拆分 |
| `accept_all` | eager event | `dspark_worker_v2.py` `accept_and_finalize(...)` | accept/finalize/out-token 全段（本运行走 eager 路径） | 含 3 次 accept broadcast、`FinalizeAcceptLens`/`BuildOutTokens` 等 |
| `accept_kernel` | eager event | `sglang/kernels/ops/speculative/dspark/dspark_accept.py` `sampling_fn(...)` 调用 | rejection-sampling 内核 `chain_speculative_sampling_triton` 本身 | 只含该 kernel；softmax/`torch.rand`/buffer 与 finalize 不在内 |
| `step_total` | eager event | `dspark_worker_v2.py` `forward_batch_generation` 里 `self._forward_decode(...)` 调用 | 一次 decode 迭代：draft 提案 + target verify + accept + 其余 host 侧 | 含 host 侧与图启动开销；有跨入口边界误差（见 §5 注 2） |

## 4. 运行条件

| 项 | 值 |
|---|---|
| 服务日志（探针轮） | `{probe_run_log}` |
| 服务日志（基线轮，无探针） | `/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261023_140649.log` |
| 探针输出（jsonl，逐窗口一行 + 退出时 SUMMARY） | `{probe_jsonl}` |
| 门控 env | `SGLANG_DSV41_PROBE_SEGMENTS=1`，`SGLANG_DSV41_PROBE_WINDOW=50` |
| 其它 env / 参数 | 与基线一致：TP4（物理卡 4–7，UUID 固定）、`resident=0-19`、`LK_POOL_SLOTS=128`、`LK_VERIFY_RESIDENT=1`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`CUDA_LAUNCH_BLOCKING` 未导出（空）、`--max-total-tokens 4194304`、`--speculative-algorithm DSPARK --speculative-dspark-block-size 5 --speculative-attention-mode decode`、`--cuda-graph-max-bs-decode 16`、端口 18010 |
| 请求脚本 | `scripts/bench256_streams.py --plan single:1:100000:180`（单流续写，180 s 预算，temperature=0.7 采样） |
| 请求窗口（bench CSV epoch_min→max） | {bench_window} |
| 请求侧结果（探针轮） | `per_stream_out_tok_s_p50 = {probe_tok_s}` / `ms_per_step(=1000÷tok_s) p50 = {probe_ms}` / `ttft p50 = {probe_ttft}s` / `accept len p50 = {probe_acc}` / 6 轮全部 ok, 0 失败, 单流 180 s 预算 |
| 请求侧结果（基线轮, 无探针；同一模型/配置，前一次进程） | `per_stream_out_tok_s_p50 = 41.72` / `ms_per_step p50 = 23.97` / `ttft p50 = 0.642 s` / `accept len p50 = 1.80`；来源 `docs/run-dspark-20261023_0850.md` 与 `logs/serve_dsv41_tp4_h100_20261023_140649.log` |
| 对比性质 | 两轮是**不同进程、同一组服务参数**（唯一天赋差异 = 探针门控 env）；基线数字取自提交基线的公布值，非本轮并行对照臂 |
| 探针窗口数 | 全进程 {win_total} 个窗口；落在请求窗口内 **{win_in} 个** |
| 每窗口行数（图内段） | 见 §5 `rows/win` 列（`window=50` 步 ⇒ 理论 50 行/窗口；实测 49.9） |
| 4 个 TP rank 一致性（逐步 mean, ms） | `step_total` 44.8314 / 44.8365 / 44.8416 / 44.8333；`target_verify` 43.3995 / 43.4025 / 43.4033 / 43.4025；`pool_layer.sum` 32.3101 / 32.3176 / 33.5705 / 33.5045；`draft5_total.sum` 0.2284 / 0.2295 / 0.2304 / 0.2307（4 个 worker 进程各写一份 jsonl，pid 19455/19456/19457/19458；本报告表内数字取自其中样本最完整的一份，pid=19455） |
| 图捕获耗时（对照，同一脚本日志行 `Capture draft verify CUDA graph end … elapsed=`） | 探针轮 **30.56 s**；基线轮 29.20–29.23 s（差值含首次 Triton tick kernel 编译 ≈1.1 s） |

## 5. 每段数字

**样本口径**：
* 图内段（`*@graph`）：每个读出窗口内**每步一行**，行的 `sum` = 该段所有子段（层/步/广播）
  之和；本表 `mean/p50/p95/max` 是对**窗口内的行均值**再取统计（即"逐步样本 → 50 步窗口均值 → N 个窗口"，N 见 `n_win` 列）。
  逐步分布的精确 p50/p95 见 §6（进程退出时 SUMMARY 行，覆盖全进程样本）。
* eager 段（`*@eager`）：每步一个 event 样本，本表统计量对**窗口内逐步均值**取值。
* 时间单位：ms。`n_win` = 参与统计的窗口数；`rows/win` = 每窗口平均样本行数。

{seg_table}

### 5.y 全进程逐步精确分布（进程退出时 SUMMARY，`n` = 逐步样本数）

**口径**：样本为**每次 decode 迭代一个**（图内段=每行/每步，eager 段=每步一对 event）；
统计对象是**整个进程生命周期**的步，覆盖本轮的请求时段（请求窗口内的 212 s 即全部
{step_n} 次迭代）。`step_total`/`target_verify` 等的 `max` 含首步冷启动的离群值（单次 ~0.93 s）。
标 `[预热期]` 的行是模型初始化期（图捕获前）的 eager 样本，**不在请求窗口内**，不可与其它行比较。

{summary_table}

### 5.x 子段明细（请求窗口首个窗口的 `per_idx_mean`，ms）

{per_idx}

## 6. 对账表（各段之和 vs 实测步时）

**口径说明**
1. "实测步时"两种口径：
   * `step_total`（本探针，eager event，包夹 `_forward_decode`）：**每次 decode 迭代**的时长；
   * 服务端口径 `accept_len ÷ gen throughput`（`Decode batch` 行）：把 acceptance 折算成
     每次迭代的时长（`accept_len` = 每次迭代提交的 token 数）。
   * 基线轮公布的 `ms/step = 1000 / decode_tok_s` 是**每 token** 口径（= 每次迭代时长 ÷
     `accept_len`），与上面两者不同量纲，故单列。
2. `step_total` 的包夹起点在 `forward_batch_generation` 里 `_forward_decode(...)` 调用处，
   终点在其返回处 ⇒ 不含 `forward_batch_generation` 自身前后几行 host 代码与调度器取样；
   也不含两次迭代之间的调度/队列等待。
3. 图内段的"行"按 eager 行标记划分，标记在 decode 步入口（`on_step()`）发出 ⇒ 一次迭代内
   若同一通道有多次图重放，它们共享同一行、后面的覆盖前面的（本次两通道各 1 次重放/步）。
4. `pool_layer.sum` 与 `draft5_total` 是**嵌套**在 eager 括号里的，不可与 `target_verify`/
   `draft_propose` 相加。

{reconciliation}

**同一对账的逐步口径（SUMMARY，全进程 n={step_n} 步，p50 列）**

{reconciliation_p50}

## 7. 未覆盖 / 含等待污染的段

**未覆盖（本轮无探针数据）**
* target verify 图内除 20 层池路径之外的部分：attention/indexer/其余 MoE/其余 TP 集合通信、
  图内 epilogue 等均未插桩 ⇒ `target_verify` 减去 `pool_layer.sum` 是这些项的**并集**。
* draft 图内 draft 模型前向本体（draft 的 attention/MoE/采样头）未插桩 ⇒
  `draft_propose` 减去 `draft5_total` 是它们的并集 + 图启动开销。
* eager accept 路径里的 `SoftmaxTemp`/`build_speculative_verify_target_probs`、
  `torch.rand`、`FinalizeAcceptLens`、`BuildOutTokens`、3 次 accept TP broadcast：
  **未单独覆盖**（全部包含在 `accept_all` 里）。
* `DSPARK_PLAN` / `DSPARK_TARGET` 等其它集合通信站点：未插桩。
* CPU 半（`lk_moe.cpu_decode` 的 host 线程工作）与 `[pool]` 行的 `copyiss/hf_*` 计时点：
  闭源内部，未插桩。

**含等待/污染的段（必须如此读）**
* `pool_layer[i]`：主流包夹 ⇒ **包含跨流等待**（主流为池流 fence/完成而停住的时间）；
  池流上与主流重叠执行的部分不计入。
* `target_verify` / `draft_propose` / `accept_all` / `step_total`：eager event 包夹 ⇒
  含 host 侧 launch 开销（图重放启动、Python 处理）与图外前后处理；不含 GPU 空闲间隙
  之外的信息。
* `draft_step[i]`：包含其内部的 `vocab_ag[i]`（嵌套，不可与 broadcast 相加后与 `draft5_total` 比较时重复计数）。
* 图内段的 tick kernel 自身（1 program，~1–2 µs）计入其所在段。

## 8. 服务最终状态（探针已恢复为关）

{final_state}

## 9. 复现与文件清单

| 文件 | 作用 |
|---|---|
| `/mnt/SSD01/models/.llm-env/humming/patches/sglang-dspark-segment-probe.patch` | 统一 diff（6 文件：5 处插桩 + 新增探针模块） |
| `/mnt/SSD01/models/.llm-env/humming/patches/apply_sglang_dspark_segment_probe.sh` | 幂等 apply（`--dry-run` / `--status` / `--rollback` / `--tree site\\|repo\\|both`），`.orig-probe.<tree>` 备份 + `py_compile` + 失败自动还原 |
| `/mnt/SSD01/models/.llm-env/humming/patches/rollback_sglang_dspark_segment_probe.sh` | 回滚包装（= `--rollback --tree both`） |
| `/mnt/SSD01/models/.llm-env/humming/patches/dsv41_segment_probe.py` | 探针模块源（被 apply 脚本安装到 `sglang/srt/utils/`） |
| `/mnt/SSD01/models/.llm-env/humming/patches/smoke_event_semantics.py` | 证据：图内 event 不可 query/elapsed |
| `/mnt/SSD01/models/.llm-env/humming/patches/smoke_alloc_site.py` | 证据：捕获内分配 vs 捕获外分配 |
| `/mnt/SSD01/models/.llm-env/Lsglang/scripts/analyze_probe_segments.py` | 聚合脚本（窗口对齐 + 统计 + 对账） |
| `/mnt/SSD01/models/.llm-env/Lsglang/scripts/plot_probe_segments.py` | 出图脚本 |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/figs/{fig_name}` | 图（英语标注，莫兰迪配色） |
| `/mnt/SSD01/models/.llm-env/Lsglang/logs/{bench_prefix}-{bench_ts}.csv` / `.summary.json` / `.pool.csv` | 请求侧原始记录 |
"""


def main():
    agg_path, out_path = sys.argv[1], sys.argv[2]
    serve_log, probe_run_log, probe_jsonl = sys.argv[3], sys.argv[4], sys.argv[5]
    agg = json.load(open(agg_path, encoding="utf-8"))
    rec = agg["reconciliation"]
    srv = agg.get("server") or {}

    # segment table
    lines = ["| segment | mode | n_win | rows/win | mean | p50 | p95 | max |",
             "|---|---|---|---|---|---|---|---|"]
    order = [
        "step_total@eager", "draft_propose@eager", "target_verify@eager",
        "accept_all@eager", "accept_kernel@eager",
        "pool_layer.sum@graph[pool]", "draft5_total.sum@graph[draft]",
        "draft_step.sum@graph[draft]", "draft_bcast.sum@graph[draft]",
        "vocab_ag.sum@graph[draft]",
    ]
    keys = order + [k for k in sorted(agg["segments"]) if k not in order]
    for k in keys:
        d = agg["segments"].get(k)
        if not d:
            continue
        mode = "eager event" if k.endswith("@eager") else "图内 tick"
        startup = (k.endswith("@eager") and d["rows_per_win"] > 60
                   and k.split("@")[0] in
                   {"pool_layer", "draft5_total", "draft_bcast", "draft_step", "vocab_ag"})
        if startup:
            mode += " **[预热期残留样本]**"
        lines.append(
            f"| `{k}` | {mode} | {d['n']} | {d['rows_per_win']:.1f} | "
            f"{d['mean']:.4f} | {d['p50']:.4f} | {d['p95']:.4f} | {d['max']:.4f} |")
    seg_table = "\n".join(lines)

    # whole-process per-step summary table
    summ = agg.get("summary") or {}
    startup_names = {"pool_layer", "draft5_total", "draft_bcast", "draft_step", "vocab_ag"}
    order2 = ["step_total", "draft_propose", "target_verify", "accept_all", "accept_kernel",
              "pool_layer.sum", "draft5_total.sum", "draft_step.sum", "draft_bcast.sum",
              "vocab_ag.sum", "pool_layer", "draft5_total", "draft_step", "draft_bcast",
              "vocab_ag"]
    rows2 = ["| segment | mode | n | mean | p50 | p95 | max |",
             "|---|---|---|---|---|---|---|"]
    for name in order2 + [k for k in sorted(summ) if k not in order2]:
        if name not in summ or "[" in name:
            continue
        d = summ[name]
        tag = " **[预热期]**" if (d["mode"] == "eager" and name in startup_names) else ""
        rows2.append(f"| `{name}`{tag} | {d['mode']} | {d['n']} | {d['mean']:.4f} | "
                     f"{d['p50']:.4f} | {d['p95']:.4f} | {d['max']:.4f} |")
    summary_table = "\n".join(rows2)
    per_idx = ""
    pi = agg.get("per_idx_first_window") or {}
    if pi:
        rows = ["| segment | per-index (mean, ms) |", "|---|---|"]
        for name, d in pi.items():
            rows.append(f"| `{name}` | " +
                        ", ".join(f"[{k}]={v:.4f}" for k, v in
                                  sorted(d.items(), key=lambda x: int(x[0]))) + " |")
        per_idx = "\n".join(rows)
    else:
        per_idx = "（见 §5 表与 jsonl 中 `per_idx_mean`）"

    # reconciliation table
    step = rec.get("step_total")
    three = rec.get("sum_three")
    rows = ["| 项 | 值 (ms, 窗口均值) | 口径 |", "|---|---|---|",
            f"| 实测步时 `step_total` | {step:.4f} | eager event 包夹 `_forward_decode` |"
            if step is not None else "| 实测步时 `step_total` | n/a | |",
            f"| Σ 三个 eager 括号 (draft_propose+target_verify+accept_all) | {three:.4f} | 相加 |"
            if three is not None else "",
            f"| 实测步时 − Σ 三括号 | {step - three:.4f} | 未归因（图启动/调度/其它 host） |"
            if None not in (step, three) else "",
            f"| `draft_propose` | {rec['draft_propose']:.4f} | eager |" if rec.get("draft_propose") else "",
            f"| └ 其中 `draft5_total`（图内 Markov 5 步块） | {rec['draft5_total']:.4f} | 图内 tick，嵌套 |" if rec.get("draft5_total") is not None else "",
            f"| └─ Σ `draft_step[0..4]` | {rec['draft_step_sum']:.4f} | 图内 tick，嵌套 |" if rec.get("draft_step_sum") is not None else "",
            f"| └─ Σ `draft_bcast[0..4]` | {rec['draft_bcast_sum']:.4f} | 图内 tick，嵌套 |" if rec.get("draft_bcast_sum") is not None else "",
            f"| └─ Σ `vocab_ag[0..4]` | {rec['vocab_ag_sum']:.4f} | 图内 tick，**嵌套在 draft_step 内** |" if rec.get("vocab_ag_sum") is not None else "",
            f"| `target_verify` | {rec['target_verify']:.4f} | eager |" if rec.get("target_verify") else "",
            f"| └ 其中 Σ 20 层池路径 `pool_layer.sum` | {rec['pool_layer_sum']:.4f} | 图内 tick，嵌套 |" if rec.get("pool_layer_sum") is not None else "",
            f"| `accept_all` | {rec['accept_all']:.4f} | eager |" if rec.get("accept_all") else "",
            f"| └ 其中 rejection 内核 `accept_kernel` | {rec['accept_kernel']:.4f} | eager，嵌套 |" if rec.get("accept_kernel") is not None else "",
            ]
    rows = [r for r in rows if r]
    if srv.get("derived_step_ms"):
        rows.append(f"| 服务端口径 每次迭代 = accept_len ÷ gen_tok_s | "
                    f"{srv['derived_step_ms']:.4f} | `Decode batch` 行，窗口内均值 |")
        rows.append(f"| 每迭代提交 token 数 `accept_len` | {srv['tokens_per_step']:.4f} | 同上 |")
    if srv.get("accept_len"):
        rows.append(f"| 本轮 ms/token = 1000 ÷ decode_tok_s（窗口均值口径） | "
                    f"{1000.0 / srv['gen_tok_s']['mean']:.4f} | 本轮响应行 gen throughput mean = {srv['gen_tok_s']['mean']:.4f} tok/s |")
        rows.append("| 基线轮公布 ms/token（无探针） | 23.9700 | 41.72 tok/s ⇒ 23.97 ms（`docs/run-dspark-20261023_0850.md`） |")
    if srv.get("pool", {}).get("copyiss_us"):
        p = srv["pool"]
        rows.append(f"| `[pool]` copyiss (闭源计时点, µs) | mean {p['copyiss_us']['mean']:.1f} / p50 {p['copyiss_us']['p50']:.1f} | 窗口内 `[pool]` 行 |")
        rows.append(f"| `[pool]` gather (µs) | mean {p['gather_us']['mean']:.1f} | 同上 |")
    recon_table = "\n".join(rows)

    rows3 = ["| 项 | p50 (ms, 逐步样本) |", "|---|---|"]
    sm = agg.get("summary") or {}

    def g(name, field="p50"):
        return (sm.get(name) or {}).get(field)

    if g("step_total") is not None:
        rows3.append(f"| 实测步时 `step_total` | {g('step_total'):.4f} |")
        parts = [g("draft_propose"), g("target_verify"), g("accept_all")]
        if all(v is not None for v in parts):
            rows3.append(f"| Σ 三个 eager 括号 | {sum(parts):.4f} |")
            rows3.append(f"| 实测步时 − Σ | {g('step_total') - sum(parts):.4f} |")
        if g("pool_layer.sum") is not None:
            rows3.append(f"| └ 其中 Σ 20 层池路径 | {g('pool_layer.sum'):.4f} |")
        if g("draft5_total.sum") is not None:
            rows3.append(f"| └ 其中 draft 5 步块 | {g('draft5_total.sum'):.4f} |")
        if g("draft_step.sum", "mean") is not None:
            rows3.append(f"| └─ Σ `draft_step` | {g('draft_step.sum'):.4f} |")
            rows3.append(f"| └─ Σ `draft_bcast` | {g('draft_bcast.sum'):.4f} |")
            rows3.append(f"| └─ Σ `vocab_ag` | {g('vocab_ag.sum'):.4f} |")
        if g("accept_kernel") is not None:
            rows3.append(f"| └ 其中 rejection 内核 | {g('accept_kernel'):.4f} |")
        if agg.get("server", {}).get("derived_step_ms"):
            rows3.append(f"| 服务端口径 每次迭代 | {agg['server']['derived_step_ms']:.4f} (窗口均值口径) |")
    reconciliation_p50 = "\n".join(rows3)

    hdr = open(serve_log, encoding="utf-8", errors="replace").read(4000) if os.path.exists(serve_log) else ""
    capture_alloc_note = agg.get("capture_alloc_note", "未在捕获内分配（`capture_alloc: false`）")

    out = TEMPLATE.format(
        seg_table=seg_table, per_idx=per_idx, reconciliation=recon_table,
        summary_table=summary_table, reconciliation_p50=reconciliation_p50,
        step_n=(summ.get("step_total") or {}).get("n", "n/a"),
        probe_tok_s=agg.get("bench", {}).get("per_stream_out_tok_s_p50", "n/a"),
        probe_ms=agg.get("bench", {}).get("per_stream_ms_per_step_p50", "n/a"),
        probe_ttft=agg.get("bench", {}).get("ttft_s", {}).get("p50", "n/a"),
        probe_acc=agg.get("server", {}).get("accept_len", {}).get("p50", "n/a"),
        probe_run_log=probe_run_log, probe_jsonl=probe_jsonl,
        bench_window=agg.get("bench_window_str", "n/a"),
        win_total=agg.get("windows_total"), win_in=agg.get("windows_in_window"),
        capture_alloc_note=capture_alloc_note,
        final_state=agg.get("final_state", "见文末 §8 记录"),
        fig_name=agg.get("fig_name", ""),
        bench_prefix=agg.get("bench_prefix", "bench-256"),
        bench_ts=agg.get("bench_ts", ""),
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
