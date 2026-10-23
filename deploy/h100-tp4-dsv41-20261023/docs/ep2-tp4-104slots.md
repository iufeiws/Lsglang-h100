# TP4 + EP2（`--ep-size 2`）启动结果：定容成功、初始化段错误

- 条件：`--tensor-parallel-size 4 --ep-size 2`、`LK_POOL_SLOTS=104`、`resident=0-19`，其余与 192 槽版一致
  （`expandable_segments:True`、`LK_VERIFY_RESIDENT=1`、`CUDA_LAUNCH_BLOCKING=""`、dspark 开、HiCache 开）
- 日志：`/mnt/SSD01/models/.llm-env/Lsglang/logs/serve_dsv41_tp4_h100_20261023_235214.log`

## 分片与内存（启动日志原文）

| 项 | EP1（192 槽，对照） | EP2（104 槽） |
|---|---|---|
| rank 标号 | `TP0..TP3` | `TP0 EP0 / TP1 EP0 / TP2 EP1 / TP3 EP1` |
| 每层每 rank 专家数 | `Initialized lk_moe with 384 experts for layer N` | `Initialized lk_moe with 192 experts for layer N` |
| `[pool] slab` | `16.81 GiB (20 layers x 860.62 MiB)` | **`18.21 GiB (20 layers x 932.34 MiB)`** |
| 单槽（由 slab 反算） | 4.4824 MiB | **8.9648 MiB（= 2×）** |
| `DSV4 memory calculation` | `available_bytes=4.43 GB` | **`available_bytes=7.19 GB`** |
| `DSV4 pool sizes` | `full=1984000` | **`full=3756544`** |

## 启动结果

```
!!!!!!! Segfault encountered !!!!!!!
  File "../../../libstdc++-v3/src/c++11/thread.cc", line 104, in execute_native_thread_routine
  File "<unknown>", line 0, in 0xffffffffffffffff
[2026-10-24 00:10:00] kill_process_tree called: parent_pid=59342, include_parent=False, pid=59342
RuntimeError: Rank 1 scheduler died during initialization (exit code: -11)
```

- 崩溃点：`[attn-prewarm] rank=1/4 buffers ok`（00:03:29）之后的初始化阶段（00:09–00:10）
- `-11` = SIGSEGV；栈只有 `std::thread` 的 `execute_native_thread_routine`，无可查符号
- 另有一条非致命告警（EP1 同样出现）：`Setup Custom allreduce failed with invalid literal for int() … 'GPU-7561b28f-…'`（`CUDA_VISIBLE_DEVICES` 为 UUID 导致回退）

## 未做

- EP2 的 decode / prefill 测量（进程未起来）
- EP2 + `LK_VERIFY_GEMM=0`、EP2 + 无池 等变体
