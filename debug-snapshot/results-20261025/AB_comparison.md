# 三臂对照 (同一模型 /mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017, TP4 EP2, DSPARK block5 + rejection sampling, 4×H100 GPU4-7)

| 指标 | #10 (10-24 23:35) | arm A (10-25 05:22) | arm B (10-25 06:56) |
|---|---|---|---|
| 代码 | e378da4b 之前 (main+fix2) | 分支 tip 7f664cf6a6 | 同 arm A |
| LK_THREAD_BINDING | CPU_CORE | NUMA_NODE | NUMA_NODE |
| LK_POOL_DET_SPLIT | 未导出(关) | 关 | 关 |
| LK_VERIFY_UVA | 未导出(关), 死代码 | 关(死代码) | **1, 实际生效** |
| _pool_verify_rows_mult | 1 | 1 | **6** |
| 单流 decode tok/s | 31.93 | **51.52** | 16.76 |
| ms/token | 31.32 | **19.41** | 59.22 |
| ms/step (服务端日志, 40步/行) | 窗口内无行 | **50.0** | 112.5 |
| accept len (bench 窗口) | ~2.4 | 2.42 | 2.25 |
| accept len (全程中位) / rate | 4.03 / 0.60 | 3.48 / 0.49 | 2.56 / 0.32 |
| prefill 328new+5888cached tok/s | 250.9 (n=70) | 305.8 (n=38) | 121.5 (n=38) |
| prefill 6216 全新 tok/s | 1781 | 2456 | 1036 |
| 复述贪心 n=20 | 19/20 | 19/20 | **20/20** |
| 复述采样 n=20 | 19/20 | 19/20 | **20/20** |
| 工具调用 | 10/10 | 1/1 | 1/1 |
| pool link / hit / sk | 45.8GB/s / 97.1% / 16671 | 393GB/s / 93.0% / 9260 | 97.0GB/s / 92.0% / 0 |
| 启动耗时 | 1235 s | 1270 s | 1160 s |
| 原始数据 | logs/serve_dsv41_tp4_h100_20261024_233544.log | .../serve_..._20261025_052223.log | .../serve_..._20261025_065642.log |
