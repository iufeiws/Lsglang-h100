# EP2 再次 SIGSEGV：获取实际故障线程

基线 `9939591` 已合入 eager decode/verify 分流修正 `87192815`。本次报告的 Python 叶帧是 `torch.ops.sgl_kernel.sparse_decode_fwd`，经 `flash_mla.py:303` 调用。它说明 Python 当时进入了 FlashMLA，不能独立证明收到 SIGSEGV 的线程也在 FlashMLA。LK 的后台原生线程可以在 Python 执行其他算子时出错。

本轮代码检查：attention 的本地头数来源是 `attn_tp_size`，未发现直接使用 `moe_tp_size` 的证据；H100 也不走只对 Blackwell 开启的 `_fast_flashmla_sched_shape` 手工 scheduler 分支。没有足够依据继续修改头数、跳过预热或替换 attention backend。

## 采集命令

新增 `test/manual/capture_native_crash.py`，用 GDB 启动原命令、跟踪 fork/exec 的 scheduler 子进程，在 SIGSEGV 交给应用 handler 之前记录：实际 PID/LWP、`siginfo`、PC、原生栈、寄存器、共享库、地址映射和各线程栈。某项读取失败不会中断其余采集。不修改推理实现、原始参数或 `CUDA_LAUNCH_BLOCKING`，仅额外开启 `LK_NATIVE_TRACE=1`。

目标机器需有 GDB（含 Python 支持）。从仓库目录运行，`--` 后保留发生此次失败的原始命令；Python console script 用相应环境的 Python 显式启动，shell script 用 `/bin/bash` 显式启动。例如，若实际使用仓库保存的 launcher 和当前配置：

```bash
python test/manual/capture_native_crash.py --output ./ep2-native-report -- \
  /bin/bash deploy/h100-tp4-dsv41-20261023/scripts/serve_dsv41_tp4_h100.sh --foreground
```

这里没有传 `--restart`：不要停止或替换其他服务。若使用不同 launcher，替换成实际使用的路径和参数，不能为了调试换一套配置。输出目录必须是新的，避免覆盖证据。保存 `ep2-native-report/native-crash.log` 和原始完整 server log；不要只截取主进程的 scheduler-died 尾栈。

这是诊断运行。GDB 在捕获后退出并结束它跟踪的进程，不保留半初始化服务；调试也会改变时序，不能用于速度评测。如果初始化阶段使用可恢复的 SIGSEGV 探测，首次捕获可能不是最终崩溃，必须结合原生栈判断。

若不确定 GDB 是否能跟踪本机的子进程和 worker，先运行无 GPU 的 smoke test：

```bash
python -m pytest -q -c /dev/null test/manual/test_native_crash_capture.py
```

它创建 fork 子进程，并让其 pthread worker 故意访问空指针，验证记录的是 worker LWP 而非主线程。当前开发环境没有 GDB，因此实际原生采集测试跳过；参数错误路径测试通过。这不是 EP2 修复提交，也没有声称完成实机复现。

GDB 行为依据：[fork/exec 跟踪](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Forks.html)、[catchpoints](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Set-Catchpoints.html)。
