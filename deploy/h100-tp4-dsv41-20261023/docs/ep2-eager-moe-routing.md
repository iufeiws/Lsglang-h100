# EP2 启动：修正 eager decode/verify 被误分到 CPU prefill

基线：部署分支 `1f90a0f`，已经包含 `30e2bc11f7` 和 `61e2ff4326`。后者应用后仍然发生原生 SIGSEGV；本次不把前两次修改当作充分修复，也不把 Python 叶帧当作原生故障指令。

## 代码确认的入口缺陷

`DeepseekV4ForCausalLM.prepare_before_cuda_graph_capture` 首先请求 `ForwardMode.DECODE` 的 dummy forward，再请求默认模式。后续检查发现，speculative 模式下 `_dummy_run` 会把显式 DECODE 再次覆盖成 TARGET_VERIFY，所以此前“首轮实际执行 DECODE”的判断不成立；修正见 [dummy-forward-mode-fix.md](dummy-forward-mode-fix.md)。两轮都在 `nullcontext()` 内运行，刻意允许首次 JIT/模块加载，不在 CUDA capture 或 `model_capture_mode()` 内。

旧版 `FusedMoE.run_moe_core` 只用 capture 标志选择 `_cpu_decode`；在 eager 情况下会落入 prefill 分支。这解释了为什么 decode 预热的栈上出现 `_cpu_prefill`，不能据此反推当前工作负载是真正的 prefill。

此外，`BaseRunner._dummy_run` 直接调用 `mr.model.forward`，绕过 `ModelRunner.forward` 中的 `set_forward_batch`。仅修改分流判断不够：预热也必须发布自己的 batch，否则模式和 verify 行布局仍不可见。

本次修改：

- dummy 执行作用域发布当前 batch，离开时恢复，包括异常退出。
- 非常驻 MoE 对 DECODE、TARGET_VERIFY、IDLE 使用 decode ABI，包括 eager 执行；原有 capture 回退保留。
- verify 从同一 batch 获取每请求行数。6 行 verify 使用宽度 6，恰好有 6 行的普通 decode 仍使用宽度 1。
- `LK_NATIVE_TRACE=1` 增加 `submit cpu_decode` 日志，记录层、行数、每请求行数和请求数。它表示提交，不能解释为原生执行已成功完成。

不通过强制 capture mode 或跳过 MoE 来绕开预热；预热仍执行完整计算。

## 常驻层并不限制 prefill 的层数

`LK_GPU_RESIDENT_MOE_LAYERS=0-19` 决定这些层的专家权重常驻 GPU，不会让模型只执行前 20 层。主模型的 forward 仍循环 `start_layer…end_layer`。

保存的配置 `dsv41_tp4_h100.effective.20261023_235214.conf` 设置 `LK_GPU_PREFILL_MIN_TOKENS=0`。在本仓库，这个值表示**关闭 GPU prefill 档位**，不是“所有 token 都走 GPU”。因此真实 EXTEND/MIXED 等工作负载在非常驻层依然可能进入 CPU prefill。本修复不修改该配置。

## 验证与尚未证明的部分

CPU 回归共 **98 项通过**：39 项新增路由/作用域测试，加原有 59 项 EP 编号及原生指针边界测试。新增测试执行生产分流方法、真实 ForwardMode 和 batch context；计算调用由替身代替。以部署基线的生产分流方法复测，eager DECODE 和 TARGET_VERIFY 均选择 `cpu_prefill`；修改后均选择 `decode`。dummy 作用域还验证嵌套 batch 的恢复和异常退出。

这些结果证明入口和 batch 传递的修正，**不能证明 LK 二进制内 SIGSEGV 已消失**。本环境不能运行 H100/NUMA 原生工作负载；原生故障指令和 EP2 特有触发条件仍未定位。检查保存的 LK 2.7.1 AMX 二进制时，MXFP4 gather/归并存在 `-1` 跳过分支，因此没有依据把混合路由中的 `-1` 直接定为根因。

部署验证需同时覆盖：

1. 原 EP2 配置加 `LK_NATIVE_TRACE=1` 启动；预热应出现 `submit cpu_decode`，DECODE 宽度 1、固定 verify 宽度 6，并到达 `[attn-prewarm] ... done`。
2. 完成初始化后实际发送请求，覆盖真实 prefill 和连续 decode/verify，检查有限输出和进程存活。仅启动成功不算 CPU prefill 问题已修复。
3. 如果真正的 CPU prefill 仍崩，保留最后的 `enter cpu_prefill`（含层、形状和本地路由数）及原生 core/线程栈；仓库已有 `test/manual/probe_lk_ep_native.py` 可独立测试原生 prefill，但不替代完整模型验证。

① FP8 表和③ PRMT 性能修改仍在独立分支，不包含于此次修复。
