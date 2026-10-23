# EP2 初始化 SIGSEGV：LK host worker 拓扑修正

基线：`67a61e8719`。这是一份后续候选修复；没有 H100 启动通过的结果，不能把 CPU 单测说成段错误已消除。

## 已确认的问题和修改

`FusedMoE._get_processes_info()` 在 EP>1 时只返回 EP size/rank。TP4+EP2 有四个 GPU worker，但 LK 收到的 `(num_processes, process_id)` 是 `(2,0), (2,0), (2,1), (2,1)`。LK 的这两个字段用于 host worker / NUMA 资源划分；专家范围由 `expert_num` 和本地权重定义。现在使用 EP×MoE-TP 的宽度及组合 rank，得到 `(4,0)…(4,3)`。纯 TP4 和纯 EP4 保持原值。

`_cpu_prefill` 另外处理两项原生边界条件：

- D2H 完成后将输入、路由 ID 和权重整理成连续内存；原生接口只有裸指针，没有 stride。输出也明确分配为连续内存。
- 整批路由在当前 EP rank 都为 `-1` 时直接返回零贡献，不向线程池提交空专家任务。不重新归一化路由权重。

没有把 `-1` 替换成假专家。检查仓库保存的 LK 2.7.1 AMX wheel 后，MXFP4 的 gather/归并函数均可见 `-1` 分支；仅凭路由计数阶段不跳过负数，不能推出整个原生实现不支持 `-1`。上一轮全局→本地 ID 修复仍保留。

## 验证范围

`test_lk_native_boundary.py` 执行生产方法，检查 TP/EP worker 唯一性、空批次、全外组路由、混合路由、非连续输入、FP32 输出到 BF16 的原有转换、trace 开关和输入不被修改。裸指针测试用 ctypes 读取实际存储；原生计算本身仍由替身完成。

与 `test_lk_ep_dispatch.py` 合计 **59 项 CPU 测试通过**。这里没有 GPU；原生 wheel 的独立尝试在本环境缺少 `/proc/cpuinfo` / 正常 NUMA 环境时，构造阶段即失败，因此不计为目标机器复现，也不计为原生验证通过。

## 目标机器最短验证

先按原命令重启 EP2，临时加 `LK_NATIVE_TRACE=1`。每次 CPU prefill 会打印 `enter/exit`、层名、worker 身份、输入形状及本地路由数量；空任务打印 `skip empty`。仍崩溃时，最后一条 `enter` 可定位到具体 LK 层和输入规模。验证完成后去掉该变量。

另附无需加载模型的原生测试：

```bash
torchrun --standalone --nproc-per-node=4 test/manual/probe_lk_ep_native.py --ep-size 2
```

它使用安装的 LK wheel、真实 H=5120 / EP2 I=1152，但缩小到每 rank 12 个专家，测试 M=1/6/16/96 的本地、混合、全外组路由。混合输出与无负数单路由原生调用之和比较，并检查有限值。`--legacy-worker-ids` 可复测旧 worker 编号，`--pool-slots 2` 可额外覆盖池初始化。它刻意直接调用原生空任务，以便判断问题发生在什么条件下；不覆盖 CUDA graph decode 或完整模型。

① H100 block-FP8 表和③ PRMT 优化保留在独立工作分支，未混入本修复。
