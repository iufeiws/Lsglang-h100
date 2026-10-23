# 修正 speculative dummy 的模式和 token 宽度

基线为部署合并提交 `9939591dc983ffbe214e3ef323e7fd37a39db0de`，已含 eager MoE 路由修复。此次仅修正从源码和 CPU 执行可确认的两处 batch 构造错误。

## 已复现的问题

1. `BaseRunner._dummy_run` 先接受显式 DECODE，随后 speculative 分支无条件改为 TARGET_VERIFY。配置宽度 6、batch size 2 时，即使调用者明确请求 DECODE，仍构造 12 个 token。使用只够 2 行的合法 decode buffer 可以立即复现容量断言失败。
2. `create_dummy_verify_input` 的 DFLASH/DSpark 分支忽略传入的每请求宽度，直接使用全局 `speculative_num_draft_tokens`。DSpark draft 在 sample-from-anchor 配置下可以比 target 少一行。测试中 runner 宽度 5、batch size 2，实际有 10 行，但 spec metadata 仍为宽度 6。`init_forward_metadata_dspark_draft_block` 会据此计算 12 行的 extend 元数据。正式 `DecodeCudaGraphRunner.get_spec_info` 已使用 `captured_req_width`，dummy 应采用同一规则。

## 修改后的行为

- 显式模式优先；只有未指定模式时，才根据 speculative 配置默认选择 TARGET_VERIFY。
- 只有实际 TARGET_VERIFY 才计算 verify 宽度和创建 verify 元数据。普通 DECODE 每请求一行，spec_info 为 None。
- DFLASH/DSpark 的 dummy verify 元数据使用调用者传入的实际宽度，与输入 buffer、MoE 行布局及正式 graph capture 一致。
- 默认 speculative verify、EAGLE target 的 dummy hidden states、PD-prefill target 的普通 decode/extend 保持相应语义。

## 验证

`test/manual/test_dummy_forward_mode.py` 执行完整的生产 `_dummy_run` 和 `create_dummy_verify_input`，使用 CPU tensors，仅替换模型、collective、attention 初始化边界及 SpecInput 数据容器。覆盖显式 decode、默认/显式 verify、DSpark draft 的 5/6 行宽度、DP token 数、PD prefill、buffer 容量以及异常后 batch context 恢复。与原有路由、EP 编号、原生参数边界回归合计 **131 项通过**。

在未修改的基线上复测，分别得到 `num_tokens=12` 对 `input_ids=2` 的断言，以及 `rows=10, batch_size=2, metadata_width=6` 的不一致。

这些结果证明两处构造错误及修正，不证明 H100 EP2 原生 SIGSEGV 已消失。最新 FlashMLA Python 叶帧不能确定出错的原生线程；本环境没有运行 H100/NUMA 工作负载。尤其不能把仅在 draft/target 宽度不同配置下触发的问题，宣称为所有 EP2 崩溃的统一根因。本补丁不要求部署者先进行长时间诊断，也不修改精度、专家放置或常驻层配置。
