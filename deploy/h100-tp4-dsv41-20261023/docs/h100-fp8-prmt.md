> **§① 已在后续提交中撤下。** 独立校验发现：`h100_block_fp8_small_m.json` 的 tile 均为
> `SWAP_AB: true`，而 SWAP_AB 内核路径在同一 kernel/形状/输入下误差为 **6.55e-4（约 5.8× 测试容差）**，
> `SWAP_AB: false` 为 9.5e-7；该内核在父提交等于本提交（属既有缺陷），但本表会让 H100 `M<=96`
> 首次进入该路径，且启动日志确认 7/7 预热形状命中该表、含专家 gate_up `(1152,5120)` 与 down `(5120,576)`。
> 因此 §① 的 profile 表、选择器与相应测试已移除，**§③ PRMT 解包保留**（已独立验证为逐位等价）。
> 本节以下内容仅作为当时的设计与离线数据留档。

# H100 block-FP8 小 M 配置与 MXFP4 PRMT 解包

本分支基于已验证启动的 `6c2b980aa8`，仅移植 `3700a46b14` 的性能增量；不包含 `61e2ff4326`、`8719281598` 或 `3eaf4bed17`。原性能分支的 `3700a46b14` 以 `61e2ff4326` 为直接父提交，因此合并原分支会带入该修复；这里通过单提交移植将两者分离。indexer 保持现有实现。

相对 `6c2b980aa8`，专家编号映射、LK worker/NUMA 配置、CPU prefill、eager MoE 路由、dummy batch 构造及 reject sampling 文件均未改动。重新执行性能回归 40 项及 EP 分流回归 31 项，共 71 项通过；这不是目标机器上的性能或启动测试。

## ① H100 block-FP8

新增 `kernels/ops/gemm/configs/h100_block_fp8_small_m.json` 和纯 Python 选择器，接入已有 Hopper swap-AB / split-K 内核。限制为 H100、block32、M=1…96 和表内 V4.1 投影形状；已有设备/形状调优文件优先，其他情形沿用原配置。覆盖 EP1/EP2 相关宽度，不把小 M 的表外推到大 prefill。

这是有边界的初始配置表，**不是 H100 实测 autotune 结果**。没有证据保证复现旧 fork 的 14.101→5.435；需要目标机器运行附带 A/B benchmark。

## ③ PRMT 解包

在本分支已经使用的 `kernels/ops/moe/lk_verify.py` 中，将 MXFP4 verify GEMM 的逐 nibble 浮点解码换成 `prmt.b32` 查表：一个 packed byte 同时生成两个精确 FP16 E2M1 值；随后仍按原 E8M0 scale 乘法和 BF16 dot 运算。每对 K 元素只加载一次 packed byte 和 scale。

保留参考 loader，通过 `launch(..., use_prmt=False)` 做 A/B。默认服务路径使用新 loader。**修改覆盖本分支的 grouped verify GEMM**；外部 LK wheel 的单 token GEMV / `_halves` 未被修改。池的冻结 snapshot、CPU miss 分工、W4A16 精度约定均保持；没有把 pool 改成 W4A8。

离线 SM90 编译得到以下资源变化（不是耗时测量）：

| 内核 | I | registers old→new | shared bytes old→new |
|---|---:|---:|---:|
| gate_up | 576 / 1152 | 96→80 | 10240→3072 |
| down | 576 | 112→59 | 10240→4096 |
| down | 1152 | 72→62 | 10240→4096 |

这些变体的 spill 均为 0；不能套用旧 fork 的 52→4。

## 验证与实机命令

- block-FP8：22 项 CPU/Triton interpreter 测试，包括 FP8 操作数、split-K 归并、尾部掩码、scale stride、配置边界与实际入口选择；48 个 SM90 shape/bucket 变体离线编译通过。
- verify：18 项测试，包括全部 256 个 packed byte、全部 E8M0 scale 字节、混合命中与未命中、原有 MoE 数值参考。CPU PTX 小解释器仅用于逐位验证，不能替代实际 GPU 执行。10 个 SM90 变体离线编译通过，检查 PRMT、BF16 MMA 和 ptxas 资源。
- 无 H100 在线测速、EP2 启动或完整模型 NaN 长测结果。

在服务环境、同一空闲 H100 上运行（benchmark 不设 `TRITON_INTERPRET=1`）：

```bash
python test/manual/test_h100_block_fp8.py --bench --output h100_block_fp8_bench.json
python test/manual/test_lk_verify_h100.py --bench --output h100_pool_prmt_bench.json
```

脚本对比旧/新完整计算路径，先检查数值及有限值，再用 CUDA graph 计时。FP8 包括 split-K reduce，verify 包括 gate_up、SiLU、down、combine。真实端到端收益仍需结合 acceptance、tokens/s 和 NaN 检查判断。
