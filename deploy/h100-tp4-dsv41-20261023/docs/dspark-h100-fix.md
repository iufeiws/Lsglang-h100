# DeepSeek-V4.1-Flash-1017 / TP4 / H100 修复

本补丁修改执行代码，不增加采样诊断开关。

- **EP 下的 LK 路由编号**：StandardDispatcher 按层的实际执行位置决定是否
  映射专家编号。非 GPU 常驻的 LK 层使用本地编号，外组专家标记 `-1`；
  常驻 FlashInfer 层继续接收全局编号。覆盖 CPU prefill、GPU pool prefill
  和 capture/decode，不更改路由权重，不在各 EP 组内重新归一化。
- **Verify 热池计算**：SM90、MXFP4、group32、SiLU、`LK_VERIFY_RESIDENT=1`
  时，按专家合并最多 16 行，用 Tensor Core GEMM 复用权重。gate/up 使用
  split-K；SiLU 每个 route 只计算一次。两次 GEMM 均为 W4A16：MXFP4 权重
  按 E8M0 scale 解码到 BF16，乘法输入为 BF16，累加为 FP32；split-K 归约
  和 SiLU 在 FP32 中计算，SiLU 输出舍入并保存为 BF16。移除原补丁中的 TF32x3。
  使用现有 frozen slot table、池 stream 和租约 fence；CPU 继续计算全部 miss。
  其他硬件、格式和普通 decode 保留原 GEMV。`LK_VERIFY_GEMM=0` 可退回 GEMV。
- **V4.1 draft 注意力**：补齐 Q 的额外 heads 置零，避免读取未初始化内存；
  sink 的长度跟随实际 Q 头数，包括不需要补齐的 FP8 路径。
- **批次分派**：从实际 forward mode/spec metadata 获取 verify 宽度，取消
  “行数被 draft width 整除就是 verify”的推断。普通 decode、draft 和 ragged
  verify 按独立行处理。CUDA graph capture 和 eager forward 都发布批次作用域。
- **池 fence**：补齐 geometry 中缺失的 layer ID，使已有 deterministic split
  的 `pool_wait_fence` 能使用正确层号。

校验（本地无 CUDA GPU）：

```bash
TRITON_INTERPRET=1 python -m pytest -q -c /dev/null test/manual/test_lk_verify_h100.py
python test/manual/test_lk_verify_h100.py --compile-sm90
```

16 项 CPU/解释器测试通过，覆盖 W4A16 数值参考、全部/部分 miss、跨 16-row tile、
TP4 的 576 维中间层与 top-k=6、普通 decode/draft 的批次分派，以及 poisoned Q
padding/sink 形状。旧版 draft 两项回归均失败。部署同版 Triton 3.8.0 已离线编译
全部 4 个 kernel；gate/up 和 down 的 SM90 PTX 含 BF16 输入、FP32 累加的 MMA，
并断言不含 TF32 指令。CPU harness 修正 Triton 3.8 解释器的两个 BF16 仿真问题：
dot 把 BF16 位模式当 uint16 数值相乘，以及 cast 的截断舍入；CUDA 测试不打补丁。

## 精度依据、开销及 W4A8 范围

当前部署 README 明确不传 `--flashinfer-mxfp4-moe-precision`，常驻 GPU 层走
FlashInfer W4A16。旧池 GEMV 确实保留 FP32 SiLU 中间值，但这只能解释原实现，
不能证明 V4.1 必须使用 FP32 GEMM 输入。随仓库证据保存的 lk_moe 2.7.1 wheel
中，`_hybrid_gemm_kernel.py` 的 MXFP4 GEMM 已采用 BF16 乘法输入和 FP32 累加，
down 输入也转为 BF16。本实现与该精度选择一致；并不承诺与 CPU/GEMV 逐位一致，
也不承诺与 FlashInfer 的分块、归约、激活融合顺序逐位一致。

单请求、6 行 verify、top-k=6、每卡 I=576 时，完整 post-SiLU 张量有
20,736 个元素：FP32 是 81 KiB，BF16 是 40.5 KiB。本实现将其紧凑地写入已有
FP32 GEMV workspace 的 BF16 view，不新增设备分配，也不释放原 GEMV 所需容量。
split-K=16 的 FP32 gate/up partial 仍有 2.53125 MiB/层；完整写入再读取约
5.0625 MiB/层、20 层约 101.25 MiB（全命中，尚未计其他流量）。所以 FP32 开销
不能只看激活张量。原 TF32x3 的乘法成本也不能用 FP32 累加合理来辩护。
离线编译的静态 MMA 数从每个投影 kernel 48 降至 8；gate/up shared memory
从 20,480 降至 10,240 B，down 从 45,056 降至 10,240 B。
这些是同形状编译结果，不是 GPU 时间测量，不能换算成端到端加速倍数。

`--flashinfer-mxfp4-moe-precision fp8` 选择常驻层的 SM90 Humming W4A8，
包括其专用权重布局、residual scales 和 FC2 activation scale；需要兼容该 API
的 FlashInfer。`process_weights_after_loading` 对非 GPU 常驻层提前返回，
因此该开关不转换 LK 专家池。本次池 verify 明确为 W4A16；与 fp8 配置一起使用
时会输出一次范围提示。全量 GPU W4A8 尚未实现：池路径还需适配两次 GEMM 的
激活量化/scale、权重表示和重缩放，再验证数值、接受长度、CUDA graph 和性能。
单独把 SiLU 输出 cast 为 FP8 不构成完整 W4A8 实现。

H100 上可去掉 `TRITON_INTERPRET=1` 运行同一数值测试。尚未运行 H100 CUDA graph
集成及端到端性能/接受长度测试，不能由这些本地测试断言已恢复 80 tok/s，或断言
Q padding 就是原先平均接受长度 1.8 的全部原因。

补丁针对仓库源码。服务必须实际导入更新后的源码并重新启动，运行中的旧进程不会
自动加载修改。无需修改 gamma、采样温度、接受阈值或丢弃 CPU miss。

## 2026-10-24：TP4 + EP2 启动修复

用户报告前版在 TP4/EP1 下接受长度约 2.7、解码约 140 tok/s；EP2 在预热后
原生线程 SIGSEGV，Rank 1 以 -11 退出。该日志尾部本身未给出原生故障函数，
也不是 OOM 的证据。

源码确认一个可复现的接口错误：`flashinfer_mxfp4` 让 StandardDispatcher
跳过全局到本地编号转换，但 `FusedMoE.run_moe_core` 对非 GPU 常驻层绕过
FlashInfer，直接调用 LK。`_fill_lk_cfg` 给 LK 的 `expert_num` 是本地专家数。
因此 EP2 下大小 192 的专家表会收到 192–383 的全局编号；EP1 的 384 个专家
掩盖了此错误。池 snapshot 也按本地专家编号寻址。

| 分组 | 持有的全局编号 | 传给 LK 的编号 | 其他组编号 |
|---|---|---|---|
| EP0（TP0/TP1） | 0–191 | 0–191 | -1 |
| EP1（TP2/TP3） | 192–383 | 0–191 | -1 |

新增 `test/manual/test_lk_ep_dispatch.py`：31 项 CPU 回归覆盖 EP1/2/4、两类
执行位置、不同 runner、负编号、shared slot、跨组加和，以及三个 LK 入口。
用修改前的源码运行两个 EP rank × 三个入口的 6 项测试，全部在 LK 调用边界
检测到越界编号；修改后通过。测试执行生产 dispatcher 和 run_moe_core 的 AST，
以替身检查原生入口参数，未实际执行 lk_moe 原生线程或复现 SIGSEGV。

```bash
python -m pytest -q -c /dev/null test/manual/test_lk_ep_dispatch.py
TRITON_INTERPRET=1 python -m pytest -q -c /dev/null test/manual/test_lk_ep_dispatch.py test/manual/test_lk_verify_h100.py
```

本地没有 H100，EP2 完整启动及吞吐仍需部署验证；此次修复不宣称已排除所有
原生线程故障。TP4/EP1 不触发专家编号转换，原 W4A16 verify kernel 未修改。
