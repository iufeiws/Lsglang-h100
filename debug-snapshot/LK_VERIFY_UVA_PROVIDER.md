# 实施说明：为 `LK_VERIFY_UVA` 提供现成的专家地址（`LK_VERIFY_UVA_PROVIDER`）

面向接手实施的工程师 / Agent。框架侧接入已完成（分支 `debug/dspark-variant-matrix-20261025`，
提交 `d7b1024c`），**实施者只需要写一个 Python 函数**，不需要改 sglang 代码。

## 1. 背景（一段话）

DSPARK 的 verify 步里，lk_moe 池层上没驻留的专家原本由 CPU 计算，导致 GPU 每层都要等。
`LK_VERIFY_UVA=1` 把草稿行的这部分改为由 GPU 经 UVA 直接读取主机上的 MXFP4 权重来计算
（kernel：`python/sglang/kernels/ops/moe/lk_verify_uva.py`）。默认做法是把 sglang 自己的 CPU 权重
原地 pin 住，每卡约 36 GiB 主机内存（不占显存）。provider 的作用是**改用已存在的一份权重**
（例如 MoE 后端自己的副本），省掉这 36 GiB。provider 不提供或给错时，框架会自动回退到 pin 副本，
**不会影响正确性**。

## 2. 要交付的东西

一个可 import 的函数，用环境变量指向它：

```bash
export LK_VERIFY_UVA=1                       # 总开关，需要 LK_VERIFY_RESIDENT=1
export LK_VERIFY_UVA_PROVIDER="pkg.module:function"
```

```python
def function(*, lk_moe, layer_id: int, num_experts: int, hidden: int, intermediate: int):
    """每个 lk_moe 池层调用一次（每个 TP rank 各自调用）。

    lk_moe:       该层的 lk_moe.MOE_MXFP4 实例（刚构造完）
    layer_id:     模型层号（池层为 20-39）
    num_experts:  本 rank 的本地专家数 E（TP4+EP2 下为 192）
    hidden:       H（5120）
    intermediate: 本 rank 的 I（TP4+EP2 下为 1152）

    返回 None：本层放弃，框架回退到 pin 副本。
    或返回 (w13, s13, w2, s2)：四个长度为 E 的整数序列，第 e 项是本地专家 e 对应块的地址。
    """
```

调用时机：`FusedMoE._process_mxfp4` 中，`MOE_MXFP4` 构造之后、sglang 删除自己的 CPU 参数之前
（见 `layer.py` 的 `_setup_verify_uva` / `_verify_uva_from_provider`）。

## 3. 地址契约（必须逐条满足）

### 3.1 可被 GPU 直接读取

地址必须是以下之一：已 `cudaHostRegister` 的主机内存、`cudaHostAlloc` / pinned 的主机内存，
或者显存。**未注册的普通主机内存会让 GPU 非法访存，进程直接崩溃**（CUDA 错误不可恢复，自检拦不住）。
地址在进程生命周期内必须一直有效，不能被释放、移动或覆写。

### 3.2 块内布局

本地专家 e 的每块数据，必须与 sglang 的 CPU 参数 `layer.w13_weight[e]` 等**逐字节相同**：

| 块 | 形状（行主序，无填充） | dtype | 字节数（H=5120, I=1152） | 含义 |
|---|---|---|---|---|
| w13 | `[2I, H/2]` | uint8 | 5,898,240 | 第 0..I-1 行为 **gate**，第 I..2I-1 行为 **up** |
| s13 | `[2I, H/32]` | E8M0（uint8） | 368,640 | w13 的块 scale |
| w2 | `[H, I/2]` | uint8 | 2,949,120 | down |
| s2 | `[H, I/32]` | E8M0（uint8） | 184,320 | w2 的块 scale |

- 打包：第 r 行第 j 个字节的**低 4 位**是 k = 2j 的 e2m1，**高 4 位**是 k = 2j+1。
- scale：第 r 行第 g 个字节作用于 k ∈ [32g, 32g+32)，值为 2^(byte-127)。
- 读取方式：kernel 读 `addr + r * (K/2) + j` 和 `scale_addr + r * (K/32) + g`（w13 时 K=H，w2 时 K=I）。
- 不同专家之间**没有**位置要求：可以不连续、乱序、分布在不同 NUMA 节点上。
  但**单个专家的一块必须是连续的**：如果后端把一个专家再切成多段，就不满足这条契约（见第 6 节）。
- "本地专家 e" 与 sglang dispatcher 的本地编号一致：EP rank r 的本地 e 对应全局 `r*E + e`。

### 3.3 与 sglang 参数的对应关系（用于自查）

sglang 的参数在 `_process_mxfp4` 中可直接取得：`layer.w13_weight`（int8，视作 uint8）、
`layer.w13_weight_scale_inv`、`layer.w2_weight`、`layer.w2_weight_scale_inv`，形状分别为
`[E, 2I, H/2]`、`[E, 2I, H/32]`、`[E, H, I/2]`、`[E, H, I/32]`。它们本身就是标准布局；
provider 给出的每块都应与对应 `[e]` 切片逐字节相同。

## 4. 验证方法

1. **离线**（不启动服务）：对任意一层调用 provider，用框架自带的自检函数比对：

   ```python
   from sglang.kernels.ops.moe.lk_verify_uva import HostExperts, matches_host_experts
   host = HostExperts.from_addresses(w13=a[0], s13=a[1], w2=a[2], s2=a[3],
                                     hidden=H, intermediate=I, device="cuda")
   assert matches_host_experts(host, w13=layer.w13_weight.view(torch.uint8),
                               s13=layer.w13_weight_scale_inv.view(torch.uint8),
                               w2=layer.w2_weight.view(torch.uint8),
                               s2=layer.w2_weight_scale_inv.view(torch.uint8),
                               experts=range(E))   # 全量比对，比启动自检更严格
   ```

2. **启动日志**（每个池层、每个 rank 各一行）：
   - `LK_VERIFY_UVA: layer N reads experts at addresses from pkg.module:function`：成功采用。
   - `LK_VERIFY_UVA: layer N: addresses from ... do not hold this layer's expert bytes (layout differs?); pinning a copy instead`：
     启动自检（比对第 0、E/2、E-1 号专家）失败，本层已回退。
   - `LK_VERIFY_UVA: layer N pinned X GiB of host experts`：provider 返回了 None，或未配置 provider。
   - 进程以 CUDA illegal address 崩溃：违反了 3.1。

3. **主机内存**：全部 20 个池层都采用 provider 后，与只开 `LK_VERIFY_UVA=1` 相比，每卡主机内存应少约 36 GiB。

4. **端到端**（与不配置 provider 时比较，两者应一致）：
   `debug-snapshot/scripts/recall_margin.py --n 50`、工具调用测试、`bench_ep2_streams.py --k 1`。

5. kernel 数值测试（与 provider 无关，在 H100 上运行）：
   `python -m pytest -q test/manual/kernels/test_lk_verify_uva.py`

## 5. 布局不符时怎么办

如果后端那份权重的块内布局与 3.2 不同（例如重排成 AMX tile 格式、额外的 scale 布局，或单个专家被切成
多段），自检会失败并回退，**不要**在 provider 里临时重排出一份新副本：那样又回到"多一份副本"的状态，
不如直接用默认的 pin 副本。正确做法是把该布局的精确定义（每个字节对应哪个权重元素）记录下来，
交给框架侧新增一个读取该布局的 kernel。
