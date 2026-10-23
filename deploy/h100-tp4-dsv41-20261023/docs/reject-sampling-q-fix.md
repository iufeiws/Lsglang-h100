# DSPARK 拒绝采样路径 q 容差修复（q 略 > 1 不再把接受长度清零）

- 日期：2026-10-23
- 修改文件（site-packages 安装版）：`/mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py`
- 安装版本：`sglang 0.5.15.dev105+g46b087b58`（`import sglang.kernels.ops.speculative.reject_sampling` 解析到上述同一文件，已核对 `module.__file__`）
- 补丁仓：`/mnt/SSD01/models/.llm-env/humming/patches/sglang-reject-sampling-q-tolerance.patch`
- 生效标记（可 grep）：`# [ENV-PATCH] reject_sampling q-tolerance EPS=1e-2`
- 容差常量：`Q_PROB_TOL_DEFAULT: tl.constexpr = 1e-2`，同时作为 kernel 参数 `Q_PROB_TOL: tl.constexpr = Q_PROB_TOL_DEFAULT` 的默认值
- 未改动：`dspark_accept.py` / `dspark_verify_window.py` / 任何 conf；未重启、未停止四卡服务（pid 43596，端口 18010，卡 4-7）；GPU0/1/2 未使用，测试全程只在 GPU3（`GPU-ccb33058-8899-4165-97a3-7cc5e661fd25`）

![修复前后对照](figs/reject-sampling-q-tolerance-20261023.png)

---

## 1. 缺陷位置与修改内容

| 位置（修复前行号） | 修复前 | 修复后 | 语义变化 |
| --- | --- | --- | --- |
| `:1-2` 之后新增 | — | `Q_PROB_TOL_DEFAULT: tl.constexpr = 1e-2`（+ 注释说明推导） | 新增容差常量 |
| `:34-35` kernel 形参 | `BLOCK_REDUCE: tl.constexpr = 4096,` | 追加 `Q_PROB_TOL: tl.constexpr = Q_PROB_TOL_DEFAULT,` | 容差可显式传参（生产入口不传，用默认值） |
| `:71`（block 验证） | `q_is_prob = (q > 0.0) & (q <= 1.0)` | `q_is_prob = (q > 0.0) & (q <= 1.0 + Q_PROB_TOL)` + `q_clamped = tl.minimum(tl.maximum(q, 0.0), 1.0)` | 容差域内被视为合法概率并夹到 [0,1]；`prefix_prob * p / q` 改用 `q_clamped` |
| `:72`（block 验证） | `prefix_prob = tl.where(q_is_prob, tl.minimum(prefix_prob * p / q, 1.0), 0.0)` | 同上，除数换成 `q_clamped` | 同上 |
| `:148`（普通验证） | `q_is_prob = (q > 0.0) & (q <= 1.0)` | `q_is_prob = (q > 0.0) & (q <= 1.0 + Q_PROB_TOL)` + `q_clamped` | 同上 |
| `:149`（普通验证） | `if q_is_prob & (coin * q < p):` | `if q_is_prob & (coin * q_clamped < p):` | `q ∈ (1, 1+EPS]` 时按 `q = 1` 参与比较，不再整条链拒绝 |
| `:194`（residual pass 1） | `q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)` | `q_in_prob = (q_val >= 0.0) & (q_val <= 1.0 + Q_PROB_TOL)`；`q_val = tl.where(q_in_prob, tl.minimum(q_val, 1.0), 0.0)` | 容差域内夹到 1.0；负值/NaN/±inf 仍取 0 |
| `:222`（residual pass 2） | 同上 | 同上 | 同上 |

安装后（修复后）文件中的实际行号：常量 `:13`、形参 `:47`、block 验证 `:89-92`、普通验证 `:173-175`、residual pass1 `:223-224`、residual pass2 `:254-255`。

保持不变的部分（原语义）：`q <= 0` 一律不可接受（零概率 draft token 不可接受）；`q = NaN` 的所有比较为 false ⇒ 仍拒绝/仍取 0；`+inf` 仍拒绝；`-inf` 仍取 0；`q > 1 + EPS` 仍走原拒绝路径。residual 里 `q` 越界为负仍视为 0（residual 回退到 `p`）。

## 2. 补丁 diff 原文

补丁文件：`/mnt/SSD01/models/.llm-env/humming/patches/sglang-reject-sampling-q-tolerance.patch`

```diff
--- a/sglang/kernels/ops/speculative/reject_sampling.py
+++ b/sglang/kernels/ops/speculative/reject_sampling.py
@@ -1,6 +1,17 @@
 import triton
 import triton.language as tl
 
+# [ENV-PATCH] reject_sampling q-tolerance EPS=1e-2
+# A draft probability q produced by an fp32 softmax may exceed 1.0 by a few ulp
+# of the row normalization. Worst-case fp32 accumulation bound for the DSPARK
+# vocab of 129,280 tokens (DeepSeek-V4.1-Flash-1017): (V - 1) * 2**-24 = 7.7e-3,
+# rounded up to 1e-2 here. Such a value is a rounding artifact of a legal
+# probability, so it is clamped into [0, 1] instead of being treated as an
+# illegal probability (which used to zero the whole accepted prefix). Values
+# above this tolerance, and non-probabilities (NaN, +-inf, <= 0), keep the
+# legacy reject behaviour.
+Q_PROB_TOL_DEFAULT: tl.constexpr = 1e-2
+
 
 @triton.jit
 def speculative_sampling_classic_kernel(
@@ -33,6 +44,7 @@
     BLOCK_V: tl.constexpr,
     BLOCK_VERIFICATION: tl.constexpr = False,
     BLOCK_REDUCE: tl.constexpr = 4096,
+    Q_PROB_TOL: tl.constexpr = Q_PROB_TOL_DEFAULT,
 ):
     pid = tl.program_id(0)
     cur_prob_row = 0
@@ -68,8 +80,17 @@
                 + (step - 1) * stride_dp_s
                 + draft_token * stride_dp_v
             )
-            q_is_prob = (q > 0.0) & (q <= 1.0)
-            prefix_prob = tl.where(q_is_prob, tl.minimum(prefix_prob * p / q, 1.0), 0.0)
+            # [ENV-PATCH] reject_sampling q-tolerance EPS=1e-2
+            # q slightly above 1.0 is a rounding artifact, not an illegal draft
+            # token: allow the tolerance band and clamp into the probability
+            # domain, so that such a q behaves exactly like q == 1.0. Values
+            # beyond 1.0 + Q_PROB_TOL and non-probabilities (NaN, +-inf, <= 0)
+            # keep the legacy reject path.
+            q_is_prob = (q > 0.0) & (q <= 1.0 + Q_PROB_TOL)
+            q_clamped = tl.minimum(tl.maximum(q, 0.0), 1.0)
+            prefix_prob = tl.where(
+                q_is_prob, tl.minimum(prefix_prob * p / q_clamped, 1.0), 0.0
+            )
             prefix_probs += (prefix_prob,)
 
         for step in tl.static_range(NUM_SLOTS - 1, 0, -1):
@@ -144,9 +165,14 @@
                 + draft_token * stride_dp_v
             )
             coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
-            # A proposal token must have positive probability under its draft distribution.
-            q_is_prob = (q > 0.0) & (q <= 1.0)
-            if q_is_prob & (coin * q < p):
+            # A proposal token must have positive probability under its draft
+            # distribution. [ENV-PATCH] reject_sampling q-tolerance EPS=1e-2:
+            # q in (1.0, 1.0 + Q_PROB_TOL] is treated as q == 1.0 (clamped)
+            # instead of being rejected; beyond the tolerance and for
+            # NaN/+-inf/<= 0 the legacy reject is unchanged.
+            q_is_prob = (q > 0.0) & (q <= 1.0 + Q_PROB_TOL)
+            q_clamped = tl.minimum(tl.maximum(q, 0.0), 1.0)
+            if q_is_prob & (coin * q_clamped < p):
                 num_accept += 1
                 cur_prob_row = step
                 tl.store(Predicts + last_accepted_global_idx, draft_token)
@@ -188,10 +214,14 @@
             else:
                 q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                 q_val = tl.load(q_ptr, mask=mask, other=0.0)
-                # Treat any non-probability q (NaN, +-inf, negative) as 0: the
-                # residual falls back to p. A comparison against NaN is false, so
-                # the range test rejects it along with the infinities.
-                q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
+                # Treat any non-probability q (NaN, +-inf, negative, or beyond
+                # 1.0 + Q_PROB_TOL) as 0: the residual falls back to p. A
+                # comparison against NaN is false, so the range test rejects it
+                # along with the infinities. [ENV-PATCH] reject_sampling
+                # q-tolerance EPS=1e-2: values inside the tolerance band above
+                # 1.0 are clamped to 1.0 instead of being zeroed.
+                q_in_prob = (q_val >= 0.0) & (q_val <= 1.0 + Q_PROB_TOL)
+                q_val = tl.where(q_in_prob, tl.minimum(q_val, 1.0), 0.0)
                 diff = residual_scale * p_val - q_val
                 val = tl.where(diff > 0.0, diff, 0.0)
 
@@ -218,8 +248,11 @@
             else:
                 q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                 q_val = tl.load(q_ptr, mask=mask, other=0.0)
-                # Same guard as pass 1.
-                q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
+                # Same guard as pass 1. [ENV-PATCH] reject_sampling q-tolerance
+                # EPS=1e-2: the tolerance band above 1.0 is clamped to 1.0
+                # instead of being zeroed; NaN/+-inf/negative still become 0.
+                q_in_prob = (q_val >= 0.0) & (q_val <= 1.0 + Q_PROB_TOL)
+                q_val = tl.where(q_in_prob, tl.minimum(q_val, 1.0), 0.0)
                 diff = residual_scale * p_val - q_val
                 val = tl.where(diff > 0.0, diff, 0.0)
 
```

文件哈希：

| 文件 | sha256 |
| --- | --- |
| 修复前（`.orig` 备份，同时存于 `docs/_evidence/q_tolerance/reject_sampling.py.pristine`） | `62caac17755082e2f89b1c81a743f55c81c31b7ff0485b7533c0267e72e9262c` |
| 修复后（当前 site-packages 生效版本） | `541e6ee85e404a4cde34cd74b7528afbb7133f40d3693aa2e83b13fa6b29afe2` |

## 3. EPS = 1e-2 的依据

### 3.1 推导：fp32 行和的最坏情况前向误差界

softmax 的 q 是"同一行 V 个指数值之和归一化"的结果，其 fp32 前向误差界为 `(V-1) * u`，`u = 2^-24 = 5.96e-08`（round-to-nearest 单位舍入误差），`V` 为 vocab：

| V | 来源 | `(V-1)*u` |
| --- | --- | --- |
| 129,280 | DeepSeek-V4.1-Flash-1017 的 `text_config.vocab_size`（本机 DSPARK 部署的实际 target 模型） | `7.71e-03` |
| 151,936 | Qwen3-235B 量级（参考） | `9.06e-03` |
| 262,144 | 2^18（参考上界） | `1.56e-02` |

取 `EPS = 1e-2`：覆盖部署 vocab 的误差界（余量 1.30 倍），并覆盖 `q ≤ 1` 的合法域之外的"归一化残差"整段；`2^18` vocab 及以上量级才需要放宽到 1.5e-2 以上。

### 3.2 本机实测量级（修复后重跑随附复现脚本）

`scripts/repro_reject_q_gt1.py` 在 GPU3 上用 DSPARK 真实 softmax 路径（`softmax_temp_flashinfer` → `flashinfer.sampling.softmax`，fp32 logits + 每行温度）采样 810 次：`max(q) = 1.000000000`，`q > 1.0` 的批次 = 0。即本机 benchmark 分布下未观测到越界，缺陷是潜在路径而非当前常态；触发下限由测试直接给出：`q = 1.0000001`（超出 1 仅 1e-7）即让接受长度从 5 变 0。

### 3.3 取值覆盖关系（与验收点的对应）

| q | `q-1` | EPS=1e-6 | EPS=1e-2（选用） |
| --- | --- | --- | --- |
| 1.0000001 | 1e-7 | 容差内 | 容差内 |
| 1.000006 | 6e-6 | 超出 ⇒ 仍拒绝 | 容差内 |
| 1.00002 | 2e-5 | 超出 ⇒ 仍拒绝 | 容差内 |
| 1.01 | 1e-2 | 超出 ⇒ 仍拒绝 | 容差内（fp32 下 `fp32(1.0)+fp32(1e-2) = 1.0099999904632568`，与 `fp32(1.01) = 1.0099999904632568` 逐位相等 ⇒ 判据为 true；fp64 同为 true） |

### 3.4 `q > 1 + EPS` 时的行为（安全性）

仍走修复前的拒绝路径（实测见 §6：`q = 1.0101 / 1.02 / 1.1 / 2.0` 接受长度 0，与修复前逐位一致；`+inf` 同样 0）。即本补丁只把"容差带内的越界值"重新解释为 `q = 1`，对明显非法的输入保持原有的保守处理。

### 3.5 夹断的数学含义

概率域内 `q ≤ 1` 恒成立，故 `q > 1` 唯一合法的解释是"`q = 1` 加上舍入偏差"；`q_clamped = min(max(q,0),1)` 把容差带内的值映射到 1.0，接受判据 `coin * q_clamped < p` 退化为 `coin < p`，与 `q = 1.0` 完全同一条规则；residual 里 `max(residual_scale*p - q_clamped, 0)` 同样只按 `q = 1` 计算。容差带外的值不会被夹断，直接判为非概率。

## 4. 等价性论证：所有 `q ∈ [0,1]` 且无 NaN 时，改动前后逐值等价

结构性论证（逐点，`EPS > 0`）：

1. 判定式 `(q > 0.0) & (q <= 1.0 + EPS)` 与 `(q > 0.0) & (q <= 1.0)` 在 `q ∈ (0,1]` 上同为 true；`q ≤ 0` 时第一项已为 false（两版同为 false）；`q = NaN` 时两项比较均为 false（两版同为 false）；`q = +inf` 时 `inf <= 1+EPS` 与 `inf <= 1` 同为 false。⇒ 判定结果逐值相同。
2. `q_clamped = tl.minimum(tl.maximum(q, 0.0), 1.0)`：对 `q ∈ [0,1]`，`max(q,0)` 与 `min(q,1)` 都只是"在有序的两个操作数中选一个"，不引入任何舍入 ⇒ `q_clamped` 与 `q` 逐位相同。
3. 普通验证：`coin * q_clamped` 与 `coin * q` 是同一对 fp32 值的同一乘法 ⇒ 比较结果与分支选择逐位相同。
4. block 验证：`prefix_prob * p / q_clamped` 与 `prefix_prob * p / q` 逐位相同（除数逐位相同 ⇒ 除法与 `tl.minimum(..., 1.0)` 结果相同）。
5. residual 两处：`q ∈ [0,1]` 时 `where(in_prob, min(q_val,1.0), 0.0)` = `where(True, q, 0.0)` = `q`，与修复前 `where(True, q_val, 0.0)` 相同；负值/NaN/±inf 时两版都取 `0.0`。

⇒ 唯一发生变化的自变量区间是 `q ∈ (1, 1 + EPS]`（以及 residual 中同一区间）。

实测对照（同一份输入、同一张卡、同一进程外重跑）：

- 随机等价性 battery：`q ~ U[0,1]`、无 NaN，`B = 16384` 条链、`gamma = 5`、`V = 32`、固定种子 `20261023`；修复前/后 `accept_token_num` **16384/16384 逐值相同**，`predicts` **16384×6 = 98304/98304 个 token 逐值相同**；输入张量 sha256 一致。
- 单链用例（`B = 1`）：16 个 `q` 用例在普通验证与 block 验证两条路径上，凡 `q ≤ 1`、`q ≤ 0`、`NaN`、`±inf`、`q > 1 + EPS` 者，`accept_token_num` 与 `predicts` 全部逐位一致。
- residual 参照用例：`q = 1.0` 的最终采样 token 修复前后同为 3。
- 接受律：`B = 262144` 条链、`p = 0.3`、`q = 0.6`（每步接受概率理论值 0.5），修复前/后 `E[接受长度] = 0.9658`（理论 0.9688，4σ = 0.0101）、首步接受率 `0.4992`（理论 0.5，4σ = 0.0039），两次完全相同。

## 5. 测试方法与规模

- 测试脚本：`/mnt/SSD01/models/.llm-env/Lsglang/scripts/test_reject_sampling_q_tolerance.py`（单卡、不加载模型/权重，只用 triton + torch，小 vocab 构造）
- 运行环境：
  - `CUDA_VISIBLE_DEVICES=GPU-ccb33058-8899-4165-97a3-7cc5e661fd25`（GPU3；脚本强制要求可见设备数为 1，否则拒绝运行，避免误落到 GPU0/1/2）
  - `PATH` 含 `/mnt/SSD01/models/.llm-env/Lsglang/env/bin`（flashinfer JIT 需要 `ninja`；复现脚本的 softmax 路径用到）
- 用例组与规模：

| 组 | 内容 | 规模 |
| --- | --- | --- |
| `accept_path` | 普通验证路径（`:148`），16 个 q 取值 | 单链 × 16 |
| `block_verification` | block 验证路径（`:71`），同一批 16 个 q 取值 | 单链 × 16 |
| `residual scan` | 强制拒绝构造，观察 residual 两遍扫描（`:194`/`:222`）输出的最终采样 token | 单链 × 2（`q_res = 1.0` / `1.0000001`） |
| `battery` | `q ~ U[0,1]` 随机等价性对照 + 不变量检查 | 16384 链 × 6 slot |
| `acceptance_law` | 接受律 `min(1, p/q)` 检查 | 262144 链 |
| `softmax probe` | 可选：真实 softmax 路径的 `max(q)`（本次由 `repro_reject_q_gt1.py` 覆盖） | 810 次采样 |

- 不变量检查（同时守护 kernel 与本测试的输入约定）：每条链的"接受前缀"必须等于被接受的 draft token 序列；非全接受链的 final token 必须落在 `[0, V)`。修复前/后各 16384 条链全部通过。
- 两种对照方式都跑过：(a) 用 `rollback`/`apply` 脚本在 site-packages 上切换修复前/后（§6.1/6.2/6.3，为主证据）；(b) 用 `--module-file` 直接按路径加载两份源码（`.pristine` 与已修复文件），不改 site-packages 即可复现同一对照。
- 测试脚本的一个输入约定（第一版测试曾写错，已修正并加了不变量检查）：`RetriveIndex` 存的是 `Predicts` 的**全局扁平下标**（与 sglang 的 `_get_or_create_chain_verify_buffers` 一致）；若误用"每行相同的 arange"，`B > 1` 时所有 program 会写同一批槽位、`predicts` 变成写入竞争的结果。

## 6. 测试输出原文

> 说明：以下为原样粘贴的终端输出（未做删改）。日志里 `[written] .../_scratch_qfix/out/*.json` 指向当轮的临时运行目录，该目录在取证后已删除，内容已原样复制到 `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/`（§9，含 sha256）。

### 6.1 修复前（先 `rollback` 还原 site-packages，再跑 `--mode pre`）

命令行：

```bash
export PATH=/mnt/SSD01/models/.llm-env/Lsglang/env/bin:$PATH
export CUDA_VISIBLE_DEVICES=GPU-ccb33058-8899-4165-97a3-7cc5e661fd25
cd /mnt/SSD01/models/.llm-env/Lsglang
/mnt/SSD01/models/.llm-env/humming/patches/rollback_sglang_reject_sampling_q_tolerance.sh
python scripts/test_reject_sampling_q_tolerance.py --mode pre \
  --dump /mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/pre_fixed_harness.json
```

```text
==============================================================================
mode            : pre
kernel module   : /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py
source sha256   : 62caac17755082e2f89b1c81a743f55c81c31b7ff0485b7533c0267e72e9262c
ENV-PATCH marker: absent
Q_PROB_TOL_DEFAULT: None
GPU             : NVIDIA H100 80GB HBM3  CUDA_VISIBLE_DEVICES='GPU-ccb33058-8899-4165-97a3-7cc5e661fd25'
==============================================================================

--- accept_path (non-block, :148) ---
case                                 accept  expected  verdict
q=1.0000001 (1e-7 over)                   0         0       ok
q=1.000006  (6e-6 over)                   0         0       ok
q=1.00002   (2e-5 over)                   0         0       ok
q=1.01      (1e-2 over)                   0         0       ok
q=1.0                                     5         5       ok
q=0.9999999                               5         5       ok
q=0.5                                     5         5       ok
q=1.0101  (over tol)                      0         0       ok
q=1.02    (over tol)                      0         0       ok
q=1.1     (over tol)                      0         0       ok
q=2.0     (over tol)                      0         0       ok
q=0.0                                     0         0       ok
q=-1e-8                                   0         0       ok
q=-0.5                                    0         0       ok
q=nan                                     0         0       ok
q=inf                                     0         0       ok
    verdict: PASS (16 cases)

--- block_verification (:71) ---
case                                 accept  expected  verdict
q=1.0000001 (1e-7 over)                   0         0       ok
q=1.000006  (6e-6 over)                   0         0       ok
q=1.00002   (2e-5 over)                   0         0       ok
q=1.01      (1e-2 over)                   0         0       ok
q=1.0                                     5         5       ok
q=0.9999999                               5         5       ok
q=0.5                                     5         5       ok
q=1.0101  (over tol)                      0         0       ok
q=1.02    (over tol)                      0         0       ok
q=1.1     (over tol)                      0         0       ok
q=2.0     (over tol)                      0         0       ok
q=0.0                                     0         0       ok
q=-1e-8                                   0         0       ok
q=-0.5                                    0         0       ok
q=nan                                     0         0       ok
q=inf                                     0         0       ok
    verdict: PASS (16 cases)

--- residual scan (:194/:222, final_token) ---
case                               finaltok  expected  verdict
residual q=1.0 (reference)                3         3       ok
residual q=1.0000001                      2         2       ok
    verdict: PASS (2 cases)

--- battery (q ∈ [0,1] randomized equivalence) ---
B=16384 gamma=5 V=32 seed=20261023
coverage: {'k_eq_0': 5831, 'k_middle': 4517, 'k_full': 6036, 'mean_accept': 2.38897705078125}
不变量(接受前缀 == draft 序列): OK
不变量(final token ∈ [0,V))  : OK
这批结果必须与对侧（pre/post）逐位一致 —— 由 --compare 校验

--- acceptance law (p=0.3, q=0.6, step accept prob = min(1,p/q) = 0.5) ---
B=262144  E[accept] = 0.9658 (theory 0.9688, ±4σ=0.0101)
P(first accepted) = 0.4992 (theory 0.5000, ±4σ=0.0039)
    verdict: PASS

[written] /mnt/SSD01/models/.llm-env/Lsglang/scripts/_scratch_qfix/out/pre_fixed_harness.json

==============================================================================
RESULT: PASS
```

### 6.2 修复后（`apply` 之后跑 `--mode post`）

命令行：

```bash
export PATH=/mnt/SSD01/models/.llm-env/Lsglang/env/bin:$PATH
export CUDA_VISIBLE_DEVICES=GPU-ccb33058-8899-4165-97a3-7cc5e661fd25
cd /mnt/SSD01/models/.llm-env/Lsglang
/mnt/SSD01/models/.llm-env/humming/patches/apply_sglang_reject_sampling_q_tolerance.sh
python scripts/test_reject_sampling_q_tolerance.py --mode post \
  --dump /mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/post_fixed_harness.json
```

```text
==============================================================================
mode            : post
kernel module   : /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py
source sha256   : 541e6ee85e404a4cde34cd74b7528afbb7133f40d3693aa2e83b13fa6b29afe2
ENV-PATCH marker: present (EPS=0.01)
Q_PROB_TOL_DEFAULT: 0.01
GPU             : NVIDIA H100 80GB HBM3  CUDA_VISIBLE_DEVICES='GPU-ccb33058-8899-4165-97a3-7cc5e661fd25'
==============================================================================

--- accept_path (non-block, :148) ---
case                                 accept  expected  verdict
q=1.0000001 (1e-7 over)                   5         5       ok
q=1.000006  (6e-6 over)                   5         5       ok
q=1.00002   (2e-5 over)                   5         5       ok
q=1.01      (1e-2 over)                   5         5       ok
q=1.0                                     5         5       ok
q=0.9999999                               5         5       ok
q=0.5                                     5         5       ok
q=1.0101  (over tol)                      0         0       ok
q=1.02    (over tol)                      0         0       ok
q=1.1     (over tol)                      0         0       ok
q=2.0     (over tol)                      0         0       ok
q=0.0                                     0         0       ok
q=-1e-8                                   0         0       ok
q=-0.5                                    0         0       ok
q=nan                                     0         0       ok
q=inf                                     0         0       ok
    verdict: PASS (16 cases)

--- block_verification (:71) ---
case                                 accept  expected  verdict
q=1.0000001 (1e-7 over)                   5         5       ok
q=1.000006  (6e-6 over)                   5         5       ok
q=1.00002   (2e-5 over)                   5         5       ok
q=1.01      (1e-2 over)                   5         5       ok
q=1.0                                     5         5       ok
q=0.9999999                               5         5       ok
q=0.5                                     5         5       ok
q=1.0101  (over tol)                      0         0       ok
q=1.02    (over tol)                      0         0       ok
q=1.1     (over tol)                      0         0       ok
q=2.0     (over tol)                      0         0       ok
q=0.0                                     0         0       ok
q=-1e-8                                   0         0       ok
q=-0.5                                    0         0       ok
q=nan                                     0         0       ok
q=inf                                     0         0       ok
    verdict: PASS (16 cases)

--- residual scan (:194/:222, final_token) ---
case                               finaltok  expected  verdict
residual q=1.0 (reference)                3         3       ok
residual q=1.0000001                      3         3       ok
    verdict: PASS (2 cases)

--- battery (q ∈ [0,1] randomized equivalence) ---
B=16384 gamma=5 V=32 seed=20261023
coverage: {'k_eq_0': 5831, 'k_middle': 4517, 'k_full': 6036, 'mean_accept': 2.38897705078125}
不变量(接受前缀 == draft 序列): OK
不变量(final token ∈ [0,V))  : OK
这批结果必须与对侧（pre/post）逐位一致 —— 由 --compare 校验

--- acceptance law (p=0.3, q=0.6, step accept prob = min(1,p/q) = 0.5) ---
B=262144  E[accept] = 0.9658 (theory 0.9688, ±4σ=0.0101)
P(first accepted) = 0.4992 (theory 0.5000, ±4σ=0.0039)
    verdict: PASS

[written] /mnt/SSD01/models/.llm-env/Lsglang/scripts/_scratch_qfix/out/post_fixed_harness.json

==============================================================================
RESULT: PASS
```

### 6.3 修复前 / 修复后对照（`--compare`，纯 CPU）

命令行：

```bash
cd /mnt/SSD01/models/.llm-env/Lsglang
python scripts/test_reject_sampling_q_tolerance.py --compare \
  docs/_evidence/q_tolerance/pre_fixed_harness.json \
  docs/_evidence/q_tolerance/post_fixed_harness.json
```

```text
==============================================================================
pre : pre  marker=False  sha=62caac17755082e2
post: post  marker=True  sha=541e6ee85e404a4c
==============================================================================

--- 修复效果（accept_token_num: pre → post）---
accept_path (non-block   q=1.0000001 (1e-7 over)        0 → 5   changed  ok
accept_path (non-block   q=1.000006  (6e-6 over)        0 → 5   changed  ok
accept_path (non-block   q=1.00002   (2e-5 over)        0 → 5   changed  ok
accept_path (non-block   q=1.01      (1e-2 over)        0 → 5   changed  ok
accept_path (non-block   q=1.0                          5 → 5   same     ok
accept_path (non-block   q=0.9999999                    5 → 5   same     ok
accept_path (non-block   q=0.5                          5 → 5   same     ok
accept_path (non-block   q=1.0101  (over tol)           0 → 0   same     ok
accept_path (non-block   q=1.02    (over tol)           0 → 0   same     ok
accept_path (non-block   q=1.1     (over tol)           0 → 0   same     ok
accept_path (non-block   q=2.0     (over tol)           0 → 0   same     ok
accept_path (non-block   q=0.0                          0 → 0   same     ok
accept_path (non-block   q=-1e-8                        0 → 0   same     ok
accept_path (non-block   q=-0.5                         0 → 0   same     ok
accept_path (non-block   q=nan                          0 → 0   same     ok
accept_path (non-block   q=inf                          0 → 0   same     ok
block_verification (:7   q=1.0000001 (1e-7 over)        0 → 5   changed  ok
block_verification (:7   q=1.000006  (6e-6 over)        0 → 5   changed  ok
block_verification (:7   q=1.00002   (2e-5 over)        0 → 5   changed  ok
block_verification (:7   q=1.01      (1e-2 over)        0 → 5   changed  ok
block_verification (:7   q=1.0                          5 → 5   same     ok
block_verification (:7   q=0.9999999                    5 → 5   same     ok
block_verification (:7   q=0.5                          5 → 5   same     ok
block_verification (:7   q=1.0101  (over tol)           0 → 0   same     ok
block_verification (:7   q=1.02    (over tol)           0 → 0   same     ok
block_verification (:7   q=1.1     (over tol)           0 → 0   same     ok
block_verification (:7   q=2.0     (over tol)           0 → 0   same     ok
block_verification (:7   q=0.0                          0 → 0   same     ok
block_verification (:7   q=-1e-8                        0 → 0   same     ok
block_verification (:7   q=-0.5                         0 → 0   same     ok
block_verification (:7   q=nan                          0 → 0   same     ok
block_verification (:7   q=inf                          0 → 0   same     ok

--- 等价性（q ≤ 1 / NaN / ±inf / q ≤ 0：必须逐位一致）---
q=1.0                              identical (accept=[5], predicts=[1, 1, 1, 1, 1, 1])
q=0.9999999                        identical (accept=[5], predicts=[1, 1, 1, 1, 1, 1])
q=0.5                              identical (accept=[5], predicts=[1, 1, 1, 1, 1, 1])
q=1.0101  (over tol)               identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=1.02    (over tol)               identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=1.1     (over tol)               identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=2.0     (over tol)               identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=0.0                              identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=-1e-8                            identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=-0.5                             identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=nan                              identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=inf                              identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=1.0                              identical (accept=[5], predicts=[1, 1, 1, 1, 1, 1])
q=0.9999999                        identical (accept=[5], predicts=[1, 1, 1, 1, 1, 1])
q=0.5                              identical (accept=[5], predicts=[1, 1, 1, 1, 1, 1])
q=1.0101  (over tol)               identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=1.02    (over tol)               identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=1.1     (over tol)               identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=2.0     (over tol)               identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=0.0                              identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=-1e-8                            identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=-0.5                             identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=nan                              identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])
q=inf                              identical (accept=[0], predicts=[1, -1, -1, -1, -1, -1])

--- residual (:194/:222) ---
residual q=1.0 (reference)         final_token 3 → 3   same     ok
residual q=1.0000001               final_token 2 → 3   changed  ok
修复后 residual: q=1.0000001 与 q=1.0 的最终采样结果一致（clamp 生效）

--- battery（随机 q ∈ [0,1]，B 条链）---
B=16384 gamma=5 V=32 seed=20261023
coverage pre={'k_eq_0': 5831, 'k_middle': 4517, 'k_full': 6036, 'mean_accept': 2.38897705078125} post={'k_eq_0': 5831, 'k_middle': 4517, 'k_full': 6036, 'mean_accept': 2.38897705078125}
input sha256 一致: 80101aeb54d9b47d
accept_num: 16384/16384 条链逐值相同（含 k=0 / 中间 / 满接受）
predicts  : 16384×6 个 token 逐值相同

--- 接受律 ---
E[accept] pre=0.9658 post=0.9658 (theory 0.9688)
首步接受率 pre=0.4992 post=0.4992 (theory 0.5000)

==============================================================================
COMPARE RESULT: PASS —— 修复消除了 q>1 的接受清零；q ≤ 1 与非概率输入逐位不变；residual 越界值改为 clamp 并与 q=1.0 对齐
```

### 6.4 复用已有复现脚本（修复后，GPU3）

命令行：

```bash
CUDA_VISIBLE_DEVICES=GPU-ccb33058-8899-4165-97a3-7cc5e661fd25 \
  /mnt/SSD01/models/.llm-env/Lsglang/env/bin/python \
  /mnt/SSD01/models/.llm-env/Lsglang/scripts/repro_reject_q_gt1.py
```

```text
device: NVIDIA H100 80GB HBM3
========================================================================
A) DSPARK 的 softmax 路径是否产出 q > 1.0
   flashinfer.sampling.softmax 签名: (logits: torch.Tensor, temperature: Union[float, torch.Tensor, NoneType] = None, enable_pdl: Optional[bool] = None) -> torch.Tensor
   N(0,1)         tries= 90  max(q)=0.230098769  q>1.0 的批次数=0
   N(0,5)         tries= 90  max(q)=0.999999166  q>1.0 的批次数=0
   N(0,20)        tries= 90  max(q)=1.000000000  q>1.0 的批次数=0
   N(0,50)        tries= 90  max(q)=1.000000000  q>1.0 的批次数=0
   N(0,100)       tries= 90  max(q)=1.000000000  q>1.0 的批次数=0
   peaked(+30)    tries= 90  max(q)=1.000000000  q>1.0 的批次数=0
   peaked(+100)   tries= 90  max(q)=1.000000000  q>1.0 的批次数=0
   peaked(+300)   tries= 90  max(q)=1.000000000  q>1.0 的批次数=0
   two-way tie    tries= 90  max(q)=0.000900580  q>1.0 的批次数=0
   ⇒ 全局 max(q) = 1.000000000   q>1.0 命中批次合计 = 0
========================================================================
B) q>1.0 时 chain_speculative_sampling_triton 的接受长度（p=0.999, coin=0.5）
   q=1.0000000     ⇒  accept_token_num = 5 / 5
   q=0.9999990     ⇒  accept_token_num = 5 / 5
   q=1.0000001     ⇒  accept_token_num = 5 / 5
   q=1.0000060     ⇒  accept_token_num = 5 / 5
   q=1.0000200     ⇒  accept_token_num = 5 / 5
   q=1.0100000     ⇒  accept_token_num = 5 / 5
========================================================================
实测结论（仅陈述数字）:
  A: softmax 全局 max(q)=1.000000000, q>1.0 批次=0
  B: q=1.0 接受 5/5;  q>1.0 最小接受 = 5/5  ⇒ 损失 0
```

### 6.5 安装 / 回滚 / 幂等性终端记录（逐字转录）

转录文件：`/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/patch_scripts_transcript.txt`

```text
$ ./rollback_sglang_reject_sampling_q_tolerance.sh --dry-run
DRY RUN: 会用 /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py.orig 覆盖 /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py（当前文件不含标记即视为还原成功）
--- current(patched)
+++ orig(pristine)
$ ./rollback_sglang_reject_sampling_q_tolerance.sh
ROLLED BACK: /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py == /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py.orig (sha256 62caac17755082e2)
$ ./rollback_sglang_reject_sampling_q_tolerance.sh   # 幂等
NOT PATCHED: /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py 无标记，无需还原
$ ./apply_sglang_reject_sampling_q_tolerance.sh --dry-run
note: 复用已有备份 /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py.orig（与当前文件一致）
DRY RUN: 不会修改 /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py
checking file sglang/kernels/ops/speculative/reject_sampling.py
DRY RUN OK: 补丁可干净应用
$ ./apply_sglang_reject_sampling_q_tolerance.sh
note: 复用已有备份 /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py.orig（与当前文件一致）
patching file sglang/kernels/ops/speculative/reject_sampling.py
APPLIED: /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py
  4:# [ENV-PATCH] reject_sampling q-tolerance EPS=1e-2
  83:            # [ENV-PATCH] reject_sampling q-tolerance EPS=1e-2
  169:            # distribution. [ENV-PATCH] reject_sampling q-tolerance EPS=1e-2:
  251:                # Same guard as pass 1. [ENV-PATCH] reject_sampling q-tolerance
回滚: /mnt/SSD01/models/.llm-env/humming/patches/rollback_sglang_reject_sampling_q_tolerance.sh
$ ./apply_sglang_reject_sampling_q_tolerance.sh   # 幂等
ALREADY APPLIED: /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py 已含标记 '[ENV-PATCH] reject_sampling q-tolerance'
$ sha256sum <site-packages>/.../reject_sampling.py
541e6ee85e404a4cde34cd74b7528afbb7133f40d3693aa2e83b13fa6b29afe2  /mnt/SSD01/models/.llm-env/Lsglang/env/lib/python3.12/site-packages/sglang/kernels/ops/speculative/reject_sampling.py
```

## 7. 安装 / 回滚 / 生效标记（路径与行为）

| 用途 | 路径 |
| --- | --- |
| 补丁 | `/mnt/SSD01/models/.llm-env/humming/patches/sglang-reject-sampling-q-tolerance.patch` |
| 应用脚本（幂等；`--dry-run`；失败自动还原） | `/mnt/SSD01/models/.llm-env/humming/patches/apply_sglang_reject_sampling_q_tolerance.sh` |
| 回滚脚本（幂等；`--dry-run`） | `/mnt/SSD01/models/.llm-env/humming/patches/rollback_sglang_reject_sampling_q_tolerance.sh` |
| 备份 | `<site-packages>/sglang/kernels/ops/speculative/reject_sampling.py.orig` |
| 生效标记 grep | `grep -n "ENV-PATCH\] reject_sampling q-tolerance" <site-packages>/sglang/kernels/ops/speculative/reject_sampling.py` |

脚本行为：先在 `site-packages` 下的临时目录里打补丁 + `py_compile` + 检查标记，全部通过后才用 `os.replace` 原子替换线上文件（线上文件不会出现半改状态，正在运行的服务即使重读源码也只会读到完整文件）；安装后自检失败会从 `.orig` 自动还原。`--dry-run` 只做检查不改文件。`ENV_PREFIX` / `SITE` 环境变量可覆盖安装前缀。

## 8. 未覆盖项与限制

1. **未重启四卡服务**：pid 43596 / 端口 18010 仍在运行（`ps`/`ss`/`nvidia-smi` 确认未受影响）。运行中的进程在启动时已导入旧模块，因此**当前运行实例仍使用修复前内核**；修复对下一次启动/重启生效。端到端（真实权重 + DSPARK 服务）验证未做。
2. **未走完整调用链**：`dspark_accept.py:140/156/157` 的 `chain_speculative_sampling_triton(...)` 与 `dspark_verify_window.py` 的调度入口未执行；测试直接调用生产入口函数（`chain_speculative_sampling_triton`），block 模式以 `block_verification=True` 传入。`threshold_single/threshold_acc` 在该版本 kernel 内未被使用。
3. **真实触发条件未在本机观测到**：810 次真实 softmax 采样 `max(q) = 1.000000000`、0 个批次越界（§3.2），因此"生产流量下接受长度被清零"的现象未在本机复现；缺陷是通过构造输入（`q = 1.0000001` 起）确认的。
4. **`q > 1 + EPS` 场景保留旧行为**：`q = 1.0101/1.02/1.1/2.0/+inf` 仍按原语义拒绝（接受长度 0）。若上游确实产出 > 1.01 的 q，接受清零的现象仍会出现——这一区间未做任何行为改动，也未评估其发生概率。
5. **vocab 维度未用真实规模**：测试用 `V = 8/32` 的小 vocab 构造（定位于 q 的判定/夹断逻辑），未跑 129,280 的真实 vocab 张量、也未覆盖 `engram_vocab`（16,000,000）等其他分支；EPS 推导只按 target vocab 129,280 给出。
6. **未验证与 CUDA graph 的交互**：未测 CUDA graph capture/重放路径下该 kernel 的行为（部署侧 `--cuda-graph-max-bs-decode/prefix 16`）。
7. **未覆盖 DSPARK 之外的其他调用方**：同一 kernel 也被 EAGLE 等其他投机解码路径复用，本报告只覆盖了 DSPARK 侧的语义要求；未测其他算法的端到端正确性。
8. **未覆盖并发/多请求批次**：测试为单/小批次（`B ≤ 262144`，batch 维独立 program），未与真实 `max-running-requests=16` 的并发调度组合验证。

## 9. 证据文件清单（绝对路径）

| 文件 | 内容 | sha256（前 16） |
| --- | --- | --- |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/pre_fixed_harness.log` | 修复前测试原文 | `7ca7c55395015508` |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/post_fixed_harness.log` | 修复后测试原文 | `089f3f0154a61ad0` |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/compare_fixed_harness.log` | 修复前/后对照原文 | `5f4a784dae2a41b1` |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/pre_fixed_harness.json` | 修复前全部用例结果 | `f388b509afe44917` |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/post_fixed_harness.json` | 修复后全部用例结果 | `c70a50494bd79959` |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/post_reapplied.json` | 再次 apply 后的结果（与首次安装逐位一致） | `c70a50494bd79959` |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/repro_postfix.log` | `repro_reject_q_gt1.py` 修复后输出 | `8830cbf1773bc521` |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/patch_scripts_transcript.txt` | apply/rollback 逐字转录（含 --dry-run 与幂等） | 见文件 |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/q_tolerance/reject_sampling.py.pristine` | 修复前源码副本 | `62caac17755082e2` |
| `/mnt/SSD01/models/.llm-env/Lsglang/docs/figs/reject-sampling-q-tolerance-20261023.png` | 结果图（英文标注，莫兰迪配色） | — |
| `/mnt/SSD01/models/.llm-env/Lsglang/scripts/test_reject_sampling_q_tolerance.py` | 单元测试脚本 | — |
| `/mnt/SSD01/models/.llm-env/Lsglang/scripts/plot_reject_sampling_q_tolerance.py` | 作图脚本 | — |

说明：上表 `docs/_evidence/q_tolerance/` 下的文件即复现命令中 `--dump` 的目标路径（本轮运行先写入一个临时运行目录，随后**原样复制**为该稳定路径；已核对 sha256，见表）。临时运行目录在取证后已删除，除上表与 `/mnt/SSD01/models/.llm-env/humming/patches/` 下的补丁/脚本外无其他中间产物。
