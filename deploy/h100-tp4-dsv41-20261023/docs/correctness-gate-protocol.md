# DeepSeek-V4.1-Flash-1017 正确性闸门协议 (GSM8K + 长上下文冒烟)

> ## 【已按用户指示取消执行 — 2026-10-22】
>
> 本协议的 **GSM8K 评测**、**W4A8/W4A16 parity 对照** 与 **1M 长上下文冒烟** 均**不执行**;
> `scripts/eval_gsm8k.sh` 与 `scripts/seed_gsm8k_cache.py` **保留备查**, 不作为待办。
> 用户判断: 问题不会来自 W4A8, 只会来自实施。本文件其余内容仅作**文档资产**。
> (曾按批准装过 `sgl-eval` 及其判分器依赖, 已全部卸载, 环境回到批准前状态；见 §7。)
>
> **以下两条是环境事实, 与"评测是否执行"无关, 仍然有效, 可继续作为结论条件引用:**
>
> 1. **数据源事实**: `sgl_eval` 的 gsm8k 默认数据源是
>    `https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl`
>    (`sgl_eval/_vendored/nemo_skills/dataset/gsm8k/prepare.py:23`), **本机实测 http=000 不可达**
>    (连带 jsdelivr×3 / ghproxy.net / raw.gitmirror / 阿里 sail-moe OSS 全 000；
>    hf-mirror `/raw/` 只返回 131 B 的 git-lfs pointer, `/resolve/` 302 到不可达的 cas-bridge)。
>    可用替代: ModelScope `AI-ModelScope/gsm8k` 的 `main/test-00000-of-00001.parquet`,
>    419088 字节 / sha256 `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`,
>    与 hf-mirror 为该文件宣告的 LFS pointer `size` 和 `oid sha256` **逐位一致**。
>    ⇒ **本机如果要跑任何 gsm8k 系评测, 必须走这个替代源并自行写 `~/.cache/sgl_eval/gsm8k/`
>    的 file+marker 双文件缓存** (只丢 jsonl 会被判定失效并重新联网)。详见 §1.3 / §1.4。
>
> 2. **参数作用范围事实** (与是否做 A/B 无关, 描述的是本仓库的实现):
>    `--flashinfer-mxfp4-moe-precision {fp8,bf16}` **只作用于 GPU 常驻层**
>    (`run_moe_core` 里 `is_gpu_resident_layer` → `quant_method.apply`, `layer.py:1730-1734`);
>    **池内专家不走这条路径** —— 它们由 lk_moe 自己的 Triton GEMV/group-GEMM 计算
>    (`layer.py:113-118` 导入, `layer.py:2489-2587` launch, 核体在 lk_moe wheel 内)。
>    因此任何以该 flag 为自变量的对照, 受影响面只等于 `LK_GPU_RESIDENT_MOE_LAYERS` 所列的层
>    (当前启动脚本为 `0-19`, 即 40 层中的 20 层; 池层 20-39 两臂构造相同) ⇒
>    测到的差异会被稀释, **不可外推为"整个模型"的结论**。详见 §4.5。

- 服务端点: `http://127.0.0.1:18010`, model 名 `DeepSeek-V4.1-Flash-1017`
- 目标硬件/模型: 4×H100 80GB (SM90), `/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017`
- 本次要守的风险: **W4A8** (`--flashinfer-mxfp4-moe-precision fp8`) 把激活量化成 block-FP8, 数值会变 —— "服务起来了" ≠ "算对了"
- 状态: **已取消执行** (见上方标注)。原计划为"只备不跑", 执行需 lead 下令且服务就绪。
- 产物目录: 脚本 `/mnt/SSD01/models/.llm-env/Lsglang/scripts/`, 文档 `.../docs/`, 日志 `.../logs/`。**不使用 /tmp。**

---

## 1. 【先决问题】gsm8k 数据从哪来 —— 结论: 不在仓库内, 默认来源本机不可达, 已找到可用的替代源

### 1.1 仓库有没有内置数据? **没有**

| 检查项 | 结论 | 证据 |
| --- | --- | --- |
| sglang 仓库内是否有 gsm8k 数据文件 | 无。只有代码 | `python/sglang/test/` 下无 `*.jsonl`/`*.parquet` 数据文件 |
| `sgl_eval` wheel 内是否 bundle gsm8k | **没有**。gsm8k 的 spec 是 `"loader": "prepare"`; 只有 aime24/25/26 是 `"loader": "bundled"` | `sgl_eval/evals/_registry.py:36-42` (`gsm8k` → `loader:"prepare", save_args:("test",)`) vs `:44-60` (`aime24/25/26` → `"loader":"bundled"`) |
| wheel 内 `dataset/gsm8k/` 目录内容 | 只有 `__init__.py` + `prepare.py`, **无数据** | 解包 `sgl_eval-0.1.3-py3-none-any.whl` 实测 |

### 1.2 默认数据来源: **raw.githubusercontent.com —— 本机不可达**

`prepare.py` 的实际下载地址 (`sgl_eval/_vendored/nemo_skills/dataset/gsm8k/prepare.py:23`):

```python
URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{}.jsonl"
```

实测 (本次分析实际执行):

| 来源 | 结果 |
| --- | --- |
| `raw.githubusercontent.com/.../test.jsonl` | **http=000 (不可达)** |
| `cdn.jsdelivr.net` / `fastly.jsdelivr.net` / `gcore.jsdelivr.net` | **http=000** |
| `ghproxy.net` / `raw.gitmirror.com` | **http=000** |
| `hf-mirror.com/datasets/openai/gsm8k/api...`(元数据) | http=200 (可用, 但只能列文件) |
| `hf-mirror.com/.../resolve/...parquet` | http=302 → 跳到 **cas-bridge.xethub.hf.co (不可达)**, 落盘 0 字节 |
| `hf-mirror.com/.../raw/...parquet` | http=200 但**只有 131 字节的 git-lfs pointer**, 不是数据 |
| `sail-moe.oss-cn-hangzhou.aliyuncs.com` (ModelScope loader 指向的阿里 OSS) | **http=000** |
| **ModelScope `AI-ModelScope/gsm8k`** | **http=200, 可下载 ✅** |

### 1.3 可用替代源 (已实测, 且**与 canonical 逐字节一致**)

```bash
modelscope download --repo-type dataset AI-ModelScope/gsm8k \
  --include 'main/test-*.parquet' --local-dir <dir>
```

得到 `main/test-00000-of-00001.parquet`:

| 项 | 实测值 | canonical 参照 | 一致? |
| --- | --- | --- | --- |
| 大小 | `419088` 字节 | hf-mirror LFS pointer 声明 `size 419088` | ✅ |
| sha256 | `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59` | hf-mirror LFS pointer 声明 `oid sha256:ee7b8da9...` | ✅ |

> **因此: 用这个源不会改变评测结果的可比性** —— 它就是要下载的那个文件本身。仓库的 0.95 阈值仍然适用。
> (本仓库已把该 parquet 与生成的 prepared 缓存落在
> `/mnt/SSD01/models/.llm-env/Lsglang/docs/_evidence/gsm8k/`, 含 sha256 校验。)

### 1.4 缓存机制: 只丢 jsonl 进去**无效**

`sgl_eval` 读 `~/.cache/sgl_eval/<name>/<file>` 且要求一个**伴随 marker** (`sgl_eval/evals/_loader.py:206-231`):

- `_cache_marker(p) = p.with_name(p.name + ".sha256")`
- `_cache_is_valid()`: 文件与 marker 都必须在, 且 `marker.read_text().strip() == sha256(file)`
- 否则 → **重新执行 prepare** → 撞回不可达的 raw.githubusercontent.com

本仓库已按此机制生成 `test.jsonl` + `test.jsonl.sha256` (1319 行, sha256 `67e9046d...`),
由 `scripts/seed_gsm8k_cache.py` 复刻 vendored `prepare.py` 的转换逻辑
(字段改名 `problem`/`expected_answer`/`reference_solution`; `####` 切分; `float/int` 归一;
`re.sub(r"<<.*?>>","")` 去计算器标注; 11 条 train 集答案修正)。

实测校验:

```
rows=1319  sha256=67e9046d913470477dfb321814d4e3240240a397cfc19c6c610af02dd2cb6456
marker matches: yes
row[0].problem = 'Janet’s ducks lay 16 eggs per day. ...'
row[0].expected_answer = 18
```

> 1319 = GSM8K test split 的标准题量 ✅; 11 条 fixes 命中 0 条属**预期**
> (prepare.py 注释写明 "some of the **training** samples have incorrect answers", 我们用的是 test split)。

---

## 2. 仓库的真实评测配方 —— 注意: **不是 5-shot**

任务下达时提到 "5-shot", 但**仓库里唯一有 DSV4.1 记录的配方是零样本 (0-shot)**。以仓库为准:

`sglang/python/sglang/test/dsv41_flash_common.py:26-45` (`DSV41FlashAccuracyMixin`):

```python
# Use the zero-shot chat evaluator, not the legacy five-shot completion path.
gsm8k_backend = "sgl_eval"
gsm8k_score_threshold = 0.95
gsm8k_num_examples = 200
gsm8k_num_threads = 64
gsm8k_max_tokens = 4096
```

| 项 | 仓库 DSV4.1 配方值 | 备注 |
| --- | --- | --- |
| backend | `sgl_eval` | 零样本 **chat** 评测 + boxed/sympy 判分器 (`GSM8KMixin` docstring, `kits/eval_accuracy_kit.py:166-167`) |
| shot 数 | **0** | `gsm8k_num_shots=5` 只对 `run_eval` backend 生效 (`:177` 注释 "run_eval backend only") |
| 题量 | 200 | 从 1319 题里取**前 200 行**(见下) |
| 阈值 | 0.95 | 仓库 CI 门槛 |
| 客户端并发 | **64 线程** | 注意: 不是串行 |
| max_tokens | 4096 | sgl_eval 默认上限是 2048, 本配方显式提到 4096 |
| thinking | **关** | 配方未覆盖 `gsm8k_thinking`, 继承 `GSM8KMixin` 的 `False` |

### 2.1 采样参数 (sgl_eval 0.1.3 `GenConfig` 默认 + 本配方覆盖)

`_registry.py` 的 gsm8k 条目只设 `thinking:False` / `default_n_repeats:1`;
其余取 `GenConfig` 默认 (`sgl_eval/types.py:20-31`); 配方把 `temperature/top_p` 传 `None`
→ **不覆盖** (`sgl_eval_utils.py:131-141` 的 `if value is not None: overrides[key]=value`)。

| 参数 | 生效值 | 来源 |
| --- | --- | --- |
| temperature | **0.0** | GenConfig 默认 |
| top_p | **0.95** | GenConfig 默认 |
| min_p | 0.0 | GenConfig 默认 |
| repetition_penalty | 1.0 | GenConfig 默认 |
| seed | **不发** (`None`) | GenConfig 默认 |
| max_tokens | **4096** | 配方显式覆盖 |
| thinking / chat_template_kwargs | 不下发 | 配方 `gsm8k_thinking=False` |
| n_repeats | 1 | gsm8k spec |
| 判分 | 符号正确性 (`symbolic_correct`) → MathMetrics | `_math.py:make_score_one_fn` |

### 2.2 题目选择是**确定的** (无 RNG)

`_loader.py::_select_rows:70-88`:
gsm8k 的 spec **没有 `sample_seed`** (`_registry.py:36-42`; 对照 mmlu/mmlu_pro 才在 `:72`/`:101` 写了 `sample_seed: 0`)
→ `sample_seed is None` → 读到第 `num_examples` 行**就停** → **取 test.jsonl 的前 200 行(文件序)**。

⇒ A/B 两臂必然落在**完全相同的 200 题**上, 无需额外固定随机种子。这是本协议最重要的一条可比性保证。

### 2.3 ⚠️ 与任务下达口径的两个差异 (需 lead 拍板)

1. **shot 数**: 仓库是 0-shot, 不是 5-shot。**不要**改用 `few_shot_gsm8k.py` —— 它在仓库里已标
   `.. deprecated:: Use sgl-eval run gsm8k ...` (`python/sglang/test/few_shot_gsm8k.py:1-9`),
   且它自己也是从同一个不可达的 raw.githubusercontent.com 下载数据 (`:59-62`)。
2. **并发**: 仓库配方是 64 客户端线程。任务下达要求"串行单请求"。二者不可混用:
   - 与 0.95 阈值可比的是 **`NUM_THREADS=64`**;
   - 串行 (`NUM_THREADS=1`) 是另一个条件, 其绝对分**不能**直接与 0.95 比。
   - 本脚本支持 `NUM_THREADS` 覆盖, 但 **A/B 两臂必须取同一个值**。

---

## 3. 怎么跑 (脚本已就位, 等令)

```bash
cd /mnt/SSD01/models/.llm-env/Lsglang

# 前置检查 (不评测): 缓存 + 服务健康 + 模型名
DRY_RUN=1 scripts/eval_gsm8k.sh

# 正式跑一次 (仓库口径: 200 题 / 0-shot / 64 线程 / max_tokens 4096)
TAG=w4a8 scripts/eval_gsm8k.sh

# 串行口径 (可选, 两臂都改)
NUM_THREADS=1 TAG=w4a8_serial scripts/eval_gsm8k.sh
```

### 3.1 唯一外部前置: `sgl-eval` 当前**没装**

实测 `/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python -c "import sgl_eval"` →
`ModuleNotFoundError`。`sgl-eval==0.1.3` 只在本仓库的 **`[test]` extra** 里
(`python/pyproject.toml:192`; `dev = ["sglang[test]"]`), 不在默认依赖。

脚本的 preflight 会**快速失败**并打印确切命令。二选一 (需 lead 批准, 本次未执行):

```bash
ENV_PY=/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python
$ENV_PY -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  'sgl-eval==0.1.3' 'antlr4-python3-runtime==4.9.3'
# 或: INSTALL_DEPS=1 scripts/eval_gsm8k.sh
```

镜像已核实有 `sgl_eval-0.1.3-py3-none-any.whl`。`antlr4-python3-runtime==4.9.3` 是
sgl-eval 的 math_verify 判分器要求的 (pyproject 注释: 4.7.x 会让 `latex2sympy2_extended` ImportError)。

### 3.2 产物落点 (每次运行独立, 可重复跑)

| 产物 | 路径 |
| --- | --- |
| 运行日志 (stdout+stderr) | `logs/gsm8k_<TAG>_<时间戳>.log` |
| 结构化结果 JSON | `docs/gsm8k_<TAG>_<时间戳>.json` |
| server_info 快照 | `logs/gsm8k_<TAG>_<时间戳>.server_info.json` |
| sgl-eval 原生产物 (predictions + metrics.json) | `logs/gsm8k_<TAG>_<时间戳>.sgl_eval/sgl_eval_gsm8k_<uuid>/` |
| CI 指标 JSONL | `logs/gsm8k_<TAG>_<时间戳>.metrics.<pid>.jsonl` |

脚本会打印: 准确率 / 题目数 / shot 数 / 温度 / top_p / seed / 并发 / max_tokens /
无效答案率 / 请求错误率 / 耗时 / 吞吐。结果 JSON 内含 `verdict` ∈ {PASS, FAIL, UNKNOWN}。

---

## 4. A/B 协议: W4A8 vs W4A16 (**两个独立条件, 必须各重启一次, 严禁混跑**)

### 4.1 两个条件

基准命令 = **`scripts/serve_dsv41_tp4_h100.sh`** (launcher 已就位, 当前即 W4A8 臂)。
该脚本实测已含以下与之相关的设置, A/B 期间**必须原样保留**:

| 项 | 当前值 | 为什么必须保留 |
| --- | --- | --- |
| `--moe-runner-backend` | `flashinfer_mxfp4` | 该精度 flag 只在 `flashinfer_mxfp4` 分支下被读取 (`fp8.py:468-490`); 改成 marlin/triton 则 flag 完全失效 |
| `LK_GPU_RESIDENT_MOE_LAYERS` | `0-19` | 脚本内注释: "必须非空, 否则两个 W4A8 开关完全失效" —— 原因见 §4.5 |
| `--ep-size 4` + `--attn-dp-size 1` | 已设 | 与仓库 DSV4.1 TP4 精度配方逐项一致 (`dsv41_flash_common.py:21-22` `TP4_EP4_ARGS`) |
| `--mem-fraction-static` / `--max-running-requests` / `--context-length` | `0.88` / `16` / `1048576` | 两臂相同即可; 且与仓库精度测试的 `mem_fraction_static="0.8"` 不同 (记录在案) |
| `LK_POOL_SLOTS` | `48` (ep4: 每卡 96 本地专家 ⇒ 50%) | 两臂相同即可 |

| | Arm A (被测) | Arm B (对照) |
| --- | --- | --- |
| 语义 | MXFP4 权重 × **FP8 激活** | MXFP4 权重 × bf16 激活 |
| flag | `--flashinfer-mxfp4-moe-precision fp8` | `--flashinfer-mxfp4-moe-precision bf16` |
| 代码映射 | `mxfp4_flashinfer_cutlass_moe.py:44-48` → `_use_sm90_humming=True`; `:164` 标签 `W4A8` | `_use_sm90_humming=False`; `:164` 标签 `W4A16` |
| 值域 | `fields/exec_.py:848-853` `Literal["default","bf16","fp8"]`, 默认 `"default"` | 同上 |

其余**全部**启动参数必须**逐字相同**。唯一允许的差异就是这一个 flag。

### 4.2 执行顺序 (每臂一次重启)

```
[准备] 冻结一份"基准启动命令"文本, A/B 只在这一行上不同; 该文件随结果一起归档。
[A] 1. 停服(由负责启停的 teammate 执行; 本脚本不碰服务)
    2. 用 Arm A 配置启动, flag = fp8
    3. 记录启动配置快照 -> logs/ab_w4a8.startup.txt  (含完整命令行/环境变量/启动时间)
    4. 等 /health == 200 且 /v1/models 含 DeepSeek-V4.1-Flash-1017
    5. TAG=w4a8 scripts/eval_gsm8k.sh
    6. 归档: docs/gsm8k_w4a8_<ts>.json + logs/gsm8k_w4a8_<ts>.log
[B] 7. 停服
    8. 用 Arm B 配置启动, flag = bf16   (同一份基准命令, 只改这一个 flag)
    9. 记录 logs/ab_w4a16.startup.txt
   10. 等 /health == 200
   11. TAG=w4a16 scripts/eval_gsm8k.sh
   12. 归档: docs/gsm8k_w4a16_<ts>.json + logs/gsm8k_w4a16_<ts>.log
```

**不得**在同一进程内切换该 flag (它是**启动期**字段, 无运行时开关) —— 必须重启。

### 4.3 两臂必须相同的量 (可比性清单)

| 量 | 要求 |
| --- | --- |
| 题目集 | 同 200 题。**自动满足**: gsm8k 无 `sample_seed` → 恒取前 200 行 (§2.2) |
| 采样 | temperature 0.0 / top_p 0.95 / min_p 0.0 / seed 不发 / max_tokens 4096 |
| n_repeats | 1 |
| thinking | 关 |
| 客户端并发 | 同一个 `NUM_THREADS` (建议两臂都用仓库值 **64**; 若做串行, 两臂都 1) |
| 服务并发上限 | 同一个 `--max-running-requests` |
| 客户端版本 | 同一个 env (`/mnt/SSD01/models/.llm-env/Lsglang/env`) |
| 数据缓存 | 同一个 `~/.cache/sgl_eval/gsm8k/test.jsonl` (sha256 `67e9046d...`) |
| 注意 | 客户端 64 线程 vs 服务 `--max-running-requests 16` ⇒ 服务侧批大小由服务决定, 两臂相同即可 |

### 4.4 判读规则

设 `A` = W4A8 准确率, `B` = W4A16 准确率, `n = 200`。

**R0 (前置有效性)** — 不满足则整轮 A/B 作废, 先查配置:
- 两臂 `error_rate == 0`, `invalid_rate` 不异常升高 (判分器正常);
- 日志中无 `finish_reason=length` 截断样本 (脚本会打印前 3 条截断样本; 若有, 调大 `MAX_TOKENS` 重跑);
- `B >= 0.95` —— 对照臂自己都没过仓库门槛, 说明是**环境/配方**问题, 不能归因给 W4A8。

**R1 (仓库门槛)** — 主闸门:
- `A >= 0.95` → PASS; 否则 FAIL。

**R2 (A/B 数值等价)** — 主判据 (两臂各 200 题, 二项标准差):
- 单臂 p≈0.95 时 `SE = sqrt(0.95×0.05/200) = 1.54` 个百分点;
- 两独立臂差值的 `SE_diff = 1.54×sqrt(2) = 2.18` 个百分点;
- 95% 置信区间半宽 `1.96 × 2.18 = 4.27` 个百分点 ≈ **8.5 题**。
- 判据:
  - `A − B >= 0` → **通过** (W4A8 不劣);
  - `−4.3pp <= A − B < 0` → **无统计显著差异**, 通过但标记 `REVIEW` (建议复跑一次确认稳定性);
  - `A − B < −4.3pp` → **不通过** (超出抽样噪声, 判为 W4A8 引入精度退化)。

> 说明: 两臂跑的是同一批 200 题, 严格做法是配对检验 (McNemar), 检验力更高;
> 上表的非配对界是**保守**闸门, 适合"先拦住明显退化"。若需要更强结论, 用
> `logs/.../sgl_eval_gsm8k_*/` 里的逐题 predictions 做 McNemar。

**R3 (建议的稳定性复核)**: 任一臂若 `A` 或 `B` 落在 `[0.94, 0.96]` 边缘, 建议同配置复跑 1 次;
两次都 PASS 才算稳定 (单次 200 题的 ±3pp 噪声足以翻转边缘结论)。

**结果记录模板** (两臂各一行):

| arm | flag | 准确率 | 题目数 | shot | T | top_p | seed | 并发 | 耗时 s | error_rate | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | `--flashinfer-mxfp4-moe-precision fp8` | | 200 | 0 | 0.0 | 0.95 | 未发 | | | | |
| B | `--flashinfer-mxfp4-moe-precision bf16` | | 200 | 0 | 0.0 | 0.95 | 未发 | | | | |

### 4.5 ⚠️ 本 A/B 的**作用范围** (判读时必须写进结论)

`--flashinfer-mxfp4-moe-precision` 只作用在 **GPU 常驻层** 的 MoE 上
(`run_moe_core` 里 `is_gpu_resident_layer` → `quant_method.apply(...)`, `layer.py:1730-1734`)。
**池内专家不走这条路径** —— 它们由 lk_moe 自己的 Triton GEMV/group-GEMM 计算
(`layer.py:113-118` 导入, `layer.py:2489-2587` launch; kernel 本体在 lk_moe wheel 内,
格式由 `geo["format"]` 驱动)。这也是启动脚本把 `LK_GPU_RESIDENT_MOE_LAYERS` 设为非空的
原因 (注释: "必须非空, 否则两个 W4A8 开关完全失效")。

当前配置 `LK_GPU_RESIDENT_MOE_LAYERS=0-19` ⇒ **40 层里只有 0-19 层参与本次 A/B**;
20-39 层 (池, `LK_POOL_SLOTS=48`) 在两臂中**构造上完全相同**。因此:

- 本 A/B 的结论只能覆盖**常驻层那半个 MoE**; 不能外推成"整个模型 W4A8 无影响";
- 换个角度看: 因为受影响面只有一半, 测到的差异是**被稀释过**的 —— 若测出显著退化, 实际常驻层的退化幅度约是测到值的 2 倍量级;
- 若 lead 希望覆盖全模型, 需要**另设一轮全常驻**配置 (`LK_GPU_RESIDENT_MOE_LAYERS=""` 会**相反地**关掉 W4A8 开关;
  正确做法是 `LK_GPU_RESIDENT_MOE_LAYERS=0-39`, 并把 `LK_POOL_SLOTS=0`), 那是第三个独立条件, 需再重启一次且显存账要重算。
  **本轮不做**, 仅记录为可选后续。

---

## 5. 长上下文冒烟方案 (1M, **只写不跑**)

**目的**: 只验证 **1M 上下文下不 OOM 且能出 token**。它**不是**正确性测试, 与 §4 的准确率无关, 不得相互代替。

### 5.1 前置条件 (任一不满足则测试无意义, 先报 lead)

1. 启动参数含 `--context-length 1048576` (或 ≥ 900K+output);
2. `--max-total-tokens` 足以容纳该请求 (KV 预算决定, 见 `docs/budget-lsglang.md`);
3. `--max-running-requests` 允许该请求进入 (测试期间建议临时降到 1, 避免别的大请求抢 KV);
4. **KV 账要预先算**: 该文档已量化 —— 在 `LK_POOL_SLOTS=288` 下每卡仅剩约 16.68 GiB KV,
   而 16×1M 需要约 27.3 GiB (swa=0.005) — 即 **1M 冒烟很可能被显存设计挡下**。
   若如此, 这是**预期结果**, 报告应写"配置容量不支持", 而不是"kernel 有问题"。

### 5.2 构造 ~900K token 输入 (确定性)

```bash
ENV_PY=/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python
# 用被测模型的 tokenizer 造一个精确 token 数的 prompt, 落盘到项目目录 (不用 /tmp)
$ENV_PY - <<'PY'
from transformers import AutoTokenizer
import json, pathlib
tok = AutoTokenizer.from_pretrained(
    "/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017", trust_remote_code=True)
target = 900_000
unit = "The quick brown fox jumps over the lazy dog. "   # 只做长度填充, 不要求语义
ids = tok(unit, add_special_tokens=False)["input_ids"]
reps = target // len(ids) + 1
prompt_ids = (ids * reps)[:target]
text = tok.decode(prompt_ids)
out = pathlib.Path("/mnt/SSD01/models/.llm-env/Lsglang/logs/longctx_900k_prompt.txt")
out.write_text(text, encoding="utf-8")
assert len(tok(text, add_special_tokens=False)["input_ids"]) >= target - 8
print("tokens:", len(tok(text, add_special_tokens=False)["input_ids"]), "->", out)
PY
```

### 5.3 请求 (单请求, 小 max_tokens)

```bash
curl -s -m 900 http://127.0.0.1:18010/v1/completions \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c "
import json,pathlib
p=pathlib.Path('/mnt/SSD01/models/.llm-env/Lsglang/logs/longctx_900k_prompt.txt').read_text()
print(json.dumps({'model':'DeepSeek-V4.1-Flash-1017','prompt':p,
                  'max_tokens':32,'temperature':0.0,'top_p':0.95}))")" \
  -o /mnt/SSD01/models/.llm-env/Lsglang/logs/longctx_900k_resp.json \
  -w 'http=%{http_code} time=%{time_total}s\n'
```

### 5.4 观测与判据

| 观测 | 采集方式 |
| --- | --- |
| HTTP 状态与耗时 | `curl -w` (上面已带) |
| 是否出 token | 响应 `choices[0].text` 非空 / `usage.completion_tokens >= 1` |
| 是否 OOM / 崩 | 服务端启动日志 + 请求期间 `/health` 仍为 200; 响应体里无 `CUDA out of memory` |
| 显存峰值 | 请求期间并行采 `nvidia-smi --query-gpu=index,memory.used --format=csv` 到 `logs/` |
| prefill 延迟 | 响应 `time_total` (单请求, 含排队) |

**判据 (全部满足才算 SMOKE-PASS)**:
1. `http=200`;
2. `completion_tokens >= 1` 且 `choices[0].text` 非空;
3. 请求返回后 `/health` 仍 200, 且服务日志无 OOM/abort;
4. `4 张卡` 的 `memory.used` 未触顶 (记录数值)。

**失败归因顺序** (避免误判):
1. 先看是否 `http=400/413` 且报"exceeds context length / max_total_tokens" → **容量配置问题**, 非 kernel 问题;
2. 再看是否 `CUDA out of memory` → 显存账问题 (§5.1.4);
3. 只有 200 但 `completion_tokens == 0` 或乱码/立即 EOS → 才升级为**内核/数值**嫌疑。

---

## 6. 本次已完成 / 未完成

| 项 | 状态 |
| --- | --- |
| 数据可得性查清 | ✅ §1 (含 6 条不可达来源的实测记录) |
| 可用的 gsm8k 源 + 字节级校验 | ✅ §1.3 (sha256 与 canonical 一致) |
| prepared 缓存 + marker 生成 | ✅ `docs/_evidence/gsm8k/prepared/` (1319 行, 已 `--check` 通过) |
| 仓库真实配方核实 (0-shot, 非 5-shot) | ✅ §2 |
| `scripts/eval_gsm8k.sh` | ✅ 已 `bash -n` 语法通过; preflight/落盘路径已就位 |
| `scripts/seed_gsm8k_cache.py` | ✅ 已实测 `--check` 通过 |
| A/B 协议 + 判读规则 | ✅ §4 |
| 1M 长上下文冒烟方案 | ✅ §5 |
| **执行评测** | ❌ 未执行, 且**已取消** (用户指示, 2026-10-22)。全程未启动/重启/触碰服务 |
| 服务启停 | ❌ 未执行 (本脚本不碰服务) |
| `sgl-eval` 依赖 | ⚠️ 曾按 lead 批准装入, 批准撤回后**已全部卸载**, 环境回到批准前状态 (详见 §7) |

---

## 7. 环境变更与回滚记录 (2026-10-22)

按 lead 批准曾执行过一次依赖安装; 该批准随后被撤回(用户指示不做评测), 已**完整卸载**, 环境已回到批准前状态并复验。

### 7.1 实际装过的包 (全部已卸载)

| 包 | 版本 | 何时装 | 为何超出原批准清单 |
| --- | --- | --- | --- |
| `sgl-eval` | 0.1.3 | 按批准 | — (批准清单内) |
| `antlr4-python3-runtime` | 4.9.3 | 按批准 | — (批准清单内) |
| `latex2sympy2_extended` | 1.11.0 | 补充 | **批准清单只有 2 个包不够**: `math_grader.py:23` 在模块级 import 它 |
| `math_verify` | 0.9.0 | 补充 | 同上, `math_grader.py:24`; 且 `math_verify==0.9.0` 恰好要求 `latex2sympy2_extended==1.11.0` |
| `editdistance` | 0.8.1 | 补充 | `_registry.py:25` → `_ruler2` → `evaluator/ruler.py:25` 在模块级 import; 缺它连注册表都起不来 |

全部用 `pip install --no-deps`(或 `--no-deps --no-build-isolation`)逐条显式指定版本安装, 未让 pip 解析依赖树。
事前已核对 `math_verify` / `latex2sympy2_extended` 的 METADATA: **都不依赖 torch**。

### 7.2 回滚与复验 (实测)

```
pip uninstall -y sgl-eval antlr4-python3-runtime latex2sympy2_extended math-verify editdistance
```

| 复验项 | 结果 |
| --- | --- |
| `torch` | `2.14.1+cu132` (未变) |
| `torchcodec` | `0.17.0+cu132` (未变) |
| `torchvision` | `0.29.1+cu132` (未变) |
| `torchaudio` | **ABSENT** (全程未装, 保持正确状态) |
| `import sglang` | OK, `0.5.15.dev105+g46b087b58` |
| `import torch` | OK, `cuda_rt 13.2`, `cuda.is_available()=True` |
| 上述 5 包 | 全部 `absent` |

### 7.3 若日后仍需该评测 -- 一次性复原命令

```bash
ENV=/mnt/SSD01/models/.llm-env/Lsglang/env
$ENV/bin/pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple \
  sgl-eval==0.1.3 antlr4-python3-runtime==4.9.3 \
  latex2sympy2_extended==1.11.0 math_verify==0.9.0 editdistance==0.8.1
```

五个包全部在清华镜像可得; 除 `editdistance`(C 扩展, 有 cp312 manylinux wheel)外均为纯 Python。
镜像上 `sgl_eval` 只到 0.1.3, `math_verify` 到 0.9.0, `latex2sympy2_extended` 到 1.11.0。

> 注意: 该环境的 `pandas 3.0.6` / `pillow 13.0.0` / `datasets 5.1.0` 高于 `sgl-eval` 声明的
> 上限(`pandas<3` / `pillow<12` / `datasets<5`)。这是环境既有状态, 未改动; gsm8k 判分路径
> 不经过这三个包, 但若日后跑 mmlu 等其它 eval 需重新评估。
