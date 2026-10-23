# 启动配置接线审查报告（只读 · fresh-context 独立审查）

审查时间：2026-10-23 00:40 – 00:56 (Asia/Shanghai)
审查者：独立接线审查者（只读，未启动任何服务，未占用 GPU，未改 conf/脚本/site-packages）

## 0. 审查对象与冻结版本

| 文件 | md5 | 行数 | mtime |
| --- | --- | --- | --- |
| `configs/dsv41_tp4_h100.conf`（v1） | `46169f8c7a9d505a63a6608caba6fecb` | 515 | 2026-10-23 00:42:50 |
| `scripts/serve_dsv41_tp4_h100.sh`（v1） | `11a20575342ce0331b88a5042c0dab4a` | 807 | 2026-10-23 00:42:58 |
| `configs/dsv41_tp4_h100.conf`（**v2 · 当前**） | `d88fc0a21f69ca8ae1da172409aab7d8` | 521 | 2026-10-23 00:58:24 |
| `scripts/serve_dsv41_tp4_h100.sh`（**v2 · 当前**） | `d57335767c36b6cd0cf07fc9f1cb30b0` | 807 | 2026-10-23 00:58:38 |
| `scripts/stop_dsv41_tp4_h100.sh` | `31adf37ffe6cbe96f6406d04a3ca209d` | — | 2026-10-22 13:26:12 |
| `scripts/status.sh` | `ac9c6ce6a9e2a275f23a771597d027bc` | — | 2026-10-22 13:50:28 |
| `scripts/health.sh` | `6090a15d916817cdb30be4870a45b453` | — | 2026-10-22 13:26:42 |

**审查期间目标被改动两次（事实）**：审查开始于 00:40:47（读到 conf 35 923 B / script 41 564 B，SELFCHECK 期望串为 W4A8）；① 00:42:50 / 00:42:58 二者被改写（SELFCHECK 期望/禁止串对调、日志头新增 W4A16 行、脚本内置 fallback 同步对调），`status.sh` 未同步；② 00:58:24 / 00:58:38 二者再次被改写（第 4 项改动的实现方式由“显式写 `bf16`”改为“**不传该 flag，用 CLI 默认 `default`**”，conf:421-427 / script:111 / script:728 随之变化），`status.sh` 仍未同步。**本报告的 §1 第 1–3 项、§2–§8 的其余条目以 v1 为准（v2 未触及这些内容，已 diff 确认仅 3 处差异）；§1 第 4 项、§3“SERVE_COMMON_ARGS”行、§4、§6-L11、§8-M17 已按 v2 复检并给出 v2 证据。** 冻结副本：`frozen/`（v1）、`frozen_v2/`（v2）。

**审查手段（全部只读 / 不启动）**：`bash -n` 语法检查；脚本自带 `--dry-run`（只做 preflight + 打印，不启动）；用已装 sglang CLI 解析器（`ServerArgs.add_cli_args` + `parse_args`，**不**构造 `ServerArgs`、`CUDA_VISIBLE_DEVICES=""`）解析生效命令行；`nvidia-smi` 查询（仅查询）；`site-packages` 源码逐点核对；`strings` 读闭源 `lk_moe` `.so`。

**副产品（事实）**：`--dry-run` 按脚本自身设计写出快照文件（本审查两次：`configs/dsv41_tp4_h100.effective.20261023_004426.conf`（v1）、`…20261023_010346.conf`（v2））；另有其它执行者在 00:58-00:59 也各写出一次快照（`…005839/005857/005907.conf`）。这些均为脚本既有行为，非本审查对 conf/脚本/site-packages 的改动。

**环境（已核实）**：`env/bin/python3` → Python 3.12.11；sglang/lsglang 1.6.1、lk_moe 2.7.1、flashinfer 0.7.2；`site-packages` 中 `python3.1` 是指向 `python3.12` 的符号链接（同一内容）。

---

## 1. 四项改动逐项判定

| 项 | 意图 | 实际（证据 file:line 或命令行原文） | 判定 | 备注 |
| --- | --- | --- | --- | --- |
| 1 `--ep-size 4` 注释掉 | MoE 走 TP4 分片，`ep_size` 回默认 1 | conf:387 `# --ep-size 4`（注释）；script:108 同；script:119 `SERVE_EP_ARGS=()`；script:260-261 为空数组则不加参数 ⇒ **生效命令行无 `--ep-size` token**（`dryrun.out:93`）。服务侧解析：`ep_size=1`（`parse_args.out`；字段默认 1：`arg_groups/fields/parallel.py:269-276`）。取值链：`moe_tp_size = tp_size // moe_ep_size // moe_dp_size = 4`（`runtime_context.py:137-157`、`:220-222`、`parallel.py:327-335`）⇒ `layer.py:425,459-460` `intermediate_size_per_partition = 2304//4 = 576`；lk_moe 侧 `_get_processes_info()` 因 `moe_ep_size==1` 返回 `(moe_tp_size, moe_tp_rank)=(4,…)`（`layer.py:2058-2061`），`cfg.expert_num = num_local_experts`（`layer.py:2107`）。每卡本地 routed 专家 = 384（模型 `config.json` text_config `n_routed_experts=384`；`layer.py:437-449`） | 一致 | 无隐式改写：`overrides.py:1625-1634`（仅 `_A2A_EP_SPANNING_BACKENDS`，我们 `moe_a2a_backend=none`）、`parallel_hook.py:52-56`（需 `enable_dp_attention`）、`parallel_hook.py:452-471`（需 `dwdp_size>1`，我们 =1）、`moe_hook.py:296`（megamoe a2a）、`deepseek_v4_hook.py:192-208`（需 `enable_prefill_cp`，默认 False 且未传）均不触发 |
| 2 `--attn-dp-size 1` 注释掉 | 回该 flag 默认 | conf:390、script:109 均注释 ⇒ 生效命令行无该 token（`dryrun.out:93`）；字段默认 **确为 1**：`arg_groups/fields/parallel.py:81-90`（`resolvable=True`）。解析值 `attn_dp_size=1`（`parse_args.out`）。派生 `attn_tp_size = tp_size // attn_dp_size // attn_cp_size = 4`（`runtime_context.py:73-76`）⇒ 注意力权重按 TP4 切分；`attn_dp_enabled_of` 为假（`runtime_context.py:191-199`，`attn_dp_size>1` 或 elastic 加入者才为真）⇒ 不进入 DP-attention 权重/缓冲路径 | 一致 | 是否被其它参数隐式改变：上述各改写门均不触发；`parallel_hook.py:320` 仅 `assert tp_size % attn_dp_size == 0`（4%1=0） |
| 3 `LK_POOL_SLOTS` 48 → 192 | 每层 GPU 侧专家缓存槽 192（无 ep 时每卡本地 384 ⇒ 50%） | conf:98 `LK_POOL_SLOTS="192"`；script:61 `192`；**实际导出** `export LK_POOL_SLOTS=192`（`dryrun.out:69`）。读取链：`lk_moe_env.py:277-278` `get_pool_slots()` → `layer.py:2098` `cfg.pool_slots` → 闭源 lib（`.so` 内 `[pump] ON pool=%s slots=%d …`、`[pool] slab: %.2f GiB (%d layers x %.2f MiB)`）。语义 = “resident expert slots per MOE layer；VRAM cost = slots × layers × expert bytes”（`lk_moe-2.7.1.dist-info/METADATA:60`）— 与意图口径一致。上限口径：192 ≤ 384 ✓（每卡本地专家 384）。显存：192×20×4,700,160 B = **16.81 GiB/卡**（preflight 实测打印 `dryrun.out:42`；conf:103、:321） | 一致 | **下界/上限校验**：仅有“>0 即开启”（`lk_moe_env.py:280-281`；0/unset = 关闭，METADATA:60），**未见** `slots ≤ expert_num` 的断言（sglang 侧 Python 与 `lk_moe` Python 均无；`.so` 内无对应校验串）⇒ 越界无校验，属“无法判定 C++ 内部行为”。另：`LK_POOL_WINDOW` 的“非法值下限 8”只作用于该旋钮（`lk_moe_env.py:124-128`），**不**作用于 `LK_POOL_SLOTS` |
| 4 `--flashinfer-mxfp4-moe-precision` fp8 → **W4A16**（v1: 显式 `bf16`；**v2: 整行注释、不传该 flag，用默认 `default`**） | 从 W4A8 切回 W4A16 做对照 | 字段定义/合法值/默认：`arg_groups/fields/exec_.py:848-853` `Literal["default","bf16","fp8"] = "default"`（默认是字面 `"default"`，不是 `"bf16"`）。v2 生效命令行**无**该 token（`effective_cmd_v2.txt`、`dryrun_v2.out:93`；conf:427、script:111 注释），服务侧解析结果 `flashinfer_mxfp4_moe_precision="default"`（`parse_args_v2.out`；v1 显式 bf16 时为 `"bf16"`）。语义：`mxfp4_flashinfer_cutlass_moe.py:48` `_use_sm90_humming = (not is_sm120) and precision=="fp8"`；H100 ⇒ `is_sm120=False` ⇒ humming=False（`"default"` 与 `"bf16"` 同为此支）⇒ 日志精度标签 `W4A16`（同文件 `:162-165`），字符串逐字为 `Preparing DSv4 MXFP4 experts for FlashInfer SM90 CUTLASS W4A16 (layer: …)`（`:166-170`）；该文件确为本配置实际实现（`fp8.py:457-484`：fp4 experts + `runner=flashinfer_mxfp4` + SM90 → `Mxfp4FlashinferCutlassMoEMethod`）；非 resident 层在 `:144` 直接 return ⇒ 只有 20 个 resident 层打此串 | 一致 | 无隐式改变精度语义：不传 flag ⇒ 字段默认 `"default"`，与 v1 的 `bf16` 在代码里走同一分支（均非 fp8）。其它 W4A8/W4A16 决定因素已核对，**无**未同步项：① `--moe-runner-backend flashinfer_mxfp4`（未改；若为 `humming` 会改走 `Mxfp4HummingMoEMethod`，`fp8.py:450-455`）；② `mxfp4.py:411,436` 第二处同规则（同读 CLI 字段，不矛盾）；③ `_use_mxfp8_act_scaling = is_sm120` ⇒ H100 为 False；④ `SGLANG_DSV4_FP4_DEQUANT` 未设（`dryrun_v2.out:40`；消费点 `model_overrides/deepseek_v4.py:80`、`model_config.py:659-671`）；⑤ `lk_moe` 无 W4A8/A16 选择权（池恒 W4A16 等价）；⑥ 量化来自权重 `quantization_config`（`quant_method=fp8`、`expert_dtype=fp4`）。自检串 conf:441-442 / script:126-127 = EXPECT W4A16、FORBID W4A8（v2 未变，与新取值相容）。⚠ 判定的性质：以上为**代码/命令行级**判定（未启动服务，故无运行期 W4A16 日志行作为实测证据） |

补充事实（与第 4 项相关、影响本轮口径）：脚本日志头与 preflight 的 W4A8 语境未同步（见 §6）；`status.sh:229-236` 仍按“W4A8≈20 行、W4A16 必须 0、出现 W4A16 即 ALARM”判定（见 §8 矛盾项）。

---

## 2. 通用问题 1：失效项（开关 → 是否 inert → 依据）

| # | 开关 / 门 | 是否 inert | 依据 |
| --- | --- | --- | --- |
| I1 | `SGLANG_DSV41_SWA_PAD_SENTINEL="1"`（conf:237；script:304 导出；`dryrun.out:82`） | **inert（无任何消费者）** | 全 `site-packages` 精确串出现 **files=0 / occurrences=0**；两个目标文件均无该 marker；`dsv4/indexer.py` mtime = 2026-10-22 13:32:54（安装时间，pristine），无 `.orig4`；`deepseek_v4_backend.py` 现为 dump 补丁版本（20:17）。即 E 补丁**已回滚** |
| I2 | `SGLANG_DSV41_AUTOTUNE_DUMMY_DECODE="1"`（conf:216） | **inert（入口被关）** | `--disable-flashinfer-autotune` 使 `should_run_flashinfer_autotune()` 直接 False（`flashinfer_autotune.py:107`）⇒ `base_runner.py:282-288` 整段不执行 ⇒ 包裹点 `base_runner.py:420-429`（`[autotune-decode]`）不可达 |
| I3 | ENV-PATCH-2 的消费分支（`layer.py:1763 elif FusedMoE._eager_decode_prewarm → _cpu_decode`） | **本轮不可达** | 两个置位点均不执行：`deepseek_v4.py:5574`（受 I4 门 `SKIP_CAPTURE_PREWARM=1` 跳过，`:5562-5572`）与 `base_runner.py:429`（受 I2 关闭）。全仓 `_eager_decode_prewarm` 仅此 4 处置位/读取 + 定义（`layer.py:375`） |
| I4 | ENV-PATCH 目标 `flashinfer/fused_moe/prepare.py:203 _process_humming_mxfp4_w4a8_payload`（int32+分块） | **本轮不执行** | 唯一调用点在 humming 专用入口 `prepare.py:262-332`（`:332`），仅 W4A8（`precision=="fp8"`）路径可达；本轮取值为非 fp8（v2 `default`／v1 `bf16`）⇒ 走 `mxfp4_flashinfer_cutlass_moe.py:221-238` 的 `interleave_moe_*` 分支 |
| I5 | `LK_GPU_PREFILL_SUB_M="16384"`（conf:166） | **inert** | `lk_moe_env.py:133-136`：`is_lk_gpu_prefill_enabled()` 为假（需 MIN_TOKENS>0，本处 = 0）⇒ `get_super_chunk_size()` 恒返回 **0**；METADATA：“Only meaningful while the prefill tier is on” |
| I6 | `LK_GPU_PREFILL_ENGINE="triton"`（conf:167） | **inert** | `lk_moe_env.py:113-122` 注释“Inert unless the tier is on”；tier 关闭（同上） |
| I7 | `LK_GPU_PREFETCH_WINDOW="1"`（conf:168） | **inert** | METADATA：triton 引擎专用（cutlass 忽略）、tier 关闭；且值即默认 1 |
| I8 | `LK_VERIFY_RESIDENT="0"` / `LK_VERIFY_SKIP_MISS="0"`（conf:172-173） | **无作用对象** | METADATA 归入 “Spec” 组；`SPEC_ENABLED="0"`（conf:171），生效命令行无任何 `--speculative-*`（`dryrun.out:93`） |
| I9 | `SGLANG_DSV41_POST_CONVERT_RELEASE`（代码侧新补丁） | **未接线 ⇒ inert** | `mem_cache/kv_cache_configurator.py` mtime = **2026-10-23 00:41:42**（`site-packages` 今日**唯一**被改文件，`.orig6` 备份在），门默认 `"0"`（`:2832-2835`）；conf/4 个脚本中均**未**设置该变量 |
| I10 | preflight 的 R2 守卫（script:613-622） | **本轮不生效** | 仅当 `SERVE_ARGS` 含字面 token `fp8` 时才断言（`:615`）；本轮不传该 flag（v2，解析值 `default`；v1 为 `bf16`）⇒ 整块跳过；且**没有**与 W4A16 对应的前检查项（W4A16 侧只由就绪后自检覆盖） |
| I11 | `--cuda-graph-max-bs-prefill 16`（由 `CUDA_GRAPH_MAX_BS="16"` 派生：conf:285 → script:250-253；实测在生效命令行中，`dryrun.out:93`） | **prefill 侧无捕获作用（自动关闭）** | 本模型 arch 经 `configs/deepseek_v41.py:38-39` 归一化为 `DeepseekV4ForCausalLM` ⇒ `is_deepseek_v4()` 为真（`configs/model_config.py:329-334`），且 CP 关闭 ⇒ `supports_prefill_cp_bcg()` 为假（`layers/cp/bcg.py:59-69`）⇒ 命中规则 “DeepSeek-V4 (heavy capture-pool memory pressure)”（规则名 `:219-224`，规则表 `:205-267`，首条命中即 `return`：`:268-282`）⇒ 声明 prefill backend = **DISABLED**；同样会命中的备用规则 “lk_moe CPU-resident experts”（`:261-267`，判定函数 `:170-192`）。prefill backend 键**未被显式锁定**（`:68-76` 只锁被显式赋值的键；默认值 `BREAKABLE`：`model_executor/cuda_graph_config.py:117-119`），故规则可改写它 ⇒ 该 token 不产生任何 prefill 图捕获；**decode 图仍为 `FULL` + `max_bs=16`**（decode backend 默认 FULL 且未被关闭）。启动日志将出现 “Breakable CUDA graph is incompatible with DeepSeek-V4 …; disabling prefill CUDA graph.”（`:272-274`） |

**相关但非失效（事实分类，便于对照）**
- 显式写成默认值的旋钮（有消费者、值等于默认）：`LK_POOL_WINDOW=32`、`LK_POOL_INFLIGHT=16`、`LK_POOL_STAGE=1`、`LK_GPU_RESERVE_GB=0`（METADATA：0 = 用 arena 自动扣减）、`LK_POOL_LOG=100`（默认 0，此处为显式开启）。
- 仍生效的“归因运行”遗留（消费者确认存在）：`CUDA_LAUNCH_BLOCKING=1`（conf:227，CUDA 运行时；全 launch 同步化）、`SGLANG_DSV41_DUMP_FLASHMLA_ARGS=1`（conf:228；消费点 `deepseek_v4_backend.py:4245,4295`，逐次调用一行 `[flashmla-args]`）。日志头未说明这两项在本轮是否仍属必需（属“未声明”，非“声明错误”）。
- 有消费者且生效（抽样核实）：`FLASHINFER_USE_CUDA_NORM=0`（`flashinfer/norm/__init__.py`）、`FLASHINFER_DISABLE_VERSION_CHECK=1`（`flashinfer/jit/env.py:83`）、`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`（`layers/engram.py`）、`SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`（`distributed/bootstrap.py`）、`SGLANG_SKIP_P2P_CHECK=1`（`custom_all_reduce_utils.py`）、`SGLANG_DSV41_SKIP_CAPTURE_PREWARM=1`（`deepseek_v4.py:5562`）、`LK_MOE_HYBRID_ENABLED`/`LK_GPU_RESIDENT_MOE_LAYERS`/`LK_POOL_LOG`/`LK_THREADS`/`LK_THREAD_BINDING`/`LK_NUMA_INTERLEAVE`/`LK_POWER_SAVING`（`lk_moe_env.py` 各处 + METADATA 启动期直读表）。
- 未设而正确（非失效）：`LK_POOL_LAYERS`（METADATA:73 明确 RETIRED，conf:109 说明一致）、`LK_POOL_DECODE=""`（不导出 ⇒ 跟随 `LK_POOL_SLOTS>0` = 开，`lk_moe_env.py:293-297`）、`LK_GPU_RESIDENT_MOE_LAYERS_SPEC=""`、`SGLANG_DSV4_FP4_DEQUANT`/`SGLANG_SM8X_MARLIN_GROUP32*`（preflight 已断言未设，`dryrun.out:40-41`）。

**“备选A”机制的活性核对（结论：机制仍在，非失效）**：SuperChunk 注意力 wrapper 安装门现位于 `model_runner_components/attention_backend_setup.py:131-137`（`get_super_chunk_size() > 0 and … > chunked_prefill_size`）⇒ 本配置 `0 > 8192` 为假，wrapper 不安装；MoE 侧 `layer.py:2126 cfg.super_batch_size = get_super_chunk_size()` = 0。

---

## 3. 通用问题 2：单一事实来源 / 重复定义

| 量 | 定义处 A | 定义处 B | 定义处 C / 其他 | 是否一致 |
| --- | --- | --- | --- | --- |
| `LK_POOL_SLOTS` | conf:98 = `192` | script:61（fallback）= `192` | `status.sh:30` fallback = `288`（conf 存在时被 `source` 覆盖为 192） | A/B 一致；C 不一致（288 正是 conf:327-328 明令禁止的值；仅 conf 缺失时可见） |
| `LK_GPU_PREFILL_SUB_M` | conf:166 = `16384` | script:75（fallback）= `8192` | — | **不一致**（生效值 = conf 的 16384；该变量本轮 inert，见 I5） |
| `SELFCHECK_EXPECT_STR` / `FORBID_STR` | conf:441-442（v2；v1 为 435-436）= W4A16 / W4A8 | script:126-127 = W4A16 / W4A8 | `status.sh:231-236` 硬编码 “W4A8 期望≈20 / W4A16 必须 0 / W4A16>0 → ALARM” | A/B 一致；C **反向不一致** |
| `SGLANG_DSV41_SKIP_CAPTURE_PREWARM`、`SGLANG_DSV41_AUTOTUNE_DUMMY_DECODE`、`SGLANG_DSV41_SWA_PAD_SENTINEL`、`SGLANG_DSV41_DUMP_FLASHMLA_ARGS`、`CUDA_LAUNCH_BLOCKING` | conf:206/216/237/228/227 = `1` | script:94-100（fallback）全部为空 = 不导出 | — | **不一致**（脚本注释声明“由 conf 置 1 生效”；即只用脚本内置默认跑时会静默少 5 个变量） |
| `SERVE_COMMON_ARGS` 整表 | conf:335-427（v1）/ conf:335-433（v2） | script:102-118 | — | **v2: 两处逐 token 完全一致（各 20 token，`conf_serve_args_v2.txt` vs `script_serve_args_v2.txt` diff 为空）**；v1 曾有 `bf16`/`--disable-flashinfer-autotune` 顺序差异，v2 已不存在 |
| `MAX_TOTAL_TOKENS` / `CUDA_GRAPH_MAX_BS` | conf:314 / :285 = `4194304` / `16` | script:121 / :122 = 同值 | — | 一致（token 级验证见 §4） |
| GPU UUID、期望物理序号、禁止卡、端口、权重路径、`GPU_IDLE_MAX_USED_MIB` | conf:57-82 | script:30-50 | `status.sh:19-40`（另有独立 fallback 副本：GPU_UUIDS/EXPECTED_INDEXES）、`stop.sh:20-30`、`health.sh:21-26` | 逐个对照**逐字一致**；`--dry-run` 实测 4/4 映射与期望一致（`dryrun.out:24-27`） |
| 池显存（同一物理量） | script preflight：`16.81 GiB/卡`（`dryrun.out:42`） | conf:103 / conf:321 = `16.81 GiB` | conf:306 = `pool 192槽×20层 = 4.203` GiB；`status.sh:174` = `LK_POOL_SLOTS*40*17280/1048576` ⇒ 192 槽打印“约 **126** GiB” | **三个口径互不一致**（4.203 / 16.81 / 126），其中 conf 内部两处即冲突 |
| 池槽/层数口径 | conf:382 “48 槽 × 20 层” | conf:321 “192 槽 × 20 层” | `status.sh:174` 用 **40** 层 | 不一致（A 为 ep4 残留） |
| 声明“内置 fallback 等值” | conf:5 | script:27 | 实测差异 6 处（`LK_GPU_PREFILL_SUB_M`、4 个 DSV41 门、`CUDA_LAUNCH_BLOCKING`；另有参数顺序） | 声明与事实**不一致** |

conf 定义变量共 65 个，**无一**在 4 个脚本中零引用（无“定义了却无人读”的 shell 级死变量）；conf 中所有需要导出的环境变量都在 script 的 `EXPORT_ENV` 中且实测均已导出（`dryrun.out:52-85`），无“conf 设了但被脚本丢弃”的项。

---

## 4. 通用问题 3：命令行等价性（dry-run 生效命令行 vs conf）

生效命令行原文（v2 · 当前，`dryrun_v2.out:93`，单行）：

```
/mnt/SSD01/models/.llm-env/Lsglang/env/bin/sglang serve --model /mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017 --tensor-parallel-size 4 --mem-fraction-static 0.88 --max-running-requests 16 --context-length 1048576 --chunked-prefill-size 8192 --moe-runner-backend flashinfer_mxfp4 --disable-flashinfer-autotune --enable-decoder-swa-bounded-replay --disable-shared-experts-fusion --reasoning-parser auto --tool-call-parser auto --trust-remote-code --max-total-tokens 4194304 --cuda-graph-max-bs-decode 16 --cuda-graph-max-bs-prefill 16 --host 0.0.0.0 --port 18010 --served-model-name DeepSeek-V4.1-Flash-1017
```

（v1 版本在此行中部多两个 token：`--flashinfer-mxfp4-moe-precision bf16`；其余逐字相同。）

| 检查 | 结果（证据） |
| --- | --- |
| token 级对照（v2） | conf `SERVE_COMMON_ARGS` 提取 **20** token，生效命令行（去 `--model <path>`）**32** token ⇒ `diff` 输出**仅** `20a21,32` 追加 12 token，即 `--max-total-tokens 4194304`、`--cuda-graph-max-bs-decode 16`、`--cuda-graph-max-bs-prefill 16`、`--host 0.0.0.0`、`--port 18010`、`--served-model-name DeepSeek-V4.1-Flash-1017`（追加来源：script:240/250-253/263/265）。**无缺失、无重复、无被覆盖值**（v1 同法对照为 22→34 token，同样只有追加） |
| 应该消失的 token | 生效命令行**无** `--ep-size`、`--attn-dp-size`、`--flashinfer-mxfp4-moe-precision`（v2 起）、`--swa-full-tokens-ratio`、`--speculative-*`、`--cuda-graph-backend-*`，且 `--max-total-tokens` 只有一份 |
| 拼写/取值合法性 | v2 的 34 个 token（v1 为 36）全部被已装 CLI 解析器接受（`parse_args_v2.out` / `parse_args.out`：`"parse": "OK (all tokens accepted)"`）；`--enable-decoder-swa-bounded-replay`（`exec_.py:140`）、`--disable-shared-experts-fusion`（`exec_.py:940`）、`--disable-flashinfer-autotune`（`exec_.py:358`）均存在 |
| 服务侧解析值（v2） | `ep_size=1`、`attn_dp_size=1`、`dp_size=1`、`attn_cp_size=1`、`moe_dp_size=1`、`dwdp_size=1`、`enable_prefill_cp=false`、`moe_tp_size=4`、`attn_tp_size=4`、`tp_size=4`、**`flashinfer_mxfp4_moe_precision="default"`**（v1 为 `"bf16"`）、`chunked_prefill_size=8192`、`max_total_tokens=4194304`、`cuda_graph_max_bs_decode/prefill=16`、`cuda_graph_backend_*=null`、`swa_full_tokens_ratio=null`（`parse_args_v2.out`） |
| `MAX_TOTAL_TOKENS` 去重逻辑 | script:232-240 的“剔除内联 `--max-total-tokens`”分支本轮**未参与**（`SERVE_COMMON_ARGS` 内本就没有该 token），结果只有 1 份，值 = conf:314 |
| preflight 的池算术所用 ep 分支 | script:641-648 从 `SERVE_ARGS` 里找 `--ep-size`，找不到 ⇒ `ep_size=1` ⇒ 每槽 `18,800,640/4 = 4,700,160 B` ⇒ 打印 `16.81 GiB/卡 (192 槽 × 20 层 × 4.48 MiB/槽)`（`dryrun.out:42`）与 conf:103 口径一致 |
| dry-run 退出码 | `0`，preflight 全绿（`dryrun.out:44,103`） |

---

## 5. 通用问题 4：前检查（preflight）完整性

| 期望覆盖项 | 是否存在 | 证据 / 实测 |
| --- | --- | --- |
| 4 个 UUID → 物理 4-7 断言 | 有 | script:561-580（解析 `nvidia-smi -L` → UUID 映射，逐个比对 `GPU_EXPECTED_INDEXES`）；实测 4/4 通过（`dryrun.out:24-27`）；禁止卡 GPU0/GPU1 检查另在 `:576-580`（实测通过） |
| 端口 18010 | 有 | script:530-536；实测空闲（`dryrun.out:18`） |
| 权重路径 | 有 | script:555-559（目录 + `config.json` + 分片计数 = 48，`dryrun.out:22`） |
| 卡空闲判定 | 有 | script:582-594（`nvidia-smi --query-gpu=index,uuid,memory.used`，阈值 `< GPU_IDLE_MAX_USED_MIB=1024`）；实测 4 卡 0 MiB（`dryrun.out:31-34`）；`--skip-gpu-idle-check` 会把超阈值降级为 WARN（`:591`） |
| 硬门槛 | 有 | preflight 失败 ⇒ dry-run 退出 1（script:690）/ 真启动拒绝（`:698`） |
| 依赖入口与仓库版本 | 有 | script:538-553（`sglang --version`、pyproject version/lk_moe 钉版本；实测 `dryrun.out:21`） |
| 池/驻留算术提示 | 有 | script:633-666（按 ep 分支、按 resident 列表算池层数；`dryrun.out:42`） |

**可能被静默跳过的分支（逐条事实）**

| 分支 | 触发条件 | 后果 |
| --- | --- | --- |
| 端口检查 | 本机无 `ss`（script:532 `command -v ss` 为假） | `port_owner_pid` 恒空 ⇒ 直接打印“[ OK ] 端口空闲”（`:535`）。本机 `/usr/bin/ss` 存在，故本轮为真实检查 |
| R2 守卫 | `SERVE_ARGS` 无字面 `fp8`（script:615） | 整块跳过（本轮 bf16 即如此）；**没有** W4A16 侧对应断言 |
| `LK_THREADS` 公式比对 | `lscpu` 失败 ⇒ `phys_cores` 非数字（script:598-600） | 比对整段跳过（无 WARN 也无 FAIL） |
| sglang 版本行 | `--version` 取不到（script:541-542） | 仅少打印一条 OK 行，不影响失败计数 |
| 仓库工作树 | 目录不存在（script:552-553） | 仅 WARN（不计入 `PREFLIGHT_FAIL`） |
| GPU 空闲检查 | `--skip-gpu-idle-check`（script:591） | 降级为 WARN，可绕过 |
| `nvidia-smi -L` 输出格式变化 | UUID 解析不到（script:566-574） | 4 个 UUID 全部 `chk_bad` ⇒ 失败（**非**静默通过） |

**前检查未覆盖的内容（事实）**：主机内存/huge pages（实测本机 `HugePages_Total: 0`，与 conf:298-299 的 “No huge pages: expect ~10x slower lookups” 告警条件相符，但 preflight 无该断言；`Mem available ≈ 1527 GiB`）；NFS 挂载/可读性与磁盘余量；目标卡上是否有**其它进程**（只比显存阈值；`pgrep` 检查仅在 `status.sh:116`）；模型 revision/文件完整性；CUDA/驱动版本；参数解析期的模型侧校验（preflight 只做 shell 级检查，不调用 sglang 解析器），抽样两条如下表。

**参数解析期校验（preflight 不覆盖，已逐条核对判定）**

| 校验 | 位置 / 触发 | 本配置的判定 |
| --- | --- | --- |
| `--enable-decoder-swa-bounded-replay` 互斥项 | `deepseek_v4_hook.py:379-407`，由 `model_hook.py:490-499` 的架构分支调用（本模型 arch 已被 `configs/deepseek_v41.py:38-39` 归一化为 `DeepseekV4ForCausalLM`）；arch hook 在 `pipeline.py:216`，晚于 cuda-graph 解析（`pipeline.py:182`） | **不命中 ⇒ 不会拒绝启动**：`incompatible[0]` 为假，因为 prefill backend 在解析后为 `DISABLED`（或最坏 `BREAKABLE`），永不等于 `FULL`（我们未显式锁 prefill backend） |
| encoder 侧 replay 预算 | `deepseek_v4_hook.py:237-249`，由 `model_hook.py:1074-1079`（`model_type == "deepseek_v41"`）调用 | 未设置 `--enable-encoder-swa-bounded-replay` ⇒ 立即 `return`，不产生约束 |

---

## 6. 通用问题 5：日志头自述 vs 实际（ENV-CHANGE / ENV-PATCH）

`dump_header()` 写入日志头（script:706-739）与**本次实际生效**配置的对照：

| # | 日志头自述（行） | 实际 | 判定 |
| --- | --- | --- | --- |
| L1 | `:717` “方案: resident=0-19 (**W4A8**) + pool=192 槽 (W4A16 等价)” | 本轮 resident 层判定为 **W4A16**（precision=bf16；`mxfp4_flashinfer_cutlass_moe.py:48,162-165`） | **不一致（标签）** |
| L2 | `:719` `[ENV-PATCH]` flashinfer `prepare.py` W4A8 预处理已改 int32+分块 | 补丁确实在（`prepare.py:203,242-256`，`.orig` 在），但本轮 W4A16 不进入该函数（唯一调用点 `:332`，仅 fp8 路径） | 声明存在、**本轮无作用**（未标注前提） |
| L3 | `:720` `[ENV-PATCH-2]` `_eager_decode_prewarm` 两文件作用域补丁已应用 | 补丁在（`layer.py:1763`），但两个置位点均关闭（`deepseek_v4.py:5574` 被 SKIP 门跳过；`base_runner.py:429` 被 autotune 关闭）⇒ 分支不可达 | 声明存在、**本轮不可达**（未标注前提） |
| L4 | `:721` `FLASHINFER_USE_CUDA_NORM 1→0` | conf:196 = 0，实测导出 `FLASHINFER_USE_CUDA_NORM=0`（`dryrun.out:56`），消费者在（`flashinfer/norm/__init__.py`） | 一致 |
| L5 | `:722` “chunked-prefill-size **8192→16384**” | 生效命令行 `--chunked-prefill-size 8192`（`dryrun.out:93`；conf:369、:366-368 已记回退） | **不一致**（该行未标注回退；其后的注记“上一条已被备选A取代并回退为 8192”挂在 LK_GPU_PREFILL 那一条上，CP 这一条自身没有回退说明） |
| L6 | `:723` `LK_GPU_PREFILL_MIN_TOKENS 1024→0` | conf:165 = 0，实测导出 0（`dryrun.out:61`） | 一致 |
| L7 | `:724` `SGLANG_DSV41_SKIP_CAPTURE_PREWARM=1` | conf:206 = 1，实测导出（`dryrun.out:81`），代码门在（`deepseek_v4.py:5562`） | 一致 |
| L8 | `:725` `SGLANG_DSV41_AUTOTUNE_DUMMY_DECODE=1` | 环境变量确已导出（`dryrun.out:79`），但语义 inert（I2） | 状态=已设置/无作用 |
| L9 | `:726` “新增 `--disable-flashinfer-autotune`” | 生效命令行含该 flag（`dryrun.out:93`） | 一致 |
| L10 | `:727` “新增 `--swa-full-tokens-ratio 0.1`” | 生效命令行**无**该 flag（conf:403-413 已撤掉） | **不一致** |
| L11 | `:728`（v2）“W4A8→W4A16 对照轮: **不传** `--flashinfer-mxfp4-moe-precision` ⇒ 用默认 default; 非 fp8 ⇒ W4A16 (厂商口径, launch/ 17/17 均不传); 自检判据已随之对调”（v1 此行为“fp8→bf16”） | conf:427 该行已注释；生效命令行无该 token（`dryrun_v2.out:93`）；解析值 `"default"`（`parse_args_v2.out`）⇒ 非 fp8 ⇒ W4A16（`mxfp4_flashinfer_cutlass_moe.py:48,162-165`）；自检串 conf:441-442 / script:126-127 为 W4A16/W4A8 | 一致（其中“17/17”计数不准确，见 §8 M17） |
| L12 | `:729` 判据“‘[attn-prewarm] rank=N done: **0** dummy forward(s)’ ×4” | 代码无条件打印该行且 SKIP 时 `n_fwd=0`（`deepseek_v4.py:5595-5601`） | 一致（**conf:204 的“按设计不再出现”与代码相反**，见 §8 M12） |
| L13 | `:736` 自检断言串 | 与 conf:441-442（v2；v1 435-436）一致（W4A16 / W4A8） | 一致 |
| L14 | 未声明项 | 日志头**未**提及本轮相关变更与解析期结果：`LK_POOL_SLOTS` 48→192（conf:98）、去掉 `--ep-size`/`--attn-dp-size`（conf:387-390）、`CUDA_LAUNCH_BLOCKING=1` 与 `SGLANG_DSV41_DUMP_FLASHMLA_ARGS=1` 仍在（conf:227-228，属归因运行遗留）、prefill CUDA graph 被解析期规则自动关闭（见 I11，日志中只有 sglang 自己的一行 warning） | 属“未声明”，非“声明错误”；全部导出 env 值仍逐行打印（`:730-733`），可回查 |
| L15 | 自检运行时文案 | `:382` “启动后自检 · W4A8 开关是否生效”、`:396` “期望串 (CUTLASS W4A8) 命中行数”、`:398/:404/:448` “W4A8 已生效/未出现 W4A8”、`:409-413` “禁止串 (CUTLASS W4A16)…没有落在 W4A16 路径”、`:434` 提示 “c) precision 是否为 fp8 ? (default → W4A16)” | 变量已对调但**文案/提示方向未对调** ⇒ 计数与通过/失败判定按变量走（功能正确），文字与结论方向相反（`:434` 的排查方向也相反） |
| L16 | 脚本文件头注释 | `:7` “前 20 层 (0-19) GPU-resident → W4A8”、`:16` “就绪后自动校验日志出现 CUTLASS W4A8 且**不出现** CUTLASS W4A16” | 与当前 conf:435-436 相反 ⇒ **不一致** |

---

## 7. 失效项（汇总）

共 **11** 条（I1–I11，明细见 §2）：① `SGLANG_DSV41_SWA_PAD_SENTINEL`（无消费者，补丁已回滚）；② `SGLANG_DSV41_AUTOTUNE_DUMMY_DECODE`（autotune 已关 ⇒ 不可达）；③ ENV-PATCH-2 消费分支（两置位点全关 ⇒ 本轮不可达）；④ ENV-PATCH 目标函数 `_process_humming_mxfp4_w4a8_payload`（仅 W4A8 可达 ⇒ 本轮不执行）；⑤ `LK_GPU_PREFILL_SUB_M`；⑥ `LK_GPU_PREFILL_ENGINE`；⑦ `LK_GPU_PREFETCH_WINDOW`；⑧ `LK_VERIFY_RESIDENT`+`LK_VERIFY_SKIP_MISS`（spec 关闭）；⑨ `SGLANG_DSV41_POST_CONVERT_RELEASE` 补丁（已 apply 于 00:41，conf/脚本未接线）；⑩ preflight R2 守卫（仅 fp8 生效；W4A16 侧无对应前检查）；⑪ `--cuda-graph-max-bs-prefill 16`（prefill 图被解析期规则自动关闭 ⇒ 无捕获作用；decode 图仍 FULL）。

附带事实（不计入 11 条）：显式写默认值 3 项（`LK_POOL_WINDOW`/`LK_POOL_INFLIGHT`/`LK_POOL_STAGE`）；`LK_GPU_RESERVE_GB=0` 等于默认语义；`LK_POOL_LOG=100` 为显式开启（默认 0）；仍生效的归因遗留 2 项（`CUDA_LAUNCH_BLOCKING=1`、`SGLANG_DSV41_DUMP_FLASHMLA_ARGS=1`）。

## 8. 矛盾项（汇总）

共 **17** 条（行号说明：conf 在 v2 中自第 421 行起整体 +6，例：自检串 v1 conf:435-436 = v2 conf:441-442；script 行号 v1/v2 相同）：

| # | 矛盾/不一致 | 位置 | 事实 |
| --- | --- | --- | --- |
| M1 | W4A8/W4A16 判定方向相反 | `status.sh:229-236` vs conf:435-436 / script:126-127 | `status.sh` 仍打印“W4A8 行数(期望≈20) / W4A16 行数(必须 0)”并在 `w4a16>0` 时 `[ALARM] 开关未生效` ⇒ 本轮必然触发 ALARM |
| M2 | 日志头声明与实际相反 | script:722 vs `dryrun.out:93` | 声明 cp 8192→16384，实际 8192 |
| M3 | 日志头声明与实际相反 | script:727 vs `dryrun.out:93` | 声明新增 `--swa-full-tokens-ratio 0.1`，实际未传 |
| M4 | resident 层精度标签 | script:717（另 script:7、:16） vs conf:421 | 标 W4A8，实际 W4A16 |
| M5 | 自检文案与对调后的变量语义相反 | script:382/396/398/404/409/411/413/448 | 计数按变量（功能正确），文字方向相反 |
| M6 | 自检失败提示方向相反 | script:434 | 提示“查 precision 是否 fp8”，在 EXPECT=W4A16 的口径下方向相反 |
| M7 | 同一物理量两个值（conf 内部） | conf:306 (`4.203`) vs conf:103/321（`16.81`） | 192 槽×20 层的显存不可能同时是 4.203 与 16.81 GiB（16.81 = 192×20×4,700,160 B 与 preflight 实测一致） |
| M8 | 池显存/层数口径 | `status.sh:174` vs script preflight / conf:321 | 公式 `LK_POOL_SLOTS*40*17280/1048576` ⇒ 192 槽打印“约 126 GiB”，且按 **40** 层（应为 20 个非 resident 层） |
| M9 | 同一旋钮第三个副本且取被禁值 | `status.sh:30` (`288`) vs conf:98 (`192`)；conf:327-328 明令“池不要调到 288” | conf 存在时被覆盖，conf 缺失时 `status.sh` 会按 288 显示 |
| M10 | “内置 fallback 等值”声明 | conf:5 / script:27 vs 实测 | fallback 与 conf 有 **6 处值差异**（v1/v2 同）：`LK_GPU_PREFILL_SUB_M`(8192/16384)、4 个 DSV41 门 + `CUDA_LAUNCH_BLOCKING`（空/1）；v1 曾另有 `SERVE_COMMON_ARGS` 参数顺序差异（v2 已消失） |
| M11 | E 补丁状态描述 | conf:229-236 vs `site-packages` 现状 | conf 称“已 apply（marker×2/marker×1）、`.orig4` 备份在”；实际该串出现 0 次、`indexer.py` 为 pristine、无 `.orig4`（已按回滚脚本恢复） |
| M12 | `[attn-prewarm] … done:` 是否出现 | conf:204 vs `deepseek_v4.py:5595-5601` / script:729 | conf 称“按设计不再出现”；代码无条件打印（SKIP 时 n_fwd=0），script:729 亦以此行为生效凭据 |
| M13 | ep4 时代核对项 vs 当前 ep1 | conf:381 vs conf:317 / 现状 | conf 称“**不应**出现 `Padding MXFP4 experts from intermediate=576 to 640`（出现即 ep4 未生效）”，但去 ep 后 `moe_tp_size=4` ⇒ `intermediate_size_per_partition=576`，`mxfp4_flashinfer_cutlass_moe.py:79-88` 必然打印该行；conf:317 自身也按 576→640 (+11.11%) 记账 |
| M14 | ep4 时代核对项 vs 当前池口径 | conf:382-383 vs conf:98/现状 | 核对项写“`[pool] slab` 应报 48 槽 × 20 层”“local experts = 96”，与 192 槽、ep=1(384) 现状不符 |
| M15 | 文件头/历史块 vs 当前决策 | conf:4、conf:19-25 vs conf:384-390、conf:421 | 头注释“48 槽专家池 (ep4)”与“ep4/W4A8 已确认有效不可回退”，与“去掉 ep”“切 W4A16”并存 |
| M16 | §3.x 与残留数值 | conf:247-265、conf:386 vs conf:387-390 | §3.x 称“定稿改为 `--ep-size 4 --attn-dp-size 1`”“ep 参数只写在第 6 节”，而第 6 节两者均已注释；conf:386 仍写“`LK_POOL_SLOTS=48` 仍远小于上限” |
| M17 | 依据中的计数与实际不符（v2 新增文案） | conf:421-427（v2 注释）、script:111、script:728 均称“厂商 `launch/` 下 **17/17** 个脚本全部不传该 flag” | 实测：`lsglang/launch/` 下共 **16** 个条目（全部为 `.sh`）；`grep -rl "flashinfer-mxfp4-moe-precision" launch/` 命中 **0** 个文件 ⇒ “全部不传”这一实质成立，但分母 17 与实测 16 不符 |

## 附录 A：本次审查使用的只读命令（可复现）

```
bash -n /mnt/SSD01/models/.llm-env/Lsglang/scripts/serve_dsv41_tp4_h100.sh
/mnt/SSD01/models/.llm-env/Lsglang/scripts/serve_dsv41_tp4_h100.sh --dry-run        # 只做 preflight + 打印, 不启动
CUDA_VISIBLE_DEVICES="" /mnt/SSD01/models/.llm-env/Lsglang/env/bin/python \
    /data3/shared/wangyf/llm-serving/dsv41-wiring-review/parse_args.py \
    /data3/shared/wangyf/llm-serving/dsv41-wiring-review/effective_cmd.txt          # 只用 argparse 校验/取值, 不构造 ServerArgs
nvidia-smi -L ; nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits
ss -ltnp 'sport = :18010'
strings -a .../lk_moe/_lk_moe_C_avx512_amx.so | grep -i "\[pool\]\|pool_slots"
```

## 附录 B：本次审查落盘证据（均在 `/data3/shared/wangyf/llm-serving/dsv41-wiring-review/`）

| 文件 | 内容 |
| --- | --- |
| `dryrun.out` / `dryrun.err`（v1）、`dryrun_v2.out` / `dryrun_v2.err`（v2） | `--dry-run` 完整输出（preflight 全绿、导出 env、生效命令行、自检断言串）|
| `effective_cmd.txt`（v1）/ `effective_cmd_v2.txt`（v2） | 从 dry-run 提取的生效命令行（供解析器复算）|
| `parse_args.out`（v1）/ `parse_args_v2.out`（v2） | CLI 解析结果：token 全接受 + `ep_size/attn_dp_size/moe_tp_size/attn_tp_size/precision` 等值（v2 precision=`default`）|
| `conf_serve_args.txt` / `eff_tokens.txt`、`conf_serve_args_v2.txt` / `eff_tokens_v2.txt` / `script_serve_args_v2.txt` | token 级对照（两版 diff 均仅追加 12 token；v2 conf 与脚本 fallback 参数表 diff 为空）|
| `conf_values.txt` / `fallback_values.txt` | conf 与脚本内置 fallback 的逐项值对照（6 处差异）|
| `frozen/`（v1）、`frozen_v2/`（v2） | 审查期间冻结的目标文件副本（含 §0 表中 md5）|
