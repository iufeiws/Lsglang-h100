#!/usr/bin/env bash
# =============================================================================
# eval_gsm8k.sh -- DeepSeek-V4.1-Flash-1017 正确性闸门 (GSM8K)
#
# 通过 HTTP 打已启动的端点, 跑仓库自带的 DSV4.1 精度配方 (sgl_eval backend)。
# 本脚本**不**启动/重启服务, **不**占 GPU; 只做 HTTP 请求 + 落盘。
#
# 仓库配方来源 (唯一有记录的 DSV4.1 精度门槛):
#   lsglang/python/sglang/test/dsv41_flash_common.py:26-45
#     gsm8k_backend      = "sgl_eval"   <- 零样本 chat 评测, 不是 5-shot 补全
#     gsm8k_score_threshold = 0.95
#     gsm8k_num_examples = 200
#     gsm8k_num_threads  = 64
#     gsm8k_max_tokens   = 4096
#     (gsm8k_thinking 未覆盖 -> 继承 GSM8KMixin 的 False)
#   lsglang/python/sglang/test/kits/eval_accuracy_kit.py:163-222 (GSM8KMixin 默认)
#   lsglang/python/sglang/test/sgl_eval_utils.py:86-129,150-225 (驱动与采样参数)
#
# 采样参数 (sgl_eval 0.1.3 的 GenConfig 默认 + 本配方的覆盖):
#   temperature = 0.0    (GenConfig 默认; 配方传 None -> 不覆盖)
#   top_p       = 0.95   (同上)
#   min_p       = 0.0    (同上)
#   repetition_penalty = 1.0
#   seed        = 未发送 (None)
#   max_tokens  = 4096   (配方显式覆盖; 仓库 CI 默认上限是 2048)
#   thinking    = 关闭 (chat_template_kwargs 不下发)
#   n_repeats   = 1      (gsm8k spec: default_n_repeats=1)
#   题目选择    = gsm8k spec **没有** sample_seed -> 取 test.jsonl 的**前 200 行**(文件序, 无 RNG)
#
# 用法:
#   scripts/eval_gsm8k.sh                 # 跑一次, 落盘到 logs/ 与 docs/
#   TAG=w4a8 scripts/eval_gsm8k.sh        # 给本次运行打标签 (A/B 对照用)
#   INSTALL_DEPS=1 scripts/eval_gsm8k.sh  # 允许自动装 sgl-eval (默认只报错退出)
#   DRY_RUN=1 scripts/eval_gsm8k.sh       # 只做前置检查, 不发评测请求
# =============================================================================
set -Eeuo pipefail

# ------------------------------- 配置 ---------------------------------------
BASE_URL="${BASE_URL:-http://127.0.0.1:18010}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4.1-Flash-1017}"
TAG="${TAG:-run}"

NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
NUM_THREADS="${NUM_THREADS:-64}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
THRESHOLD="${THRESHOLD:-0.95}"

ENV_PY="${ENV_PY:-/mnt/SSD01/models/.llm-env/Lsglang/env/bin/python}"
ROOT=/mnt/SSD01/models/.llm-env/Lsglang
REPO="$ROOT/lsglang"
LOGS="$ROOT/logs"
DOCS="$ROOT/docs"
SEED="$ROOT/scripts/seed_gsm8k_cache.py"
# 预置的 gsm8k 缓存 (源数据已按 sha256 校验与 canonical openai/gsm8k 逐字节一致)
SEEDED_ARTIFACT="$ROOT/docs/_evidence/gsm8k/prepared"
MIRROR_PARQUET="$ROOT/docs/_evidence/gsm8k/ms_aimirror/main/test-00000-of-00001.parquet"
CANONICAL_SHA256="ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
SGL_EVAL_CACHE="${SGL_EVAL_CACHE:-$HOME/.cache/sgl_eval/gsm8k}"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ID="gsm8k_${TAG}_${STAMP}"
RUN_LOG="$LOGS/${RUN_ID}.log"
RESULT_JSON="$DOCS/${RUN_ID}.json"
SGL_OUT="$LOGS/${RUN_ID}.sgl_eval"

mkdir -p "$LOGS" "$DOCS"
# 让 stdout+stderr 同时进日志与终端
exec > >(tee -a "$RUN_LOG") 2>&1

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

say "==================== $RUN_ID ===================="
say "base_url=$BASE_URL model=$MODEL_NAME tag=$TAG"
say "num_examples=$NUM_EXAMPLES num_threads=$NUM_THREADS max_tokens=$MAX_TOKENS threshold=$THRESHOLD"

# ------------------------- 0. 固定前置: sgl-eval ----------------------------
# sgl_eval 只在本仓库的 [test] extra 里 (python/pyproject.toml:192 `sgl-eval==0.1.3`),
# 默认环境**没有**装。缺它评测脚本会直接 skip/失败。
if ! "$ENV_PY" -c "import sgl_eval" >/dev/null 2>&1; then
  if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
    say "sgl_eval 缺失, INSTALL_DEPS=1 -> 安装 (清华镜像)"
    "$ENV_PY" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
      "sgl-eval==0.1.3" "antlr4-python3-runtime==4.9.3" \
      || die "sgl-eval 安装失败"
  else
    die "环境缺 sgl_eval (它只在 sglang[test] extra 里)。
  这是本脚本唯一的外部前置。二选一:
    (a) 显式装:  $ENV_PY -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 'sgl-eval==0.1.3' 'antlr4-python3-runtime==4.9.3'
    (b) 让脚本装: INSTALL_DEPS=1 $0
  (评测器 math_verify 由 sgl-eval 拉入; math_verify 用 SIGALRM, sgl_eval_utils 已处理主线程约束)"
  fi
fi
say "sgl_eval: ok ($("$ENV_PY" -c 'import sgl_eval,importlib.metadata as m;print(m.version("sgl-eval"))'))"

# ------------------------- 1. 数据可用性 (离线化) ---------------------------
# sgl_eval 的 gsm8k 用 "prepare" loader, 默认从 raw.githubusercontent.com 下载 ->
# 本机不可达。这里把预置缓存装进 sgl_eval 读取的目录, 并校验 marker。
ensure_cache() {
  if "$ENV_PY" "$SEED" --check --dest "$SGL_EVAL_CACHE" --num-examples "$NUM_EXAMPLES" >/dev/null 2>&1; then
    say "gsm8k 缓存已就绪: $SGL_EVAL_CACHE"
    return 0
  fi
  mkdir -p "$SGL_EVAL_CACHE"
  if [[ -f "$SEEDED_ARTIFACT/test.jsonl" && -f "$SEEDED_ARTIFACT/test.jsonl.sha256" ]]; then
    say "安装预置 gsm8k 缓存 (源 = ModelScope AI-ModelScope/gsm8k, sha256 与 canonical 一致)"
    cp "$SEEDED_ARTIFACT/test.jsonl" "$SGL_EVAL_CACHE/test.jsonl"
    cp "$SEEDED_ARTIFACT/test.jsonl.sha256" "$SGL_EVAL_CACHE/test.jsonl.sha256"
  elif [[ -f "$MIRROR_PARQUET" ]]; then
    say "从镜像 parquet 现场生成 gsm8k 缓存"
    "$ENV_PY" "$SEED" --from-parquet "$MIRROR_PARQUET" --verify-sha256 "$CANONICAL_SHA256" --dest "$SGL_EVAL_CACHE"
  else
    die "没有任何可用的 gsm8k 数据源。
  默认来源 https://raw.githubusercontent.com/... 在本机不可达 (http=000)。
  请恢复以下任一文件后重跑:
    预置产物: $SEEDED_ARTIFACT/test.jsonl (+ .sha256)
    镜像 parquet: $MIRROR_PARQUET
  取镜像: modelscope download --repo-type dataset AI-ModelScope/gsm8k --include 'main/test-*.parquet' --local-dir <dir>"
  fi
  "$ENV_PY" "$SEED" --check --dest "$SGL_EVAL_CACHE" --num-examples "$NUM_EXAMPLES" \
    || die "缓存校验失败 (marker 不匹配)"
}
ensure_cache

# ------------------------- 2. 服务就绪检查 ---------------------------------
say "服务健康检查: $BASE_URL"
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$BASE_URL/health" || true)
  [[ "$code" == "200" ]] && break
  [[ "$i" == "30" ]] && die "服务未就绪: $BASE_URL/health 返回 $code (等服务起来后再跑; 本脚本不启动服务)"
  sleep 5
done
say "health: 200"

MODELS=$(curl -s -m 10 "$BASE_URL/v1/models" || true)
echo "$MODELS" | grep -q "$MODEL_NAME" \
  || die "/v1/models 里没有 '$MODEL_NAME'。实际返回: $(echo "$MODELS" | head -c 400)"
say "/v1/models: 找到 $MODEL_NAME"

SRV_INFO=$(curl -s -m 15 "$BASE_URL/server_info" || true)
echo "$SRV_INFO" > "$LOGS/${RUN_ID}.server_info.json"
say "server_info 已存 $LOGS/${RUN_ID}.server_info.json"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  say "DRY_RUN=1 -> 前置检查全部通过, 不发送评测请求。"
  say "log: $RUN_LOG"
  exit 0
fi

# ------------------------- 3. 跑评测 --------------------------------------
# 用仓库自带的驱动函数, 参数与 dsv41_flash_common.py 的 DSV41FlashAccuracyMixin 逐项对齐。
say "开始评测 (200 题, 零样本, 串行线程数=$NUM_THREADS) ..."
set +e
SGLANG_TEST_METRICS_OUTPUT="$LOGS/${RUN_ID}.metrics" \
"$ENV_PY" - "$BASE_URL" "$MODEL_NAME" "$NUM_EXAMPLES" "$NUM_THREADS" "$MAX_TOKENS" "$SGL_OUT" <<'PY'
import json, sys, time
sys.path.insert(0, "/mnt/SSD01/models/.llm-env/Lsglang/lsglang/python")
from types import SimpleNamespace
from sglang.test.sgl_eval_utils import run_sgl_eval

base_url, model, n, threads, max_tokens, out_dir = sys.argv[1:7]
args = SimpleNamespace(
    eval_name="gsm8k",
    base_url=base_url,
    model=model,
    num_examples=int(n),
    num_threads=int(threads),
    max_tokens=int(max_tokens),
    temperature=None,          # -> GenConfig 默认 0.0
    top_p=None,                # -> GenConfig 默认 0.95
    chat_template_kwargs=None, # thinking 关闭 (配方 gsm8k_thinking=False)
    sgl_eval_out_dir=out_dir,
)
t0 = time.time()
metrics = run_sgl_eval(args)
wall = time.time() - t0
print("__METRICS_JSON__" + json.dumps({"metrics": metrics, "wall_seconds": wall}))
PY
rc=$?
set -e

# ------------------------- 4. 汇总落盘 -------------------------------------
if [[ $rc -ne 0 ]]; then
  say "评测进程退出码 $rc (见上面的 sgl-eval 报错; 常见: error_rate>0 / partial)"
fi

"$ENV_PY" - "$RUN_LOG" "$RESULT_JSON" "$RUN_ID" "$TAG" "$BASE_URL" "$MODEL_NAME" \
           "$NUM_EXAMPLES" "$NUM_THREADS" "$MAX_TOKENS" "$THRESHOLD" "$rc" "$SGL_OUT" <<'PY'
import json, sys, glob, os, re
(log_path, out_json, run_id, tag, base_url, model, n, threads, mx, thres, rc, sgl_out) = sys.argv[1:13]
text = open(log_path, encoding="utf-8", errors="replace").read()
m = re.findall(r"__METRICS_JSON__(\{.*\})", text)
payload = json.loads(m[-1]) if m else {}
metrics = payload.get("metrics", {}) or {}
# sgl-eval 自己写的 metrics.json 更权威 (含 generation/num_examples/n_repeats)
per_run = sorted(glob.glob(os.path.join(sgl_out, "sgl_eval_gsm8k_*", "metrics.json")))
inner = json.load(open(per_run[-1])) if per_run else {}

rec = {
    "run_id": run_id, "tag": tag, "dataset": "gsm8k", "backend": "sgl_eval",
    "shots": 0, "num_examples": int(n), "num_threads": int(threads),
    "max_tokens": int(mx),
    "sampling": {"temperature": 0.0, "top_p": 0.95, "min_p": 0.0,
                 "repetition_penalty": 1.0, "seed": None, "thinking": False,
                 "n_repeats": 1, "selection": "first-N rows in file order (no sample_seed)"},
    "score": metrics.get("score", inner.get("aggregate", {}).get("score")),
    "threshold": float(thres),
    "invalid_rate": metrics.get("invalid"),
    "error_rate": metrics.get("error_rate", inner.get("aggregate", {}).get("error_rate")),
    "wall_seconds": payload.get("wall_seconds"),
    "latency_seconds": metrics.get("latency"),
    "output_throughput_tps": metrics.get("output_throughput"),
    "eval_rc": int(rc),
    "base_url": base_url, "model": model,
    "generation_effective": inner.get("generation"),
    "num_examples_scored": inner.get("num_examples"),
    "sgl_eval_metrics_path": inner and per_run[-1],
    "log": log_path,
}
score = rec["score"]
rec["verdict"] = ("PASS" if isinstance(score, (int, float)) and score >= float(thres)
                  else ("FAIL" if isinstance(score, (int, float)) else "UNKNOWN"))
json.dump(rec, open(out_json, "w"), indent=2, ensure_ascii=False)
print("\n================ 结果 ================")
print(f"  run_id      : {rec['run_id']}")
print(f"  数据集      : gsm8k ({rec['shots']}-shot, backend={rec['backend']})")
print(f"  题目数      : {rec['num_examples']} (实际计分 {rec['num_examples_scored']})")
print(f"  阈值        : {rec['threshold']}")
print(f"  采样        : T={rec['sampling']['temperature']} top_p={rec['sampling']['top_p']} "
      f"seed={rec['sampling']['seed']} thinking={rec['sampling']['thinking']} reps={rec['sampling']['n_repeats']}")
print(f"  并发线程    : {rec['num_threads']}")
print(f"  ---- 准确率 : {rec['score']}  -> {rec['verdict']}")
print(f"  无效答案率  : {rec['invalid_rate']}")
print(f"  请求错误率  : {rec['error_rate']}")
print(f"  耗时        : {rec['wall_seconds']} s (评测内部 latency={rec['latency_seconds']})")
print(f"  吞吐        : {rec['output_throughput_tps']} tok/s")
print(f"  JSON        : {out_json}")
print(f"  日志        : {log_path}")
print(f"  sgl-eval 产物: {rec['sgl_eval_metrics_path']}")
print("=====================================")
PY

say "完成。结果: $RESULT_JSON"
say "日志: $RUN_LOG"
