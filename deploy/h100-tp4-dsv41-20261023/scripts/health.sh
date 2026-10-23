#!/usr/bin/env bash
# =============================================================================
# health.sh
#   对已启动的 Lsglang 服务做三层健康检查:
#     1) GET  /health            —— 引擎是否就绪
#     2) GET  /v1/models         —— 模型是否注册 (自动发现模型 id)
#     3) POST /v1/chat/completions —— 一次最小真实推理 (验证端到端可生成)
#   全程只发 HTTP 请求, 不额外占显存。默认 3 项全过才 exit 0。
#
# 用法:
#   ./health.sh                    # 默认 (就绪探测: 最多等 --wait 秒; 默认 0=不等)
#   ./health.sh --wait 60          # /health 未就绪时最多等 60s
#   ./health.sh --no-infer         # 跳过第 3 步 (只探活, 不推理)
#   ./health.sh --prompt "..." --max-tokens 16
#   ./health.sh --help
# 退出码: 0 全过 ; 1 有失败 ; 4 端口无人监听
# =============================================================================
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
: "${LSGLANG_ROOT:=/mnt/SSD01/models/.llm-env/Lsglang}"
CONF="${LSGLANG_CONF:-${LSGLANG_ROOT}/configs/dsv41_tp4_h100.conf}"
CONF_LOADED="(none — 内置默认)"
[[ -f "$CONF" ]] && { source "$CONF"; CONF_LOADED="$CONF"; }

: "${PORT:=18010}"
: "${HOST:=0.0.0.0}"
: "${RUN_DIR:=${LSGLANG_ROOT}/logs}"
: "${HEALTH_CURL_TIMEOUT_S:=5}"
: "${PID_FILE:=${RUN_DIR}/serve_dsv41_tp4_h100.pid}"

WAIT_S=0
DO_INFER=1
PROMPT="用一句话说明你是谁。"
MAX_TOKENS=32
MODEL_OVERRIDE=""

usage() {
  cat <<'USAGE_EOF'
health.sh
  对已启动的 Lsglang 服务做三层健康检查:
    1) GET  /health              —— 引擎是否就绪
    2) GET  /v1/models           —— 模型是否注册 (自动发现模型 id)
    3) POST /v1/chat/completions —— 一次最小真实推理
  全程只发 HTTP 请求, 不额外占显存。

用法:
  ./health.sh                    # 默认 (不等待; /health 未就绪即报错)
  ./health.sh --wait 60          # /health 未就绪时最多等 60s
  ./health.sh --no-infer         # 跳过第 3 步
  ./health.sh --prompt "..." --max-tokens 16
  ./health.sh --model <id>       # 指定模型 id
  ./health.sh --port 18010
  ./health.sh --help

退出码: 0 全过 ; 1 有失败 ; 4 端口无人监听
USAGE_EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait)       WAIT_S="$2"; shift ;;
    --no-infer)   DO_INFER=0 ;;
    --prompt)     PROMPT="$2"; shift ;;
    --max-tokens) MAX_TOKENS="$2"; shift ;;
    --model)      MODEL_OVERRIDE="$2"; shift ;;
    --port)       PORT="$2"; shift ;;
    --help|-h)    usage; exit 0 ;;
    *) echo "未知参数: $1 (用 --help)" >&2; exit 2 ;;
  esac
  shift
done

BASE="http://127.0.0.1:${PORT}"
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }
hr()  { printf '%s\n' "--------------------------------------------------------------"; }
FAILED=0
pass() { printf '  [ OK ]   %s\n' "$*"; }
fail() { printf '  [FAIL]   %s\n' "$*"; FAILED=$((FAILED + 1)); }

mkdir -p "$RUN_DIR"
HLOG="${RUN_DIR}/health_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$HLOG") 2>&1

echo "=============================================================="
echo " Lsglang health · DeepSeek-V4.1-Flash-1017 · TP4 · ${BASE}"
echo " 时间: $(ts)   conf: $CONF_LOADED"
echo " 本次记录: $HLOG"
echo "=============================================================="

# 0. 端口
echo
echo "0) 前置: 端口监听"
if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${PORT}" 2>/dev/null | awk 'NR>1' | grep -q .; then
  pass "端口 ${PORT} 正在监听"
elif curl -sf -m 2 "$BASE/health" >/dev/null 2>&1; then
  pass "端口 ${PORT} 可连通"
else
  fail "端口 ${PORT} 无监听。服务未启动? 先跑 ./serve_dsv41_tp4_h100.sh 或 ./status.sh"
  echo
  echo "结果: 1 项失败 —— 服务未运行。"
  hr
  exit 4
fi

# 1. /health
echo
echo "1) GET /health"
t0=$(date +%s.%N)
health_code=""
deadline=$(( SECONDS + WAIT_S ))
while :; do
  health_code="$(curl -s -o /dev/null -w '%{http_code}' -m "$HEALTH_CURL_TIMEOUT_S" "$BASE/health" 2>/dev/null || echo 000)"
  [[ "$health_code" == "200" ]] && break
  (( SECONDS >= deadline )) && break
  log "  /health 返回 ${health_code}, 重试中… (剩余 $(( deadline - SECONDS ))s)"
  sleep 5
done
t1=$(date +%s.%N)
dt="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')"
if [[ "$health_code" == "200" ]]; then
  pass "GET /health -> 200 (${dt}s)"
else
  fail "GET /health -> ${health_code} (${dt}s)。引擎未就绪或加载中; 用 ./status.sh 看日志。"
fi

# 2. /v1/models
echo
echo "2) GET /v1/models"
models_json="$(curl -s -m "$HEALTH_CURL_TIMEOUT_S" "$BASE/v1/models" 2>/dev/null || true)"
if [[ -n "$models_json" ]] && printf '%s' "$models_json" | grep -q '"object"'; then
  ids="$(printf '%s' "$models_json" | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*"\([^"]*\)"$/\1/' || true)"
  n="$(printf '%s\n' "$ids" | grep -c . || true)"
  pass "GET /v1/models -> 200, ${n} 个模型"
  printf '%s\n' "$ids" | sed 's/^/           id: /'
  if [[ -n "$MODEL_OVERRIDE" ]]; then MODEL="$MODEL_OVERRIDE"; else MODEL="$(printf '%s\n' "$ids" | head -1)"; fi
  [[ -n "$MODEL" ]] && pass "使用模型 id: ${MODEL}" || fail "无法确定模型 id"
else
  fail "GET /v1/models 无有效响应: $(printf '%s' "$models_json" | cut -c1-160)"
  MODEL="$MODEL_OVERRIDE"
fi

# 3. 最小 chat 推理
echo
echo "3) POST /v1/chat/completions (最小真实推理)"
if [[ "$DO_INFER" -eq 0 ]]; then
  log "  (--no-infer: 跳过)"
elif [[ -z "$MODEL" ]]; then
  fail "无模型 id, 跳过推理"
elif [[ "$health_code" != "200" ]]; then
  log "  引擎未就绪, 跳过推理"
else
  req="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"temperature":0,"stream":false}' \
          "$MODEL" "$PROMPT" "$MAX_TOKENS")"
  resp_file="${RUN_DIR}/health_response_$(date '+%Y%m%d_%H%M%S').json"
  t0=$(date +%s.%N)
  code="$(curl -s -o "$resp_file" -w '%{http_code}' -m 300 \
            -H 'Content-Type: application/json' \
            -H 'Authorization: Bearer EMPTY' \
            -d "$req" "$BASE/v1/chat/completions" 2>/dev/null || echo 000)"
  t1=$(date +%s.%N)
  dt="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')"
  if [[ "$code" == "200" ]]; then
    content="$(grep -oE '"content"[[:space:]]*:[[:space:]]*"[^"]*"' "$resp_file" | head -1 | sed 's/.*:[[:space:]]*"\(.*\)"$/\1/' || true)"
    usage="$(grep -oE '"usage"[[:space:]]*:[[:space:]]*\{[^}]*\}' "$resp_file" | head -1 | tr -d '\n' | tr -s ' ' || true)"
    pass "POST /v1/chat/completions -> 200 (${dt}s)"
    printf '           回复: %s\n' "${content:0:300}"
    [[ -n "$usage" ]] && printf '           usage: %s\n' "$usage"
    printf '           原始响应: %s\n' "$resp_file"
    if [[ -z "$content" ]]; then
      if grep -qE '"(reasoning_content|tool_calls)"' "$resp_file"; then
        log "  (content 为空但存在 reasoning_content/tool_calls —— 属该模型正常输出形态)"
      else
        fail "响应 200 但未解析到 content, 见 $resp_file"
      fi
    fi
  else
    fail "POST /v1/chat/completions -> ${code} (${dt}s), 响应见 $resp_file"
    head -c 600 "$resp_file" | sed 's/^/           /'
  fi
fi

echo
hr
if [[ "$FAILED" -eq 0 ]]; then
  echo " 结果: 全部通过 ✅   ${BASE}/v1  (Authorization: Bearer <任意非空>)"
  hr; exit 0
else
  echo " 结果: ${FAILED} 项失败 ❌"
  echo " 排查: ./status.sh ; 日志尾 tail -n 80 \"${LOG_FILE_RECORDED:-${RUN_DIR}/serve_dsv41_tp4_h100.log}\""
  hr; exit 1
fi
