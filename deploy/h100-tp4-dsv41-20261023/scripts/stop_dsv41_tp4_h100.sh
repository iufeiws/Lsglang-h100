#!/usr/bin/env bash
# =============================================================================
# stop_dsv41_tp4_h100.sh
#   优雅停止 serve_dsv41_tp4_h100.sh 拉起的 Lsglang 服务。
#
# 策略: SIGTERM(进程组) -> 轮询最多 ${STOP_SIGTERM_WAIT_S}s (默认 60) -> 仍未退则 SIGKILL(进程组)
# 安全: 只操作 logs/serve_dsv41_tp4_h100.pid 里记录的 PID/PGID, 且二次校验该进程确实是
#       本服务的 `sglang serve --model <权重路径>`; 不匹配则拒绝杀。
# 幂等: 未运行 / PID 文件缺失 / 陈旧 → 打印说明并以 0 退出 (除非 --fail-if-not-running)。
#
# 用法:
#   ./stop_dsv41_tp4_h100.sh                # 默认
#   ./stop_dsv41_tp4_h100.sh --timeout 120  # 改 SIGTERM 等待秒数
#   ./stop_dsv41_tp4_h100.sh --force        # 跳过 cmdline 校验 (仅在确认 PID 归属时用)
#   ./stop_dsv41_tp4_h100.sh --help
# =============================================================================
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
: "${LSGLANG_ROOT:=/mnt/SSD01/models/.llm-env/Lsglang}"
CONF="${LSGLANG_CONF:-${LSGLANG_ROOT}/configs/dsv41_tp4_h100.conf}"
CONF_LOADED="(none — 内置默认)"
[[ -f "$CONF" ]] && { source "$CONF"; CONF_LOADED="$CONF"; }

: "${RUN_DIR:=${LSGLANG_ROOT}/logs}"
: "${PID_FILE:=${RUN_DIR}/serve_dsv41_tp4_h100.pid}"
: "${STOP_SIGTERM_WAIT_S:=60}"
: "${PORT:=18010}"
: "${MODEL_PATH:=/mnt/SSD01/models/DeepSeek/DeepSeek-V4.1-Flash-1017}"
: "${STATUS_TAIL_LINES:=40}"

FORCE=0
FAIL_IF_NOT_RUNNING=0
usage() {
  cat <<'USAGE_EOF'
stop_dsv41_tp4_h100.sh
  优雅停止 serve_dsv41_tp4_h100.sh 拉起的 Lsglang 服务。
  策略: SIGTERM(进程组) → 等 60s → SIGKILL(进程组)
  安全: 只杀 PID 文件里记录的 PID/PGID, 且校验进程 cmdline 确为本服务。

用法:
  ./stop_dsv41_tp4_h100.sh                # 默认
  ./stop_dsv41_tp4_h100.sh --timeout 120  # 改 SIGTERM 等待秒数
  ./stop_dsv41_tp4_h100.sh --force        # 跳过 cmdline 校验
  ./stop_dsv41_tp4_h100.sh --fail-if-not-running
  ./stop_dsv41_tp4_h100.sh --help
USAGE_EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout)  STOP_SIGTERM_WAIT_S="$2"; shift ;;
    --force)    FORCE=1 ;;
    --fail-if-not-running) FAIL_IF_NOT_RUNNING=1 ;;
    --help|-h)  usage; exit 0 ;;
    *) echo "未知参数: $1 (用 --help)" >&2; exit 2 ;;
  esac
  shift
done

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { printf '[%s] ERROR: %s\n' "$(ts)" "$*" >&2; exit 1; }

echo "=============================================================="
echo " 停止 Lsglang serve · DeepSeek-V4.1-Flash-1017 · TP4 (端口 ${PORT})"
echo " conf      : $CONF_LOADED"
echo " PID 文件  : $PID_FILE"
echo " 策略      : SIGTERM(进程组) → 等 ${STOP_SIGTERM_WAIT_S}s → SIGKILL(进程组)"
echo "=============================================================="

PID=""; PGID=""; START_TS=""; LOG_FILE_RECORDED=""; SNAP_FILE=""
if [[ -f "$PID_FILE" ]]; then
  # shellcheck source=/dev/null
  . "$PID_FILE"
else
  log "PID 文件不存在: $PID_FILE"
  log "服务未由本套件启动 (或已被停止)。无操作。"
  (( FAIL_IF_NOT_RUNNING == 1 )) && exit 1
  exit 0
fi

[[ -n "${PID:-}" ]] || die "PID 文件内容异常 (无 PID): $PID_FILE"

if ! kill -0 "$PID" 2>/dev/null; then
  log "PID ${PID} 已不存在 (进程已退出)。清理 PID 文件。"
  rm -f "$PID_FILE"
  (( FAIL_IF_NOT_RUNNING == 1 )) && exit 1
  exit 0
fi

cmdline="$(tr '\0' ' ' < "/proc/${PID}/cmdline" 2>/dev/null || true)"
if [[ -z "$cmdline" ]]; then
  log "无法读取 /proc/${PID}/cmdline (权限或进程刚退出)"
  [[ "$FORCE" -eq 1 ]] || die "无法校验 PID ${PID} 归属, 拒绝杀。确认可用 --force。"
else
  log "PID ${PID} cmdline: $(printf '%s' "$cmdline" | cut -c1-160)"
  if [[ "$FORCE" -ne 1 ]]; then
    ok=1
    printf '%s' "$cmdline" | grep -q -- 'sglang' || { ok=0; log "校验失败: cmdline 不含 'sglang'"; }
    printf '%s' "$cmdline" | grep -q -- 'serve'  || { ok=0; log "校验失败: cmdline 不含 'serve'"; }
    printf '%s' "$cmdline" | grep -qF -- "$MODEL_PATH" || { ok=0; log "校验失败: cmdline 不含权重路径 $MODEL_PATH"; }
    (( ok == 1 )) || die "PID ${PID} 不是本套件启动的服务, 拒绝杀 (确认无误可加 --force)"
    log "归属校验通过 ✓"
  else
    log "已加 --force: 跳过归属校验"
  fi
fi

target_kind="pid"; target="$PID"
if [[ -n "${PGID:-}" ]] && [[ "$PGID" =~ ^[0-9]+$ ]] && [[ "$PGID" == "$PID" ]]; then
  target_kind="pgid"; target="$PGID"
  member_count="$(ps -o pid= -g "$PGID" 2>/dev/null | grep -c . || true)"
  log "将向进程组 -${PGID} 发信号 (setsid 子会话, PGID==PID); 成员数 ${member_count}"
else
  log "PGID 未记录或与 PID 不一致 (PGID='${PGID:-}'), 只对单个 PID ${PID} 发信号"
fi

log "发送 SIGTERM 到 $( [[ "$target_kind" == pgid ]] && echo "进程组 -${target}" || echo "PID ${target}" )…"
if [[ "$target_kind" == "pgid" ]]; then
  kill -TERM -- "-${target}" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
else
  kill -TERM "$PID" 2>/dev/null || true
fi

waited=0; graceful_ok=0
while (( waited < STOP_SIGTERM_WAIT_S )); do
  if ! kill -0 "$PID" 2>/dev/null; then graceful_ok=1; break; fi
  if (( waited % 5 == 0 )); then
    mem="$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ' || echo '')"
    [[ -n "$mem" ]] && log "  等待退出… ${waited}s / ${STOP_SIGTERM_WAIT_S}s (PID ${PID} RSS=$(( mem / 1024 )) MiB)"
  fi
  sleep 1; waited=$(( waited + 1 ))
done

if (( graceful_ok == 0 )); then
  log "等满 ${STOP_SIGTERM_WAIT_S}s 仍未退出 → SIGKILL"
  if [[ "$target_kind" == "pgid" ]]; then
    kill -KILL -- "-${target}" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
  else
    kill -KILL "$PID" 2>/dev/null || true
  fi
  for _ in $(seq 1 15); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$PID" 2>/dev/null && die "SIGKILL 后 PID ${PID} 仍在 (D 状态/僵死?)。请人工介入: ps -fp ${PID}"
  log "SIGKILL 后已退出 (可能残留部分显存, 请用 ./status.sh 复核)"
else
  log "✅ SIGTERM 后 ${waited}s 内优雅退出"
fi

leftover="$(ps -eo pid=,pgid=,cmd= 2>/dev/null | awk -v g="${PGID:-$PID}" '$2==g' | grep -v "awk\|ps -eo" || true)"
if [[ -n "$leftover" ]]; then
  log "⚠ 同进程组仍有残留 (可能是我方 worker/scheduler, 请人工确认):"
  printf '%s\n' "$leftover" | sed 's/^/    /'
else
  log "同进程组无残留 ✓"
fi

rm -f "$PID_FILE"
log "已删除 PID 文件: $PID_FILE"

sleep 1
if command -v ss >/dev/null 2>&1; then
  if ss -ltn "sport = :${PORT}" 2>/dev/null | awk 'NR>1' | grep -q .; then
    log "⚠ 端口 ${PORT} 仍在监听 (TIME_WAIT 属正常; 若持续占用请查 status.sh)"
  else
    log "端口 ${PORT} 已释放 ✓"
  fi
fi

if [[ -n "${LOG_FILE_RECORDED:-}" && -f "$LOG_FILE_RECORDED" ]]; then
  log "日志尾部 ($LOG_FILE_RECORDED):"
  tail -n "$STATUS_TAIL_LINES" "$LOG_FILE_RECORDED" | sed 's/^/    /' || true
fi

log "停止流程结束。"
