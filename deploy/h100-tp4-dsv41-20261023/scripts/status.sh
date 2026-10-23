#!/usr/bin/env bash
# =============================================================================
# status.sh
#   DeepSeek-V4.1-Flash-1017 · Lsglang TP4 (物理卡 4-7) 运行状态总览。
#   输出: 进程 / 端口 / 目标卡显存 / NUMA / /health / lk_moe 池关键行 / 日志尾部。
#   全程只读, 不占显存, 可随时执行。
#
# 用法:
#   ./status.sh                # 完整状态
#   ./status.sh --tail 100     # 改日志尾部行数 (默认 40)
#   ./status.sh --brief        # 精简
#   ./status.sh --help
# 退出码: 0 = 在跑且 /health OK ; 3 = 在跑但 /health 未就绪 ; 4 = 未运行
# =============================================================================
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
: "${LSGLANG_ROOT:=/mnt/SSD01/models/.llm-env/Lsglang}"
CONF="${LSGLANG_CONF:-${LSGLANG_ROOT}/configs/dsv41_tp4_h100.conf}"
CONF_LOADED="(none — 内置默认)"
[[ -f "$CONF" ]] && { source "$CONF"; CONF_LOADED="$CONF"; }

: "${RUN_DIR:=${LSGLANG_ROOT}/logs}"
: "${PID_FILE:=${RUN_DIR}/serve_dsv41_tp4_h100.pid}"
: "${PORT:=18010}"
: "${HOST:=0.0.0.0}"
: "${HEALTH_CURL_TIMEOUT_S:=5}"
: "${STATUS_TAIL_LINES:=40}"
: "${LOG_PREFIX:=serve_dsv41_tp4_h100}"
: "${LK_POOL_SLOTS:=288}"
# 数组不能用 ${VAR:=(…)} 兜底 (那会得到字符串而非数组), 显式判存在
if ! declare -p GPU_EXPECTED_INDEXES >/dev/null 2>&1; then GPU_EXPECTED_INDEXES=(4 5 6 7); fi
if ! declare -p GPU_UUIDS >/dev/null 2>&1; then
  GPU_UUIDS=(
    GPU-7561b28f-b049-bbdb-31f8-16efeb06663d
    GPU-fb59db94-0e4e-4c4f-8ae5-4d0fdd1e0f0b
    GPU-3c5d7267-445c-4679-bce9-503bedd0c920
    GPU-3c0f2856-a6e3-b9d5-fd10-a08f55579fff
  )
fi

BRIEF=0
usage() {
  cat <<'USAGE_EOF'
status.sh
  DeepSeek-V4.1-Flash-1017 · Lsglang TP4 (物理卡 4-7) 运行状态总览。
  输出: 进程 / 端口 / 目标卡显存 / NUMA / /health / lk_moe 池关键行 / 日志尾部。
  全程只读, 不占显存, 可随时执行。

用法:
  ./status.sh                # 完整状态
  ./status.sh --tail 100     # 改日志尾部行数 (默认 40)
  ./status.sh --brief        # 精简
  ./status.sh --help

退出码: 0 = 在跑且 /health OK ; 3 = 在跑但 /health 未就绪 ; 4 = 未运行
USAGE_EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tail)  STATUS_TAIL_LINES="$2"; shift ;;
    --brief) BRIEF=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "未知参数: $1 (用 --help)" >&2; exit 2 ;;
  esac
  shift
done

ts() { date '+%Y-%m-%d %H:%M:%S'; }
hr()  { printf '%s\n' "--------------------------------------------------------------"; }
sec() { printf '\n=== %s ===\n' "$*"; }

echo "=============================================================="
echo " Lsglang status · DeepSeek-V4.1-Flash-1017 · TP4 × 4×H100 · 端口 ${PORT}"
echo " 采样时间: $(ts)"
echo " conf: $CONF_LOADED"
echo "=============================================================="

# -----------------------------------------------------------------------------
sec "1. 进程 (PID 文件 $PID_FILE)"
PID=""; PGID=""; START_TS=""; LOG_FILE_RECORDED=""; SNAP_FILE=""
RUNNING=0
if [[ -f "$PID_FILE" ]]; then
  # shellcheck source=/dev/null
  . "$PID_FILE"
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    RUNNING=1
    etime="$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ' || echo '?')"
    rss_mib="$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ' || echo 0)"
    pcpu="$(ps -o %cpu= -p "$PID" 2>/dev/null | tr -d ' ' || echo '?')"
    nlwp="$(ps -o nlwp= -p "$PID" 2>/dev/null | tr -d ' ' || echo '?')"
    printf '  状态      : RUNNING\n'
    printf '  PID/PGID  : %s / %s\n' "$PID" "${PGID:-?}"
    printf '  启动时间  : %s (已运行 %s)\n' "${START_TS:-?}" "$etime"
    printf '  RSS       : %s MiB   CPU%%: %s   线程数: %s\n' "$(( rss_mib / 1024 ))" "$pcpu" "$nlwp"
    printf '  日志      : %s\n' "${LOG_FILE_RECORDED:-?}"
    printf '  配置快照  : %s\n' "${SNAP_FILE:-?}"
    printf '  cmdline   : %s\n' "$(tr '\0' ' ' < "/proc/${PID}/cmdline" 2>/dev/null | cut -c1-200 || true)"
    # 同进程组的子进程 (sglang 有 scheduler / detokenizer 子进程)
    if [[ -n "${PGID:-}" ]]; then
      kids="$(ps -o pid=,rss=,comm= -g "$PGID" 2>/dev/null | grep -v "^\s*${PID}\b" || true)"
      if [[ -n "$kids" ]]; then
        printf '  同组子进程:\n'
        printf '%s\n' "$kids" | sed 's/^/      /'
      fi
    fi
  else
    printf '  状态      : PID 文件存在但进程已死 (陈旧)\n'
    printf '  PID       : %s\n' "${PID:-?}"
  fi
else
  printf '  状态      : NOT RUNNING (无 PID 文件)\n'
fi

if command -v pgrep >/dev/null 2>&1; then
  stray="$(pgrep -af "sglang" 2>/dev/null | grep -F -- "$PORT" || true)"
  if [[ -n "$stray" && "$RUNNING" -eq 0 ]]; then
    printf '  ⚠ 发现非本套件记录的 sglang 进程 (端口 %s):\n' "$PORT"
    printf '%s\n' "$stray" | sed 's/^/      /'
  fi
fi

# -----------------------------------------------------------------------------
sec "2. 端口 ${PORT}"
PORT_OK=0
if command -v ss >/dev/null 2>&1; then
  line="$(ss -ltnp "sport = :${PORT}" 2>/dev/null | awk 'NR>1' || true)"
  if [[ -n "$line" ]]; then PORT_OK=1; printf '  LISTEN 中:\n'; printf '%s\n' "$line" | sed 's/^/      /'
  else printf '  未监听\n'; fi
else
  printf '  (无 ss, 跳过)\n'
fi

# -----------------------------------------------------------------------------
sec "3. HTTP 探活"
HEALTH_OK=0
if curl -sf -m "$HEALTH_CURL_TIMEOUT_S" "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  HEALTH_OK=1; printf '  /health            : 200 OK  (就绪)\n'
else
  printf '  /health            : 未响应 / 未就绪\n'
fi
if curl -sf -m "$HEALTH_CURL_TIMEOUT_S" "http://127.0.0.1:${PORT}/health_generate" >/dev/null 2>&1; then
  printf '  /health_generate   : 200 OK  (可生成)\n'
else
  printf '  /health_generate   : 未响应 / 未就绪\n'
fi
mids="$(curl -sf -m "$HEALTH_CURL_TIMEOUT_S" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
        | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*"\([^"]*\)"$/\1/' | paste -sd, - || true)"
if [[ -n "$mids" ]]; then printf '  /v1/models         : OK (id: %s)\n' "$mids"
else printf '  /v1/models         : 未响应\n'; fi

# -----------------------------------------------------------------------------
sec "4. GPU (物理卡 ${GPU_EXPECTED_INDEXES[*]}, 只读查询, 不占显存)"
if command -v nvidia-smi >/dev/null 2>&1; then
  q="$(nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "$q" ]]; then
    printf '  phys  uuid                                     used/total MiB    util%%  temp\n'
    tot_used=0
    for i in "${!GPU_UUIDS[@]}"; do
      u="${GPU_UUIDS[$i]}"; exp="${GPU_EXPECTED_INDEXES[$i]}"
      row="$(printf '%s\n' "$q" | awk -F', *' -v u="$u" '$2==u{print $1", "$3", "$4", "$5", "$6}' | head -1)"
      if [[ -n "$row" ]]; then
        IFS=', ' read -r idx used total util temp <<<"$row"
        mark=""; [[ "$idx" != "$exp" ]] && mark="  ⚠期望 GPU${exp}"
        printf '  %-4s  %-40s  %6s/%-6s  %5s  %4s%s\n' "$idx" "$u" "$used" "$total" "$util" "$temp" "$mark"
        tot_used=$(( tot_used + used ))
      else
        printf '  %-4s  %-40s  <UUID 未找到>\n' "$exp" "$u"
      fi
    done
    printf '  4 卡合计已用: %s MiB (%.1f GiB)  ——  池预算参考: %s 槽 → 约 %s GiB\n' \
      "$tot_used" "$(awk -v a="$tot_used" 'BEGIN{printf "%.1f", a/1024}')" \
      "${LK_POOL_SLOTS}" "$(( LK_POOL_SLOTS * 40 * 17280 / 1024 / 1024 ))"
    apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "$apps" ]]; then
      mine="$(printf '%s\n' "$apps" | while IFS= read -r l; do
                for u in "${GPU_UUIDS[@]}"; do [[ "$l" == *"$u"* ]] && { printf '%s\n' "$l"; break; }; done
              done || true)"
      printf '\n  卡上的计算进程 (仅目标 4 卡):\n'
      if [[ -n "$mine" ]]; then printf '%s\n' "$mine" | sed 's/^/      /'; else printf '      (无)\n'; fi
    fi
  else
    printf '  nvidia-smi 查询失败\n'
  fi
  printf '\n  参照: 物理卡 0/1 属他人占用, 本套件不得触碰。\n'
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{printf "      GPU%-2s used=%s MiB\n", $1, $2}'
else
  printf '  无 nvidia-smi\n'
fi

# -----------------------------------------------------------------------------
sec "5. NUMA / CPU"
if command -v numactl >/dev/null 2>&1; then
  numactl -H 2>/dev/null | grep -E '^node [0-9]+ (cpus|size|free)' | sed 's/^/  /'
  printf '  (卡 4-7 均在 node1; LK_NUMA_INTERLEAVE=1 用于跨节点交织主机内存)\n'
else
  printf '  无 numactl\n'
fi

if [[ "$BRIEF" -eq 1 ]]; then
  echo; hr
  printf 'BRIEF: running=%s port=%s health=%s\n' "$RUNNING" "$PORT_OK" "$HEALTH_OK"
  hr
  if [[ "$RUNNING" -eq 1 && "$HEALTH_OK" -eq 1 ]]; then exit 0
  elif [[ "$RUNNING" -eq 1 ]]; then exit 3
  else exit 4; fi
fi

# -----------------------------------------------------------------------------
sec "6. 日志"
LATEST=""
if [[ -n "${LOG_FILE_RECORDED:-}" && -f "$LOG_FILE_RECORDED" ]]; then
  LATEST="$LOG_FILE_RECORDED"
else
  LATEST="$(ls -1t "${RUN_DIR}/${LOG_PREFIX}"_*.log 2>/dev/null | head -1 || true)"
fi
if [[ -n "$LATEST" && -f "$LATEST" ]]; then
  printf '  文件      : %s\n' "$LATEST"
  printf '  大小/行数 : %s / %s 行\n' "$(du -h "$LATEST" | cut -f1)" "$(wc -l < "$LATEST")"
  printf '  最后修改  : %s\n' "$(date -r "$LATEST" '+%Y-%m-%d %H:%M:%S')"
  printf '\n  --- 尾部 %s 行 ---\n' "$STATUS_TAIL_LINES"
  tail -n "$STATUS_TAIL_LINES" "$LATEST" | sed 's/^/  | /'
  printf '\n  --- 关键事件 ---\n'
  hits="$(grep -nE "Traceback|ERROR|Error:|CUDA out of memory|OOM|Killed|\[pool\]|lk_moe|hybrid|KV Cache is allocated|max_total_num_tokens|The server is fired up|Available memory" "$LATEST" 2>/dev/null | tail -n 30 || true)"
  if [[ -n "$hits" ]]; then printf '%s\n' "$hits" | sed 's/^/  | /'; else printf '  | (无匹配)\n'; fi

  printf '\n  --- W4A8 生效串 (lead 必读 1) ---\n'
  # ⚠ 排除 serve 脚本自己写的 "自检断言" 横幅 (该行原文引用了这两个串 → 否则双向假阳性)
  w4a8="$(grep -v '自检断言' "$LATEST" 2>/dev/null | grep -cE "CUTLASS W4A8 \(layer: " || true)"
  w4a16="$(grep -v '自检断言' "$LATEST" 2>/dev/null | grep -cE "CUTLASS W4A16 \(layer: " || true)"
  printf '  CUTLASS W4A8 行数 = %s (期望 ≈20, 每个 resident 层一行)\n' "$w4a8"
  printf '  CUTLASS W4A16 行数 = %s (必须 0)\n' "$w4a16"
  if (( w4a16 > 0 )); then printf '  [ALARM] 出现 W4A16 —— 开关未生效!\n'; fi
  grep -v '自检断言' "$LATEST" 2>/dev/null | grep -oE "Preparing DSv4 MXFP4 experts for FlashInfer [^(]*" | sort | uniq -c | sed 's/^/  | /' || true

  printf '\n  --- lk_moe 池 (lead 必读 2): slab / 逐层初始化 ---\n'
  grep -oE '\[pool\] slab:[^(]*\([^)]*\)' "$LATEST" 2>/dev/null | tail -3 | sed 's/^/  | /' || true
  n_cpu="$(grep -c 'Initialized lk_moe with .* \[CPU\]' "$LATEST" 2>/dev/null || true)"
  n_gpu="$(grep -c 'Initialized lk_moe with .* \[GPU\]' "$LATEST" 2>/dev/null || true)"
  printf '  Initialized lk_moe [CPU] 行数 = %s (期望 20 = 池层)\n' "$n_cpu"
  printf '  Initialized lk_moe [GPU] 行数 = %s (期望 20 = resident 层)\n' "$n_gpu"
  printf '  ⚠ 注: [GPU] 分支未真正构造 lk_moe (layer.py:2010-2013 直接 return), 勿读作"已初始化"\n'

  printf '\n  --- CUDA graph (lead 必读 3): 图捕获真实开销 ---\n'
  gsum="$(grep -oE 'Capture .* end\..*mem usage=([0-9.]+) GB' "$LATEST" 2>/dev/null | tail -10 || true)"
  if [[ -n "$gsum" ]]; then
    printf '%s\n' "$gsum" | sed 's/^/  | /'
    tot="$(printf '%s\n' "$gsum" | grep -oE 'mem usage=([0-9.]+) GB' | grep -oE '[0-9.]+' | awk '{s+=$1} END{printf "%.2f", s}')"
    printf '  图捕获 mem usage 合计(尾部10条) = %s GB   ⚠ 余量约 10 GB/卡\n' "${tot:-0}"
    awk -v t="${tot:-0}" 'BEGIN{ if (t+0 > 10) print "  [ALARM] 图捕获已超 ~10 GB 余量 —— 有 OOM 风险"; }'
  else
    printf '  | (未找到 Capture 行; 可能是启动未到捕获阶段)\n'
  fi
  grep -nE "Disable (prefill|decode) CUDA graph|cuda-graph-max-bs|graph_backend" "$LATEST" 2>/dev/null | tail -5 | sed 's/^/  | /' || true

  printf '\n  --- KV / 内存 breakdown (lead 必读 4) ---\n'
  grep -oE 'max_total_num_tokens=[0-9]+[^|]*' "$LATEST" 2>/dev/null | tail -2 | sed 's/^/  | /' || true
  grep -oE 'DeepGEMM masked layout budget:[^.]*\.' "$LATEST" 2>/dev/null | tail -2 | sed 's/^/  | /' || true
  printf '  KV/token = 1630 + ratio×23400; ratio=0.005 → 1747 B; 4×1M 需 6.83 GiB\n'

  printf '\n  --- Engram host table (lead 必读 5) ---\n'
  grep -oE 'engram host table [^:]*: [^|]*' "$LATEST" 2>/dev/null | tail -3 | sed 's/^/  | /' || true
  if grep -q "No huge pages: expect ~10x slower lookups" "$LATEST" 2>/dev/null; then
    printf '  [WARN] 出现 "No huge pages: expect ~10x slower lookups" —— 生效但 lookup 退化约 10×\n'
  fi

  printf '\n  --- lk_moe 池命中率 (需 LK_POOL_LOG>0) ---\n'
  ph="$(grep -oE '\[pool\][^|]*hit=[0-9.]+%?' "$LATEST" 2>/dev/null | tail -n 8 || true)"
  if [[ -n "$ph" ]]; then printf '%s\n' "$ph" | sed 's/^/  | /'; else printf '  | (无; 设 LK_POOL_LOG=100 后才会打)\n'; fi
else
  printf '  未找到日志文件 (%s/%s_*.log)\n' "$RUN_DIR" "$LOG_PREFIX"
fi

echo; hr
printf ' 总结: '
if [[ "$RUNNING" -eq 1 && "$HEALTH_OK" -eq 1 ]]; then
  printf 'RUNNING + 就绪  →  http://%s:%s/v1  (Authorization: Bearer <任意非空>)\n' "$HOST" "$PORT"
  hr; exit 0
elif [[ "$RUNNING" -eq 1 ]]; then
  printf 'RUNNING 但 /health 未就绪 (可能仍在加载 475 GiB 权重 + 上池; 看上面日志)\n'
  hr; exit 3
else
  printf 'NOT RUNNING\n'
  hr; exit 4
fi
