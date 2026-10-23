#!/usr/bin/env bash
# =============================================================================
# run_pool6row_bench.sh  --  6-row pool contract: one measured round.
#
#   Drives scripts/bench256_streams.py against the LIVE service on :18010 with a
#   plan built from (streams, budget_s) pairs, writes every artefact under
#   logs/pool6row_<RUN_TS>/ and hard-fails if a foreign client is connected.
#
# Usage:
#   run_pool6row_bench.sh --tag <name> --plan "1:120,2:90,4:90"
#                         [--run-ts <ts>] [--serve-log <path>] [--reps 1]
#   plan item = <streams>:<budget_seconds>
# =============================================================================
set -Eeuo pipefail
ROOT=/mnt/SSD01/models/.llm-env/Lsglang
SCRIPTS="$ROOT/scripts"
PY="$ROOT/env/bin/python"
PORT=18010

TAG=""; PLAN=""; RUN_TS=""; SERVE_LOG=""; REPS=1
args=("$@")
for i in "${!args[@]}"; do
  case "${args[$i]}" in
    --tag) TAG="${args[$i+1]}" ;;
    --plan) PLAN="${args[$i+1]}" ;;
    --run-ts) RUN_TS="${args[$i+1]}" ;;
    --serve-log) SERVE_LOG="${args[$i+1]}" ;;
    --reps) REPS="${args[$i+1]}" ;;
  esac
done
[[ -n "$TAG" && -n "$PLAN" ]] || { echo "need --tag and --plan" >&2; exit 2; }
[[ -n "$RUN_TS" ]] || RUN_TS="$(cat "$ROOT/logs/.pool6row_run_ts" 2>/dev/null || date +%Y%m%d_%H%M)"
[[ -n "$SERVE_LOG" ]] || SERVE_LOG="$(ls -t "$ROOT"/logs/serve_dsv41_tp4_h100_*.log | head -1)"

OUT="$ROOT/logs/pool6row_$RUN_TS"
mkdir -p "$OUT"

conns=$(ss -tn state established "( sport = :$PORT or dport = :$PORT )" 2>/dev/null | tail -n +2 | wc -l)
if [[ "$conns" != "0" ]]; then
  echo "[run_pool6row] REFUSE: $conns established connection(s) to :$PORT (must be 0)" >&2
  ss -tnp state established "( sport = :$PORT or dport = :$PORT )" >&2 || true
  exit 3
fi

# plan -> bench args
BENCH_PLAN=""
for item in ${PLAN//,/ }; do
  K="${item%%:*}"; B="${item##*:}"
  BENCH_PLAN+="${BENCH_PLAN:+,}single:${K}:100000:${B}"
done

TS="$(date +%Y%m%d_%H%M%S)"
PREFIX="bench-pool6row-${TAG}"
echo "[run_pool6row] tag=$TAG plan=$PLAN bench_plan=$BENCH_PLAN reps=$REPS"
echo "[run_pool6row] serve_log=$SERVE_LOG outdir=$OUT ts=$TS"

for r in $(seq 1 "$REPS"); do
  for item in ${PLAN//,/ }; do
    K="${item%%:*}"; B="${item##*:}"
    conns=$(ss -tn state established "( sport = :$PORT or dport = :$PORT )" 2>/dev/null | tail -n +2 | wc -l)
    [[ "$conns" == "0" ]] || { echo "[run_pool6row] REFUSE mid-round: $conns conns" >&2; exit 3; }
    echo "[run_pool6row] === rep$r streams=$K budget=${B}s $(date +%H:%M:%S) ==="
    "$PY" "$SCRIPTS/bench256_streams.py" \
      --plan "single:${K}:100000:${B}" \
      --outdir "$OUT" --ts "${TS}-K${K}-r${r}" \
      --prefix "$PREFIX" --serve-log "$SERVE_LOG" \
      2>&1 | tail -25
  done
done
echo "[run_pool6row] done: $OUT"
ls -la "$OUT" | tail -20
